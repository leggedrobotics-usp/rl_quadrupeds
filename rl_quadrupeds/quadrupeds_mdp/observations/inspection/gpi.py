from typing import Sequence, Tuple
import torch

class OnlineGaussianProcessRFF:
    """
    Fast batched GP approximation using Random Fourier Features (RFF).
    We run E × M independent models, fully vectorized.

    Posterior is Bayesian linear regression in feature space (F features, F << N):
        w ~ N(0, I)
        y = phi(x)^T w + eps
        A = (1/sigma^2) * Phi^T Phi + I
        b = (1/sigma^2) * Phi^T y

    Complexity per step is O(E * M * F^3) for the Cholesky (small F)
    and O(E * M * T * F) for prediction, independent of total history length.

    Key additions to keep it fast & stable over time:
        - Fixed F (num_features), so no cubic growth with N.
        - Adaptive jitter on Cholesky.
        - Optional exponential forgetting (decay).
        - Ability to return only diagonal covariance (big perf win).
    """

    def __init__(
        self,
        max_points: int,                 # kept for API compatibility, unused
        input_dim: int,
        num_envs: int,
        num_objects: int,
        device: torch.device,
        lengthscale: float = 0.5 / 20.0,
        variance: float = 1.0,
        noise: float = 1e-2,
        num_features: int = 128,         # <-- your new default
        dtype: torch.dtype = torch.float32,
        jitter: float = 1e-6,
        max_chol_tries: int = 5,
        decay: float = 0.995,            # slightly stronger forgetting to help conditioning
    ):
        self.device = device
        self.dtype = dtype

        self.D = input_dim
        self.E = num_envs
        self.M = num_objects
        self.F = num_features

        self.lengthscale = torch.tensor(lengthscale, device=device, dtype=dtype)
        self.variance = torch.tensor(variance, device=device, dtype=dtype)
        self.noise = torch.tensor(noise, device=device, dtype=dtype)

        self.jitter0 = jitter
        self.max_chol_tries = max_chol_tries
        self.decay = torch.tensor(decay, device=device, dtype=dtype)

        # Precompute constants
        self._inv_noise2 = 1.0 / (self.noise ** 2)
        self.phi_scale = torch.sqrt(torch.tensor(2.0 / self.F, device=device, dtype=dtype))
        self.eyeF = torch.eye(self.F, device=device, dtype=dtype).unsqueeze(0)  # [1, F, F]

        # RFF params (shared across all E x M)
        self.W = torch.randn(self.F, self.D, device=device, dtype=dtype) / self.lengthscale
        self.b = 2.0 * torch.pi * torch.rand(self.F, device=device, dtype=dtype)

        # Per-(E,M) accumulators
        eyeF = torch.eye(self.F, device=device, dtype=dtype)
        self.A = eyeF.expand(self.E, self.M, self.F, self.F).clone()
        self.bvec = torch.zeros(self.E, self.M, self.F, device=device, dtype=dtype)

        # Cache
        self._posterior_dirty = torch.ones(self.E, self.M, dtype=torch.bool, device=device)
        self._L = torch.empty(self.E, self.M, self.F, self.F, device=device, dtype=dtype)
        self._w_mean = torch.empty(self.E, self.M, self.F, device=device, dtype=dtype)

        self.min_dist = 0.0  # kept for API compat

    # ------------------------- Features -------------------------

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., D]
        proj = torch.matmul(x, self.W.T) + self.b  # [*, F]
        return self.phi_scale * torch.cos(proj)

    # ------------------------- Public API -----------------------

    @torch.no_grad()
    def predict(
        self,
        X_test: torch.Tensor,    # [E, M, T, D]
        test_mask: torch.Tensor, # [E, M, T] (bool)
        return_diag_only: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            if return_diag_only:
                mu:      [E, M, T]
                var_diag:[E, M, T]
            else:
                mu:  [E, M, T]
                cov: [E, M, T, T]  (diagonal-only covariance packed in a full matrix)
        """
        E, M, T, D = X_test.shape
        assert D == self.D
        if not test_mask.any():
            # Nothing to predict; just return zeros with correct shapes
            if return_diag_only:
                return (torch.zeros(E, M, T, device=X_test.device, dtype=X_test.dtype),
                        torch.zeros(E, M, T, device=X_test.device, dtype=X_test.dtype))
            else:
                return (torch.zeros(E, M, T, device=X_test.device, dtype=X_test.dtype),
                        torch.zeros(E, M, T, T, device=X_test.device, dtype=X_test.dtype))

        self._ensure_posterior()

        B = E * M
        X_test_b = X_test.view(B, T, D)
        test_mask_b = test_mask.view(B, T)

        Phi_test = self._phi(X_test_b.reshape(-1, D)).view(B, T, self.F)  # [B, T, F]
        w_mean = self._w_mean.view(B, self.F)

        mu = torch.einsum("btf,bf->bt", Phi_test, w_mean) * self.variance.sqrt()

        # Diagonal covariance only
        L = self._L.view(B, self.F, self.F)
        v = torch.linalg.solve_triangular(L, Phi_test.transpose(1, 2), upper=False)  # [B, F, T]
        var_diag = (v ** 2).sum(dim=1) * self.variance  # [B, T]

        # Mask out invalid test points
        mu = torch.where(test_mask_b, mu, torch.zeros_like(mu))
        var_diag = torch.where(test_mask_b, var_diag, torch.zeros_like(var_diag))

        if return_diag_only:
            return mu.view(E, M, T), var_diag.view(E, M, T)

        # (Slower path) Pack diagonal into full matrix to preserve old API
        cov = torch.zeros(B, T, T, device=X_test.device, dtype=X_test.dtype)
        idx = torch.arange(T, device=X_test.device)
        cov[:, idx, idx] = var_diag
        cov = torch.where(
            test_mask_b.unsqueeze(-1) & test_mask_b.unsqueeze(-2),
            cov,
            torch.zeros_like(cov),
        )
        return mu.view(E, M, T), cov.view(E, M, T, T)

    @torch.no_grad()
    def update(self, x_new: torch.Tensor, y_new: torch.Tensor, object_mask: torch.Tensor):
        """
        x_new: [E, M, T, D]
        y_new: [E, M, T]
        object_mask: [E, M] (bool)
        """
        if not object_mask.any():
            return

        E, M, T, D = x_new.shape
        assert D == self.D

        valid_mask = (y_new != 0.0)

        B = E * M
        x_new_b = x_new.view(B, T, D)
        y_new_b = y_new.view(B, T)
        valid_mask_b = valid_mask.view(B, T)
        object_mask_b = object_mask.view(B)

        # Exponential forgetting to keep matrices well conditioned
        if self.decay < 1.0:
            A_b = self.A.view(B, self.F, self.F)
            b_b = self.bvec.view(B, self.F)
            A_b[object_mask_b] = (
                self.decay * A_b[object_mask_b]
                + (1.0 - self.decay) * torch.eye(self.F, device=self.device, dtype=self.dtype)
            )
            b_b[object_mask_b] = self.decay * b_b[object_mask_b]

        phi = self._phi(x_new_b.reshape(-1, D)).view(B, T, self.F)
        phi = phi * valid_mask_b.unsqueeze(-1)
        y_new_b = y_new_b * valid_mask_b

        PhiT = phi.transpose(1, 2)                             # [B, F, T]
        A_delta = self._inv_noise2 * torch.matmul(PhiT, phi)   # [B, F, F]
        b_delta = self._inv_noise2 * torch.matmul(PhiT, y_new_b.unsqueeze(-1)).squeeze(-1)  # [B, F]

        A = self.A.view(B, self.F, self.F)
        bvec = self.bvec.view(B, self.F)

        A[object_mask_b] = A[object_mask_b] + A_delta[object_mask_b]
        bvec[object_mask_b] = bvec[object_mask_b] + b_delta[object_mask_b]

        self._posterior_dirty.view(-1)[object_mask_b] = True

    @torch.no_grad()
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        eye = torch.eye(self.F, device=self.device, dtype=self.dtype)
        if env_ids is None:
            self.A.copy_(eye)
            self.bvec.zero_()
            self._posterior_dirty.fill_(True)
        else:
            self.A[env_ids] = eye
            self.bvec[env_ids] = 0.0
            self._posterior_dirty[env_ids] = True

    # ------------------------- Internals ------------------------

    @torch.no_grad()
    def _ensure_posterior(self):
        dirty = self._posterior_dirty
        if not dirty.any():
            return

        E, M, F = self.E, self.M, self.F
        B = E * M

        A = self.A.view(B, F, F)
        b = self.bvec.view(B, F)
        dirty_b = dirty.view(B)

        if not dirty_b.any():
            return

        A_dirty = A[dirty_b]
        b_dirty = b[dirty_b]

        # Symmetrize (floating point drift)
        A_dirty = 0.5 * (A_dirty + A_dirty.transpose(-1, -2))

        # Slight diagonal clamp to reduce chances of ill-conditioning
        diag = torch.diagonal(A_dirty, dim1=-2, dim2=-1)
        diag.clamp_min_(1e-9)
        A_dirty = A_dirty.clone()
        A_dirty[torch.arange(A_dirty.shape[0], device=A_dirty.device).unsqueeze(-1),
                torch.arange(F, device=A_dirty.device),
                torch.arange(F, device=A_dirty.device)] = diag

        # Adaptive jitter
        jitter = self.jitter0
        L = None
        info = None
        for _ in range(self.max_chol_tries):
            A_try = A_dirty + (self.eyeF * jitter)
            L, info = torch.linalg.cholesky_ex(A_try)
            if (info == 0).all():
                break
            jitter *= 10.0

        if (info != 0).any():
            # Fallback: pinv is expensive but guaranteed; try to avoid getting here
            A_inv = torch.linalg.pinv(A_dirty)
            w = torch.matmul(A_inv, b_dirty.unsqueeze(-1)).squeeze(-1)
            big_jitter = jitter * 100.0
            L_fallback = torch.linalg.cholesky(
                A_dirty + torch.eye(F, device=self.device, dtype=self.dtype) * big_jitter
            )
            # Write back
            self._w_mean.view(B, F)[dirty_b] = w
            self._L.view(B, F, F)[dirty_b] = L_fallback
        else:
            # Solve A w = b via the Cholesky factors
            z = torch.linalg.solve_triangular(L, b_dirty.unsqueeze(-1), upper=False)
            w = torch.linalg.solve_triangular(L.transpose(1, 2), z, upper=True).squeeze(-1)

            self._w_mean.view(B, F)[dirty_b] = w
            self._L.view(B, F, F)[dirty_b] = L

        dirty.fill_(False)