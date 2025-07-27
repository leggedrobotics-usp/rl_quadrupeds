from typing import Sequence, Tuple
import torch

class OnlineGaussianProcess:
    """
    Batched RFF GP (Bayesian linear regression in feature space), vectorized over
    (num_envs × num_objects). Designed to avoid Python scalar extraction so it
    plays nice with Dynamo/JIT (if you want to enable it later).

    Args:
        max_points: kept for API compatibility (unused here)
        input_dim: D
        num_envs: E
        num_objects: M
        device: torch device
        lengthscale, variance, noise: kernel hyperparams
        num_features: number of RFF features F (speed/accuracy trade-off)
    """

    def __init__(
        self,
        max_points: int,
        input_dim: int,
        num_envs: int,
        num_objects: int,
        device: torch.device,
        lengthscale: float = 0.5 / 20.0,
        variance: float = 1.0,
        noise: float = 1e-2,
        num_features: int = 256,
        dtype: torch.dtype = torch.float32,
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

        # Random Fourier Features params
        self.W = torch.randn(self.F, self.D, device=device, dtype=dtype) / self.lengthscale
        self.b = 2.0 * torch.pi * torch.rand(self.F, device=device, dtype=dtype)

        # Per (E, M) accumulators
        self.A = torch.eye(self.F, device=device, dtype=dtype).expand(self.E, self.M, self.F, self.F).clone()
        self.bvec = torch.zeros(self.E, self.M, self.F, device=device, dtype=dtype)

        self._posterior_dirty = torch.ones(self.E, self.M, dtype=torch.bool, device=device)
        self._L = torch.zeros(self.E, self.M, self.F, self.F, device=device, dtype=dtype)
        self._w_mean = torch.zeros(self.E, self.M, self.F, device=device, dtype=dtype)

    # ------------------------- Features -------------------------

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        proj = torch.matmul(x, self.W.T) + self.b
        return torch.sqrt(torch.tensor(2.0 / self.F, device=x.device, dtype=x.dtype)) * torch.cos(proj)

    # ------------------------- Public API -----------------------

    @torch.no_grad()
    def predict(self, X_test: torch.Tensor, test_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        X_test:    [E, M, T, D]
        test_mask: [E, M, T] (bool)

        Returns:
            mu:  [E, M, T]
            cov: [E, M, T, T] (diagonal-filled; off-diagonals are zero to match your API)
        """
        E, M, T, D = X_test.shape
        B = E * M

        self._ensure_posterior()

        Xb = X_test.view(B, T, D)
        maskb = test_mask.view(B, T)

        Phi = self._phi(Xb.reshape(-1, D)).view(B, T, self.F)
        Phi = Phi * maskb.unsqueeze(-1)

        w_mean = self._w_mean.view(B, self.F)

        mu = torch.einsum("btf,bf->bt", Phi, w_mean) * torch.sqrt(self.variance)
        mu = mu.view(E, M, T)

        # Diagonal variance only
        L = self._L.view(B, self.F, self.F)
        v = torch.linalg.solve_triangular(L, Phi.transpose(1, 2), upper=False)
        var_diag = (v ** 2).sum(dim=1) * self.variance  # [B, T]
        var_diag = var_diag.view(E, M, T)

        cov = torch.zeros(E, M, T, T, device=X_test.device, dtype=X_test.dtype)
        idx = torch.arange(T, device=X_test.device)
        cov[:, :, idx, idx] = var_diag

        mu = torch.where(test_mask, mu, torch.zeros_like(mu))
        cov = torch.where(test_mask.unsqueeze(-1) & test_mask.unsqueeze(-2), cov, torch.zeros_like(cov))

        return mu, cov

    @torch.no_grad()
    def update(self, x_new: torch.Tensor, y_new: torch.Tensor, object_mask: torch.Tensor):
        """
        x_new:       [E, M, T, D]
        y_new:       [E, M, T]
        object_mask: [E, M]
        """
        E, M, T, D = x_new.shape
        B = E * M

        valid_mask = (y_new != 0.0)  # [E, M, T]

        Xb = x_new.view(B, T, D)
        yb = y_new.view(B, T)
        vb = valid_mask.view(B, T)
        ob = object_mask.view(B)

        Phi = self._phi(Xb.reshape(-1, D)).view(B, T, self.F)
        Phi = Phi * vb.unsqueeze(-1)
        yb = yb * vb

        inv_noise2 = 1.0 / (self.noise ** 2)

        Pt = Phi.transpose(1, 2)
        A_delta = inv_noise2 * torch.matmul(Pt, Phi)            # [B, F, F]
        b_delta = inv_noise2 * torch.matmul(Pt, yb.unsqueeze(-1)).squeeze(-1)  # [B, F]

        A = self.A.view(B, self.F, self.F)
        bvec = self.bvec.view(B, self.F)

        A[ob] = A[ob] + A_delta[ob]
        bvec[ob] = bvec[ob] + b_delta[ob]

        self._posterior_dirty.view(B)[ob] = True

    @torch.no_grad()
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        eye = torch.eye(self.F, device=self.device, dtype=self.dtype)
        if env_ids is None:
            self.A[:] = eye
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

        L = torch.linalg.cholesky(A[dirty_b])
        self._L.view(B, F, F)[dirty_b] = L

        z = torch.linalg.solve_triangular(L, b[dirty_b].unsqueeze(-1), upper=False)
        w = torch.linalg.solve_triangular(L.transpose(1, 2), z, upper=True).squeeze(-1)
        self._w_mean.view(B, F)[dirty_b] = w

        dirty.fill_(False)