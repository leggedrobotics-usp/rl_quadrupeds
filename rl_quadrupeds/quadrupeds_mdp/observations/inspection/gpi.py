from typing import Sequence
import torch

class OnlineGaussianProcess:
    def __init__(self, max_points, input_dim, num_envs, num_objects, device,
                 lengthscale=0.001, variance=1., noise=1e-3):
        """
        Runs E × M independent Gaussian Processes in a vectorized way.

        Args:
            max_points (int): Maximum number of observations per GP (N).
            input_dim (int): Input dimensionality (D).
            num_envs (int): Number of environments (E).
            num_objects (int): Number of objects per environment (M).
            lengthscale (float): RBF kernel lengthscale.
            variance (float): RBF kernel variance.
            noise (float): Observation noise.
            device (str): Torch device.
        """
        self.lengthscale = lengthscale
        self.variance = variance
        self.noise = noise
        self.device = device

        self.N = max_points
        self.D = input_dim
        self.E = num_envs
        self.M = num_objects

        # Shapes:
        # [E, M, N, D] — Inputs
        self.X = torch.zeros(self.E, self.M, self.N, self.D, device=device)
        # [E, M, N, 1] — Targets
        self.Y = torch.zeros(self.E, self.M, self.N, 1, device=device)
        # [E, M] — Counts
        self.counts = torch.zeros(self.E, self.M, dtype=torch.long, device=device)
        # [E, M, N] — Valid mask
        self.masks = torch.zeros(self.E, self.M, self.N, device=device)

        self.min_dist = 0.5/20.  # Minimum distance between points

    def _scale_inputs(self, X):
        return X / self.lengthscale

    def _rbf_kernel(self, X1, X2):
        # X1, X2: [E, M, N1, D], [E, M, N2, D]
        diff = X1.unsqueeze(3) - X2.unsqueeze(2)  # [E, M, N1, N2, D]
        dist_sq = diff.pow(2).sum(-1)             # [E, M, N1, N2]
        return self.variance * torch.exp(-0.5 * dist_sq)

    def _apply_mask(self, K, mask1, mask2):
        # K: [E, M, N1, N2], mask1: [E, M, N1], mask2: [E, M, N2]
        return K * mask1.unsqueeze(3) * mask2.unsqueeze(2)

    def _compute_rbf(self, X1, X2, mask1, mask2):
        K = self._rbf_kernel(self._scale_inputs(X1), self._scale_inputs(X2))
        return self._apply_mask(K, mask1, mask2)

    def _cholesky_solve(self, A, B, mask):
        """
        A: [E, M, N, N], B: [E, M, N, 1], mask: [E, M, N]
        """
        E, M, N, _ = A.shape
        eye = torch.eye(N, device=self.device).expand(E, M, N, N)
        A_reg = A + self.noise * eye * mask.unsqueeze(-1) * mask.unsqueeze(-2) + 1e-6 * eye
        L = torch.linalg.cholesky(A_reg)
        return torch.cholesky_solve(B, L), L

    def predict(self, X_test, test_mask):
        """
        Args:
            X_test: [E, M, T, D] — test inputs
            test_mask: [E, M, T] — mask for test points

        Returns:
            mu: [E, M, T]
            cov: [E, M, T, T]
        """
        K_xx = self._compute_rbf(self.X, self.X, self.masks, self.masks)       # [E, M, N, N]
        alpha, L = self._cholesky_solve(K_xx, self.Y, self.masks)              # [E, M, N, 1]

        K_xxs = self._compute_rbf(self.X, X_test, self.masks, test_mask)       # [E, M, N, T]
        K_xsx = K_xxs.transpose(2, 3)                                          # [E, M, T, N]
        K_xsxs = self._compute_rbf(X_test, X_test, test_mask, test_mask)       # [E, M, T, T]

        mu = torch.matmul(K_xsx, alpha).squeeze(-1)                            # [E, M, T]
        v = torch.cholesky_solve(K_xxs, L)                                     # [E, M, N, T]
        cov = K_xsxs - torch.matmul(K_xsx, v)                                  # [E, M, T, T]

        return mu, cov

    # Update with min_dist filtering (removing for now)
    def update(self, x_new, y_new, object_mask):
        E, M, T, D = x_new.shape
        env_ids, obj_ids = torch.where(object_mask > 0)

        for e, m in zip(env_ids.tolist(), obj_ids.tolist()):
            count = self.counts[e, m].item()
            existing_count = int(self.masks[e, m].sum().item())
            if existing_count == 0:
                existing_X = None
            else:
                existing_X = self.X[e, m, :existing_count]  # [N_existing, D]

            for t in range(T):
                x = x_new[e, m, t]

                # Skip padding
                if torch.all(x == 0):
                    continue

                # Check distance from all existing points
                if existing_X is not None:
                    dists = torch.norm(existing_X - x, dim=-1)  # [N_existing]
                    if torch.any(dists < self.min_dist):
                        continue  # Too close to an existing point

                idx = count % self.N
                self.X[e, m, idx] = x
                self.Y[e, m, idx, 0] = y_new[e, m, t]
                self.masks[e, m, idx] = 1.0
                count += 1

                # Update the list of existing inputs for next comparisons
                if existing_X is not None and idx < existing_count:
                    existing_X[idx] = x  # Replace in-place if cyclic
                else:
                    existing_X = torch.cat([existing_X, x.unsqueeze(0)], dim=0) if existing_X is not None else x.unsqueeze(0)

            self.counts[e, m] = min(self.N, count)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.X[env_ids] = 0.0
        self.Y[env_ids] = 0.0
        self.counts[env_ids] = 0
        self.masks[env_ids] = 0.0