"""
**randomized range finding**:

1. Draw (\Omega \in \mathbb{R}^{m\times (r+p)}) with i.i.d. Gaussian entries (small oversampling (p)).
2. Form (Y = W \Omega\in\mathbb{R}^{n\times(r+p)}).
3. Compute (Y = Q R) (QR), and let (L = Q_{[:,1:r]}\in\mathbb{R}^{n\times r}).
4. Set (R = I_r), (B = L^\top W).

Then
[
\widehat W = L B = L L^\top W
]
is a rank–(r) approximation whose error is provably close to the optimal truncated SVD in Frobenius and spectral norm under mild conditions on (W). ([arXiv][1])

A two–sided version sketches columns as well (using another random test matrix on the left of (W^\top)) and yields (W \approx L C R^\top) with small core (C\in\mathbb{R}^{r\times r}). This fits your (L R B) template up to reparameterization.

[1]: https://arxiv.org/abs/0909.4061
"""

import torch


def _orthonormalize(A: torch.Tensor) -> torch.Tensor:
    """
    Orthonormal basis for columns of A via QR.
    Returns Q with Q^T Q = I.
    """
    Q, _ = torch.linalg.qr(A, mode="reduced")
    return Q


def compress(
    W: torch.Tensor,
    rank: int,
    finder_oversampling: int = 5,
    finder_n_power_iter: int = 0,
    absorb_R: bool = True,
    **ignored_kwargs,
):
    """
    Two-sided, data-dependent randomized projection.

    W          : (n x m)
    rank       : target rank r
    oversampling : extra dimensions p (k = r + p)
    n_power_iter : number of power iterations on each side

    Returns:
      L    : (n x r), left projector (columns orthonormal)
      R    : (m x r), right projector (columns orthonormal)
      B    : (r x r), core = L^T W R
      W_hat: (n x m), approximation = L B R^T
    """
    assert W.dim() == 2
    n, m = W.shape
    assert 1 <= rank <= min(n, m)

    print(f"[randomized_range_finder] Config: oversampling={finder_oversampling}, n_power_iter={finder_n_power_iter}")

    k = min(rank + finder_oversampling, n, m)

    # ----- left subspace (rows of W) -----
    Omega_right = torch.randn(m, k, device=W.device, dtype=W.dtype)  # (m x k)
    Y = W @ Omega_right  # (n x k)

    for _ in range(finder_n_power_iter):
        Y = W @ (W.T @ Y)

    Q_left = _orthonormalize(Y)  # (n x k)
    A = Q_left[:, :rank]  # (n x r)

    # ----- right subspace (columns of W) -----
    Omega_left = torch.randn(n, k, device=W.device, dtype=W.dtype)  # (n x k)
    Z = W.T @ Omega_left  # (m x k)

    for _ in range(finder_n_power_iter):
        Z = W.T @ (W @ Z)

    Q_right = _orthonormalize(Z)  # (m x k)
    B = Q_right[:, :rank]  # (m x r)

    # ----- core and reconstruction -----
    R = A.T @ W @ B  # (r x r)
    # W_hat = L @ B @ R.T             # (n x m)

    B = B.T  # return R as (r x m)

    if absorb_R:
        A = A @ R  # n x r
        R = torch.eye(rank, device=W.device, dtype=W.dtype)

    return A, R, B


def reconstruct(A, R, B):
    """
    Reconstructs the original matrix from the compressed representation.

    Computes: W_reconstructed = A @ R @ B

    Returns:
        torch.Tensor: The reconstructed 2D matrix.
    """
    reconstructed_matrix = A @ R @ B
    return reconstructed_matrix


def randomized_left_projection(
    W: torch.Tensor,
    rank: int,
    oversampling: int = 5,
    n_power_iter: int = 0,
    absorb_R: bool = True,
):
    """
    One-sided, data-dependent randomized projection (randomized range finder).

    W          : (n x m)
    rank       : target rank r
    oversampling : extra dimensions p (k = r + p)
    n_power_iter : number of power iterations to sharpen the subspace

    Returns:
      L    : (n x r), approximate leading left subspace of W
      B    : (r x m), core = L^T W
      W_hat: (n x m), approximation = L B
    """
    assert W.dim() == 2
    n, m = W.shape
    assert 1 <= rank <= min(n, m)

    k = min(rank + oversampling, n, m)  # total sketch dimension

    # Random test matrix on the right
    Omega = torch.randn(m, k, device=W.device, dtype=W.dtype)

    # Sketch Y = W Ω
    Y = W @ Omega  # (n x k)

    # Optional power iterations (W W^T)^q W Ω
    for _ in range(n_power_iter):
        Y = W @ (W.T @ Y)

    # Orthonormal basis for the range of W
    Q = _orthonormalize(Y)  # (n x k)

    # Truncate to desired rank
    A = Q[:, :rank]  # (n x r)

    # Core and reconstruction
    R = A.T @ W  # (r x m)
    # W_hat = A @ R  # (n x m)

    if absorb_R:
        A = A @ R  # n x r
        R = torch.eye(rank, device=W.device, dtype=W.dtype)

    return A, R
