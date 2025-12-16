import torch


def haar_orthogonal_q(m: int, n: int, device=None, dtype=None):
    """
    Draw a Haar-orthogonal matrix Q with shape (m, n), m >= n:
      - sample Gaussian G ~ N(0,1)^{m x n}
      - QR-decompose G = Q R
      - fix signs to make distribution exactly Haar
    """
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32

    G = torch.randn(m, n, device=device, dtype=dtype)
    # mode='reduced' gives Q: (m x n), R: (n x n)
    Q, R = torch.linalg.qr(G, mode="reduced")

    # Fix signs (make diagonal of R non-negative)
    diag = torch.diag(R)
    sign = torch.sign(diag)
    # Avoid zeros: treat 0 as +1
    sign[sign == 0] = 1.0
    Q = Q * sign  # broadcast over rows

    return Q


def compress(W: torch.Tensor, rank: int, absorb_R: bool = True, **ignored_kwargs):
    """
    Two-sided Haar-orthogonal random projection of W.

    Input:
      W    : (n x m) weight matrix
      rank : target rank r (1 <= r <= min(n,m))

    Output:
      A    : (n x r) left Haar-orthogonal projector
      B    : (r x m) right Haar-orthogonal projector
      R    : (r x r) inner core = L^T W R

    Decomposition:
      W ≈ L B R^T
    """
    assert W.dim() == 2
    n, m = W.shape
    assert 1 <= rank <= min(n, m)

    device = W.device
    dtype = W.dtype

    # Left and right Haar-orthogonal factors
    A = haar_orthogonal_q(n, rank, device=device, dtype=dtype)  # n x r
    B = haar_orthogonal_q(m, rank, device=device, dtype=dtype)  # m x r

    # Inner core
    R = A.transpose(0, 1) @ W @ B  # r x r

    B = B.transpose(0, 1)  # m x r -> r x m 

    if absorb_R:
        A = A @ R  # n x r
        R = torch.eye(rank, device=device, dtype=dtype)

    # Reconstruction in the random subspace
    # W_hat = A @ R @ B  # n x m

    print(
        f"[random_projections]: W {W.shape} -> A {A.shape}, R {R.shape}, B {B.shape}"
    )

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
