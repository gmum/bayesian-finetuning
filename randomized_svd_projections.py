import torch


def compress(
    W,
    rank,
    rsvd_n_iter=10,
    rsvd_oversample=10,
    absorb_R=True,
    **ignored_kwargs,
):
    """
    Decomposes matrix W (n x m) into L (n x r), R (r x r), and B (r x m)
    using Randomized SVD (Halko et al., 2011).

    Args:
        W (torch.Tensor): The input weight matrix (n x m).
        rank (int): The target rank 'r'.
        n_iter (int): Number of power iterations. Increase for matrices where
                      singular values decay slowly. Default 2 is usually sufficient.
        oversample (int): Safety buffer. We calculate (rank + oversample) vectors
                          internally, then trim to 'rank' at the end.

    Returns:
        A (torch.Tensor): Left projection (n x r). Approximate Left Singular Vectors.
        R (torch.Tensor): Center mixing matrix (r x r). Diagonal Singular Values.
        B (torch.Tensor): Right projection (r x m). Approximate Right Singular Vectors.
    """
    n, m = W.shape
    k = rank + rsvd_oversample

    print(
        f"[randomized_svd] Decomposing matrix of shape {W.shape} to rank {rank}. "
        f"Configuration n_iter={rsvd_n_iter}, oversample={rsvd_oversample} (k={k})"
    )

    # 1. Generate Random Test Matrix (Gaussian)
    # Omega shape: (m x k)
    Omega = torch.randn(m, k, device=W.device, dtype=W.dtype)

    # 2. Compute the Sketch Y (Range Approximation)
    # Y shape: (n x k)
    Y = W @ Omega

    # 3. Power Iterations (Halko et al., 2011, Algorithm 4.3)
    # This reduces the error if singular values don't decay rapidly.
    # We multiply by W W^T to push Y towards the dominant singular vectors.
    for _ in range(rsvd_n_iter):
        Y = W @ (W.T @ Y)

    # 4. Orthogonalize the Sketch (QR Decomposition)
    # Q shape: (n x k). Q forms an orthonormal basis for the range of W.
    Q, _ = torch.linalg.qr(Y)

    # 5. Project W into the smaller subspace
    # B_temp shape: (k x m)
    B_temp = Q.T @ W

    # 6. Compute deterministic SVD on the small matrix
    # U_small: (k x k), S: (k,), Vt: (k x m)
    U_small, S, Vt = torch.linalg.svd(B_temp, full_matrices=False)

    # 7. Reassemble and Trim to target rank 'r'
    # We only keep the top 'rank' components

    # L = Q * U_small (trimmed)
    # Projects the abstract SVD basis back to original row space
    A = (Q @ U_small)[:, :rank]  # Shape: (n x r)

    # R = Diagonal matrix of singular values (trimmed)
    R = torch.diag(S[:rank])  # Shape: (r x r)

    # B = Right singular vectors (trimmed)
    B = Vt[:rank, :]  # Shape: (r x m)

    if absorb_R:
        A = A @ R  # n x r
        R = torch.eye(rank, device=W.device, dtype=W.dtype)

    return A, R, B


def reconstruct(A, R, B):
    """Reconstructs the original matrix from the compressed representation."""
    reconstructed_matrix = A @ R @ B
    return reconstructed_matrix
