import torch
import numpy as np


def create_dct_matrix(N, device="cpu", dtype=torch.float32):
    """
    Creates the N×N DCT-II transformation matrix explicitly (vectorized).

    The DCT matrix D is defined such that:
    y = D @ x performs the DCT transform
    x = D.T @ y performs the inverse DCT (since D is orthogonal)

    Args:
        N (int): Size of the matrix
        device (str or torch.device): Device to create the matrix on
        dtype (torch.dtype): Data type for the matrix

    Returns:
        torch.Tensor: N×N DCT transformation matrix
    """
    # Create indices k (rows) and n (columns)
    k = torch.arange(N, device=device, dtype=dtype).unsqueeze(1)  # Shape: (N, 1)
    n = torch.arange(N, device=device, dtype=dtype).unsqueeze(0)  # Shape: (1, N)

    # Compute the DCT matrix using broadcasting
    D = torch.sqrt(torch.tensor(2 / N, device=device, dtype=dtype)) * torch.cos(
        torch.pi * k * (2 * n + 1) / (2 * N)
    )

    # Fix the first row (k=0)
    D[0, :] = torch.sqrt(torch.tensor(1 / N, device=device, dtype=dtype))

    return D


def compress(
    matrix,
    rank=None,
    keep_dims=(10, 10),
    dct_select_by="energy",
    absorb_R=True,
    **ignored_kwargs,
):
    """
    Compresses a 2D matrix using the Discrete Cosine Transform (DCT)
    by keeping coefficients with the highest energy (magnitude).

    This function explicitly computes projection matrices A and B such that:
    W ≈ A @ R @ B, where R is the compressed representation.

    All DCT operations are computed explicitly using matrix multiplication.

    Args:
        matrix (torch.Tensor): The 2D input matrix (e.g., weights), shape (m, n).
        keep_dims (tuple): A tuple (k_rows, k_cols) specifying the number
                           of coefficients to keep in each dimension.

    Returns:
            - 'A': Left projection matrix (m x k_rows)
            - 'R': The compressed representation (k_rows x k_cols)
            - 'B': Right projection matrix (k_cols x n)
            # - 'row_indices': Indices of selected rows in the DCT space
            # - 'col_indices': Indices of selected columns in the DCT space
            # - 'D_row': Full DCT matrix for rows (for reference)
            # - 'D_col': Full DCT matrix for columns (for reference)
    """
    if rank is not None:
        keep_dims = (rank, rank)

    m, n = matrix.shape
    device = matrix.device
    dtype = matrix.dtype

    # 1. Create DCT transformation matrices
    # @TODO: Cache these matrices if compressing multiple matrices of the same size
    D_row = create_dct_matrix(m, device=device, dtype=dtype)  # m×m DCT matrix for rows
    D_col = create_dct_matrix(
        n, device=device, dtype=dtype
    )  # n×n DCT matrix for columns

    # 2. Apply 2D-DCT explicitly: DCT_coeffs = D_row @ W @ D_col.T
    dct_coeffs = D_row @ matrix @ D_col.T

    # 3. Get the number of rows/cols to keep
    k_rows = min(keep_dims[0], m)
    k_cols = min(keep_dims[1], n)

    print(f"[dct] DCT coefficient selection method: {dct_select_by}")
    if dct_select_by == "energy":
        # 4. Calculate energy (squared magnitude) for each row and column
        row_energy = torch.sum(dct_coeffs**2, dim=1)  # Energy per row
        col_energy = torch.sum(dct_coeffs**2, dim=0)  # Energy per column

        # 5. Find indices of rows and columns with highest energy
        row_indices = torch.argsort(row_energy, descending=True)[:k_rows]  # Top k_rows
        col_indices = torch.argsort(col_energy, descending=True)[:k_cols]  # Top k_cols

    elif dct_select_by == "top-left":
        # Simply take the first k_rows and k_cols
        row_indices = torch.arange(k_rows, device=device)
        col_indices = torch.arange(k_cols, device=device)

    else:
        raise ValueError(
            f"Unknown selection method: {dct_select_by}" " (choose 'energy' or 'top-left')!"
        )

    print(f"[dct] Selected row indices (DCT space): {row_indices}")
    print(f"[dct] Selected column indices (DCT space): {col_indices}")

    # Sort indices to maintain relative order
    row_indices, _ = torch.sort(row_indices)
    col_indices, _ = torch.sort(col_indices)

    # 6. Extract the selected coefficients (compressed representation R)
    R = dct_coeffs[row_indices[:, None], col_indices[None, :]]

    # 7. Construct the projection matrices A and B
    # For reconstruction: W ≈ A @ R @ B
    # We have: W ≈ D_row.T @ [selected rows/cols of DCT_coeffs] @ D_col
    #
    # A is formed by selecting columns of D_row.T (inverse DCT basis for rows)
    # B is formed by selecting rows of D_col (DCT basis for columns)

    A = D_row.T[:, row_indices]  # m × k_rows: IDCT basis for selected row frequencies
    B = D_col[col_indices, :]  # k_cols × n: DCT basis for selected col frequencies

    print(f"[dct] A shape: {A.shape}, R shape: {R.shape}, B shape: {B.shape}")

    if absorb_R:
        # Absorb R into A and B for simplified representation
        A = A @ R  # Now A is m × k_cols
        R = torch.eye(k_cols, device=device, dtype=dtype)  # Identity matrix
        print(
            f"[dct] After absorbing R, new A shape: {A.shape}, R shape: {R.shape}"
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


def absorb_permutations_into_factors(A, B, row_perm_indices, col_perm_indices):
    """
    Given factors A, B such that for the *permuted* matrix
        W_perm ≈ A @ R @ B,
    and permutations used as
        data = torch.gather(data, dim=1, index=col_perm_indices)
        data = torch.gather(data, dim=0, index=row_perm_indices),
    return A_new, B_new such that for the *original*-order matrix
        W ≈ A_new @ R @ B_new
    without applying inverse permutations to the result.

    Assumes a single row permutation and a single column permutation
    shared across all rows/columns (possibly broadcasted).

    Args:
        A: torch.Tensor of shape (n_rows, rank)
        B: torch.Tensor of shape (rank, n_cols)
        row_perm_indices: torch.Tensor, 1D or 2D
        col_perm_indices: torch.Tensor, 1D or 2D

    Returns:
        A_new: torch.Tensor of shape (n_rows, rank)
        B_new: torch.Tensor of shape (rank, n_cols)
    """

    # --- normalize permutations to 1D vectors ---

    # row permutation: used with dim=0
    if row_perm_indices.ndim == 2:
        # assume same permutation across columns; take first column
        row_perm = row_perm_indices[:, 0]
    else:
        row_perm = row_perm_indices

    # column permutation: used with dim=1
    if col_perm_indices.ndim == 2:
        # assume same permutation across rows; take first row
        col_perm = col_perm_indices[0, :]
    else:
        col_perm = col_perm_indices

    # --- inverse permutations ---
    row_inv = row_perm.argsort()
    col_inv = col_perm.argsort()

    # --- fold permutations into A, B ---
    # W_perm = A R B
    # W      = P_row^{-1} W_perm P_col^{-1} = (P_row^{-1} A) R (B P_col^{-1})
    A_new = A[row_inv, :]  # P_row^{-1} A
    B_new = B[:, col_inv]  # B P_col^{-1}

    return A_new, B_new


def permute_to_sort_rows_and_cols_by_sums(data):
    """
    Prepare permutation indices for rows and columns based on their sums.

    Args:
        data: torch.Tensor of shape (n_rows, n_cols)

    Returns:
        total_row_perm_indices: torch.Tensor of shape (n_rows, n_cols)
        total_col_perm_indices: torch.Tensor of shape (n_rows, n_cols)
    """
    # Sort columns based on total column sums (same for each row)
    col_sums = abs(data).sum(dim=0)
    col_perm_indices = col_sums.argsort()
    col_perm_indices = col_perm_indices.unsqueeze(0).expand(data.shape[0], -1)

    # Sort rows based on total row sums (same for each column)
    row_sums = abs(data).sum(dim=1)
    row_perm_indices = row_sums.argsort()
    row_perm_indices = row_perm_indices.unsqueeze(1).expand(-1, data.shape[1])

    return row_perm_indices, col_perm_indices


def permute_and_compress(data, permutations="global", **compress_kwargs):
    """
    Permute the input data matrix to sort rows and columns by their sums,
    then compress using DCT-based compression.
    Args:
        data: torch.Tensor of shape (n_rows, n_cols)
        permutations: str, either "global" or "local"
        **compress_kwargs: additional arguments for the compress function
    Returns:
        A: torch.Tensor, left projection matrix
        R: torch.Tensor, compressed representation
        B: torch.Tensor, right projection matrix
    """
    # Prepare global permutations, e.g., based on total row/column sums
    row_perm_indices, col_perm_indices = permute_to_sort_rows_and_cols_by_sums(data)
    if permutations == "global":
        # Apply permutations
        data_sorted = torch.gather(data, 1, col_perm_indices)
        data_sorted = torch.gather(data_sorted, 0, row_perm_indices)

    elif permutations == "local":
        # Get sorting indices (permutation for each column)
        local_col_perm_indices = torch.argsort(data, dim=1)
        # Apply permutation to obtain sorted data (cols)
        data_sorted = torch.gather(data, 1, local_col_perm_indices)
        # Get sorting indices (permutation for each row)
        local_row_perm_indices = torch.argsort(data_sorted, dim=0)
        # Apply permutation to obtain sorted data (rows)
        data_sorted = torch.gather(data_sorted, 0, local_row_perm_indices)

    else:
        raise ValueError(f"Unknown permutations method: {permutations}")

    A, R, B = compress(data_sorted, **compress_kwargs)
    A, B = absorb_permutations_into_factors(A, B, row_perm_indices, col_perm_indices)
    return A, R, B


######################################################################################
# # NumPy versions of the permutation functions for use outside PyTorch
######################################################################################


def prepare_permutations_np(data):
    # # # The same sorting for each row based on total column sums
    col_sums = np.sum(data, axis=0)
    total_col_perm_indices = np.argsort(col_sums)
    total_col_perm_indices = np.tile(total_col_perm_indices, (data.shape[0], 1))

    # The same sorting for each column based on total row sums
    row_sums = np.sum(data, axis=1)
    total_row_perm_indices = np.argsort(row_sums)
    total_row_perm_indices = np.tile(total_row_perm_indices, (data.shape[1], 1)).T

    return total_row_perm_indices, total_col_perm_indices


def absorb_permutations_into_factors_np(A, B, row_perm_indices, col_perm_indices):
    """
    Given factors A, B such that for the *permuted* matrix
        W_perm ≈ A @ R @ B,
    and permutations used as
        data = np.take_along_axis(data, col_perm_indices, axis=1)
        data = np.take_along_axis(data, row_perm_indices, axis=0),
    return A_new, B_new such that for the *original*-order matrix
        W ≈ A_new @ R @ B_new
    without applying inverse permutations to the result.

    Assumes a single row permutation and a single column permutation
    shared across all rows/columns (possibly broadcasted).
    """

    # --- normalize permutations to 1D vectors ---

    row_perm_indices = np.asarray(row_perm_indices)
    col_perm_indices = np.asarray(col_perm_indices)

    # row permutation: used with axis=0
    if row_perm_indices.ndim == 2:
        # assume same permutation across columns; take first column
        row_perm = row_perm_indices[:, 0]
    else:
        row_perm = row_perm_indices

    # column permutation: used with axis=1
    if col_perm_indices.ndim == 2:
        # assume same permutation across rows; take first row
        col_perm = col_perm_indices[0, :]
    else:
        col_perm = col_perm_indices

    # --- inverse permutations ---
    row_inv = np.argsort(row_perm)
    col_inv = np.argsort(col_perm)

    # --- fold permutations into A, B ---
    # W_perm = A R B
    # W      = P_row^{-1} W_perm P_col^{-1} = (P_row^{-1} A) R (B P_col^{-1})
    A_new = A[row_inv, :]  # P_row^{-1} A
    B_new = B[:, col_inv]  # B P_col^{-1}

    return A_new, B_new


def permute_and_compress_np(data, **compress_kwargs):
    row_perm_indices, col_perm_indices = prepare_permutations_np(data)
    data_sorted = np.take_along_axis(data, col_perm_indices, axis=1)
    data_sorted = np.take_along_axis(data_sorted, row_perm_indices, axis=0)
    A, R, B = compress(data_sorted, **compress_kwargs)
    A, B = absorb_permutations_into_factors_np(A, B, row_perm_indices, col_perm_indices)
    return A, R, B


######################################################################################
