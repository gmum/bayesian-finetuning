import numpy as np
from scipy.fft import dct, idct
import matplotlib.pyplot as plt

def compress_matrix_dct(matrix, keep_dims=(dct_rows_to_keep, dct_cols_to_keep)):
    """
    Compresses a 2D matrix using the Discrete Cosine Transform (DCT)
    by truncating high-frequency components.

    This is analogous to SVD truncation, but it "projects" the matrix
    onto a standard basis of cosine functions and keeps only the
    coefficients for the lowest-frequency functions.

    Args:
        matrix (np.ndarray): The 2D input matrix (e.g., weights).
        keep_dims (tuple): A tuple (k_rows, k_cols) specifying the number
                           of low-frequency coefficients to keep in each
                           dimension.

    Returns:
        np.ndarray: The truncated 2D DCT coefficient matrix. This is the
                    "compressed" representation.
    """
    # 1. Apply the 2D-DCT. We apply the 1D-DCT to each axis.
    # 'norm="ortho"' makes the transform orthogonal, so idct(dct(x)) == x
    dct_coeffs = dct(dct(matrix, type=2, axis=1, norm='ortho'), 
                     type=2, axis=0, norm='ortho')

    # 2. Create the truncation mask
    mask = np.zeros_like(dct_coeffs)
    
    # Get the number of rows/cols to keep, ensuring they are not
    # larger than the matrix dimensions
    k_rows = min(keep_dims[0], matrix.shape[0])
    k_cols = min(keep_dims[1], matrix.shape[1])
    
    # 3. Keep the top-left (k_rows x k_cols) block of coefficients
    mask[0:k_rows, 0:k_cols] = 1
    # mask[-k_rows:0, -k_cols:0] = 1


    # 4. Apply the mask to truncate (zero-out) high-frequency coefficients
    # print(f"dct_coeffs = {dct_coeffs.shape}: {dct_coeffs}")
    truncated_coeffs = dct_coeffs * mask
    # print(f"truncated_coeffs = {truncated_coeffs.shape}: {truncated_coeffs}")

    return truncated_coeffs


def reconstruct_matrix_dct(truncated_coeffs, original_shape=None):
    """
    Reconstructs the original matrix from its truncated DCT coefficients.

    Args:
        truncated_coeffs (np.ndarray): The compressed coefficient matrix
                                     from compress_matrix_dct.

    Returns:
        np.ndarray: The reconstructed 2D matrix.
    """


    # Apply the Inverse 2D-DCT, in reverse order of the forward DCT
    reconstructed_matrix = idct(idct(truncated_coeffs, type=2, axis=0, norm='ortho'), 
                                type=2, axis=1, norm='ortho')
    
    return reconstructed_matrix

#######################################################################################################################################

import numpy as np
from scipy.fft import dct, idct
import matplotlib.pyplot as plt

def compress_matrix_dct_adaptive(matrix, keep_dims=(dct_rows_to_keep, dct_cols_to_keep)):
    """
    Compresses a 2D matrix using the Discrete Cosine Transform (DCT)
    by keeping coefficients with the highest energy (magnitude).

    Unlike the standard approach that keeps the top-left corner,
    this adaptive version selects the rows and columns with highest
    total energy, regardless of their position in the frequency space.

    Args:
        matrix (np.ndarray): The 2D input matrix (e.g., weights).
        keep_dims (tuple): A tuple (k_rows, k_cols) specifying the number
                           of high-energy coefficients to keep in each
                           dimension.

    Returns:
        tuple: (truncated_coeffs, row_indices, col_indices)
            - truncated_coeffs: The compressed representation (k_rows x k_cols)
            - row_indices: Indices of selected rows in the original DCT
            - col_indices: Indices of selected columns in the original DCT
    """
    # 1. Apply the 2D-DCT
    dct_coeffs = dct(dct(matrix, type=2, axis=1, norm='ortho'), 
                     type=2, axis=0, norm='ortho')

    # 2. Get the number of rows/cols to keep
    k_rows = min(keep_dims[0], matrix.shape[0])
    k_cols = min(keep_dims[1], matrix.shape[1])
    
    # 3. Calculate energy (squared magnitude) for each row and column
    row_energy = np.sum(dct_coeffs**2, axis=1)  # Energy per row
    col_energy = np.sum(dct_coeffs**2, axis=0)  # Energy per column
    
    # 4. Find indices of rows and columns with highest energy
    row_indices = np.argsort(row_energy)[-k_rows:]  # Top k_rows
    col_indices = np.argsort(col_energy)[-k_cols:]  # Top k_cols
    
    # Sort indices to maintain relative order
    row_indices = np.sort(row_indices)
    col_indices = np.sort(col_indices)
    
    # 5. Extract the selected coefficients
    # This creates a k_rows x k_cols matrix
    truncated_coeffs = dct_coeffs[np.ix_(row_indices, col_indices)]

    return truncated_coeffs, row_indices, col_indices


def reconstruct_matrix_dct_adaptive(truncated_coeffs, row_indices, col_indices, 
                                    original_shape):
    """
    Reconstructs the original matrix from adaptively truncated DCT coefficients.

    Args:
        truncated_coeffs (np.ndarray): The compressed coefficient matrix (k_rows x k_cols)
        row_indices (np.ndarray): Indices of the selected rows in the full DCT
        col_indices (np.ndarray): Indices of the selected columns in the full DCT
        original_shape (tuple): Shape of the original matrix (rows, cols)

    Returns:
        np.ndarray: The reconstructed 2D matrix with original_shape.
    """
    # 1. Create a zero matrix of the original DCT coefficient size
    full_dct_coeffs = np.zeros(original_shape)
    
    # 2. Place the truncated coefficients back at their original positions
    full_dct_coeffs[np.ix_(row_indices, col_indices)] = truncated_coeffs
    
    # 3. Apply the Inverse 2D-DCT
    reconstructed_matrix = idct(idct(full_dct_coeffs, type=2, axis=0, norm='ortho'), 
                                type=2, axis=1, norm='ortho')
    
    return reconstructed_matrix


top_row_indices, top_col_indices = None, None

def compress_matrix_dct(*args, **kwargs):
    global top_row_indices, top_col_indices
    R, top_row_indices, top_col_indices = compress_matrix_dct_adaptive(*args, **kwargs)
    return R

def reconstruct_matrix_dct(truncated_coeffs, original_shape):
    W_approx = reconstruct_matrix_dct_adaptive(
        truncated_coeffs, top_row_indices, top_col_indices, original_shape)
    return W_approx


#######################################################################################################################################


import torch
import math

def _get_dct_basis(N, device='cpu'):
    """
    Generates the orthonormal 1D DCT-II basis matrix of size (N, N).
    Row k is the k-th frequency component.
    """
    # Create the grid for n (0 to N-1) and k (0 to N-1)
    n = torch.arange(float(N), device=device)
    k = torch.arange(float(N), device=device)
    
    # Use broadcasting to create the (N, N) matrix arguments
    # argument: (pi * k * (2n + 1)) / (2N)
    arg = (math.pi * k[:, None] * (2 * n + 1)) / (2 * N)
    basis = torch.cos(arg)
    
    # Apply orthonormalization factors
    # For k=0: sqrt(1/N)
    # For k>0: sqrt(2/N)
    norm_factors = torch.full((N, 1), math.sqrt(2 / N), device=device)
    norm_factors[0] = math.sqrt(1 / N)
    
    return basis * norm_factors

def compress_matrix_dct(matrix, keep_dims=(dct_rows_to_keep, dct_cols_to_keep)):
    """
    Projects the matrix into the DCT domain and truncates high frequencies.
    
    Args:
        matrix: The input weight matrix W of size (M, N).
        keep_dims: Tuple (h, w) indicating the size of the core R.
    
    Returns:
        R: The compressed core matrix of size (h, w).
    """
    if not torch.is_tensor(matrix):
        matrix = torch.tensor(matrix)

    M, N = matrix.shape
    h, w = keep_dims
    
    # 1. Generate the fixed bases U (MxM) and V (NxN)
    # Note: For massive matrices, FFT implementations are preferred 
    # to save memory, but this explicit form aligns with W = U R V^T.
    U_full = _get_dct_basis(M, device=matrix.device)
    V_full = _get_dct_basis(N, device=matrix.device)
    
    # 2. Truncate the bases (Keep low frequency components)
    # U_k is (M, h) -> The first h columns of U^T (or rows of U)
    # V_k is (N, w) -> The first w columns of V^T (or rows of V)
    # Because our _get_dct_basis returns rows as frequencies, 
    # we transpose them to treat columns as basis vectors.
    U_k = U_full.T[:, :h] 
    V_k = V_full.T[:, :w]
    
    # 3. Project: R = U_k^T @ W @ V_k
    # This finds the optimal R for the fixed subspace.
    R = U_k.T @ matrix @ V_k
    
    return R

def reconstruct_matrix_dct(truncated_coeffs, original_shape):
    """
    Reconstructs the approximation of the original matrix from the core.
    
    Args:
        truncated_coeffs: The core matrix R of size (h, w).
        original_shape: Tuple (M, N) required to regenerate the correct basis.
    
    Returns:
        W_approx: The approximated matrix of size (M, N).
    """
    M, N = original_shape
    h, w = truncated_coeffs.shape
    
    # 1. Regenerate the fixed bases
    U_full = _get_dct_basis(M, device=truncated_coeffs.device)
    V_full = _get_dct_basis(N, device=truncated_coeffs.device)
    
    # 2. Slice to get the active subspace
    U_k = U_full.T[:, :h]
    V_k = V_full.T[:, :w]
    print(f"U_k.shape = {U_k.shape}, V_k.shape = {V_k.shape}")
    
    # 3. Reconstruct: W = U_k @ R @ V_k^T
    W_approx = U_k @ truncated_coeffs @ V_k.T
    
    return W_approx


#######################################################################################################################################


import torch
import math

def _get_dct_basis(N, device='cpu'):
    """Generates full (N, N) DCT basis."""
    n = torch.arange(float(N), device=device)
    k = torch.arange(float(N), device=device)
    arg = (math.pi * k[:, None] * (2 * n + 1)) / (2 * N)
    basis = torch.cos(arg)
    norm_factors = torch.full((N, 1), math.sqrt(2 / N), device=device)
    norm_factors[0] = math.sqrt(1 / N)
    return basis * norm_factors

def compress_matrix_dct_adaptive(matrix, keep_dims=(dct_rows_to_keep, dct_cols_to_keep)):
    """
    Adaptive DCT: Selects the 'best' basis vectors based on the specific
    energy distribution of the input matrix, rather than just low-freqs.
    """
    if not torch.is_tensor(matrix):
        matrix = torch.tensor(matrix)

    M, N = matrix.shape
    h, w = keep_dims
    
    # 1. Generate full bases
    U_full = _get_dct_basis(M, device=matrix.device) # (M, M)
    V_full = _get_dct_basis(N, device=matrix.device) # (N, N)
    
    # 2. Compute the full spectrum (Transform W into frequency domain)
    # Spectrum = U^T * W * V
    spectrum = U_full.T @ matrix @ V_full
    print(f"spectrum shape: {spectrum.shape}")
    
    # 3. Determine importance of Rows (U components) and Cols (V components)
    # We calculate the L2 norm (energy) of every row and column in the spectrum
    row_energy = torch.norm(spectrum, dim=1) # Shape (M,)
    col_energy = torch.norm(spectrum, dim=0) # Shape (N,)
    
    # 4. Select the indices with the highest energy
    # We pick the top 'h' rows and top 'w' columns
    top_row_indices = torch.topk(row_energy, h).indices
    top_col_indices = torch.topk(col_energy, w).indices
    
    # Sort indices to keep the core matrix somewhat ordered (optional but cleaner)
    top_row_indices, _ = torch.sort(top_row_indices)
    top_col_indices, _ = torch.sort(top_col_indices)
    
    # 5. Build the Adaptive Bases
    # Select specific columns from the bases corresponding to high energy
    print(f"U_full shape: {U_full.shape}, V_full shape: {V_full.shape}, top_row_indices={top_row_indices}, top_col_indices={top_col_indices}")
    U_adaptive = U_full.T[:, top_row_indices] # (M, h)
    V_adaptive = V_full.T[:, top_col_indices] # (N, w)

    print(f"U_adaptive shape: {U_adaptive.shape}, V_adaptive shape: {V_adaptive.shape}")
    
    # 6. Project to get the core R
    R = U_adaptive.T @ matrix @ V_adaptive
    
    return R, top_row_indices, top_col_indices


def reconstruct_matrix_dct_adaptive(R, indices_u, indices_v, original_shape):
    """
    Reconstructs W using the specific frequencies we selected earlier.
    """
    M, N = original_shape
    
    U_full = _get_dct_basis(M, device=R.device)
    V_full = _get_dct_basis(N, device=R.device)
    
    # Retrieve the specific basis vectors we saved
    U_adaptive = U_full.T[:, indices_u]
    V_adaptive = V_full.T[:, indices_v]
    
    W_approx = U_adaptive @ R @ V_adaptive.T
    return W_approx




top_row_indices, top_col_indices = None, None

def compress_matrix_dct(*args, **kwargs):
    global top_row_indices, top_col_indices
    R, top_row_indices, top_col_indices = compress_matrix_dct_adaptive(*args, **kwargs)
    return R

def reconstruct_matrix_dct(truncated_coeffs, original_shape):
    W_approx = reconstruct_matrix_dct_adaptive(
        truncated_coeffs, top_row_indices, top_col_indices, original_shape)
    return W_approx