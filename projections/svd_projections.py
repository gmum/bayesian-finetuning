from typing import Tuple
import numpy as np
from sklearn.decomposition import TruncatedSVD
import torch


def run_svd(
    input_matrix: np.ndarray, rank: int, n_iter: int, random_state: int
) -> Tuple[np.ndarray, TruncatedSVD]:
    svd = TruncatedSVD(n_components=rank, n_iter=n_iter, random_state=random_state)
    svd.fit(input_matrix)
    reduced_matrix = svd.transform(input_matrix)
    return reduced_matrix, svd


def get_linear_rec_svd(
    input_matrix: np.ndarray, rank: int, n_iter: int, random_state: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    print(
        f"[svd_projections.get_linear_rec_svd] input_matrix={input_matrix.shape}, "
        f"rank={rank}, n_iter={n_iter}, random_state={random_state}"
    )
    reduced_matrix, svd = run_svd(input_matrix, rank, n_iter, random_state)

    reconstructed_matrix = svd.inverse_transform(reduced_matrix)
    return reconstructed_matrix, reduced_matrix, svd.components_


def compress(W, rank, n_iter=10, random_state=42, **ignored_kwargs):
    reconstructed_matrix, enc, dec = get_linear_rec_svd(
        W.cpu().detach().numpy(),
        rank,
        n_iter,
        random_state,
    )
    final_enc = torch.tensor(enc, dtype=W.dtype, device=W.device)
    final_dec = torch.tensor(dec, dtype=W.dtype, device=W.device)

    core = torch.eye(rank, dtype=W.dtype, device=W.device)
    return final_enc, core, final_dec


def reconstruct(A, R, B):
    return A @ R @ B
