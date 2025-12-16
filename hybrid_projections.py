import torch
import math
from typing import Tuple, Dict, Callable

import svd_projections
import random_projections
import dct_projections
import randomized_svd_projections
import range_finder_projections
import cca_projections


# Map the string name to the corresponding function
DECOMPOSITION_METHODS: Dict[
    str, Callable[[torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
] = {
    "dct": dct_projections.permute_and_compress,
    "svd": svd_projections.compress,
    "random": random_projections.compress,
    "rsvd": randomized_svd_projections.compress,
    "finder": range_finder_projections.compress,
    "cca": cca_projections.compress,
}


def _calculate_component_ranks(specification: str, total_rank: int) -> Dict[str, int]:
    """
    Parses the specification string and calculates the concrete rank for each component.

    Returns:
        A dictionary mapping method name (str) to allocated rank (int).
    """
    print(
        f"[hybrid] Calculating component ranks: specification {specification}, total_rank {total_rank}"
    )
    parts = specification.split("_")

    specified_fractions = {}
    known_fraction_sum = 0.0

    # First pass: parse specified methods and calculate known fractions
    for part in parts:
        if "-" in part:
            method, fraction_str = part.split("-")
            if method not in DECOMPOSITION_METHODS:
                raise ValueError(f"Unknown decomposition method: {method}")

            try:
                if "/" in fraction_str:
                    numerator, denominator = map(int, fraction_str.split("/"))
                    fraction = numerator / denominator
                else:
                    fraction = float(fraction_str)
            except ValueError:
                raise ValueError(
                    f"Invalid fraction format: {fraction_str}. Must be X/Y or a float."
                )

            specified_fractions[method] = fraction
            known_fraction_sum += fraction
        else:
            if "default_method" in specified_fractions:
                raise ValueError(
                    "Only one default (unspecified fraction) method is allowed."
                )
            if part not in DECOMPOSITION_METHODS:
                raise ValueError(f"Unknown decomposition method: {part}")

            specified_fractions["default_method"] = part

    if known_fraction_sum > 1.0 + 1e-9:
        raise ValueError(
            f"Total specified fractions exceed 1.0: {known_fraction_sum:.4f}"
        )

    # Second pass: calculate concrete ranks
    component_ranks = {}
    current_rank_sum = 0
    default_method_name = None

    # Calculate ranks for methods with specified fractions
    for method_name, fraction in specified_fractions.items():
        if method_name != "default_method":
            # Use floor to guarantee we don't exceed total_rank before the default
            rank = math.floor(total_rank * fraction)
            component_ranks[method_name] = rank
            current_rank_sum += rank
        else:
            default_method_name = specified_fractions["default_method"]

    # Assign remaining rank to the default method
    if default_method_name:
        remaining_rank = total_rank - current_rank_sum
        if remaining_rank < 0:
            raise RuntimeError(
                "Internal rank calculation error: Remaining rank is negative."
            )

        component_ranks[default_method_name] = remaining_rank
        current_rank_sum += remaining_rank
    elif current_rank_sum != total_rank:
        raise ValueError(
            f"Total rank mismatch. Specified ranks sum to {current_rank_sum}, but total rank is {total_rank}. Missing default component?"
        )

    if current_rank_sum != total_rank:
        raise RuntimeError(
            f"Internal error: Final allocated rank {current_rank_sum} does not match total rank {total_rank}."
        )

    return component_ranks


def compress(
    W: torch.Tensor,
    rank: int,
    specification: str = "dct-1/3_random-1/3_svd",
    **cfg_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes a mixed low-rank decomposition W approx A R B based on a string specification.

    Args:
        W (torch.Tensor): The input weight matrix (n x m).
        specification (str): e.g., "svd-1/3_random-1/3_dct".
        total_rank (int): The maximum target rank 'r'.

    Returns:
        A_combined (torch.Tensor): Left projection (n x r).
        R_combined (torch.Tensor): Center mixing matrix (r x r, block diagonal).
        B_combined (torch.Tensor): Right projection (r x m).
    """
    n, m = W.shape
    parts = specification.split("_")

    # 1. Calculate Ranks
    component_ranks = _calculate_component_ranks(specification, rank)
    print(f"[hybrid] Component Ranks: {component_ranks}")
    print(f"[hybrid] Configuration per part: {cfg_kwargs}")

    # --- 2. Execute Decompositions and Combine Results ---

    A_components = []
    R_components = []
    B_components = []

    # Track methods already processed to handle the default component correctly
    processed_methods = {}

    # Iterate through the original specification parts to maintain order
    for part in parts:
        method_name = part.split("-")[0]

        if method_name in processed_methods:
            continue

        # Determine the rank allocated to this method
        rank = component_ranks.get(method_name)

        # Skip if allocated rank is zero or missing
        if rank is None or rank == 0:
            continue

        # Execute the decomposition
        decomp_func = DECOMPOSITION_METHODS[method_name]
        A_i, R_i, B_i = decomp_func(W, rank=rank, **cfg_kwargs)

        # Store components
        A_components.append(A_i)
        R_components.append(R_i)
        B_components.append(B_i)

        processed_methods[method_name] = True

    # --- 3. Block Concatenation ---

    if not A_components:
        return torch.empty(n, 0), torch.empty(0, 0), torch.empty(0, m)

    # A: Concatenate along columns (dim=1) -> n x r_total
    A_combined = torch.cat(A_components, dim=1)

    # R: Create a Block-Diagonal matrix -> r_total x r_total
    R_combined = torch.block_diag(*R_components)

    # B: Concatenate along rows (dim=0) -> r_total x m
    B_combined = torch.cat(B_components, dim=0)

    return A_combined, R_combined, B_combined


def reconstruct(A, R, B):
    """
    Reconstructs the original matrix from the compressed representation.

    Computes: W_reconstructed = A @ R @ B

    Returns:
        torch.Tensor: The reconstructed 2D matrix.
    """
    reconstructed_matrix = A @ R @ B
    return reconstructed_matrix
