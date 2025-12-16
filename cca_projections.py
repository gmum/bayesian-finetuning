import torch
import torch.nn.functional as F

from covariance_computation import CovarianceAccumulator

import sys
from tqdm import tqdm


def capture_module_input_hook(module, input, output, module_name):
    """
    Passes the input tensors X to CovarianceAccumulators stored in a dictionary.

    The 'input' is a tuple, even if only one tensor is passed to the module.
    We are interested in the first element (the tensor X).
    """
    if capture_module_input_hook.print_line_counter < 70:
        capture_module_input_hook.print_line_counter += 1
        print(
            f"[cca.capture_input_hook] Capturing input for module: {module_name} = {module}, "
            f"len(input) = {len(input)} "
            f"input shape: {input[0].shape}, output shape: {output.shape}"
        )

    assert len(input) == 1
    input = input[0]

    if module_name not in capture_module_input_hook.covariance_accumulators:
        print(
            f"[cca.capture_input_hook] Creating CovarianceAccumulator input_dim={input.shape[-1]}, device={input.device}"
        )
        acm = CovarianceAccumulator(input_dim=input.shape[-1], device=input.device)
        capture_module_input_hook.covariance_accumulators[module_name] = acm

    # Update stats
    cov = capture_module_input_hook.covariance_accumulators[module_name]
    flattened_input = input.view(
        -1, input.shape[-1]
    )  # (batch*input_length) x attention_size
    cov.update(flattened_input)


capture_module_input_hook.print_line_counter = 0
capture_module_input_hook.covariance_accumulators = {}


def move_covariances_to_cpu():
    print("[cca.move_covariances_to_cpu] moving tensors to CPU")
    for acm in capture_module_input_hook.covariance_accumulators.values():
        acm.to_cpu()
    torch.cuda.empty_cache()


def register_module_hooks(model, target_modules):
    print(
        f"[cca.register_hooks_for_cca] Registering hooks for target modules: {target_modules} (model={model})"
    )

    handles = []  # Keep track of handles to remove hooks later (important for cleanup!)
    for module_name, module in model.named_modules():
        # We only care about the target modules within the entire model structure.
        # The names in model.named_modules() will be prefixed, e.g., 'encoder.layer.0.output.dense'

        target_module_found = any(
            module_name.endswith(target_key) for target_key in target_modules
        )

        if target_module_found:
            print(
                f"[cca.register_hooks_for_cca] Registering hook for module: {module_name} = {module}"
            )

            # We use a lambda to pass the specific 'name' of the module to the generic hook function
            handle = module.register_forward_hook(
                lambda mod, input, output, name=module_name: capture_module_input_hook(
                    mod, input, output, name
                )
            )
            handles.append(handle)
            print(f"[cca.register_hooks_for_cca] Hook registered on: {module_name}")

    return handles


def remove_module_hooks(handles):
    # Remove the hooks when you're done to restore the model's original state and prevent memory leaks.
    for handle in handles:
        handle.remove()


@torch.no_grad
def precompute_covariances(
    pretrained_model, target_modules, dataloader, max_steps=None, device=None
):
    handles = register_module_hooks(pretrained_model, target_modules)

    device = device or pretrained_model.device

    pretrained_model.eval()
    print("[cca] Pre-computing layers input covariances by pushing data through model")
    for step, batch in enumerate(tqdm(dataloader)):
        _ = pretrained_model(
            **{
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
        )

        if max_steps is not None and step >= max_steps:
            print(
                f"WARNINIG: pre-mature stopping @step={step} of covariance computation!"
            )
            break
    pretrained_model.train()

    remove_module_hooks(handles)

    for name, acc in list(capture_module_input_hook.covariance_accumulators.items()):
        print(f"[precompute_cca] covariance for {name} = {acc.get_sigma_xx().shape}")

        # Add aliases
        capture_module_input_hook.covariance_accumulators[
            "base_model.model." + name
        ] = acc


def compress_whitened_svd_matmul(
    W: torch.Tensor, keep_dims: tuple, Sigma_xx: torch.Tensor
):
    """
    CCA-based compression specifically for Y = X @ W convention.

    ### The Optimization Goal (The Loss)

    The code minimizes the **Expected Output Error**, not just the weight reconstruction error.

    Given:
    * Input $x$ (row vector)
    * Weights $W$ (matrix $N_{in} \\times N_{out}$)
    * Output $y = xW$

    We seek an approximation $\\hat{W}$ of rank $k$ that minimizes:
    $$L = \\mathbb{E}_x [ || xW - x\\hat{W} ||^2 ]$$

    This expands to a weighted Frobenius norm involving the input covariance $\\Sigma_{xx}$:
    $$L = || (W - \\hat{W})^T \\Sigma_{xx}^{1/2} ||_F^2$$

    This effectively means we want to perform SVD on the **Whitened Weight Matrix** $\\tilde{W}$:
    $$\\tilde{W} = W^T \\Sigma_{xx}^{1/2}$$

    ---

    ### The SVD Step

    The code computes the SVD of this whitened matrix:
    $$\\tilde{W} \\approx U \\cdot S \\cdot \\tilde{V}^T$$

    Where:
    * $U$: Principal directions of the *Output*.
    * $S$: Singular values (importance).
    * $\\tilde{V}$: Principal directions of the *Whitened Input*.

    Substitute this back into the definition of $\\tilde{W}$:
    $$W^T \\Sigma_{xx}^{1/2} \\approx U S \\tilde{V}^T$$

    To solve for the actual weights $W^T$, we right-multiply by the inverse whitener $\\Sigma_{xx}^{-1/2}$:
    $$W^T \\approx U S (\\tilde{V}^T \\Sigma_{xx}^{-1/2})$$

    Let's transpose the whole equation to match the code's $Y=XW$ layout:
    $$W \\approx (\\Sigma_{xx}^{-1/2} \\tilde{V}) \\cdot S \\cdot U^T$$

    ---

    ### The Derivation of `V_raw`

    The term in the parentheses above is exactly what `V_raw` represents in the code. It maps the orthogonal directions found in the "whitened" space back to the "raw" input feature space.

    $$V_{raw} = \\Sigma_{xx}^{-1/2} \\tilde{V}$$

    In the code:
    * `Whitener_inverse` is $\\Sigma_{xx}^{-1/2}$.
    * `Vh_k.T` is $\\tilde{V}$ (the right singular vectors).
    * Therefore: `V_raw = Whitener_inverse @ Vh_k.T`.

    ---

    ### The Scaling and Normalization 

    This is the subtle but critical step.

    **The Problem:**
    While $\\tilde{V}$ (from SVD) contains orthonormal vectors (unit length), $V_{raw}$ does **not**. The matrix $\\Sigma_{xx}^{-1/2}$ stretches and shrinks vectors based on the variance of the input features.
    * Inputs with **low variance** get multiplied by large numbers (inverse of small sigma), resulting in **long vectors** in $V_{raw}$.
    * Inputs with **high variance** get multiplied by small numbers, resulting in **short vectors**.

    Neural networks generally prefer projection matrices (like $A$) to have normalized columns (unit norm) for numerical stability.

    **The Operation:**
    1.  **Calculate Norms:** `scales = torch.norm(V_raw, dim=0)` calculates the length of each vector in $V_{raw}$.
    2.  **Normalize:** `Input_Basis_A = V_raw / scales` forces the input basis $A$ to have unit length.

    Mathematically, we are splitting $V_{raw}$ into a direction matrix ($A$) and a scaling diagonal matrix ($D_{scales}$):
    $$V_{raw} = A \\cdot D_{scales}$$

    ---

    ### Compensating in the Core (Diagonal Scaling)

    If we replace $V_{raw}$ with the normalized $A$, we change the magnitude of the approximation. To preserve mathematical exactness, we must move the "missing length" into the Core matrix $R$.

    Recall the reconstruction formula from step 2:
    $$W \\approx \\underbrace{V_{raw}}_{\\text{Input Basis}} \\cdot \\underbrace{S}_{\\text{Singular Values}} \\cdot \\underbrace{U^T}_{\\text{Output Basis}}$$

    Substitute $V_{raw} = A \\cdot D_{scales}$:
    $$W \\approx (A \\cdot D_{scales}) \\cdot S \\cdot U^T$$

    Since $S$ is diagonal and $D_{scales}$ is diagonal (and diagonal matrices commute):
    $$W \\approx A \\cdot (S \\cdot D_{scales}) \\cdot U^T$$

    This is why the Core $R$ is calculated as `S_k * scales_k` in the subsequent lines. We are absorbing the magnitude information—which represents how "sensitive" the network is to specific input directions—directly into the singular values.

    Args:
        W: Weight matrix of shape (N_in, N_out).
        keep_dims: Tuple (rank_in, rank_out).
        Sigma_xx: Input covariance matrix of shape (N_in, N_in).

    Returns:
        A (Input Projection): Shape (N_in, rank_in)
        R (Core):             Shape (rank_in, rank_out)
        B (Output Basis):     Shape (rank_out, N_out)

        Reconstruction: W_approx = A @ R @ B
    """
    N_in, N_out = W.shape
    k_in, k_out = keep_dims

    assert Sigma_xx.shape == (
        N_in,
        N_in,
    ), f"Sigma shape mismatch: {Sigma_xx.shape} vs ({N_in}, {N_in})"

    # --- Whitening Preparation ---
    # We transpose W to (N_out, N_in) to align with standard derivation: y = W_proc * x
    W_proc = W.T

    L, Q = torch.linalg.eigh(Sigma_xx)
    L = torch.clamp(L, min=1e-10)

    # Forward Whitener (Sigma^0.5): Transforms weights to 'whitened input space'
    Whitener_forward = Q @ torch.diag(torch.sqrt(L)) @ Q.T

    # Inverse Whitener (Sigma^-0.5): Transforms basis back to 'raw input space'
    L_inv_sqrt = torch.rsqrt(L)
    Whitener_inverse = Q @ torch.diag(L_inv_sqrt) @ Q.T

    # --- CCA / Whitened SVD ---
    # W_tilde = W_proc @ Sigma^0.5
    # Represents weights acting on decorrelated inputs.
    W_tilde = W_proc @ Whitener_forward

    # SVD: W_tilde approx U_tilde @ S @ Vh_tilde
    U_tilde, S_tilde, Vh_tilde = torch.linalg.svd(W_tilde, full_matrices=False)

    # --- Compute Input Projection (A) ---
    # Select top k_in singular vectors for input
    Vh_k = Vh_tilde[:k_in, :]  # (k_in, N_in)

    # Un-whiten: V_raw = Sigma^-0.5 @ V_tilde_transposed
    V_raw = Whitener_inverse @ Vh_k.T  # (N_in, k_in)

    # Calculate norms for scaling correction
    # We normalize A to ensure stable activations, but we must record the scale
    # to put it back into the Core R.
    scales = torch.norm(V_raw, dim=0)

    # Avoid division by zero
    scales = torch.clamp(scales, min=1e-8)

    # A is normalized V_raw
    Input_Basis_A = V_raw / scales.view(1, -1)

    # --- Compute Core (R) via Diagonal Scaling ---
    # The analytical CCA solution implies R is diagonal containing the singular values.
    # Because we normalized A (divided by 'scales'), we must multiply R by 'scales'.

    # Get the singular values corresponding to the overlap of rank_in and rank_out
    k_min = min(k_in, k_out)
    S_k = S_tilde[:k_min]

    # The corresponding scales for the active columns of A
    scales_k = scales[:k_min]

    # Construct Diagonal Core
    # R_diag = S * scales
    R_values = S_k * scales_k

    # Initialize Core with zeros
    Core_R = torch.zeros((k_in, k_out), device=W.device, dtype=W.dtype)

    # Fill diagonal
    Core_R[:k_min, :k_min] = torch.diag(R_values)

    # --- Compute Output Basis (B) ---
    # Standard SVD output vectors
    # U_k shape: (N_out, k_out) -> B shape: (k_out, N_out)
    U_k = U_tilde[:, :k_out]
    Output_Basis_B = U_k.T

    return Input_Basis_A, Core_R, Output_Basis_B


def compress_whitened_svd_noisy(
    W: torch.Tensor, keep_dims: tuple, Sigma_xx: torch.Tensor
):
    """
    CCA-based compression specifically for Y = X @ W convention.

    Args:
        W: Weight matrix of shape (N_in, N_out).
        keep_dims: Tuple (rank_in, rank_out).
        Sigma_xx: Input covariance matrix of shape (N_in, N_in).

    Returns:
        A (Input Projection): Shape (N_in, rank_in)
        R (Core):             Shape (rank_in, rank_out)
        B (Output Basis):     Shape (rank_out, N_out)

        Reconstruction: W_approx = A @ R @ B
    """
    N_in, N_out = W.shape
    k_in, k_out = keep_dims
    print(f"[cca] fitting for {N_in} x {N_out} -> {k_in} x {k_out}")

    assert Sigma_xx.shape == (
        N_in,
        N_in,
    ), f"Sigma shape mismatch: {Sigma_xx.shape} vs ({N_in}, {N_in})"

    # --- Whitening Preparation ---
    # We need to process W as (Out, In) for the derivation logic, so we transpose.
    # W_proc shape: (N_out, N_in)
    W_proc = W.T

    L, Q = torch.linalg.eigh(Sigma_xx)
    L = torch.clamp(L, min=1e-10)

    # Forward Whitener (Sigma^0.5): Transforms weights to 'whitened input space'
    Whitener_forward = Q @ torch.diag(torch.sqrt(L)) @ Q.T

    # Inverse Whitener (Sigma^-0.5): Transforms basis back to 'raw input space'
    L_inv_sqrt = torch.rsqrt(L)
    Whitener_inverse = Q @ torch.diag(L_inv_sqrt) @ Q.T

    # --- CCA / Whitened SVD ---
    # W_tilde = W_proc @ Sigma^0.5
    # Represents weights acting on decorrelated inputs.
    W_tilde = W_proc @ Whitener_forward

    # SVD: W_tilde approx U_tilde @ S @ Vh_tilde
    U_tilde, _, Vh_tilde = torch.linalg.svd(W_tilde, full_matrices=False)

    # Truncate
    # U_k: Output directions (N_out, k_out)
    # Vh_k: Whitened Input directions (rows) (k_in, N_in)
    U_k = U_tilde[:, :k_out]
    Vh_k = Vh_tilde[:k_in, :]

    # --- Compute Input Projection (A) ---
    # We must Un-whiten the V basis to apply it to raw X.
    # V_k = Sigma^-0.5 @ V_tilde_transposed
    V_k = Whitener_inverse @ Vh_k.T

    # Normalize columns (N_in, k_in)
    Input_Basis_A = F.normalize(V_k, dim=0)

    # --- Compute Core (R) ---
    # R_core (math view) = U^T @ W_proc @ V
    # Shape: (k_out, N_out) @ (N_out, N_in) @ (N_in, k_in) -> (k_out, k_in)
    R_core_math = U_k.T @ W_proc @ Input_Basis_A

    # For Y = XW, we want the transpose of the core to match A @ R @ B
    Core_R = R_core_math.T  # Shape (k_in, k_out)

    # --- Compute Output Basis (B) ---
    # U_k is (N_out, k_out). For W approx A @ R @ B, B must be (k_out, N_out).
    Output_Basis_B = U_k.T

    return Input_Basis_A, Core_R, Output_Basis_B


def compress(
    layer_weight,
    rank=None,
    keep_dims=(10, 10),
    module_name=None,
    input_batch=None,
    Sigma_xx=None,
    absorb_R=True,
    compress_whitened_svd=compress_whitened_svd_matmul,
):
    """
    Compresses W using CCA-inspired bases (SVD on whitened weights).
    """
    print(
        f"[cca] Applying for module_name={module_name} layer_weight={layer_weight.shape}"
    )

    if rank is not None:
        keep_dims = (rank, rank)

    if module_name is not None:
        print(f"[cca] Retrieving Sigma_xx for module_name = {module_name}")
        Sigma_xx = capture_module_input_hook.covariance_accumulators[
            module_name
        ].get_sigma_xx()
    assert Sigma_xx is not None or input_batch is not None

    if Sigma_xx is None:
        # 1. Center the inputs
        X = input_batch - input_batch.mean(dim=0, keepdim=True)

        # 2. Compute Input Covariance Sigma_xx
        # (Using a small epsilon for numerical stability in inversion)
        N = X.shape[0]
        Sigma_xx = (X.T @ X) / (N - 1)
        epsilon = 1e-6 * torch.eye(Sigma_xx.shape[0], device=X.device)
        Sigma_xx += epsilon

    Sigma_xx = Sigma_xx.to(layer_weight.device)

    A, R, B = compress_whitened_svd(layer_weight, keep_dims, Sigma_xx)

    if absorb_R:
        # Absorb R into A and B for simplified representation
        A = A @ R  # Now A is m × k_cols
        R = torch.eye(
            rank, device=layer_weight.device, dtype=layer_weight.dtype
        )  # Identity matrix
        print(f"[cca] After absorbing R, new A shape: {A.shape}, R shape: {R.shape}")

    return A, R, B
