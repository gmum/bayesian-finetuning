import torch


class CovarianceAccumulator:
    """
    Incrementally calculates the global mean and covariance matrix (Sigma_XX)
    of input data using a stable online algorithm.

    Welford's Algorithm Extension: The code correctly implements an online algorithm for computing covariance, 
    which is a generalization of Welford's algorithm for variance.
    """

    def __init__(self, input_dim, device="cpu"):
        """
        Initializes the accumulator state.

        Args:
            input_dim (int): The dimensionality of the input vector (D).
            device (str): The device for calculations.
        """
        self.total_samples = 0
        self.device = device

        # M_k: The current mean vector (D,)
        self.M_k = torch.zeros(input_dim, dtype=torch.double, device=device)

        # C_k: The current sum of outer products (D, D), equivalent to
        # (N-1) * Sigma_XX for the accumulated data.
        self.C_k = torch.zeros(
            (input_dim, input_dim), dtype=torch.double, device=device
        )

        self.dtype = None

    def update(self, X_batch):
        """
        Updates the running mean and covariance statistics with a new batch of data.

        Args:
            X_batch (torch.Tensor): A batch of input data (Batch_size, Input_Dim).
        """
        self.dtype = X_batch.dtype

        # Ensure the batch is properly formatted and on the correct device
        X_batch = X_batch.to(self.device).double()

        # Flatten batch if necessary (e.g., if input is an image)
        # if X_batch.dim() > 2:
        #     X_batch = X_batch.reshape(X_batch.size(0), -1)
        assert len(X_batch.shape) == 2

        batch_size = X_batch.size(0)

        # --- Update Global Mean (M_k) and Covariance Sum (C_k) ---

        # Calculate the mean of the current batch
        batch_mean = X_batch.mean(dim=0)

        # Difference between current batch mean and global mean
        delta = batch_mean - self.M_k

        # Update mean (M_k)
        new_total = self.total_samples + batch_size
        self.M_k += delta * (batch_size / new_total)

        # Update Covariance Sum (C_k)
        X_centered = X_batch - batch_mean  # Batch data centered w.r.t batch mean

        # The cross term correction factor
        cross_term = torch.outer(delta, delta) * (
            self.total_samples * batch_size / new_total
        )

        self.C_k += X_centered.T @ X_centered + cross_term
        self.total_samples = new_total

    def get_sigma_xx(self):
        """
        Returns the final calculated covariance matrix Sigma_XX.
        """
        if self.total_samples < 2:
            # Cannot calculate covariance with less than 2 samples
            print("Warning: Insufficient samples to calculate covariance.")
            return torch.zeros_like(self.C_k, dtype=torch.float32)

        # Sigma_XX = C_k / (N - 1)
        Sigma_xx = self.C_k / (self.total_samples - 1)

        # Return in standard float32 precision for use with neural networks
        # return Sigma_xx.float()
        return Sigma_xx.type(self.dtype)
