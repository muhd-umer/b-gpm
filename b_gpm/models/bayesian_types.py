from dataclasses import dataclass
import torch


@dataclass
class BayesianEmbedding:
    """
    Container for Bayesian preference embeddings.

    Attributes:
        sample: Sampled embedding produced via reparameterization.
        mean: Posterior mean of the embedding.
        logvar: Log-variance (diagonal) of the embedding.
    """

    sample: torch.Tensor
    mean: torch.Tensor
    logvar: torch.Tensor

    def select(self, start: int, end: int) -> "BayesianEmbedding":
        """Return a slice of the embedding along the batch dimension."""
        return BayesianEmbedding(
            sample=self.sample[start:end],
            mean=self.mean[start:end],
            logvar=self.logvar[start:end],
        )
