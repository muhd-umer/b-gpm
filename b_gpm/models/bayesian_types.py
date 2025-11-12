from dataclasses import dataclass
import torch


@dataclass
class BayesianEmbedding:
    """
    Container for Bayesian preference embeddings.

    Attributes:
        sample: Sampled embedding produced via reparameterization (normalized).
        mean: Posterior mean on the unit sphere (used for deterministic inference).
        logvar: Log-variance (diagonal) in the ambient space (used for KL/uncertainty).
        raw_mean: Unnormalized posterior mean in the ambient space (used for KL).
    """

    sample: torch.Tensor
    mean: torch.Tensor
    logvar: torch.Tensor
    raw_mean: torch.Tensor

    def select(self, start: int, end: int) -> "BayesianEmbedding":
        """Return a slice of the embedding along the batch dimension."""
        return BayesianEmbedding(
            sample=self.sample[start:end],
            mean=self.mean[start:end],
            logvar=self.logvar[start:end],
            raw_mean=self.raw_mean[start:end],
        )
