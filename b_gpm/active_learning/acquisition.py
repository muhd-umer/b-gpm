from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import math

import torch
import torch.nn.functional as F

from b_gpm.models.bayesian_types import BayesianEmbedding


@dataclass
class AcquisitionResult:
    """Container for acquisition function outputs."""

    scores: torch.Tensor  # acquisition scores for each pair
    mean_scores: Optional[torch.Tensor] = None  # preference score means
    variances: Optional[torch.Tensor] = None  # preference score variances


class AcquisitionFunction(ABC):
    """Base class for acquisition functions."""

    @abstractmethod
    def __call__(
        self,
        chosen_embed: BayesianEmbedding,
        reject_embed: BayesianEmbedding,
        value_head_dim: int,
    ) -> AcquisitionResult:
        """Compute acquisition scores for a batch of preference pairs.

        Args:
            chosen_embed: Bayesian embedding for chosen responses.
            reject_embed: Bayesian embedding for rejected responses.
            value_head_dim: Dimension of the preference embedding.

        Returns:
            AcquisitionResult with scores and optional statistics.
        """
        pass

    @staticmethod
    def compute_preference_mean(
        mu_i: torch.Tensor,
        mu_j: torch.Tensor,
        value_head_dim: int,
    ) -> torch.Tensor:
        """Compute E[s] = mu_i^T R mu_j using the skew-symmetric block matrix."""
        device, dtype = mu_i.device, mu_i.dtype
        R = _create_skew_symmetric_block_matrix(value_head_dim, device, dtype)
        transformed = torch.matmul(mu_i, R.T)
        return (transformed * mu_j).sum(dim=-1)

    @staticmethod
    def compute_preference_variance(
        mu_i: torch.Tensor,
        mu_j: torch.Tensor,
        var_i: torch.Tensor,
        var_j: torch.Tensor,
        value_head_dim: int,
    ) -> torch.Tensor:
        """Compute Var[s] using closed-form expression.

        Var[s] = <mu_i^2, S*var_j> + <mu_j^2, S*var_i> + <var_i, S*var_j>

        where S is the swap permutation.
        """
        swap_idx = torch.arange(value_head_dim, device=var_i.device)
        swap_idx[0::2] = torch.arange(1, value_head_dim, 2, device=var_i.device)
        swap_idx[1::2] = torch.arange(0, value_head_dim, 2, device=var_i.device)

        swapped_var_j = var_j[:, swap_idx]
        swapped_var_i = var_i[:, swap_idx]

        term1 = (mu_i.pow(2) * swapped_var_j).sum(dim=-1)
        term2 = (mu_j.pow(2) * swapped_var_i).sum(dim=-1)
        term3 = (var_i * swapped_var_j).sum(dim=-1)

        return term1 + term2 + term3


class MaxVariance(AcquisitionFunction):
    """Maximum Variance acquisition: alpha(y_i, y_j) = Var[s(y_i > y_j)]."""

    def __call__(
        self,
        chosen_embed: BayesianEmbedding,
        reject_embed: BayesianEmbedding,
        value_head_dim: int,
    ) -> AcquisitionResult:
        mu_i, mu_j = chosen_embed.mean, reject_embed.mean
        var_i = torch.exp(chosen_embed.logvar)
        var_j = torch.exp(reject_embed.logvar)

        variance = self.compute_preference_variance(
            mu_i, mu_j, var_i, var_j, value_head_dim
        )
        mean_score = self.compute_preference_mean(mu_i, mu_j, value_head_dim)

        return AcquisitionResult(
            scores=variance,
            mean_scores=mean_score,
            variances=variance,
        )


class BALD(AcquisitionFunction):
    """Bayesian Active Learning by Disagreement.

    BALD = H[E_q[P(Y|v)]] - E_q[H[P(Y|v)]]

    This measures the mutual information between the prediction and the
    model parameters, capturing epistemic uncertainty.
    """

    def __init__(self, n_samples: int = 20, tau: float = 0.1):
        """Initialize BALD acquisition.

        Args:
            n_samples: Number of MC samples for estimating expectations.
            tau: Temperature for Bradley-Terry probability.
        """
        self.n_samples = n_samples
        self.tau = tau

    def __call__(
        self,
        chosen_embed: BayesianEmbedding,
        reject_embed: BayesianEmbedding,
        value_head_dim: int,
    ) -> AcquisitionResult:
        mu_i, mu_j = chosen_embed.mean, reject_embed.mean
        std_i = torch.exp(0.5 * chosen_embed.logvar)
        std_j = torch.exp(0.5 * reject_embed.logvar)
        var_i = std_i.pow(2)
        var_j = std_j.pow(2)

        device, dtype = mu_i.device, mu_i.dtype
        batch_size = mu_i.shape[0]
        R = _create_skew_symmetric_block_matrix(value_head_dim, device, dtype)

        # monte carlo sampling for BALD
        probs_samples = []
        for _ in range(self.n_samples):
            eps_i = torch.randn_like(mu_i)
            eps_j = torch.randn_like(mu_j)
            sample_i = F.normalize(mu_i + eps_i * std_i, p=2, dim=-1)
            sample_j = F.normalize(mu_j + eps_j * std_j, p=2, dim=-1)

            # compute preference score for this sample
            transformed = torch.matmul(sample_i, R.T)
            score = (transformed * sample_j).sum(dim=-1)
            prob = torch.sigmoid(score / self.tau)
            probs_samples.append(prob)

        probs_samples = torch.stack(probs_samples, dim=0)  # [n_samples, batch]

        # H[E[P(Y|v)]]: entropy of mean prediction
        mean_prob = probs_samples.mean(dim=0)
        mean_prob = mean_prob.clamp(1e-7, 1 - 1e-7)
        entropy_of_mean = _binary_entropy(mean_prob)

        # E[H[P(Y|v)]]: expected entropy
        probs_clamped = probs_samples.clamp(1e-7, 1 - 1e-7)
        entropies = _binary_entropy(probs_clamped)
        mean_of_entropy = entropies.mean(dim=0)

        # BALD = mutual information
        bald_scores = entropy_of_mean - mean_of_entropy

        # also compute closed-form stats for reference
        mean_score = self.compute_preference_mean(mu_i, mu_j, value_head_dim)
        variance = self.compute_preference_variance(
            mu_i, mu_j, var_i, var_j, value_head_dim
        )

        return AcquisitionResult(
            scores=bald_scores,
            mean_scores=mean_score,
            variances=variance,
        )


class UCB(AcquisitionFunction):
    """Upper Confidence Bound: alpha = |E[s]| + kappa * sqrt(Var[s]).

    Balances exploitation (high |E[s]|) with exploration (high variance).
    """

    def __init__(self, kappa: float = 1.0):
        """Initialize UCB acquisition.

        Args:
            kappa: Exploration coefficient. Higher = more exploration.
        """
        self.kappa = kappa

    def __call__(
        self,
        chosen_embed: BayesianEmbedding,
        reject_embed: BayesianEmbedding,
        value_head_dim: int,
    ) -> AcquisitionResult:
        mu_i, mu_j = chosen_embed.mean, reject_embed.mean
        var_i = torch.exp(chosen_embed.logvar)
        var_j = torch.exp(reject_embed.logvar)

        mean_score = self.compute_preference_mean(mu_i, mu_j, value_head_dim)
        variance = self.compute_preference_variance(
            mu_i, mu_j, var_i, var_j, value_head_dim
        )

        # UCB: prioritize uncertain pairs, especially those near decision boundary
        ucb_scores = mean_score.abs() + self.kappa * variance.sqrt()

        return AcquisitionResult(
            scores=ucb_scores,
            mean_scores=mean_score,
            variances=variance,
        )


class RandomAcquisition(AcquisitionFunction):
    """Random acquisition baseline for comparison."""

    def __call__(
        self,
        chosen_embed: BayesianEmbedding,
        reject_embed: BayesianEmbedding,
        value_head_dim: int,
    ) -> AcquisitionResult:
        batch_size = chosen_embed.mean.shape[0]
        device = chosen_embed.mean.device

        # random scores
        scores = torch.rand(batch_size, device=device)

        mu_i, mu_j = chosen_embed.mean, reject_embed.mean
        var_i = torch.exp(chosen_embed.logvar)
        var_j = torch.exp(reject_embed.logvar)

        mean_score = self.compute_preference_mean(mu_i, mu_j, value_head_dim)
        variance = self.compute_preference_variance(
            mu_i, mu_j, var_i, var_j, value_head_dim
        )

        return AcquisitionResult(
            scores=scores,
            mean_scores=mean_score,
            variances=variance,
        )


def _create_skew_symmetric_block_matrix(
    dim: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Create the skew-symmetric block matrix R."""
    R = torch.zeros((dim, dim), device=device, dtype=dtype)
    for i in range(0, dim, 2):
        R[i, i + 1] = -1
        R[i + 1, i] = 1
    return R


def _binary_entropy(p: torch.Tensor) -> torch.Tensor:
    """Compute binary entropy H(p) = -p*log(p) - (1-p)*log(1-p)."""
    return -p * torch.log(p) - (1 - p) * torch.log(1 - p)
