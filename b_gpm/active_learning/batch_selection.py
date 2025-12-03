from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np
import torch


class BatchSelector(ABC):
    """Base class for batch selection strategies."""

    @abstractmethod
    def select(
        self,
        scores: torch.Tensor,
        features: Optional[torch.Tensor],
        k: int,
    ) -> List[int]:
        """Select k indices from candidates based on scores and diversity.

        Args:
            scores: Acquisition scores for each candidate. Shape: [N].
            features: Optional feature vectors for diversity. Shape: [N, D].
            k: Number of samples to select.

        Returns:
            List of selected indices.
        """
        pass


class TopKSelector(BatchSelector):
    """Simple top-k selection by acquisition score (no diversity)."""

    def select(
        self,
        scores: torch.Tensor,
        features: Optional[torch.Tensor],
        k: int,
    ) -> List[int]:
        k = min(k, len(scores))
        _, indices = torch.topk(scores, k)
        return indices.cpu().tolist()


class ClusterDiverseSelector(BatchSelector):
    """Cluster-based diverse selection.

    1. Cluster candidates by feature similarity
    2. Allocate budget proportionally to cluster uncertainty mass
    3. Select top samples within each cluster
    """

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        cluster_weight: float = 0.5,
    ):
        """Initialize cluster-diverse selector.

        Args:
            n_clusters: Number of clusters. If None, uses sqrt(k).
            cluster_weight: Weight given to diversity vs pure score (0-1).
        """
        self.n_clusters = n_clusters
        self.cluster_weight = cluster_weight

    def select(
        self,
        scores: torch.Tensor,
        features: Optional[torch.Tensor],
        k: int,
    ) -> List[int]:
        n = len(scores)
        k = min(k, n)

        if features is None or k <= 2:
            return TopKSelector().select(scores, features, k)

        scores_np = scores.cpu().float().numpy()
        features_np = features.cpu().float().numpy()

        # determine number of clusters
        n_clusters = self.n_clusters or max(2, int(np.sqrt(k)))
        n_clusters = min(n_clusters, n, k)

        # simple k-means clustering
        cluster_labels = self._kmeans(features_np, n_clusters)

        # allocate budget per cluster based on uncertainty mass
        cluster_scores = []
        for c in range(n_clusters):
            mask = cluster_labels == c
            if mask.sum() > 0:
                cluster_scores.append(scores_np[mask].sum())
            else:
                cluster_scores.append(0.0)

        total_score = sum(cluster_scores) + 1e-8
        allocations = [max(1, int(round(k * s / total_score))) for s in cluster_scores]

        # adjust to exactly k
        while sum(allocations) > k:
            max_idx = np.argmax(allocations)
            allocations[max_idx] -= 1
        while sum(allocations) < k:
            min_idx = np.argmin(allocations)
            allocations[min_idx] += 1

        # select top samples within each cluster
        selected = []
        for c in range(n_clusters):
            mask = cluster_labels == c
            cluster_indices = np.where(mask)[0]
            if len(cluster_indices) == 0:
                continue

            cluster_scores_c = scores_np[cluster_indices]
            n_select = min(allocations[c], len(cluster_indices))

            if n_select > 0:
                top_local = np.argsort(cluster_scores_c)[-n_select:]
                selected.extend(cluster_indices[top_local].tolist())

        # if we're short (due to empty clusters), fill with top remaining
        if len(selected) < k:
            remaining = set(range(n)) - set(selected)
            remaining_scores = [(i, scores_np[i]) for i in remaining]
            remaining_scores.sort(key=lambda x: x[1], reverse=True)
            for i, _ in remaining_scores[: k - len(selected)]:
                selected.append(i)

        return selected[:k]

    def _kmeans(
        self,
        X: np.ndarray,
        n_clusters: int,
        max_iter: int = 50,
    ) -> np.ndarray:
        """Simple k-means clustering."""
        n = len(X)
        rng = np.random.default_rng(42)

        # initialize centroids
        idx = rng.choice(n, size=min(n_clusters, n), replace=False)
        centroids = X[idx].copy()

        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            # assign labels
            dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
            new_labels = np.argmin(dists, axis=1)

            if np.all(new_labels == labels):
                break
            labels = new_labels

            # update centroids
            for c in range(n_clusters):
                mask = labels == c
                if mask.sum() > 0:
                    centroids[c] = X[mask].mean(axis=0)

        return labels


class GreedyCoreset(BatchSelector):
    """Greedy coreset selection for maximum diversity.

    Iteratively selects the point furthest from already selected points,
    weighted by acquisition score.
    """

    def __init__(self, score_weight: float = 0.5):
        """Initialize greedy coreset selector.

        Args:
            score_weight: Balance between score (1.0) and diversity (0.0).
        """
        self.score_weight = score_weight

    def select(
        self,
        scores: torch.Tensor,
        features: Optional[torch.Tensor],
        k: int,
    ) -> List[int]:
        n = len(scores)
        k = min(k, n)

        if features is None:
            return TopKSelector().select(scores, features, k)

        scores_np = scores.cpu().float().numpy()
        features_np = features.cpu().float().numpy()

        # normalize features and scores
        features_norm = features_np / (
            np.linalg.norm(features_np, axis=1, keepdims=True) + 1e-8
        )
        scores_norm = (scores_np - scores_np.min()) / (
            scores_np.max() - scores_np.min() + 1e-8
        )

        selected = []
        # start with highest scoring point
        first = np.argmax(scores_np)
        selected.append(first)

        for _ in range(k - 1):
            # compute min distance to selected set
            selected_feats = features_norm[selected]
            dists = np.linalg.norm(
                features_norm[:, None, :] - selected_feats[None, :, :], axis=2
            )
            min_dists = dists.min(axis=1)

            # mask already selected
            min_dists[selected] = -np.inf

            # combine diversity and score
            combined = self.score_weight * scores_norm + (
                1 - self.score_weight
            ) * min_dists / (min_dists.max() + 1e-8)
            combined[selected] = -np.inf

            next_idx = np.argmax(combined)
            selected.append(next_idx)

        return selected
