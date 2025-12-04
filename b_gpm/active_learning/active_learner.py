from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any, Tuple
import logging
import os
import json
import math
from datetime import datetime

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from b_gpm.models.bayesian_types import BayesianEmbedding
from .acquisition import AcquisitionFunction, AcquisitionResult, MaxVariance
from .batch_selection import BatchSelector, TopKSelector, ClusterDiverseSelector
from .pool import UnlabeledPool, PoolDataset, PreferencePair

logger = logging.getLogger(__name__)


def _is_rank_0() -> bool:
    """Check if current process is rank 0."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


@dataclass
class ActiveLearningConfig:
    """Configuration for active learning."""

    # acquisition
    batch_size: int = 32  # samples to select per iteration
    acquisition_batch_size: int = 64  # batch size for computing embeddings
    use_diverse_selection: bool = True

    max_iterations: int = 100
    max_labels: int = 10000
    target_accuracy: Optional[float] = None

    eval_every: int = 5
    save_every: int = 10

    output_dir: str = "./al_results"
    checkpoint_dir: Optional[str] = None


@dataclass
class ActiveLearningState:
    """State of the active learning process."""

    iteration: int = 0
    total_labels: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    best_accuracy: float = 0.0


class ActiveLearner:
    """Active learning loop for Bayesian GPM.

    Workflow:
    1. Compute embeddings for unlabeled pool
    2. Calculate acquisition scores
    3. Select diverse batch of high-scoring samples
    4. Query oracle for labels
    5. Update training set and retrain model
    6. Evaluate and repeat
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        pool: UnlabeledPool,
        acquisition_fn: AcquisitionFunction = None,
        batch_selector: BatchSelector = None,
        config: ActiveLearningConfig = None,
        oracle: Callable[[List[PreferencePair]], List[int]] = None,
        eval_fn: Callable[[torch.nn.Module], Dict[str, float]] = None,
        train_fn: Callable[[torch.nn.Module, List[Dict]], torch.nn.Module] = None,
        value_head_dim: int = 6,
    ):
        """Initialize active learner.

        Args:
            model: Trained Bayesian GPM model.
            tokenizer: Tokenizer for the model.
            pool: Pool of unlabeled preference pairs.
            acquisition_fn: Acquisition function (default: MaxVariance).
            batch_selector: Batch selection strategy (default: ClusterDiverseSelector).
            config: Active learning configuration.
            oracle: Function to query labels. Takes pairs, returns labels (0 or 1).
                    If None, uses ground truth from pool (for simulation).
            eval_fn: Function to evaluate model. Returns dict of metrics.
            train_fn: Function to train/fine-tune model on labeled data.
            value_head_dim: Dimension of preference embeddings.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.pool = pool
        self.acquisition_fn = acquisition_fn or MaxVariance()
        self.batch_selector = batch_selector or ClusterDiverseSelector()
        self.config = config or ActiveLearningConfig()
        self.oracle = oracle or self._simulated_oracle
        self.eval_fn = eval_fn
        self.train_fn = train_fn
        self.value_head_dim = value_head_dim

        self.state = ActiveLearningState()
        self.device = next(model.parameters()).device

        if _is_rank_0():
            os.makedirs(self.config.output_dir, exist_ok=True)

    def _log(self, msg: str):
        """Log message only on rank 0."""
        if _is_rank_0():
            logger.info(msg)

    def run(self) -> ActiveLearningState:
        """Run the active learning loop."""
        self._log("Starting active learning loop")
        self._log(f"Pool size: {len(self.pool)}, Labeled: {self.pool.n_labeled()}")

        while not self._should_stop():
            self.state.iteration += 1
            iter_start = datetime.now()

            self._log(f"Iteration {self.state.iteration}")

            unlabeled_indices = self.pool.get_unlabeled_indices()
            if len(unlabeled_indices) == 0:
                self._log("No more unlabeled samples")
                break

            self._log(
                f"Computing acquisition scores for {len(unlabeled_indices)} samples..."
            )
            scores, features = self._compute_acquisition_scores(unlabeled_indices)

            n_select = min(self.config.batch_size, len(unlabeled_indices))
            selected_local = self.batch_selector.select(scores, features, n_select)
            selected_pool_indices = [unlabeled_indices[i] for i in selected_local]

            self._log(f"Selected {len(selected_pool_indices)} samples for labeling")

            selected_pairs = self.pool.get_pairs(selected_pool_indices)
            labels = self.oracle(selected_pairs)

            self.pool.label_pairs(selected_pool_indices, labels)
            self.state.total_labels += len(labels)

            if self.train_fn is not None:
                self._log("Retraining model...")
                training_data = self.pool.to_training_format()
                self.model = self.train_fn(self.model, training_data)

            metrics = {}
            if (
                self.eval_fn is not None
                and self.state.iteration % self.config.eval_every == 0
            ):
                self._log("Evaluating model...")
                metrics = self.eval_fn(self.model)
                if "accuracy" in metrics:
                    if metrics["accuracy"] > self.state.best_accuracy:
                        self.state.best_accuracy = metrics["accuracy"]
                self._log(f"Metrics: {metrics}")

            iter_record = {
                "iteration": self.state.iteration,
                "n_labeled": self.pool.n_labeled(),
                "n_selected": len(selected_pool_indices),
                "selected_indices": selected_pool_indices,
                "acquisition_scores": scores[selected_local].tolist(),
                "duration_seconds": (datetime.now() - iter_start).total_seconds(),
                **metrics,
            }
            self.state.history.append(iter_record)

            if self.state.iteration % self.config.save_every == 0:
                self._save_checkpoint()

        self._log(f"\nActive learning complete!")
        self._log(f"Total iterations: {self.state.iteration}")
        self._log(f"Total labels: {self.state.total_labels}")
        self._log(f"Best accuracy: {self.state.best_accuracy:.4f}")

        self._save_checkpoint()
        return self.state

    def _compute_acquisition_scores(
        self, indices: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute acquisition scores for given pool indices."""
        self.model.eval()

        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()

            chunk_size = math.ceil(len(indices) / world_size)
            start_idx = rank * chunk_size
            end_idx = min(start_idx + chunk_size, len(indices))
            local_indices = indices[start_idx:end_idx]
        else:
            local_indices = indices
            world_size = 1
            rank = 0

        dataset = PoolDataset(
            self.pool,
            local_indices,
            self.tokenizer,
            max_length=2048,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.acquisition_batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

        all_scores = []
        all_features = []

        with torch.no_grad():
            for batch in tqdm(
                dataloader, desc="Computing scores", leave=False, disable=rank != 0
            ):
                chosen_ids, chosen_mask, reject_ids, reject_mask, _ = batch
                chosen_ids = chosen_ids.to(self.device)
                chosen_mask = chosen_mask.to(self.device)
                reject_ids = reject_ids.to(self.device)
                reject_mask = reject_mask.to(self.device)

                chosen_embed = self._get_embedding(chosen_ids, chosen_mask)
                reject_embed = self._get_embedding(reject_ids, reject_mask)

                result = self.acquisition_fn(
                    chosen_embed, reject_embed, self.value_head_dim
                )
                all_scores.append(result.scores)

                features = torch.cat(
                    [
                        chosen_embed.mean,
                        reject_embed.mean,
                        torch.exp(0.5 * chosen_embed.logvar),
                        torch.exp(0.5 * reject_embed.logvar),
                    ],
                    dim=-1,
                )
                all_features.append(features)

        if len(all_scores) > 0:
            local_scores = torch.cat(all_scores, dim=0)
            local_features = torch.cat(all_features, dim=0)
        else:
            local_scores = torch.tensor([], device=self.device)
            feature_dim = 4 * self.value_head_dim
            local_features = torch.zeros((0, feature_dim), device=self.device)

        if world_size > 1:

            local_size = torch.tensor([local_scores.size(0)], device=self.device)
            all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
            dist.all_gather(all_sizes, local_size)

            max_size = max([s.item() for s in all_sizes])

            padded_scores = torch.zeros(
                max_size, device=self.device, dtype=local_scores.dtype
            )
            if local_scores.size(0) > 0:
                padded_scores[: local_scores.size(0)] = local_scores

            gathered_scores = [
                torch.zeros_like(padded_scores) for _ in range(world_size)
            ]
            dist.all_gather(gathered_scores, padded_scores)

            final_scores = []
            for i, size in enumerate(all_sizes):
                final_scores.append(gathered_scores[i][: size.item()])
            scores = torch.cat(final_scores, dim=0)

            padded_features = torch.zeros(
                (max_size, local_features.size(1)),
                device=self.device,
                dtype=local_features.dtype,
            )
            if local_features.size(0) > 0:
                padded_features[: local_features.size(0)] = local_features

            gathered_features = [
                torch.zeros_like(padded_features) for _ in range(world_size)
            ]
            dist.all_gather(gathered_features, padded_features)

            final_features = []
            for i, size in enumerate(all_sizes):
                final_features.append(gathered_features[i][: size.item()])
            features = torch.cat(final_features, dim=0)
        else:
            scores = local_scores
            features = local_features

        return scores.cpu(), features.cpu()

    def _get_embedding(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> BayesianEmbedding:
        """Get Bayesian embedding from model."""
        model = self.model
        if hasattr(model, "module"):
            model = model.module

        embedding, _ = model.custom_forward(
            input_ids, attention_mask=attention_mask, return_output=False
        )
        if not isinstance(embedding, BayesianEmbedding):
            raise ValueError(
                "Model must return BayesianEmbedding. "
                "Ensure is_bayesian_gpm=True when loading the model."
            )
        return embedding

    def _simulated_oracle(self, pairs: List[PreferencePair]) -> List[int]:
        """Simulated oracle using ground truth labels from pool."""
        labels = []
        for pair in pairs:
            if pair.label is not None:
                labels.append(pair.label)
            else:

                labels.append(1)
        return labels

    def _should_stop(self) -> bool:
        """Check if active learning should stop."""
        if self.state.iteration >= self.config.max_iterations:
            self._log("Reached max iterations")
            return True

        if self.state.total_labels >= self.config.max_labels:
            self._log("Reached max labels")
            return True

        if (
            self.config.target_accuracy is not None
            and self.state.best_accuracy >= self.config.target_accuracy
        ):
            self._log(f"Reached target accuracy: {self.state.best_accuracy:.4f}")
            return True

        if self.pool.n_unlabeled() == 0:
            self._log("Pool exhausted")
            return True

        return False

    def _save_checkpoint(self) -> None:
        """Save active learning state."""
        if not _is_rank_0():
            return

        checkpoint = {
            "iteration": self.state.iteration,
            "total_labels": self.state.total_labels,
            "best_accuracy": self.state.best_accuracy,
            "history": self.state.history,
            "config": {
                "batch_size": self.config.batch_size,
                "max_iterations": self.config.max_iterations,
                "max_labels": self.config.max_labels,
            },
        }

        path = os.path.join(
            self.config.output_dir, f"checkpoint_iter_{self.state.iteration}.json"
        )
        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        pool_path = os.path.join(self.config.output_dir, "pool_state.json")
        self.pool.save(pool_path)

        self._log(f"Saved checkpoint to {path}")

    def get_acquisition_stats(self) -> Dict[str, Any]:
        """Get statistics about acquisition over iterations."""
        if not self.state.history:
            return {}

        return {
            "iterations": len(self.state.history),
            "total_labels": self.state.total_labels,
            "labels_per_iteration": [h["n_selected"] for h in self.state.history],
            "accuracies": [
                h.get("accuracy") for h in self.state.history if "accuracy" in h
            ],
        }
