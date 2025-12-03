from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any, Tuple
import logging
import os
import json
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from b_gpm.models.bayesian_types import BayesianEmbedding
from .acquisition import AcquisitionFunction, AcquisitionResult, MaxVariance
from .batch_selection import BatchSelector, TopKSelector, ClusterDiverseSelector
from .pool import UnlabeledPool, PoolDataset, PreferencePair


logger = logging.getLogger(__name__)


@dataclass
class ActiveLearningConfig:
    """Configuration for active learning."""

    # acquisition
    batch_size: int = 32  # samples to select per iteration
    acquisition_batch_size: int = 64  # batch size for computing embeddings
    use_diverse_selection: bool = True

    # stopping criteria
    max_iterations: int = 100
    max_labels: int = 10000
    target_accuracy: Optional[float] = None

    # evaluation
    eval_every: int = 5
    save_every: int = 10

    # paths
    output_dir: str = "./active_learning_results"
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

        os.makedirs(self.config.output_dir, exist_ok=True)

    def run(self) -> ActiveLearningState:
        """Run the active learning loop."""
        logger.info("Starting active learning loop")
        logger.info(f"Pool size: {len(self.pool)}, Labeled: {self.pool.n_labeled()}")

        while not self._should_stop():
            self.state.iteration += 1
            iter_start = datetime.now()

            logger.info(f"Iteration {self.state.iteration}")

            # step 1: compute acquisition scores for unlabeled samples
            unlabeled_indices = self.pool.get_unlabeled_indices()
            if len(unlabeled_indices) == 0:
                logger.info("No more unlabeled samples")
                break

            logger.info(
                f"Computing acquisition scores for {len(unlabeled_indices)} samples..."
            )
            scores, features = self._compute_acquisition_scores(unlabeled_indices)

            # step 2: select batch
            n_select = min(self.config.batch_size, len(unlabeled_indices))
            selected_local = self.batch_selector.select(scores, features, n_select)
            selected_pool_indices = [unlabeled_indices[i] for i in selected_local]

            logger.info(f"Selected {len(selected_pool_indices)} samples for labeling")

            # step 3: query oracle
            selected_pairs = self.pool.get_pairs(selected_pool_indices)
            labels = self.oracle(selected_pairs)

            # step 4: update pool with labels
            self.pool.label_pairs(selected_pool_indices, labels)
            self.state.total_labels += len(labels)

            # step 5: retrain model (if train_fn provided)
            if self.train_fn is not None:
                logger.info("Retraining model...")
                training_data = self.pool.to_training_format()
                self.model = self.train_fn(self.model, training_data)

            # step 6: evaluate (if eval_fn provided)
            metrics = {}
            if (
                self.eval_fn is not None
                and self.state.iteration % self.config.eval_every == 0
            ):
                logger.info("Evaluating model...")
                metrics = self.eval_fn(self.model)
                if "accuracy" in metrics:
                    if metrics["accuracy"] > self.state.best_accuracy:
                        self.state.best_accuracy = metrics["accuracy"]
                logger.info(f"Metrics: {metrics}")

            # record history
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

            # save checkpoint
            if self.state.iteration % self.config.save_every == 0:
                self._save_checkpoint()

        logger.info(f"\nActive learning complete!")
        logger.info(f"Total iterations: {self.state.iteration}")
        logger.info(f"Total labels: {self.state.total_labels}")
        logger.info(f"Best accuracy: {self.state.best_accuracy:.4f}")

        self._save_checkpoint()
        return self.state

    def _compute_acquisition_scores(
        self, indices: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute acquisition scores for given pool indices."""
        self.model.eval()

        dataset = PoolDataset(
            self.pool,
            indices,
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
            for batch in tqdm(dataloader, desc="Computing scores", leave=False):
                chosen_ids, chosen_mask, reject_ids, reject_mask, _ = batch
                chosen_ids = chosen_ids.to(self.device)
                chosen_mask = chosen_mask.to(self.device)
                reject_ids = reject_ids.to(self.device)
                reject_mask = reject_mask.to(self.device)

                # get embeddings
                chosen_embed = self._get_embedding(chosen_ids, chosen_mask)
                reject_embed = self._get_embedding(reject_ids, reject_mask)

                # compute acquisition scores
                result = self.acquisition_fn(
                    chosen_embed, reject_embed, self.value_head_dim
                )
                all_scores.append(result.scores.cpu())

                # collect features for diverse selection
                features = torch.cat(
                    [
                        chosen_embed.mean,
                        reject_embed.mean,
                        torch.exp(0.5 * chosen_embed.logvar),
                        torch.exp(0.5 * reject_embed.logvar),
                    ],
                    dim=-1,
                )
                all_features.append(features.cpu())

        scores = torch.cat(all_scores, dim=0)
        features = torch.cat(all_features, dim=0)
        return scores, features

    def _get_embedding(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> BayesianEmbedding:
        """Get Bayesian embedding from model."""
        model = self.model
        if hasattr(model, "module"):
            model = model.module

        # use the model's custom_forward method
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
                # if no ground truth, default to 1 (chosen > rejected)
                labels.append(1)
        return labels

    def _should_stop(self) -> bool:
        """Check if active learning should stop."""
        if self.state.iteration >= self.config.max_iterations:
            logger.info("Reached max iterations")
            return True

        if self.state.total_labels >= self.config.max_labels:
            logger.info("Reached max labels")
            return True

        if (
            self.config.target_accuracy is not None
            and self.state.best_accuracy >= self.config.target_accuracy
        ):
            logger.info(f"Reached target accuracy: {self.state.best_accuracy:.4f}")
            return True

        if self.pool.n_unlabeled() == 0:
            logger.info("Pool exhausted")
            return True

        return False

    def _save_checkpoint(self) -> None:
        """Save active learning state."""
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

        # also save pool state
        pool_path = os.path.join(self.config.output_dir, "pool_state.json")
        self.pool.save(pool_path)

        logger.info(f"Saved checkpoint to {path}")

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
