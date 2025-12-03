from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import json
import torch
from torch.utils.data import Dataset


@dataclass
class PreferencePair:
    """A single preference pair (possibly unlabeled)."""

    prompt: str
    chosen: str
    rejected: str
    label: Optional[int] = (
        None  # 1 if chosen > rejected, 0 otherwise, None if unlabeled
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_labeled(self) -> bool:
        return self.label is not None


class UnlabeledPool:
    """Pool of unlabeled preference pairs for active learning."""

    def __init__(self, pairs: List[PreferencePair] = None):
        self.pairs: List[PreferencePair] = []
        self._labeled_indices: set = set()
        if pairs:
            for pair in pairs:
                self.add_pair(pair)

    def __len__(self) -> int:
        return len(self.pairs)

    def n_unlabeled(self) -> int:
        return len(self.pairs) - len(self._labeled_indices)

    def n_labeled(self) -> int:
        return len(self._labeled_indices)

    def add_pair(self, pair: PreferencePair) -> int:
        """Add a pair to the pool, returns its index."""
        idx = len(self.pairs)
        self.pairs.append(pair)
        if pair.is_labeled():
            self._labeled_indices.add(idx)
        return idx

    def add_pairs(self, pairs: List[PreferencePair]) -> List[int]:
        """Add multiple pairs, returns their indices."""
        return [self.add_pair(p) for p in pairs]

    def get_unlabeled_indices(self) -> List[int]:
        """Get indices of all unlabeled pairs."""
        return [i for i in range(len(self.pairs)) if i not in self._labeled_indices]

    def get_labeled_indices(self) -> List[int]:
        """Get indices of all labeled pairs."""
        return list(self._labeled_indices)

    def reset_labels(self, keep_ground_truth: bool = True) -> None:
        """Reset all pairs to unlabeled state.

        Args:
            keep_ground_truth: If True, keeps the label values in pairs
                              but clears _labeled_indices. Useful for simulation
                              where ground truth is stored but we simulate unlabeled.
        """
        if not keep_ground_truth:
            for pair in self.pairs:
                pair.label = None
        self._labeled_indices = set()

    def label_pair(self, idx: int, label: int) -> None:
        """Assign a label to a pair."""
        if idx < 0 or idx >= len(self.pairs):
            raise IndexError(f"Index {idx} out of range")
        self.pairs[idx].label = label
        self._labeled_indices.add(idx)

    def label_pairs(self, indices: List[int], labels: List[int]) -> None:
        """Assign labels to multiple pairs."""
        if len(indices) != len(labels):
            raise ValueError("indices and labels must have same length")
        for idx, label in zip(indices, labels):
            self.label_pair(idx, label)

    def get_pairs(self, indices: List[int]) -> List[PreferencePair]:
        """Get pairs by indices."""
        return [self.pairs[i] for i in indices]

    def get_labeled_pairs(self) -> List[PreferencePair]:
        """Get all labeled pairs."""
        return [self.pairs[i] for i in sorted(self._labeled_indices)]

    def to_training_format(self) -> List[Dict[str, Any]]:
        """Convert labeled pairs to training data format."""
        data = []
        for pair in self.get_labeled_pairs():
            if pair.label == 1:
                data.append(
                    {
                        "prompt": pair.prompt,
                        "chosen": pair.chosen,
                        "rejected": pair.rejected,
                    }
                )
            else:
                # swap chosen/rejected for label=0
                data.append(
                    {
                        "prompt": pair.prompt,
                        "chosen": pair.rejected,
                        "rejected": pair.chosen,
                    }
                )
        return data

    def save(self, path: str) -> None:
        """Save pool to JSON file."""
        data = {
            "pairs": [
                {
                    "prompt": p.prompt,
                    "chosen": p.chosen,
                    "rejected": p.rejected,
                    "label": p.label,
                    "metadata": p.metadata,
                }
                for p in self.pairs
            ],
            "labeled_indices": list(self._labeled_indices),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "UnlabeledPool":
        """Load pool from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        pairs = [
            PreferencePair(
                prompt=p["prompt"],
                chosen=p["chosen"],
                rejected=p["rejected"],
                label=p.get("label"),
                metadata=p.get("metadata", {}),
            )
            for p in data["pairs"]
        ]
        pool = cls(pairs)
        pool._labeled_indices = set(data.get("labeled_indices", []))
        return pool

    @classmethod
    def from_dataset(
        cls,
        dataset,
        prompt_key: str = "prompt",
        chosen_key: str = "chosen",
        rejected_key: str = "rejected",
        include_labels: bool = False,
    ) -> "UnlabeledPool":
        """Create pool from a HuggingFace dataset."""
        pairs = []
        for item in dataset:
            pair = PreferencePair(
                prompt=item[prompt_key],
                chosen=item[chosen_key],
                rejected=item[rejected_key],
                label=1 if include_labels else None,
            )
            pairs.append(pair)
        return cls(pairs)


class PoolDataset(Dataset):
    """PyTorch Dataset wrapper for pool subset."""

    def __init__(
        self,
        pool: UnlabeledPool,
        indices: List[int],
        tokenizer,
        max_length: int = 2048,
    ):
        self.pool = pool
        self.indices = indices
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair = self.pool.pairs[self.indices[idx]]
        return {
            "prompt": pair.prompt,
            "chosen": pair.chosen,
            "rejected": pair.rejected,
            "pool_idx": self.indices[idx],
        }

    def collate_fn(self, batch: List[Dict]) -> Tuple:
        """Collate function for DataLoader."""
        prompts = [b["prompt"] for b in batch]
        chosen = [b["chosen"] for b in batch]
        rejected = [b["rejected"] for b in batch]
        pool_indices = [b["pool_idx"] for b in batch]

        # format as chat for tokenization
        chosen_texts = [
            self._format_conversation(p, c) for p, c in zip(prompts, chosen)
        ]
        rejected_texts = [
            self._format_conversation(p, r) for p, r in zip(prompts, rejected)
        ]

        chosen_tokens = self.tokenizer(
            chosen_texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        rejected_tokens = self.tokenizer(
            rejected_texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return (
            chosen_tokens["input_ids"],
            chosen_tokens["attention_mask"],
            rejected_tokens["input_ids"],
            rejected_tokens["attention_mask"],
            pool_indices,
        )

    def _format_conversation(self, prompt: str, response: str) -> str:
        """Format prompt-response as conversation."""
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            return f"User: {prompt}\nAssistant: {response}"
