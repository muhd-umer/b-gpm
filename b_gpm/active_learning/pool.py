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

    def to_training_format(
        self, use_message_format: bool = True
    ) -> List[Dict[str, Any]]:
        """Convert labeled pairs to training data format.

        Args:
            use_message_format: If True, output chosen/rejected as message lists
                               for chat template compatibility. If False, output
                               as plain strings with prompt field.
        """
        data = []
        for pair in self.get_labeled_pairs():
            chosen_text = pair.chosen if pair.label == 1 else pair.rejected
            rejected_text = pair.rejected if pair.label == 1 else pair.chosen

            if use_message_format:
                data.append(
                    {
                        "chosen": [
                            {"role": "user", "content": pair.prompt},
                            {"role": "assistant", "content": chosen_text},
                        ],
                        "rejected": [
                            {"role": "user", "content": pair.prompt},
                            {"role": "assistant", "content": rejected_text},
                        ],
                    }
                )
            else:
                data.append(
                    {
                        "prompt": pair.prompt,
                        "chosen": chosen_text,
                        "rejected": rejected_text,
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
        self._preformatted = self._preformat_all()

    def _preformat_all(self) -> List[Tuple[str, str]]:
        """Pre-format all conversations to avoid repeated chat template calls."""
        formatted = []
        for idx in self.indices:
            pair = self.pool.pairs[idx]
            chosen_text = self._format_conversation(pair.prompt, pair.chosen)
            rejected_text = self._format_conversation(pair.prompt, pair.rejected)
            formatted.append((chosen_text, rejected_text))
        return formatted

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        chosen_text, rejected_text = self._preformatted[idx]
        return {
            "chosen_text": chosen_text,
            "rejected_text": rejected_text,
            "pool_idx": self.indices[idx],
        }

    def collate_fn(self, batch: List[Dict]) -> Tuple:
        """Collate function for DataLoader."""
        chosen_texts = [b["chosen_text"] for b in batch]
        rejected_texts = [b["rejected_text"] for b in batch]
        pool_indices = [b["pool_idx"] for b in batch]

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
