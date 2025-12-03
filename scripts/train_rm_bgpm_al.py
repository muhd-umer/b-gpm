import argparse
import json
import logging
import math
import os
import random
from datetime import datetime
from typing import Any, Callable, Dict, List

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers.trainer import get_scheduler

from b_gpm.active_learning import (
    BALD,
    UCB,
    ActiveLearner,
    ActiveLearningConfig,
    ClusterDiverseSelector,
    MaxVariance,
    PreferencePair,
    RandomAcquisition,
    TopKSelector,
    UnlabeledPool,
)
from b_gpm.datasets import GeneralRewardDataset
from b_gpm.models import get_reward_model
from b_gpm.trainer import GeneralPreferenceRewardTrainer
from b_gpm.utils import blending_datasets, get_strategy, get_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_acquisition_fn(args):
    """Get acquisition function based on args."""
    if args.acquisition == "mv":
        return MaxVariance()
    elif args.acquisition == "bald":
        return BALD(n_samples=args.bald_n_samples, tau=args.bald_tau)
    elif args.acquisition == "ucb":
        return UCB(kappa=args.ucb_kappa)
    elif args.acquisition == "random":
        return RandomAcquisition()
    else:
        raise ValueError(f"Unknown acquisition: {args.acquisition}")


def load_pool(args, strategy) -> UnlabeledPool:
    """Load dataset and create unlabeled pool."""
    strategy.print(f"Loading dataset for active learning pool: {args.dataset}")

    total_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
        stopping_strategy="all_exhausted",
    )

    # Convert to PreferencePair objects
    pairs = []
    for item in tqdm(
        total_data, desc="Creating pool", disable=strategy.get_rank() != 0
    ):
        # Handle different dataset formats
        prompt = item.get("prompt") or item.get("instruction", "")
        chosen = item.get("chosen") or item.get("chosen_response", "")
        rejected = item.get("rejected") or item.get("rejected_response", "")

        # Handle message list format
        if isinstance(chosen, list):
            for msg in chosen:
                if msg.get("role") == "assistant":
                    chosen = msg.get("content", "")
                    break
            else:
                chosen = str(chosen[-1]) if chosen else ""

        if isinstance(rejected, list):
            for msg in rejected:
                if msg.get("role") == "assistant":
                    rejected = msg.get("content", "")
                    break
            else:
                rejected = str(rejected[-1]) if rejected else ""

        if isinstance(prompt, list):
            prompt = prompt[0].get("content", str(prompt)) if prompt else ""

        pair = PreferencePair(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            label=1,  # ground truth: chosen > rejected
        )
        pairs.append(pair)

    pool = UnlabeledPool(pairs)

    # Reset to unlabeled for simulation (keeps ground truth in .label)
    pool.reset_labels(keep_ground_truth=True)

    strategy.print(f"Pool created with {len(pool)} samples")
    return pool


def create_eval_fn(model, tokenizer, strategy, value_head_dim: int):
    """Create evaluation function for active learning."""

    def eval_fn(model) -> Dict[str, float]:
        model.eval()
        device = next(model.parameters()).device
        return {"placeholder": 0.0}  # Actual eval done by trainer

    return eval_fn


def simulated_oracle_factory(pool: UnlabeledPool):
    """Create oracle that uses ground truth labels."""

    def oracle(pairs: List[PreferencePair]) -> List[int]:
        labels = []
        for pair in pairs:
            for p in pool.pairs:
                if p.prompt == pair.prompt and p.chosen == pair.chosen:
                    labels.append(p.label if p.label is not None else 1)
                    break
            else:
                labels.append(1)
        return labels

    return oracle


def create_train_fn(args, strategy, tokenizer):
    """Create training function for active learning iterations."""

    def train_fn(model, labeled_data: List[Dict]) -> torch.nn.Module:
        """Fine-tune model on labeled data."""
        from datasets import Dataset

        # Convert to HF dataset format
        ds = Dataset.from_list(labeled_data)

        train_dataset = GeneralRewardDataset(
            ds,
            tokenizer,
            args.max_len,
            strategy,
            is_custom=args.is_custom_dataset,
            return_prompt_length=args.return_prompt_length,
        )
        train_dataloader = strategy.setup_dataloader(
            train_dataset,
            args.micro_train_batch_size,
            True,
            True,
            train_dataset.collate_fn,
            group_size=args.group_size,
        )

        # Create optimizer for this iteration
        optim = strategy.create_optimizer(
            model, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.l2
        )

        num_update_steps = len(train_dataloader) // strategy.accumulated_gradient
        max_steps = math.ceil(args.al_epochs_per_iter * num_update_steps)

        scheduler = get_scheduler(
            "cosine",
            optim,
            num_warmup_steps=math.ceil(max_steps * 0.03),
            num_training_steps=max_steps,
        )

        (model, optim, scheduler) = strategy.prepare((model, optim, scheduler))

        trainer = GeneralPreferenceRewardTrainer(
            model=model,
            strategy=strategy,
            optim=optim,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            eval_dataloader=None,
            scheduler=scheduler,
            max_epochs=args.al_epochs_per_iter,
            is_general_preference=args.is_general_preference,
            is_bayesian_gpm=args.is_bayesian_gpm,
            tau=args.general_preference_tau,
            value_head_dim=args.value_head_dim,
        )

        # Create minimal args for fit
        train_args = argparse.Namespace(
            save_path=args.save_path,
            save_steps=-1,
            eval_steps=-1,
            logging_steps=args.logging_steps,
            save_best_model=args.save_best_model,
            save_on_epoch_end=args.save_on_epoch_end,
            add_pretrain_loss=args.add_pretrain_loss,
            ptx_loss_coef=args.ptx_loss_coef,
            reward_scaler_beta=args.reward_scaler_beta,
            reward_margin=args.reward_margin,
            regression_target_margin=args.regression_target_margin,
            bayesian_kl_warmup_steps=args.bayesian_kl_warmup_steps,
            bayesian_max_kl_weight=args.bayesian_max_kl_weight,
            bayesian_prior_variance=args.bayesian_prior_variance,
            bayesian_regularize_mean=args.bayesian_regularize_mean,
            bayesian_sample_mix_ratio=args.bayesian_sample_mix_ratio,
            use_wandb=args.use_wandb,
        )

        trainer.fit(train_args)
        return model

    return train_fn


def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure model
    model = get_reward_model(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(),
        init_value_head=True,
        is_general_preference=args.is_general_preference,
        is_bayesian_gpm=args.is_bayesian_gpm,
        value_head_dim=args.value_head_dim,
        bayesian_init_logvar=args.bayesian_init_logvar,
        bayesian_min_logvar=args.bayesian_min_logvar,
        bayesian_max_logvar=args.bayesian_max_logvar,
        init_prompt_head=True,
        add_prompt_head=args.add_prompt_head,
    )

    # configure tokenizer
    tokenizer = get_tokenizer(
        args.pretrain, model, "left", strategy, use_fast=not args.disable_fast_tokenizer
    )
    tokenizer.truncation_side = "right"

    strategy.print(model)

    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": args.gradient_checkpointing_use_reentrant
            }
        )

    # Load unlabeled pool
    pool = load_pool(args, strategy)

    # Add initial random labels
    if args.initial_labels > 0:
        random.seed(args.seed)
        initial_indices = random.sample(
            range(len(pool)), min(args.initial_labels, len(pool))
        )
        for idx in initial_indices:
            pool._labeled_indices.add(idx)
        strategy.print(f"Added {len(initial_indices)} initial random labels")

    # Setup active learning components
    acquisition_fn = get_acquisition_fn(args)
    batch_selector = (
        ClusterDiverseSelector(n_clusters=args.al_n_clusters)
        if args.use_diverse_selection
        else TopKSelector()
    )

    os.makedirs(args.save_path, exist_ok=True)

    config = ActiveLearningConfig(
        batch_size=args.al_batch_size,
        acquisition_batch_size=args.micro_train_batch_size * 2,
        use_diverse_selection=args.use_diverse_selection,
        max_iterations=args.al_max_iterations,
        max_labels=args.al_max_labels,
        target_accuracy=args.al_target_accuracy,
        eval_every=args.al_eval_every,
        save_every=args.al_save_every,
        output_dir=args.save_path,
    )

    eval_fn = create_eval_fn(model, tokenizer, strategy, args.value_head_dim)
    oracle = simulated_oracle_factory(pool)
    train_fn = create_train_fn(args, strategy, tokenizer) if args.al_retrain else None

    # Prepare model for active learning
    optim = strategy.create_optimizer(
        model, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.l2
    )
    (model, optim, _) = strategy.prepare((model, optim, None))

    learner = ActiveLearner(
        model=model,
        tokenizer=tokenizer,
        pool=pool,
        acquisition_fn=acquisition_fn,
        batch_selector=batch_selector,
        config=config,
        oracle=oracle,
        eval_fn=eval_fn,
        train_fn=train_fn,
        value_head_dim=args.value_head_dim,
    )

    # Run active learning loop
    strategy.print("Starting active learning loop...")
    state = learner.run()

    # Save final results
    results = {
        "final_labels": state.total_labels,
        "final_accuracy": state.best_accuracy,
        "iterations": state.iteration,
        "history": state.history,
        "acquisition": args.acquisition,
        "config": vars(args),
    }

    results_path = os.path.join(args.save_path, "al_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    strategy.print(f"Active learning complete. Results saved to {results_path}")

    # Save final model
    strategy.save_model(model, tokenizer, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument("--pretrain", type=str, default="google/gemma-2b")
    parser.add_argument("--save_path", type=str, default="../results/saved_model/al")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument(
        "--ckpt_path", type=str, default="../results/saved_model/checkpoint"
    )
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1000)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--micro_train_batch_size", type=int, default=8)
    parser.add_argument("--accumulated_gradient", type=int, default=1)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument(
        "--save_on_epoch_end",
        action="store_true",
        default=False,
        help="Save a checkpoint at the end of every training epoch.",
    )
    parser.add_argument(
        "--save_best_model",
        type=int,
        default=None,
        help="Save the top N models with the lowest evaluation loss.",
    )

    # DeepSpeed arguments
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--zpg", type=int, default=1)
    parser.add_argument("--adam_offload", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--compute_fp32_loss", action="store_true", default=False)
    parser.add_argument("--margin_loss", action="store_true", default=False)
    parser.add_argument("--grad_accum_dtype", type=str, default=None)
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # Dataset arguments
    parser.add_argument(
        "--dataset", type=str, default="../data/test_data/test_data.jsonl"
    )
    parser.add_argument("--dataset_probs", type=str, default="1.0")
    parser.add_argument("--max_samples", type=int, default=100000)
    parser.add_argument("--is_custom_dataset", action="store_true", default=False)
    parser.add_argument("--return_prompt_length", action="store_true", default=False)
    parser.add_argument("--group_size", type=int, default=1)

    # General Preference / Bayesian GPM arguments
    parser.add_argument("--is_general_preference", action="store_true", default=False)
    parser.add_argument("--is_bayesian_gpm", action="store_true", default=False)
    parser.add_argument("--general_preference_tau", type=float, default=0.1)
    parser.add_argument("--value_head_dim", type=int, default=6)
    parser.add_argument("--add_prompt_head", action="store_true", default=False)
    parser.add_argument("--add_pretrain_loss", action="store_true", default=False)
    parser.add_argument("--ptx_loss_coef", type=float, default=0.1)
    parser.add_argument("--reward_scaler_beta", type=float, default=2.0)
    parser.add_argument("--reward_margin", type=float, default=1.0)
    parser.add_argument("--regression_target_margin", type=float, default=10.0)

    # Bayesian GPM specific arguments
    parser.add_argument("--bayesian_kl_warmup_steps", type=int, default=500)
    parser.add_argument("--bayesian_max_kl_weight", type=float, default=0.001)
    parser.add_argument("--bayesian_prior_variance", type=float, default=0.01)
    parser.add_argument(
        "--bayesian_regularize_mean", action="store_true", default=False
    )
    parser.add_argument("--bayesian_init_logvar", type=float, default=-5.0)
    parser.add_argument("--bayesian_min_logvar", type=float, default=-8.0)
    parser.add_argument("--bayesian_max_logvar", type=float, default=1.0)
    parser.add_argument("--bayesian_sample_mix_ratio", type=float, default=0.7)

    # Active Learning arguments
    parser.add_argument(
        "--acquisition",
        type=str,
        default="mv",
        choices=["mv", "bald", "ucb", "random"],
        help="Acquisition function: mv (MaxVariance), bald, ucb, or random",
    )
    parser.add_argument(
        "--al_batch_size",
        type=int,
        default=32,
        help="Number of samples to select per active learning iteration",
    )
    parser.add_argument(
        "--al_max_iterations",
        type=int,
        default=100,
        help="Maximum number of active learning iterations",
    )
    parser.add_argument(
        "--al_max_labels",
        type=int,
        default=10000,
        help="Maximum total labels to acquire",
    )
    parser.add_argument(
        "--al_target_accuracy",
        type=float,
        default=None,
        help="Stop when this accuracy is reached",
    )
    parser.add_argument(
        "--al_eval_every",
        type=int,
        default=5,
        help="Evaluate every N iterations",
    )
    parser.add_argument(
        "--al_save_every",
        type=int,
        default=10,
        help="Save checkpoint every N iterations",
    )
    parser.add_argument(
        "--al_retrain",
        action="store_true",
        default=False,
        help="Retrain model after each iteration (expensive)",
    )
    parser.add_argument(
        "--al_epochs_per_iter",
        type=int,
        default=1,
        help="Training epochs per active learning iteration",
    )
    parser.add_argument(
        "--initial_labels",
        type=int,
        default=100,
        help="Number of random samples to label initially",
    )
    parser.add_argument(
        "--use_diverse_selection",
        action="store_true",
        default=True,
        help="Use cluster-based diverse batch selection",
    )
    parser.add_argument(
        "--al_n_clusters",
        type=int,
        default=None,
        help="Number of clusters for diverse selection (default: sqrt(batch_size))",
    )

    # UCB specific
    parser.add_argument("--ucb_kappa", type=float, default=1.0)

    # BALD specific
    parser.add_argument("--bald_n_samples", type=int, default=20)
    parser.add_argument("--bald_tau", type=float, default=0.1)

    # WandB arguments
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="bgpm_active_learning")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="al_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    args = parser.parse_args()
    train(args)
