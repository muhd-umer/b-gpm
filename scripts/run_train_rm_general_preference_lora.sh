#!/usr/bin/env bash

deepspeed train_rm_general_preference.py \
--save_path ../results/saved_model/2b_gemma_lora/rm \
--save_steps -1 \
--logging_steps 10 \
--eval_steps 30000 \
--accumulated_gradient 128 \
--micro_train_batch_size 1 \
--pretrain google/gemma-2b-it \
--bf16 \
--max_epochs 2 \
--max_len 2048 \
--learning_rate 2e-6 \
--general_preference_tau 0.1 \
--load_in_4bit \
--lora_rank 16 \
--lora_alpha 32 \
--lora_dropout 0.05 \
--dataset ../data/test_data/test_data.jsonl \
--dataset_probs 1.0 \
--adam_offload \
--flash_attn \
--gradient_checkpointing \
--group_size 1 \
--value_head_dim 6 \
--ptx_loss_coef 0.1 \
--train_split_ratio 0.98 \
--is_general_preference