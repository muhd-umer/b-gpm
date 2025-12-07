
deepspeed --force_multi --hostfile hostfile train_rm_bgpm.py \
--save_path ../results/saved_model/2b_gemma/rm \
--save_steps -1 \
--logging_steps 1 \
--eval_steps -1 \
--accumulated_gradient 1 \
--micro_train_batch_size 4 \
--pretrain google/gemma-2b-it \
--bf16 \
--max_epochs 2 \
--max_len 2048 \
--zero_stage 3 \
--learning_rate 2e-6 \
--general_preference_tau 0.1 \
--dataset natolambert/skywork-preferences-80k-v0.1-cleaned \
--dataset_probs 1 \
--flash_attn \
--gradient_checkpointing \
--group_size 1 \
--value_head_dim 6 \
--save_best_model 2 \
--ptx_loss_coef 0.1 \
--train_split_ratio 0.98 \
--is_general_preference \
--is_bayesian_gpm

# --is_general_preference \
# --add_pretrain_loss \
# --is_custom_dataset \
# --return_prompt_length \
# --add_prompt_head \
