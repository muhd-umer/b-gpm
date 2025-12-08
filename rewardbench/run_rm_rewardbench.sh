export CUDA_VISIBLE_DEVICES=0
python run_rm_rewardbench.py \
--model muhd-umer/bgpm-gemma-2-2b \
--chat_template raw \
--bf16 \
--flash_attn \
--is_custom_model \
--do_not_save \
--model_name "muhd-umer/2b_gemma_bgpm" \
--batch_size 32 \
--value_head_dim 12 \
--max_length 4096 \
--is_general_preference \
--is_bayesian_gpm \
--add_prompt_head 
