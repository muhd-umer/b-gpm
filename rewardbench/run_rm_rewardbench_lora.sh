export CUDA_VISIBLE_DEVICES=0
python run_rm_rewardbench.py \
--model ../results/saved_model/2b_gemma_lora/rm \
--chat_template raw \
--bf16 \
--flash_attn \
--is_custom_model \
--do_not_save \
--model_name "general-preference/GPM-Gemma-2B" \
--batch_size 4 \
--value_head_dim 6 \
--max_length 2048 \
--is_general_preference \
--add_prompt_head 

