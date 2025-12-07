python batch_inference_rm_bgpm.py \
--pretrain ../results/2b_gemma_bsmr00/rm \
--dataset  ../data/test_data/test_data.jsonl  \
--max_samples 100000 \
--general_preference_tau 0.1 \
--micro_batch_size 3 \
--max_len 2048 \
--value_head_dim 12 \
--is_general_preference \
--is_bayesian_gpm
