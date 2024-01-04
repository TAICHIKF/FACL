export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 main.py --k 5 \
--exp_code fl_noise_0.15  \
--noise_level 0.15 \
--task classification \
--model_type attention_mil \
--weighted_fl_avg \
--data_root_dir '/mnt/group-ai-medical-SHARD/private/feifkong/data/prostate_data' \
--split_dir 'classification_prostate' \
--max_epochs 50 \
--mu 0 \
--alpha 0.5