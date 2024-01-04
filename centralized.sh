export CUDA_VISIBLE_DEVICES=7
python3 main.py --k 5 \
--exp_code no_fl \
--task classification \
--no_fl \
--data_root_dir '/mnt/group-ai-medical-SHARD/private/feifkong/data/prostate_data' \
--split_dir 'classification_prostate' \
--max_epochs 50 \
--alpha 0.05