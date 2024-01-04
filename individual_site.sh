export CUDA_VISIBLE_DEVICES=6
python3 main.py --k 5 \
--inst_name 6 \
--exp_code institute_6  \
--model_type CLAM_MB \
--task classification \
--data_root_dir '/mnt/group-ai-medical-SHARD/private/feifkong/data/prostate_data' \
--split_dir 'classification_prostate' \
--max_epochs 30 \
--alpha 0.05