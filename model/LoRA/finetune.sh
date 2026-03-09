HF_HOME=/home/wuwl/data/huggingface TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --dataset_path data/xgj_lora/class_3/data_segmentation/split_5/  \
    --lora_rank 8  \
    --per_device_train_batch_size 8  \
    --gradient_accumulation_steps 1  \
    --save_strategy epoch \
    --num_train_epochs 10 \
    --save_total_limit 10  \
    --learning_rate 1e-4  \
    --fp16  \
    --remove_unused_columns false  \
    --logging_steps 10  \
    --output_dir output/xgj_lora/class_3/data_segmentation/split_5/

