data_type=xgj
data_class=class_3

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
DISTRIBUTED_ARGS="deepspeed --include localhost:0,1,2,3 --master_port $MASTER_PORT"

HF_HOME=/home/wuwl/data/huggingface TRANSFORMERS_OFFLINE=1 ${DISTRIBUTED_ARGS} main.py \
    --deepspeed deepspeed.json \
    --do_train \
    --train_file /mnt/nvme_share/wuwl/project/ChatGLM-6B-main/data/xgj/class_3/data_segmentation/merged_train_0.03.json  \
    --test_file /mnt/nvme_share/wuwl/project/ChatGLM-6B-main/data/xgj/class_3/ChatGLM_Ptuning_data_inference.json \
    --prompt_column instruction \
    --response_column output \
    --overwrite_cache \
    --model_name_or_path /mnt/nvme_share/wuwl/chatglm-6B/ \
    --output_dir /mnt/nvme_share/wuwl/project/ChatGLM-6B-main/ptuning/output/xgj/class_3/data_filte/0.03/ \
    --overwrite_output_dir \
    --label_smoothing_factor 0 \
    --max_source_length 350 \
    --max_target_length 350 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --num_train_epochs 10 \
    --logging_steps 50 \
    --save_total_limit 10 \
    --save_strategy epoch \
    --learning_rate 2e-2 \
    --pre_seq_len 128 \
    --quantization_bit 4 \
    --num_beams 3