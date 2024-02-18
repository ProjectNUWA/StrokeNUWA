OUTPUT_DIR=./ckpt/EDM

mkdir -p ${OUTPUT_DIR}

deepspeed --num_gpus 16 \
    --num_nodes 4 \
    --master_addr worker-0 \
    --master_port 6668 \
    --hostfile configs/machine/hostfile_v64_sxm4 \
    train_vq_seq2seq_aug.py \
    --model_name_or_path "./ckpt/flan-t5-xl" \
    --data_path "./example_dataset/data_sample_edm.pkl" \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 20 \
    --model_max_length 512 \
    --per_device_train_batch_size 25 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 20 \
    --learning_rate 5e-6 \
    --warmup_steps 60 \
    --logging_steps 1 \
    --dataloader_num_workers 12 \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed configs/deepspeed/stage3.json \
    --fp16 False \
    --remove_unused_columns False \
    --freezen_llm True \
    --config_path "configs/deepspeed/vqvae_config.yaml";
