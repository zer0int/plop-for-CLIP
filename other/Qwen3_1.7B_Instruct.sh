
python grpo_metamath.py \
    --model_id Qwen/Qwen3-1.7B \
    --output_dir .. \
    --target_modules q_proj k_proj gate_proj \
    --learning_rate 5e-6 \
    --num_generations 8 \
    --gradient_accumulation_steps 4 \
    --per_device_train_batch_size 16 \
    --logging_steps 2
