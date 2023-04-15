set PYTHON_PATH=venv\Scripts
set HF_DATASETS_CACHE=.cache\huggingface\datasets

set MODEL_PATH=F:\ChatGLM-6B\huggingface\chatglm-6b
set LOG_PATH=logs

%PYTHON_PATH%\python.exe finetune.py ^
  --dataset_path output\data ^
  --path %MODEL_PATH% ^
  --logging_dir %LOG_PATH% ^
  --lora_rank 32 ^
  --lora_alpha 32 ^
  --per_device_train_batch_size 16 ^
  --gradient_accumulation_steps 1 ^
  --max_steps 1000 ^
  --save_steps 100 ^
  --save_total_limit 2 ^
  --learning_rate 5e-4 ^
  --fp16 ^
  --remove_unused_columns false ^
  --logging_steps 1 ^
  --output_dir output\checkpoint

pause
