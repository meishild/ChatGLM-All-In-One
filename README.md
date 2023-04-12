## ChatGLM lORA训练 for windows

### 0.install

通过`installer.bat`可以将python本地环境安装到venv下。

### 1.make_dataset.bat

在线生成问题，并进行回答。输出目录在output/dataset下。

### 2.conver_jsonl.bat

将1中的问题转换成训练使用的训练集。输出目录在output/data下。

### train.bat

执行训练，具体对应训练参数注意修改。

```
python.exe finetune.py ^
  --dataset_path output\data ^
  --path %MODEL_PATH% ^
  --logging_dir %LOG_PATH% ^
  --lora_rank 32 ^
  --lora_alpha 32 ^
  --per_device_train_batch_size 6 ^
  --gradient_accumulation_steps 1 ^
  --max_steps 400 ^
  --save_steps 50 ^
  --save_total_limit 2 ^
  --learning_rate 1e-4 ^
  --fp16 ^
  --remove_unused_columns false ^
  --logging_steps 50 ^
  --output_dir output\checkpoint
```

### web_ui.bat

可以打开对应相关页面，在页面上默认加载output内checkpoint最后生成的lora模型。