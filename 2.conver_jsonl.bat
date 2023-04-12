set PYTHON_PATH=venv\Scripts
set HF_DATASETS_CACHE=.cache\huggingface\datasets
set MODEL_PATH=F:\ChatGLM-6B\huggingface\chatglm-6b

rm -r .cache

%PYTHON_PATH%\python.exe cover_alpaca2jsonl.py ^
   --data_path output\dataset\answers.json ^
   --save_path output\dataset\answers.jsonl

%PYTHON_PATH%\python.exe tokenize_dataset_rows.py ^
   --jsonl_path output\dataset\answers.jsonl ^
   --save_path output\data ^
   --path %MODEL_PATH% ^
   --max_seq_length 200 
pause