@echo off
set PYTHON_PATH=venv\Scripts
set MODEL_PATH=F:\ChatGLM-6B\huggingface\chatglm-6b

%PYTHON_PATH%\python.exe web.py --port 7880 --path %MODEL_PATH% --lora output/checkpoint
pause
