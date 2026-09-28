@echo off
setlocal

if "%MARK_LIV_MODEL%"=="" set "MARK_LIV_MODEL=Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M"

echo [Mark-LIV] Starting local llama.cpp server (CPU-only)...

llama.exe serve ^
  -hf "%MARK_LIV_MODEL%" ^
  --jinja ^
  --alias mark-liv ^
  --host 127.0.0.1 ^
  --port 8080 ^
  -c 4096 ^
  -np 1 ^
  --device none
