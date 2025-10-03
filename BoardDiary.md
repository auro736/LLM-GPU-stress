- lo faccio girare per più iterazioni con llama
- cambia la temperatura dei modelli a 0.5 (tutti e 3) con llama
- cambia il rateo t/t_max a 0.7 (GPU-burn in 5 minuti fa 0.6187)
- cambia il modello di meta-llama/- Llama-4-Scout-17B-16E-Instruct e openai/gpt-oss-120b


- dirgli di utilizzare cutlass

### primo prompt
```bash
python3 main.py \
    --model meta-llama/Llama-3.3-70B-Instruct-Turbo \
    --api_key <api_key> \
    --model_type together \
    --epsilon 0.7 \
    --cuda_temperature 0.2 \
    --compiling_temperature 0.2 \
    --performance_temperature 0.2 \
    --optimization_mode temp \
    --test_duration 300 \
    --max_correction_attempts 5 \
    --max_optimization_attempts 5 \
    --folder_name Root \
    --gpu "one RTX 4060 Ada generation GPU with CUDA 12 and 8GB VRAM" > logTemp.log
```

### lo faccio girare per più iterazioni con llama
```bash
python3 main.py \
    --model meta-llama/Llama-3.3-70B-Instruct-Turbo \
    --api_key <api_key> \
    --model_type together \
    --epsilon 0.7 \
    --cuda_temperature 0.2 \
    --compiling_temperature 0.2 \
    --performance_temperature 0.2 \
    --optimization_mode temp \
    --test_duration 300 \
    --max_correction_attempts 10 \
    --max_optimization_attempts 10 \
    --folder_name Iterations \
    --gpu "one RTX 4060 Ada generation GPU with CUDA 12 and 8GB VRAM" > logIT.log
```

### cambia la temperatura dei modelli a 0.5 (tutti e 3) con llama
```bash
python3 main.py \
    --model meta-llama/Llama-3.3-70B-Instruct-Turbo \
    --api_key <api_key> \
    --model_type together \
    --epsilon 0.7 \
    --cuda_temperature 0.5 \
    --compiling_temperature 0.5 \
    --performance_temperature 0.5 \
    --optimization_mode temp \
    --test_duration 300 \
    --max_correction_attempts 5 \
    --max_optimization_attempts 5 \
    --folder_name Creativity \
    --gpu "one RTX 4060 Ada generation GPU with CUDA 12 and 8GB VRAM" > logCreativity.log
```

### cambia il modello di meta-llama/Llama-4-Scout-17B-16E-Instruct
```bash
python3 main.py \
    --model meta-llama/Llama-4-Scout-17B-16E-Instruct \
    --api_key <api_key> \
    --model_type together \
    --epsilon 0.7 \
    --cuda_temperature 0.2 \
    --compiling_temperature 0.2 \
    --performance_temperature 0.2 \
    --optimization_mode temp \
    --test_duration 300 \
    --max_correction_attempts 5 \
    --max_optimization_attempts 5 \
    --folder_name LLama4 \
    --gpu "one RTX 4060 Ada generation GPU with CUDA 12 and 8GB VRAM" > logLLama4.log
```

### cambia il modello di openai/gpt-oss-120b
```bash
python3 main.py \
    --model openai/gpt-oss-120b \
    --api_key <api_key> \
    --model_type together \
    --epsilon 0.7 \
    --cuda_temperature 0.2 \
    --compiling_temperature 0.2 \
    --performance_temperature 0.2 \
    --optimization_mode temp \
    --test_duration 300 \
    --max_correction_attempts 5 \
    --max_optimization_attempts 5 \
    --folder_name GPT \
    --gpu "one RTX 4060 Ada generation GPU with CUDA 12 and 8GB VRAM" > logGPT.log
```