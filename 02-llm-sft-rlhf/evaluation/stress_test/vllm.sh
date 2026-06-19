python -m vllm.entrypoints.openai.api_server \
    --model '/root/autodl-tmp/40_rlhf_on_llama/LLaMA-Factory/output/llama3_lora_ppo' \
    --served-model-name 'qwen3-4b-ppo' \
    --host 0.0.0.0 \
    --port 6006 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --dtype=half \