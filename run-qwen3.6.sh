#!/bin/bash

MODEL="protoLabsAI/Qwen3.6-35B-A3B-uncensored-heretic-FP8"
PORT=8080
VLLM_PORT=8000

docker run -ti --rm --name vllm-server \
    --gpus all --net=host --ipc=host \
    -v ~/models:/models \
    -e NCCL_P2P_DISABLE=1 \
    -e NCCL_DEBUG=INFO \
    -e VLLM_USE_DEEP_GEMM=0 \
    -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    vllm-qwen35-v2 \
    serve ${MODEL} \
    --served-model-name qwen/qwen3.5 \
    --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.88 \
    --port 8000 \
    --host 0.0.0.0 \
    --max-model-len 262144 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --attention-backend FLASHINFER \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
