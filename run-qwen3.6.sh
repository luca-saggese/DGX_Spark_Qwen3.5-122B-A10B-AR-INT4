#!/bin/bash

MODEL="Qwen3.6-35B-A3B-FP8"
PORT=8080
VLLM_PORT=8000

docker run -d --rm --name vllm-server \
    --gpus all --net=host --ipc=host \
    -v ~/models:/models \
    -e NCCL_P2P_DISABLE=1 \
    -e NCCL_DEBUG=INFO \
    -e VLLM_USE_DEEP_GEMM=0 \
    -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    vllm-qwen35-v2 \
    /models/${MODEL} \
    --disable-custom-all-reduce \
    --attention-backend FLASHINFER \
    --max-model-len 524288 \
    --tensor-parallel-size 2 \
    --max-num-seqs 11 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.926 \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --host 0.0.0.0 \
    --port ${VLLM_PORT} \
    --dtype auto \
    --tokenizer-mode auto \
    --limit-mm-per-prompt '{"image":5, "video":0}' \
    --speculative-config '{"method":"mtp", "num_speculative_tokens":2}' \
    --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}' \
    --served-model-name qwen/qwen3.5 
