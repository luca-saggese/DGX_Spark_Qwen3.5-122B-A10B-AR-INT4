
#!/bin/bash
docker rm vllm-qwen35
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
docker run -d --name vllm-qwen35 \
  --gpus all --net=host --ipc=host \
  -v ~/models:/models \
  vllm-qwen35-v2 \
  serve /models/qwen35-122b-hybrid-int4fp8 \
  --served-model-name qwen/qwen3.5 \
  --max-model-len 196608 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.88 \
  --port 8000 \
  --host 0.0.0.0 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --attention-backend FLASHINFER \
  --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":2}'

#   --load-format fastsafetensors \
#   --attention-backend FLASHINFER \
#   --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
#   --enable-chunked-prefill \
#   --enable-auto-tool-choice \
#   --tool-call-parser qwen3_coder \
#   --generation-config auto \
#   --override-generation-config '{"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 0.0, "repetition_penalty": 1.0}'

#   LAUNCH_CMD="docker run -d --name vllm-qwen35 \\
#     --gpus all --net=host --ipc=host \\
#     -v ${MODELS_PARENT}:/models \\
#     vllm-qwen35-v2 \\
#     serve /models/${MODEL_BASENAME} \\
#     --served-model-name qwen \\
#     --port 8000 \\
#     --max-model-len 262144 \\
#     --gpu-memory-utilization 0.90 \\
#     --reasoning-parser qwen3 \\
#     --enable-auto-tool-choice \\
#     --tool-call-parser qwen3_coder \\
#     --attention-backend FLASHINFER \\
#     --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":2}'"