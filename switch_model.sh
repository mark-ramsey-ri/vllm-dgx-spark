#!/usr/bin/env bash
set -euo pipefail

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# vLLM Model Switching Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Allows switching between different models on the vLLM cluster.
# Handles tensor parallelism, model downloading, and rsync to worker nodes.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load configuration
if [ -f "${SCRIPT_DIR}/setup-env.sh" ]; then
  # Get WORKER_HOST and WORKER_USER from environment setup
  WORKER_HOST="${WORKER_HOST:-}"
  WORKER_USER="${WORKER_USER:-$(whoami)}"
  HF_CACHE="${HF_CACHE:-/raid/hf-cache}"
  HF_TOKEN="${HF_TOKEN:-}"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model Definitions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Model HuggingFace IDs
#
# Models tagged [NVIDIA] are from the official Spark vLLM playbook
# (https://build.nvidia.com/spark/vllm/instructions). [community] entries are
# additional models that work but are not on NVIDIA's published matrix.
#
# Adding new entries: append to ALL parallel arrays (MODELS, MODEL_NAMES,
# MODEL_QUANT, MODEL_IMAGE, MODEL_TP, MODEL_NODES, MODEL_GPU_MEM,
# MODEL_MAX_LEN, MODEL_TRUST_REMOTE, MODEL_NEEDS_TOKEN, MODEL_EXPERT_PARALLEL).
MODELS=(
  # ── [community] entries (existing, not in NVIDIA's official matrix) ──
  "openai/gpt-oss-120b"
  "openai/gpt-oss-20b"
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen2.5-14B-Instruct"
  "Qwen/Qwen2.5-32B-Instruct"
  "Qwen/Qwen2.5-72B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "mistralai/Mistral-Nemo-Instruct-2407"
  "mistralai/Mixtral-8x7B-Instruct-v0.1"
  "meta-llama/Llama-3.1-8B-Instruct"
  "meta-llama/Llama-3.1-70B-Instruct"
  "microsoft/phi-4"
  "google/gemma-2-27b-it"
  "CohereForAI/c4ai-command-r-plus-08-2024"
  "nvidia/Llama-3.1-405B-Instruct-FP4"
  "meta-llama/Llama-3.3-70B-Instruct"
  # ── [NVIDIA] entries (from build.nvidia.com/spark/vllm/instructions) ──
  # Llama family
  "nvidia/Llama-3.1-8B-Instruct-FP8"
  "nvidia/Llama-3.1-8B-Instruct-NVFP4"
  "nvidia/Llama-3.3-70B-Instruct-NVFP4"
  # Qwen3 family
  "nvidia/Qwen3-8B-FP8"
  "nvidia/Qwen3-8B-NVFP4"
  "nvidia/Qwen3-14B-FP8"
  "nvidia/Qwen3-14B-NVFP4"
  "nvidia/Qwen3-32B-NVFP4"
  # Qwen multimodal / reranker / embedding
  "nvidia/Qwen2.5-VL-7B-Instruct-NVFP4"
  "Qwen/Qwen3-VL-Reranker-2B"
  "Qwen/Qwen3-VL-Reranker-8B"
  "Qwen/Qwen3-VL-Embedding-2B"
  # Phi-4 family
  "nvidia/Phi-4-multimodal-instruct-FP8"
  "nvidia/Phi-4-multimodal-instruct-NVFP4"
  "nvidia/Phi-4-reasoning-plus-FP8"
  "nvidia/Phi-4-reasoning-plus-NVFP4"
  # Nemotron-3
  "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
  "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
  "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
  # Gemma 4 (uses a different vLLM image - see MODEL_IMAGE below)
  "google/gemma-4-E2B-it"
  "google/gemma-4-E4B-it"
  "google/gemma-4-26B-A4B-it"
  "google/gemma-4-31B-it"
  "nvidia/Gemma-4-31B-IT-NVFP4"
  # NVIDIA's TP=4 example for switched-Spark setups
  "MiniMaxAI/MiniMax-M2.5"
)

# Human-readable model descriptions
MODEL_NAMES=(
  # ── [community] ──
  "[community] GPT-OSS-120B (120B MoE, native MXFP4 ~65GB, high quality)"
  "[community] GPT-OSS-20B (21B MoE, ~16-20GB, fast)"
  "[community] Qwen2.5-7B (7B, ~7GB, very fast)"
  "[community] Qwen2.5-14B (14B, ~14GB, fast)"
  "[community] Qwen2.5-32B (32B, ~30GB, strong mid-size)"
  "[community] Qwen2.5-72B (72B, ~70GB, slow, high quality)"
  "[community] Mistral-7B v0.3 (7B, ~7GB, very fast)"
  "[community] Mistral-Nemo-12B (12B, ~12GB, 128k context)"
  "[community] Mixtral-8x7B (47B total, 12B active, ~45GB, MoE)"
  "[community] Llama-3.1-8B (8B, ~8GB, very fast)"
  "[community] Llama-3.1-70B (70B, ~65GB, high quality)"
  "[community] Phi-4 (15B, ~14-16GB, small but smart)"
  "[community] Gemma2-27B (27B, ~24-28GB, strong mid-size)"
  "[community] Command-R-Plus (104B BF16 ~208GB, 2 Sparks)"
  "[community] Llama-3.1-405B-FP4 (405B FP4 ~200GB, 2 Sparks)"
  "[community] Llama-3.3-70B (70B BF16 ~141GB, 2 Sparks)"
  # ── [NVIDIA] ──
  "[NVIDIA] Llama-3.1-8B-Instruct FP8 (~8GB)"
  "[NVIDIA] Llama-3.1-8B-Instruct NVFP4 (~4GB)"
  "[NVIDIA] Llama-3.3-70B-Instruct NVFP4 (~35GB)"
  "[NVIDIA] Qwen3-8B FP8 (~8GB)"
  "[NVIDIA] Qwen3-8B NVFP4 (~4GB)"
  "[NVIDIA] Qwen3-14B FP8 (~14GB)"
  "[NVIDIA] Qwen3-14B NVFP4 (~7GB)"
  "[NVIDIA] Qwen3-32B NVFP4 (~16GB)"
  "[NVIDIA] Qwen2.5-VL-7B-Instruct NVFP4 (multimodal, ~4GB)"
  "[NVIDIA] Qwen3-VL-Reranker-2B (reranker, ~4GB)"
  "[NVIDIA] Qwen3-VL-Reranker-8B (reranker, ~16GB)"
  "[NVIDIA] Qwen3-VL-Embedding-2B (embedding, ~4GB)"
  "[NVIDIA] Phi-4-multimodal-instruct FP8 (multimodal, trust_remote)"
  "[NVIDIA] Phi-4-multimodal-instruct NVFP4 (multimodal, trust_remote)"
  "[NVIDIA] Phi-4-reasoning-plus FP8 (reasoning, ~14GB)"
  "[NVIDIA] Phi-4-reasoning-plus NVFP4 (reasoning, ~7GB)"
  "[NVIDIA] Nemotron-3-Nano-30B-A3B BF16 (MoE, ~60GB)"
  "[NVIDIA] Nemotron-3-Nano-30B-A3B FP8 (MoE, ~30GB)"
  "[NVIDIA] Nemotron-3-Super-120B-A12B NVFP4 (MoE, 2 Sparks)"
  "[NVIDIA] Gemma 4 E2B IT (Base, ~4GB, gemma4 image)"
  "[NVIDIA] Gemma 4 E4B IT (Base, ~8GB, gemma4 image)"
  "[NVIDIA] Gemma 4 26B A4B IT (Base, ~52GB, gemma4 image)"
  "[NVIDIA] Gemma 4 31B IT (Base, ~62GB, gemma4 image)"
  "[NVIDIA] Gemma 4 31B IT NVFP4 (~16GB)"
  "[NVIDIA] MiniMax-M2.5 (MoE, TP=4 across 4 Sparks per NVIDIA example)"
)

# Quantization label (purely informational — vLLM picks up actual quant from
# the model's config.json). Use one of: BF16, FP8, NVFP4, MXFP4, FP4, AWQ.
MODEL_QUANT=(
  # [community]
  "MXFP4" "MXFP4" "BF16" "BF16" "BF16" "BF16" "BF16" "BF16" "BF16"
  "BF16" "BF16" "BF16" "BF16" "BF16" "FP4"  "BF16"
  # [NVIDIA]
  "FP8"   "NVFP4" "NVFP4"
  "FP8"   "NVFP4" "FP8"   "NVFP4" "NVFP4"
  "NVFP4" "BF16"  "BF16"  "BF16"
  "FP8"   "NVFP4" "FP8"   "NVFP4"
  "BF16"  "FP8"   "NVFP4"
  "BF16"  "BF16"  "BF16"  "BF16"  "NVFP4"
  "BF16"
)

# Optional per-model VLLM_IMAGE override. Leave empty ("") to use the default
# IMAGE from config.env / start_cluster.sh. NVIDIA ships Gemma 4 in a separate
# image because the family currently requires CUDA 13.0 specific kernels.
MODEL_IMAGE=(
  # [community] - all use default
  "" "" "" "" "" "" "" "" ""
  "" "" "" "" "" "" ""
  # [NVIDIA] - Gemma 4 family overrides
  "" "" ""
  "" "" "" "" ""
  "" "" "" ""
  "" "" "" ""
  "" "" ""
  "vllm/vllm-openai:gemma4-cu130"
  "vllm/vllm-openai:gemma4-cu130"
  "vllm/vllm-openai:gemma4-cu130"
  "vllm/vllm-openai:gemma4-cu130"
  ""
  ""
)

# Tensor Parallelism (number of GPUs needed)
# Models that fit in a single DGX Spark's ~120GB unified memory use TP=1;
# larger models that must be split across multiple Sparks use TP=N.
MODEL_TP=(
  # ── [community] ──
  1    # gpt-oss-120b - native MXFP4 ~65GB
  1    # gpt-oss-20b - ~16-20GB
  1    # Qwen2.5-7B - ~7GB
  1    # Qwen2.5-14B - ~14GB
  1    # Qwen2.5-32B - ~30GB
  1    # Qwen2.5-72B - ~70GB
  1    # Mistral-7B - ~7GB
  1    # Mistral-Nemo-12B - ~12GB
  1    # Mixtral-8x7B - ~45GB
  1    # Llama-3.1-8B - ~8GB
  1    # Llama-3.1-70B - ~65GB
  1    # Phi-4 - ~14-16GB
  1    # Gemma2-27B - ~24-28GB
  2    # Command-R-Plus - BF16 ~208GB, 2 Sparks
  2    # Llama-3.1-405B-FP4 - ~200GB, 2 Sparks
  2    # Llama-3.3-70B - BF16 ~141GB, 2 Sparks
  # ── [NVIDIA] ──
  1    # Llama-3.1-8B-FP8 - ~8GB
  1    # Llama-3.1-8B-NVFP4 - ~4GB
  1    # Llama-3.3-70B-NVFP4 - ~35GB
  1    # Qwen3-8B-FP8 - ~8GB
  1    # Qwen3-8B-NVFP4 - ~4GB
  1    # Qwen3-14B-FP8 - ~14GB
  1    # Qwen3-14B-NVFP4 - ~7GB
  1    # Qwen3-32B-NVFP4 - ~16GB
  1    # Qwen2.5-VL-7B NVFP4 - ~4GB
  1    # Qwen3-VL-Reranker-2B - ~4GB
  1    # Qwen3-VL-Reranker-8B - ~16GB
  1    # Qwen3-VL-Embedding-2B - ~4GB
  1    # Phi-4-multimodal FP8 - ~14GB
  1    # Phi-4-multimodal NVFP4 - ~7GB
  1    # Phi-4-reasoning-plus FP8 - ~14GB
  1    # Phi-4-reasoning-plus NVFP4 - ~7GB
  1    # Nemotron-3-Nano-30B-A3B BF16 - ~60GB
  1    # Nemotron-3-Nano-30B-A3B FP8 - ~30GB
  2    # Nemotron-3-Super-120B-A12B NVFP4 - 2 Sparks
  1    # Gemma 4 E2B - ~4GB
  1    # Gemma 4 E4B - ~8GB
  1    # Gemma 4 26B A4B - ~52GB
  1    # Gemma 4 31B - ~62GB
  1    # Gemma 4 31B NVFP4 - ~16GB
  4    # MiniMax-M2.5 - NVIDIA's TP=4 example, 4 Sparks
)

# Minimum number of Sparks required (1 = fits on a single Spark, 2+ = needs
# multi-Spark TP). The menu groups models by this count for display.
MODEL_NODES=(
  # ── [community] ──
  1 1 1 1 1 1 1 1 1 1 1 1 1
  2 2 2
  # ── [NVIDIA] ──
  1 1 1                  # Llama-3.1-8B FP8/NVFP4, Llama-3.3-70B NVFP4
  1 1 1 1 1              # Qwen3-{8B FP8, 8B NVFP4, 14B FP8, 14B NVFP4, 32B NVFP4}
  1 1 1 1                # Qwen2.5-VL NVFP4 + Qwen3-VL Reranker x2 + Embedding
  1 1 1 1                # Phi-4 multimodal x2 + reasoning-plus x2
  1 1 2                  # Nemotron-3-Nano BF16 + FP8, Nemotron-3-Super (2 Sparks)
  1 1 1 1 1              # Gemma 4 E2B, E4B, 26B, 31B, 31B-NVFP4
  4                      # MiniMax-M2.5 (NVIDIA's 4-Spark example)
)

# GPU Memory Utilization (0.90 default)
MODEL_GPU_MEM=(
  # [community]
  0.90 0.90 0.90 0.90 0.90 0.90 0.90 0.90 0.90 0.90 0.90 0.90 0.90
  0.90 0.90 0.90
  # [NVIDIA]
  0.90 0.90 0.90
  0.90 0.90 0.90 0.90 0.90
  0.90 0.90 0.90 0.90
  0.90 0.90 0.90 0.90
  0.90 0.90 0.90
  0.90 0.90 0.90 0.90 0.90
  0.90
)

# Max model length (context window) - chosen to balance the model's published
# limit against KV-cache memory pressure on 120GB unified memory.
MODEL_MAX_LEN=(
  # ── [community] ──
  8192   # gpt-oss-120b
  8192   # gpt-oss-20b
  32768  # Qwen2.5-7B
  32768  # Qwen2.5-14B
  32768  # Qwen2.5-32B
  32768  # Qwen2.5-72B
  32768  # Mistral-7B v0.3
  131072 # Mistral-Nemo-12B (native 128k)
  32768  # Mixtral-8x7B
  131072 # Llama-3.1-8B (native 128k)
  131072 # Llama-3.1-70B (native 128k)
  16384  # Phi-4
  8192   # Gemma2-27B
  32768  # Command-R-Plus (native 128k)
  16384  # Llama-3.1-405B-FP4 (very large)
  32768  # Llama-3.3-70B (native 128k)
  # ── [NVIDIA] ──
  131072 # Llama-3.1-8B-FP8
  131072 # Llama-3.1-8B-NVFP4
  32768  # Llama-3.3-70B-NVFP4
  32768  # Qwen3-8B-FP8
  32768  # Qwen3-8B-NVFP4
  32768  # Qwen3-14B-FP8
  32768  # Qwen3-14B-NVFP4
  32768  # Qwen3-32B-NVFP4
  32768  # Qwen2.5-VL-7B-NVFP4
  8192   # Qwen3-VL-Reranker-2B (reranker - short ctx is fine)
  8192   # Qwen3-VL-Reranker-8B
  8192   # Qwen3-VL-Embedding-2B (embedder)
  16384  # Phi-4-multimodal-FP8
  16384  # Phi-4-multimodal-NVFP4
  16384  # Phi-4-reasoning-plus-FP8
  16384  # Phi-4-reasoning-plus-NVFP4
  32768  # Nemotron-3-Nano BF16
  32768  # Nemotron-3-Nano FP8
  32768  # Nemotron-3-Super-NVFP4
  8192   # Gemma 4 E2B
  8192   # Gemma 4 E4B
  8192   # Gemma 4 26B
  8192   # Gemma 4 31B
  8192   # Gemma 4 31B NVFP4
  129000 # MiniMax-M2.5 (NVIDIA's example uses 129000)
)

# Trust remote code flag (required for some custom architectures)
MODEL_TRUST_REMOTE=(
  # [community]
  false false false false false false false false false
  false false true   # phi-4
  false false false false
  # [NVIDIA] - Phi-4-multimodal and MiniMax need trust_remote_code per docs
  false false false              # Llama-3.1-8B FP8/NVFP4, Llama-3.3-70B
  false false false false false  # Qwen3 family
  false false false false        # Qwen2.5-VL + Qwen3-VL
  true  true  false false        # Phi-4-multimodal (trust_remote), Phi-4-reasoning
  false false false              # Nemotron-3-Nano + Super
  false false false false false  # Gemma 4 family
  true                           # MiniMax-M2.5 (NVIDIA explicitly uses --trust-remote-code)
)

# Requires HF token (gated models). NVIDIA-quantized derivatives of gated
# upstream models (e.g. nvidia/Llama-3.1-8B-Instruct-NVFP4) may still need a
# token for the original Meta/Google license check.
MODEL_NEEDS_TOKEN=(
  # [community]
  false false false false false false false false false
  true  true  false true  true  true  true     # Llama, Gemma2, CommandR, 405B-FP4, 3.3-70B
  # [NVIDIA] Llama family - gated (Meta license)
  true  true  true
  # [NVIDIA] Qwen3 - publicly redistributable
  false false false false false
  # [NVIDIA] Qwen2.5-VL public, Qwen3-VL Reranker/Embedding from Qwen org
  false false false false
  # [NVIDIA] Phi-4 family - public
  false false false false
  # [NVIDIA] Nemotron-3 - public
  false false false
  # [NVIDIA] Gemma 4 family - gated (Google license)
  true  true  true  true  true
  # MiniMax public
  false
)

# Enable expert parallel for MoE models (one entry per index in MODELS).
MODEL_EXPERT_PARALLEL=(
  # [community]
  true  true  false false false false false false true
  false false false false false false false
  # [NVIDIA] Llama family - dense
  false false false
  # [NVIDIA] Qwen3 - dense
  false false false false false
  # [NVIDIA] Qwen multimodal/reranker/embedding - dense
  false false false false
  # [NVIDIA] Phi-4 - dense
  false false false false
  # [NVIDIA] Nemotron-3-Nano-30B-A3B is MoE (A3B = active 3B), Super-120B-A12B is MoE
  true  true  true
  # [NVIDIA] Gemma 4 26B-A4B is MoE; rest are dense
  false false true  false false
  # MiniMax-M2.5 is MoE
  true
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

get_current_model() {
  # Try to get from running container
  if docker ps | grep -q ray-head; then
    LOADED_MODEL=$(docker exec ray-head bash -lc "curl -sf http://127.0.0.1:8000/v1/models 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin)[\"data\"][0][\"id\"])'" 2>/dev/null || echo "")
    if [ -n "${LOADED_MODEL}" ]; then
      echo "${LOADED_MODEL}"
      return
    fi
  fi

  # Fall back to reading start_cluster.sh
  if [ -f "${SCRIPT_DIR}/start_cluster.sh" ]; then
    grep '^MODEL=' "${SCRIPT_DIR}/start_cluster.sh" 2>/dev/null | head -1 | sed 's/MODEL="\${MODEL:-//' | sed 's/}"$//' || echo ""
  else
    echo ""
  fi
}

check_hf_token() {
  if [ -n "${HF_TOKEN:-}" ]; then
    return 0
  fi
  return 1
}

# Get the HF cache path for a model
get_model_cache_path() {
  local model="$1"
  local cache_name="models--$(echo "${model}" | sed 's|/|--|g')"
  echo "${HF_CACHE}/hub/${cache_name}"
}

# Check if model is downloaded locally
is_model_downloaded() {
  local model="$1"
  local cache_path=$(get_model_cache_path "${model}")

  if [ -d "${cache_path}/snapshots" ]; then
    # Check if there's at least one snapshot with model files
    local snapshot_count=$(find "${cache_path}/snapshots" -name "config.json" 2>/dev/null | wc -l)
    [ "${snapshot_count}" -gt 0 ]
  else
    return 1
  fi
}

# Download model using the modern `hf` CLI (huggingface-cli is deprecated).
download_model() {
  local model="$1"
  local token_arg=""

  if [ -n "${HF_TOKEN:-}" ]; then
    token_arg="--token ${HF_TOKEN}"
  fi

  log "Downloading model: ${model}"
  log "  Destination: ${HF_CACHE}/hub/"
  log "  Excluding: original/*, metal/* (to save space)"

  HF_HOME="${HF_CACHE}" hf download "${model}" ${token_arg} --exclude "original/*" --exclude "metal/*" 2>&1 | tail -10
}

# Rsync model to all worker nodes (1 to N). Sequential, with progress per worker.
rsync_model_to_worker() {
  local model="$1"
  local worker_user="${WORKER_USER:-$(whoami)}"
  local worker_hf_cache="${WORKER_HF_CACHE:-${HF_CACHE}}"

  # Split WORKER_HOST into array (matches start_cluster.sh's array convention).
  local -a worker_hosts=()
  if [ -n "${WORKER_HOST:-}" ]; then
    read -ra worker_hosts <<< "${WORKER_HOST}"
  fi

  if [ "${#worker_hosts[@]}" -eq 0 ]; then
    log "  No WORKER_HOST configured, skipping rsync"
    return 0
  fi

  local cache_name="models--$(echo "${model}" | sed 's|/|--|g')"
  local local_path="${HF_CACHE}/hub/${cache_name}"

  if [ ! -d "${local_path}" ]; then
    log "  ERROR: Model not found at ${local_path}"
    return 1
  fi

  log "Syncing model to ${#worker_hosts[@]} worker(s)..."
  log "  Source: ${local_path}"
  log "  Dest:   ${worker_hf_cache}/hub/ on each worker"

  local rsync_failures=0
  for i in "${!worker_hosts[@]}"; do
    local host="${worker_hosts[i]}"
    log "  [$((i+1))/${#worker_hosts[@]}] -> ${worker_user}@${host}"
    ssh "${worker_user}@${host}" "mkdir -p ${worker_hf_cache}/hub" 2>/dev/null || true
    if ! rsync -a --info=progress2 --human-readable \
      --no-perms --no-owner --no-group \
      --exclude='.locks' \
      --exclude='*.lock' \
      "${local_path}" \
      "${worker_user}@${host}:${worker_hf_cache}/hub/"; then
      log "  WARNING: rsync to ${host} failed"
      rsync_failures=$((rsync_failures + 1))
    fi
  done

  if [ "${rsync_failures}" -gt 0 ]; then
    log "  ${rsync_failures} of ${#worker_hosts[@]} worker(s) failed - cluster may not start"
    return 1
  fi
  log "  Model synced to all ${#worker_hosts[@]} worker(s)"
  return 0
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Parse Arguments
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SKIP_RESTART=false
LIST_ONLY=false
DOWNLOAD_ONLY=false
SKIP_DOWNLOAD=false
MODEL_NUMBER=""

usage() {
  cat << EOF
Usage: $0 [OPTIONS] [MODEL_NUMBER]

Switch between different models on the vLLM cluster.

Options:
  -l, --list          List available models without switching
  -s, --skip-restart  Update config only, don't restart cluster
  -d, --download-only Download model only, don't switch or restart
  --skip-download     Skip download step (use existing cached model)
  -h, --help          Show this help

Examples:
  $0                  # Interactive model selection
  $0 1                # Switch to model #1 (GPT-OSS-120B)
  $0 --list           # List all available models
  $0 -s 3             # Update config for model #3 without restarting
  $0 -d 5             # Download model #5 only (no restart)

Environment:
  WORKER_HOST         Worker node hostname/IP for rsync
  WORKER_USER         SSH username for worker (default: current user)
  HF_CACHE            HuggingFace cache directory (default: /raid/hf-cache)
  HF_TOKEN            HuggingFace token for gated models

EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -l|--list)
      LIST_ONLY=true
      shift
      ;;
    -s|--skip-restart)
      SKIP_RESTART=true
      shift
      ;;
    -d|--download-only)
      DOWNLOAD_ONLY=true
      shift
      ;;
    --skip-download)
      SKIP_DOWNLOAD=true
      shift
      ;;
    -h|--help)
      usage
      ;;
    [0-9]*)
      MODEL_NUMBER="$1"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " vLLM Model Switcher"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Show current model
CURRENT_MODEL=$(get_current_model)
if [ -n "${CURRENT_MODEL}" ]; then
  echo "Current model: ${CURRENT_MODEL}"
else
  echo "Current model: (not configured)"
fi
echo ""

# Display available models
echo "Available models:"
echo ""
echo "  Single-Node Models (TP=1):"
for i in "${!MODELS[@]}"; do
  if [ "${MODEL_NODES[$i]}" -eq 1 ]; then
    MARKER=""
    if [ "${MODELS[$i]}" = "${CURRENT_MODEL}" ]; then
      MARKER=" [CURRENT]"
    fi
    if [ "${MODEL_NEEDS_TOKEN[$i]}" = "true" ]; then
      MARKER="${MARKER} [HF TOKEN]"
    fi
    # Check if downloaded
    if is_model_downloaded "${MODELS[$i]}"; then
      MARKER="${MARKER} [CACHED]"
    fi
    printf "    %2d. %s%s\n" "$((i+1))" "${MODEL_NAMES[$i]}" "${MARKER}"
  fi
done

echo ""
echo "  Multi-Spark Models (require 2+ DGX Spark nodes, TP scales with N):"
for i in "${!MODELS[@]}"; do
  if [ "${MODEL_NODES[$i]}" -gt 1 ]; then
    MARKER=" [needs ${MODEL_NODES[$i]} Sparks, TP=${MODEL_TP[$i]}]"
    if [ "${MODELS[$i]}" = "${CURRENT_MODEL}" ]; then
      MARKER=" [CURRENT]${MARKER}"
    fi
    if [ "${MODEL_NEEDS_TOKEN[$i]}" = "true" ]; then
      MARKER="${MARKER} [HF TOKEN]"
    fi
    # Check if downloaded
    if is_model_downloaded "${MODELS[$i]}"; then
      MARKER="${MARKER} [CACHED]"
    fi
    printf "    %2d. %s%s\n" "$((i+1))" "${MODEL_NAMES[$i]}" "${MARKER}"
  fi
done
echo ""

# Exit if list only
if [ "${LIST_ONLY}" = "true" ]; then
  exit 0
fi

# Get model selection
if [ -z "${MODEL_NUMBER}" ]; then
  read -p "Select model (1-${#MODELS[@]}), or 'q' to quit: " MODEL_NUMBER
fi

if [ "${MODEL_NUMBER}" = "q" ] || [ "${MODEL_NUMBER}" = "Q" ]; then
  echo "Cancelled."
  exit 0
fi

# Validate selection
if ! [[ "${MODEL_NUMBER}" =~ ^[0-9]+$ ]] || [ "${MODEL_NUMBER}" -lt 1 ] || [ "${MODEL_NUMBER}" -gt "${#MODELS[@]}" ]; then
  echo "ERROR: Invalid selection. Please enter a number between 1 and ${#MODELS[@]}."
  exit 1
fi

# Get model configuration
IDX=$((MODEL_NUMBER - 1))
NEW_MODEL="${MODELS[$IDX]}"
NEW_MODEL_NAME="${MODEL_NAMES[$IDX]}"
NEW_QUANT="${MODEL_QUANT[$IDX]:-BF16}"
NEW_IMAGE="${MODEL_IMAGE[$IDX]:-}"
NEW_TP="${MODEL_TP[$IDX]}"
NEW_NODES="${MODEL_NODES[$IDX]}"
NEW_GPU_MEM="${MODEL_GPU_MEM[$IDX]}"
NEW_MAX_LEN="${MODEL_MAX_LEN[$IDX]}"
NEW_TRUST="${MODEL_TRUST_REMOTE[$IDX]}"
NEEDS_TOKEN="${MODEL_NEEDS_TOKEN[$IDX]}"
NEW_EXPERT_PARALLEL="${MODEL_EXPERT_PARALLEL[$IDX]}"

# Check if model needs HF token
if [ "${NEEDS_TOKEN}" = "true" ]; then
  if ! check_hf_token; then
    echo ""
    echo "WARNING: ${NEW_MODEL} requires a HuggingFace token."
    echo ""
    echo "Please set HF_TOKEN before continuing:"
    echo "  export HF_TOKEN=hf_your_token_here"
    echo ""
    read -p "Continue anyway? (y/N): " CONTINUE
    if [ "${CONTINUE}" != "y" ] && [ "${CONTINUE}" != "Y" ]; then
      echo "Cancelled."
      exit 1
    fi
  fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Switching to: ${NEW_MODEL_NAME}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Configuration:"
echo "  Model:             ${NEW_MODEL}"
echo "  Tensor Parallel:   ${NEW_TP}"
echo "  Nodes Required:    ${NEW_NODES}"
echo "  GPU Memory Util:   ${NEW_GPU_MEM}"
echo "  Max Context:       ${NEW_MAX_LEN}"
[ "${NEW_TRUST}" = "true" ] && echo "  Trust Remote Code: yes"
[ "${NEW_EXPERT_PARALLEL}" = "true" ] && echo "  Expert Parallel:   yes (MoE)"
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1: Download Model (if needed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [ "${SKIP_DOWNLOAD}" != "true" ]; then
  log "Step 1: Checking/Downloading model..."

  if is_model_downloaded "${NEW_MODEL}"; then
    log "  Model already cached locally"
  else
    log "  Model not found in cache, downloading..."
    if ! download_model "${NEW_MODEL}"; then
      echo "ERROR: Failed to download model"
      exit 1
    fi
    log "  Download complete"
  fi

  # Rsync to worker if multi-node
  if [ "${NEW_NODES}" -gt 1 ] && [ -n "${WORKER_HOST:-}" ]; then
    log ""
    log "Step 2: Syncing model to worker node..."
    rsync_model_to_worker "${NEW_MODEL}" || true
  fi
else
  log "Skipping download (--skip-download specified)"
fi

# Exit if download only
if [ "${DOWNLOAD_ONLY}" = "true" ]; then
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " Download Complete"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
  echo "Model downloaded: ${NEW_MODEL}"
  echo "Cache location:   $(get_model_cache_path "${NEW_MODEL}")"
  echo ""
  echo "To switch to this model and start the cluster:"
  echo "  $0 --skip-download ${MODEL_NUMBER}"
  echo ""
  exit 0
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2: Update Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log ""
log "Updating start_cluster.sh configuration..."

# Update the MODEL, TENSOR_PARALLEL, MAX_MODEL_LEN, GPU_MEMORY_UTIL in start_cluster.sh
START_SCRIPT="${SCRIPT_DIR}/start_cluster.sh"

if [ -f "${START_SCRIPT}" ]; then
  # Update MODEL
  sed -i "s|^MODEL=.*|MODEL=\"\${MODEL:-${NEW_MODEL}}\"|" "${START_SCRIPT}"

  # Update TENSOR_PARALLEL
  sed -i "s|^TENSOR_PARALLEL=.*|TENSOR_PARALLEL=\"\${TENSOR_PARALLEL:-${NEW_TP}}\"|" "${START_SCRIPT}"

  # Update MAX_MODEL_LEN
  sed -i "s|^MAX_MODEL_LEN=.*|MAX_MODEL_LEN=\"\${MAX_MODEL_LEN:-${NEW_MAX_LEN}}\"|" "${START_SCRIPT}"

  # Update GPU_MEMORY_UTIL
  sed -i "s|^GPU_MEMORY_UTIL=.*|GPU_MEMORY_UTIL=\"\${GPU_MEMORY_UTIL:-${NEW_GPU_MEM}}\"|" "${START_SCRIPT}"

  # Update ENABLE_EXPERT_PARALLEL (for MoE models)
  sed -i "s|^ENABLE_EXPERT_PARALLEL=.*|ENABLE_EXPERT_PARALLEL=\"\${ENABLE_EXPERT_PARALLEL:-${NEW_EXPERT_PARALLEL}}\"|" "${START_SCRIPT}"

  log "  Configuration updated in ${START_SCRIPT}"
else
  log "  WARNING: ${START_SCRIPT} not found"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 3: Restart Cluster (if not skipped)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [ "${SKIP_RESTART}" = "true" ]; then
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " Configuration Updated (restart skipped)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
  echo "To start the cluster with the new model:"
  echo "  ./start_cluster.sh"
  echo ""
  exit 0
fi

# Stop existing cluster
echo ""
log "Stopping existing cluster..."

# Stop head container
if docker ps | grep -q ray-head; then
  docker stop ray-head >/dev/null 2>&1 || true
  docker rm ray-head >/dev/null 2>&1 || true
  log "  Head container stopped"
else
  log "  No head container running"
fi

# Stop worker containers if WORKER_HOST is set
if [ -n "${WORKER_HOST:-}" ]; then
  log "  Stopping worker container on ${WORKER_HOST}..."
  ssh "${WORKER_USER:-$(whoami)}@${WORKER_HOST}" "docker stop ray-worker >/dev/null 2>&1; docker rm ray-worker >/dev/null 2>&1" 2>/dev/null || true
fi

# Start new cluster
echo ""
log "Starting cluster with new model..."
echo ""

# Set environment for start script
export MODEL="${NEW_MODEL}"
export TENSOR_PARALLEL="${NEW_TP}"
export MAX_MODEL_LEN="${NEW_MAX_LEN}"
export GPU_MEMORY_UTIL="${NEW_GPU_MEM}"
export ENABLE_EXPERT_PARALLEL="${NEW_EXPERT_PARALLEL}"
export TRUST_REMOTE_CODE="${NEW_TRUST}"
export SKIP_MODEL_DOWNLOAD=1  # We already downloaded
# Per-model image override (e.g. Gemma 4 needs vllm/vllm-openai:gemma4-cu130).
# Empty string means use the default IMAGE from config.env.
if [ -n "${NEW_IMAGE}" ]; then
  export VLLM_IMAGE="${NEW_IMAGE}"
  log "  Using model-specific image: ${VLLM_IMAGE}"
fi

if [ "${NEW_NODES}" -gt 1 ]; then
  echo "  Starting multi-node cluster (this may take 3-5 minutes)..."
else
  echo "  Starting single-node cluster (this may take 2-3 minutes)..."
fi

"${SCRIPT_DIR}/start_cluster.sh" 2>&1 | tee /tmp/model_switch.log &
STARTUP_PID=$!

# Wait for API
echo ""
log "Waiting for API to become ready..."

MAX_WAIT=1800  # 30 minutes for large models (70B+ need more time)
ELAPSED=0
API_URL="http://127.0.0.1:8000"

while [ $ELAPSED -lt $MAX_WAIT ]; do
  if curl -sf "${API_URL}/health" >/dev/null 2>&1; then
    echo ""
    echo "  API is ready!"
    break
  fi
  sleep 10
  ELAPSED=$((ELAPSED + 10))
  if [ $((ELAPSED % 30)) -eq 0 ]; then
    # Check if startup process is still running
    if ! kill -0 $STARTUP_PID 2>/dev/null; then
      # Check if it succeeded
      if curl -sf "${API_URL}/health" >/dev/null 2>&1; then
        echo ""
        echo "  API is ready!"
        break
      fi
    fi
    echo "  Still waiting... ${ELAPSED}s elapsed"
  fi
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
  echo ""
  echo "  WARNING: API not ready after ${MAX_WAIT}s"
  echo "  Check logs: docker logs ray-head"
  echo "  Or: cat /tmp/model_switch.log"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Verify and Display Results
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Model Switch Complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Get loaded model info
LOADED_MODEL=$(curl -sf "${API_URL}/v1/models" 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || echo "unknown")

echo "  Model:        ${LOADED_MODEL}"
echo "  API:          ${API_URL}"
echo "  Health:       ${API_URL}/health"
echo "  Time:         ${ELAPSED}s"
echo ""

# Quick test
echo "Testing inference..."
TEST_RESPONSE=$(curl -sf "${API_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"'"${NEW_MODEL}"'","messages":[{"role":"user","content":"Say OK"}],"max_tokens":5}' 2>/dev/null || echo "{}")

if echo "${TEST_RESPONSE}" | grep -q '"choices"'; then
  echo "  Inference test: PASSED"
else
  echo "  Inference test: FAILED (check logs)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
