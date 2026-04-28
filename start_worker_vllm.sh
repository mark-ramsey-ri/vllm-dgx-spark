#!/usr/bin/env bash
set -euo pipefail

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DGX Spark vLLM Worker Node - Production Setup Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NOTE: This script is automatically copied and executed on worker nodes
# by start_cluster.sh via SSH. You should NOT run this script manually.
# Instead, run start_cluster.sh on the head node with WORKER_HOST set.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Configuration
IMAGE="${IMAGE:-nvcr.io/nvidia/vllm:26.04-py3}"
RAY_VERSION="${RAY_VERSION:-2.55.1}"
RAY_PORT="${RAY_PORT:-6385}"
HF_CACHE="${HF_CACHE:-/raid/hf-cache}"
HF_TOKEN="${HF_TOKEN:-}"  # Set via: export HF_TOKEN=hf_xxx

# Model configuration - MUST match the head node's MODEL setting
MODEL="${MODEL:-openai/gpt-oss-120b}"

# Skip model download (set by head script when model is synced via rsync)
SKIP_MODEL_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-}"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Auto-detect Network Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Uses ibdev2netdev to discover active InfiniBand/RoCE interfaces.
# The IP address on the IB/RoCE interface can be any valid IP (not limited
# to link-local addresses). We rely on ibdev2netdev output to identify the
# correct network interface for RDMA communication.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Discover primary RoCE/IB interface using ibdev2netdev
discover_ib_interface() {
  if command -v ibdev2netdev >/dev/null 2>&1; then
    # Get the first active (Up) interface from ibdev2netdev
    local active_line
    active_line=$(ibdev2netdev 2>/dev/null | awk '/\(Up\)/ {print; exit}')

    if [ -n "$active_line" ]; then
      # Extract interface name (5th field, removing parentheses)
      echo "$active_line" | awk '{print $5}' | tr -d '()'
    fi
  fi
}

# Get all active IB/RoCE HCAs (comma-separated for NCCL_IB_HCA)
discover_all_ib_hcas() {
  if command -v ibdev2netdev >/dev/null 2>&1; then
    ibdev2netdev 2>/dev/null | grep "(Up)" | awk '{print $1}' | sort | tr '\n' ',' | sed 's/,$//'
  fi
}

# Get the first IPv4 address from an interface
get_interface_ip() {
  local iface="$1"
  if [ -n "$iface" ]; then
    ip -o addr show "$iface" 2>/dev/null | grep "inet " | awk '{print $4}' | cut -d'/' -f1 | head -1
  fi
}

# Auto-detect primary IB/RoCE interface
PRIMARY_IB_IF=$(discover_ib_interface)

# HEAD_IP is required - must be provided (it's the head node's IB/RoCE IP)
if [ -z "${HEAD_IP:-}" ]; then
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "❌ ERROR: HEAD_IP is not set"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
  echo "⚠️  This script should be run automatically by start_cluster.sh on the head node."
  echo ""
  echo "To start the cluster with workers:"
  echo "  1. On the head node, set WORKER_HOST to the worker's InfiniBand IP"
  echo "  2. Run: bash start_cluster.sh"
  echo ""
  echo "The head node will automatically SSH to workers and start them."
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
  exit 1
fi

# Auto-detect WORKER_IP from IB interface (or use override)
if [ -z "${WORKER_IP:-}" ]; then
  if [ -n "${PRIMARY_IB_IF}" ]; then
    WORKER_IP=$(get_interface_ip "${PRIMARY_IB_IF}")
  fi
  # Final fallback if auto-detection fails
  if [ -z "${WORKER_IP:-}" ]; then
    echo "ERROR: Could not auto-detect WORKER_IP from InfiniBand/RoCE interface."
    echo ""
    echo "Please ensure:"
    echo "  1. The InfiniBand/RoCE cable is connected between nodes"
    echo "  2. Run 'ibdev2netdev' to verify IB/RoCE interfaces are Up"
    echo "  3. Check that an IP is assigned to the IB/RoCE interface"
    echo ""
    echo "Then either:"
    echo "  - Fix the interface and re-run this script, OR"
    echo "  - Set WORKER_IP manually: export WORKER_IP=<your_ib_ip>"
    exit 1
  fi
fi

# Auto-detect network interfaces from active IB/RoCE devices
if [ -z "${GLOO_IF:-}" ] || [ -z "${TP_IF:-}" ] || [ -z "${NCCL_IF:-}" ] || [ -z "${UCX_DEV:-}" ]; then
  if [ -n "${PRIMARY_IB_IF}" ]; then
    # Use primary IB interface for all NCCL/GLOO/TP/UCX communication
    GLOO_IF="${GLOO_IF:-${PRIMARY_IB_IF}}"
    TP_IF="${TP_IF:-${PRIMARY_IB_IF}}"
    NCCL_IF="${NCCL_IF:-${PRIMARY_IB_IF}}"
    UCX_DEV="${UCX_DEV:-${PRIMARY_IB_IF}}"
  else
    # Error if no IB interface detected and not manually specified
    echo "ERROR: No active InfiniBand/RoCE interface detected."
    echo "Run 'ibdev2netdev' to check interface status."
    exit 1
  fi
fi

# Auto-detect InfiniBand HCAs using ibdev2netdev (or use override)
if [ -z "${NCCL_IB_HCA:-}" ]; then
  IB_DEVICES=$(discover_all_ib_hcas)
  if [ -n "${IB_DEVICES}" ]; then
    NCCL_IB_HCA="${IB_DEVICES}"
  else
    # Fallback: use all IB devices from sysfs
    IB_DEVICES=$(ls -1 /sys/class/infiniband/ 2>/dev/null | tr '\n' ',' | sed 's/,$//')
    NCCL_IB_HCA="${IB_DEVICES:-}"
    if [ -z "${NCCL_IB_HCA}" ]; then
      echo "ERROR: No InfiniBand HCAs detected."
      echo "Run 'ibdev2netdev' or check /sys/class/infiniband/"
      exit 1
    fi
  fi
fi

# Set OMPI_MCA for MPI-based communication (needed for some frameworks)
OMPI_MCA_IF="${NCCL_IF}"

# Generate unique worker name based on hostname
WORKER_NAME="ray-worker-$(hostname -s)"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

error() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
  exit 1
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log "Starting DGX Spark vLLM Worker Setup"
log "Configuration:"
log "  Image:         ${IMAGE}"
log "  Worker Name:   ${WORKER_NAME}"
log "  Head IP:       ${HEAD_IP}"
log "  Worker IP:     ${WORKER_IP} (auto-detected)"
log "  Ray Version:   ${RAY_VERSION}"
log "  Model:         ${MODEL}"
log ""
if [ -n "${HF_TOKEN}" ]; then
  log "  HF Auth:       ✅ Token provided"
else
  log "  HF Auth:       ⚠️  No token (gated models will fail)"
fi
log ""
log "Network Configuration (auto-detected from ibdev2netdev):"
log "  Primary IB IF:   ${PRIMARY_IB_IF:-<not detected>}"
log "  GLOO Interface:  ${GLOO_IF}"
log "  TP Interface:    ${TP_IF}"
log "  NCCL Interface:  ${NCCL_IF}"
log "  UCX Device:      ${UCX_DEV}"
log "  OMPI MCA IF:     ${OMPI_MCA_IF}"
log "  NCCL IB HCAs:    ${NCCL_IB_HCA}"
log ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log "Step 1/8: Testing connectivity to head"
if ! nc -zv -w 3 "${HEAD_IP}" "${RAY_PORT}" 2>&1 | grep -q "succeeded"; then
  error "Cannot reach Ray head at ${HEAD_IP}:${RAY_PORT}. Check network connectivity and firewall."
fi
log "  ✅ Head is reachable"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log "Step 2/8: Pulling Docker image"
if ! docker pull "${IMAGE}"; then
  error "Failed to pull image ${IMAGE}"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log "Step 3/8: Cleaning old container"
if docker ps -a --format '{{.Names}}' | grep -qx "${WORKER_NAME}"; then
  log "  Removing existing container: ${WORKER_NAME}"
  docker rm -f "${WORKER_NAME}" >/dev/null
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log "Step 4/8: Starting worker container"

# Build environment variable args for IB/NCCL configuration
# These are passed into the container to ensure NCCL uses the IB/RoCE link
# Note: We do NOT set HF_HUB_OFFLINE=1 because we need to download model weights first

# Build HF token env arg if provided
HF_TOKEN_ENV=""
if [ -n "${HF_TOKEN}" ]; then
  HF_TOKEN_ENV="-e HF_TOKEN=${HF_TOKEN}"
fi

# Run container as root (required for NVIDIA/vLLM)
docker run -d \
  --restart unless-stopped \
  --name "${WORKER_NAME}" \
  --gpus all \
  --network host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --cap-add=SYS_NICE \
  --device=/dev/infiniband \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -e VLLM_HOST_IP="${WORKER_IP}" \
  -e GLOO_SOCKET_IFNAME="${GLOO_IF}" \
  -e TP_SOCKET_IFNAME="${TP_IF}" \
  -e NCCL_SOCKET_IFNAME="${NCCL_IF}" \
  -e UCX_NET_DEVICES="${UCX_DEV}" \
  -e OMPI_MCA_btl_tcp_if_include="${OMPI_MCA_IF}" \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA="${NCCL_IB_HCA}" \
  -e NCCL_NET_GDR_LEVEL=5 \
  -e NCCL_DEBUG="${NCCL_DEBUG:-INFO}" \
  -e NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e RAY_memory_usage_threshold=0.995 \
  -e VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-1800}" \
  -e HF_HOME=/root/.cache/huggingface \
  ${HF_TOKEN_ENV} \
  "${IMAGE}" sleep infinity

if ! docker ps | grep -q "${WORKER_NAME}"; then
  error "Container failed to start"
fi

log "  Container started successfully"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log "Step 5/8: Verifying RDMA/InfiniBand libraries for NCCL"
log "  These libraries are required for NCCL to use InfiniBand/RoCE instead of Ethernet"
# NVIDIA vLLM container already includes RDMA libraries (libibverbs, librdmacm, ibverbs-providers)
# We just verify they're present rather than installing
if docker exec "${WORKER_NAME}" bash -lc "ldconfig -p 2>/dev/null | grep -q libibverbs"; then
  log "  ✅ RDMA libraries available (libibverbs, librdmacm)"
else
  log "  ⚠️  Warning: RDMA libraries may not be properly installed"
  log "     NCCL may fall back to Socket transport (slower)"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log "Step 6/8: Verifying container dependencies"
# The nvidia vLLM container already has vLLM and Ray properly built
# Do NOT pip install vllm as it would overwrite the CUDA-enabled version with CPU-only PyPI version

# Verify vLLM is available with CUDA
CUDA_AVAILABLE=$(docker exec "${WORKER_NAME}" python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
if [ "${CUDA_AVAILABLE}" != "True" ]; then
  error "PyTorch CUDA not available - container may be corrupted. Try: docker pull nvcr.io/nvidia/vllm:26.04-py3"
fi
log "  ✅ PyTorch CUDA available"

INSTALLED_VLLM_VERSION=$(docker exec "${WORKER_NAME}" python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
if [ "${INSTALLED_VLLM_VERSION}" == "unknown" ]; then
  error "vLLM not found in container"
fi
log "  ✅ vLLM ${INSTALLED_VLLM_VERSION} available"

# Install Ray if missing (26.04 container does not ship Ray)
if ! docker exec "${WORKER_NAME}" python3 -c "import ray" 2>/dev/null; then
  log "  Installing Ray ${RAY_VERSION} (not bundled in container)..."
  if ! docker exec "${WORKER_NAME}" pip install --quiet "ray[default]==${RAY_VERSION}" 2>&1 | tail -3; then
    error "Failed to install Ray ${RAY_VERSION}"
  fi
fi

# Verify Ray version
INSTALLED_RAY_VERSION=$(docker exec "${WORKER_NAME}" python3 -c "import ray; print(ray.__version__)" 2>/dev/null || echo "unknown")
if [ "${INSTALLED_RAY_VERSION}" == "unknown" ]; then
  error "Ray not found in container after install"
fi
log "  ✅ Ray ${INSTALLED_RAY_VERSION} available"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [ -n "${SKIP_MODEL_DOWNLOAD}" ]; then
  log "Step 7/8: Verifying model weights (synced from head)"
  log "  Model: ${MODEL}"
  log "  Skipping download - model was synced from head node via rsync"

  # Verify model was synced by checking for config.json
  if ! docker exec "${WORKER_NAME}" bash -lc "
    export HF_HOME=/root/.cache/huggingface
    python3 -c \"
from huggingface_hub import snapshot_download
import os
path = snapshot_download('${MODEL}', local_files_only=True)
config_path = os.path.join(path, 'config.json')
if not os.path.exists(config_path):
    raise FileNotFoundError(f'Model config not found at {config_path}')
print(f'  ✅ Model verified at: {path}')
\"
  "; then
    error "Model verification failed - model may not have been synced correctly from head"
  fi

  log "  Model verification complete"
else
  log "Step 7/8: Pre-downloading model weights"
  log "  Model: ${MODEL}"
  log "  This may take a while for large models on first download..."
  log ""
  log "  ⚠️  IMPORTANT: This model MUST match the head node's MODEL setting!"
  log ""

  # Build HF token arg if provided
  HF_TOKEN_ARG=""
  if [ -n "${HF_TOKEN}" ]; then
    HF_TOKEN_ARG="--token ${HF_TOKEN}"
  fi

  # Download model with verification (using 'hf download' instead of deprecated 'huggingface-cli download')
  if ! docker exec "${WORKER_NAME}" bash -lc "
    export HF_HOME=/root/.cache/huggingface
    echo '  Downloading model files (excluding original/* and metal/* to save space)...'
    hf download ${MODEL} ${HF_TOKEN_ARG} --exclude 'original/*' --exclude 'metal/*' 2>&1 | tail -5
  "; then
    error "Failed to download model ${MODEL}"
  fi

  # Verify model was downloaded by checking for config.json
  if ! docker exec "${WORKER_NAME}" bash -lc "
    export HF_HOME=/root/.cache/huggingface
    python3 -c \"
from huggingface_hub import snapshot_download
import os
path = snapshot_download('${MODEL}', local_files_only=True)
config_path = os.path.join(path, 'config.json')
if not os.path.exists(config_path):
    raise FileNotFoundError(f'Model config not found at {config_path}')
print(f'  ✅ Model verified at: {path}')
\"
  "; then
    error "Model verification failed - config.json not found"
  fi

  log "  Model download complete and verified"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log "Step 8/8: Joining Ray cluster"
docker exec "${WORKER_NAME}" bash -lc "
  ray stop --force 2>/dev/null || true
  ray start --address=${HEAD_IP}:${RAY_PORT} --node-ip-address=${WORKER_IP}
" >/dev/null

log "  Worker started, waiting for cluster registration..."

# Wait for worker to join cluster with progress indicator
# Check for multiple nodes in the cluster (more reliable than "Healthy:" which may not appear immediately)
START_TIME=$(date +%s)
CONNECTED=false
for i in {1..60}; do
  CURRENT_TIME=$(date +%s)
  ELAPSED=$((CURRENT_TIME - START_TIME))

  # Check if ray status shows more than 1 node (head + this worker)
  NODE_COUNT=$(docker exec "${WORKER_NAME}" bash -lc "ray status --address=${HEAD_IP}:${RAY_PORT} 2>/dev/null | grep -E '^ [0-9]+ node_' | wc -l" 2>/dev/null || echo "0")
  if [ "${NODE_COUNT}" -ge 2 ]; then
    echo ""
    log "  ✅ Worker connected to cluster (${ELAPSED}s) - ${NODE_COUNT} nodes active"
    CONNECTED=true
    break
  fi

  # Show spinner
  SPINNER="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
  SPIN_CHAR="${SPINNER:$((i % 10)):1}"
  printf "\r  %s Connecting to Ray cluster... [%ds elapsed]  " "${SPIN_CHAR}" "${ELAPSED}"

  sleep 1
done

if [ "${CONNECTED}" != "true" ]; then
  echo ""
  log "  ⚠️  Connection check timed out after 60s"
  log "     The worker may still be connected - verify from head node:"
  log "     docker exec ray-head ray status --address=127.0.0.1:${RAY_PORT}"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Worker ${WORKER_NAME} is ready!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🔍 Verify from head node:"
echo "  docker exec ray-head ray status --address=127.0.0.1:${RAY_PORT}"
echo ""
echo "📊 Expected output should show multiple 'Healthy' nodes"
echo ""
echo "🌐 Ray Dashboard: http://${HEAD_IP}:8265"
echo "   (Check 'Cluster' tab to see all nodes)"
echo ""
echo "⚙️  To increase parallelism, update head vLLM with:"
echo "  --tensor-parallel-size <num_total_gpus>"
echo ""
echo "🔧 Worker logs:"
echo "  docker logs -f ${WORKER_NAME}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
