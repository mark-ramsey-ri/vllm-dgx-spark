#!/usr/bin/env bash
set -euo pipefail

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# vLLM Cluster Startup Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Starts vLLM/Ray on head node and orchestrates worker nodes via SSH.
# Run from head node - it will handle workers automatically.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load configuration (config.local.env overrides config.env)
if [ -f "${SCRIPT_DIR}/config.local.env" ]; then
  source "${SCRIPT_DIR}/config.local.env"
elif [ -f "${SCRIPT_DIR}/config.env" ]; then
  source "${SCRIPT_DIR}/config.env"
fi

# Configuration (can be overridden by config.env or environment)
IMAGE="${VLLM_IMAGE:-${IMAGE:-nvcr.io/nvidia/vllm:26.03-py3}}"
NAME="${HEAD_CONTAINER_NAME:-${NAME:-ray-head}}"
HF_CACHE="${HF_CACHE:-/raid/hf-cache}"
HF_TOKEN="${HF_TOKEN:-}"
RAY_VERSION="${RAY_VERSION:-2.54.0}"
# Offline mode: skip the HF download step and run vLLM/HF in offline mode.
# Set SKIP_MODEL_DOWNLOAD=1 when the HF cache is already populated (air-gapped
# Spark, or model copied from another machine). The script will verify the
# model is present in ${HF_CACHE}/hub/ before launching vLLM.
SKIP_MODEL_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-}"

# Worker node configuration (for orchestrated setup)
# WORKER_HOST: Ethernet IP for SSH access (e.g., 192.168.7.111)
# WORKER_IB_IP: InfiniBand IP for NCCL communication (e.g., 169.254.216.8)
# Legacy WORKER_IPS is supported for backwards compatibility
WORKER_HOST="${WORKER_HOST:-}"
WORKER_IB_IP="${WORKER_IB_IP:-${WORKER_IPS:-}}"  # Fallback to WORKER_IPS for backwards compat
WORKER_USER="${WORKER_USER:-$(whoami)}"
WORKER_HF_CACHE="${WORKER_HF_CACHE:-${HF_CACHE}}"

# Model configuration
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.90}"
SWAP_SPACE="${SWAP_SPACE:-16}"
SHM_SIZE="${SHM_SIZE:-16g}"

# Auto-detect MoE models for expert parallelism
# Known MoE models: gpt-oss, mixtral, deepseek-moe, qwen-moe
if [ -z "${ENABLE_EXPERT_PARALLEL:-}" ]; then
  if echo "${MODEL}" | grep -qiE "(gpt-oss|mixtral|deepseek.*moe|qwen.*moe)"; then
    ENABLE_EXPERT_PARALLEL="true"
  else
    ENABLE_EXPERT_PARALLEL="false"
  fi
fi

TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"
# Model loading format (safetensors is recommended)
LOAD_FORMAT="${LOAD_FORMAT:-safetensors}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Ports
VLLM_PORT="${VLLM_PORT:-8000}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_PORT="${RAY_PORT:-6385}"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Detect Single-Node Mode Early
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Single-node mode is enabled when:
#   1. WORKER_HOST is not set (no remote workers)
#   2. TENSOR_PARALLEL <= number of local GPUs
# In single-node mode, InfiniBand is not required.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Detect local GPU count
LOCAL_GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
if [ "${LOCAL_GPU_COUNT}" -eq 0 ]; then
  # Fallback: try nvidia-smi query
  LOCAL_GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo "1")
fi

# Determine if we're in single-node mode
# Single-node: no WORKER_HOST and TENSOR_PARALLEL fits on local GPUs
if [ -z "${WORKER_HOST}" ] && [ "${TENSOR_PARALLEL}" -le "${LOCAL_GPU_COUNT}" ]; then
  SINGLE_NODE_MODE=true
else
  SINGLE_NODE_MODE=false
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Auto-detect Network Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# For multi-node: Uses ibdev2netdev to discover active InfiniBand/RoCE interfaces.
# For single-node: Uses localhost/loopback - InfiniBand not required.
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Network Configuration (conditional on single-node vs multi-node)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [ "${SINGLE_NODE_MODE}" = "true" ]; then
  # Single-node mode: InfiniBand not required
  # Use loopback or first available ethernet interface
  PRIMARY_IB_IF=""

  # Use provided HEAD_IP or default to 127.0.0.1
  if [ -z "${HEAD_IP:-}" ]; then
    # Try to get a non-loopback IP for Ray (preferred for dashboard access)
    HEAD_IP=$(ip -o addr show | grep "inet " | grep -v "127.0.0.1" | grep -v "172.17" | awk '{print $4}' | cut -d'/' -f1 | head -1)
    if [ -z "${HEAD_IP}" ]; then
      HEAD_IP="127.0.0.1"
    fi
  fi

  # For single-node, use loopback for all NCCL/GLOO interfaces
  # NCCL will use NVLink/PCIe for GPU-to-GPU communication
  GLOO_IF="${GLOO_IF:-lo}"
  TP_IF="${TP_IF:-lo}"
  NCCL_IF="${NCCL_IF:-lo}"
  UCX_DEV="${UCX_DEV:-lo}"
  NCCL_IB_HCA=""
  OMPI_MCA_IF="lo"

else
  # Multi-node mode: InfiniBand required for performance

  # Auto-detect primary IB/RoCE interface
  PRIMARY_IB_IF=$(discover_ib_interface)

  # Auto-detect HEAD_IP from IB interface (or use override)
  if [ -z "${HEAD_IP:-}" ]; then
    if [ -n "${PRIMARY_IB_IF}" ]; then
      HEAD_IP=$(get_interface_ip "${PRIMARY_IB_IF}")
    fi
    # Final fallback if auto-detection fails
    if [ -z "${HEAD_IP:-}" ]; then
      echo "ERROR: Could not auto-detect HEAD_IP from InfiniBand/RoCE interface."
      echo ""
      echo "Please ensure:"
      echo "  1. The InfiniBand/RoCE cable is connected between nodes"
      echo "  2. Run 'ibdev2netdev' to verify IB/RoCE interfaces are Up"
      echo "  3. Check that an IP is assigned to the IB/RoCE interface"
      echo ""
      echo "Then either:"
      echo "  - Fix the interface and re-run this script, OR"
      echo "  - Set HEAD_IP manually: export HEAD_IP=<your_ib_ip>"
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
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

error() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
  exit 1
}

# Step counter - incremented before each step
CURRENT_STEP=0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pre-flight Checks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Pre-flight Checks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

PREFLIGHT_FAILED=false

# Check 1: Worker configuration for distributed setup
echo "Checking worker configuration..."
# If WORKER_HOST not set but WORKER_IB_IP is, fall back for backwards compat
if [ -z "${WORKER_HOST}" ] && [ -n "${WORKER_IB_IP}" ]; then
  echo "  ⚠️  WORKER_HOST not set, using WORKER_IB_IP (${WORKER_IB_IP}) for SSH"
  WORKER_HOST="${WORKER_IB_IP}"
  # Re-evaluate single-node mode since WORKER_HOST changed
  SINGLE_NODE_MODE=false
fi

if [ "${SINGLE_NODE_MODE}" = "true" ]; then
  echo "  ℹ️  Running in SINGLE-NODE mode (no InfiniBand required)"
  echo ""
  echo "  Local GPUs: ${LOCAL_GPU_COUNT}"
  echo "  Tensor Parallel: ${TENSOR_PARALLEL}"
  echo ""
  echo "  For multi-node distributed inference, set:"
  echo "    export WORKER_HOST=\"192.168.x.x\"    # Ethernet IP for SSH"
  echo "    export WORKER_IB_IP=\"169.254.x.x\"   # InfiniBand IP for NCCL"
  echo ""
elif [ -z "${WORKER_HOST}" ]; then
  # TENSOR_PARALLEL > LOCAL_GPU_COUNT but no WORKER_HOST
  echo "  ⚠️  TENSOR_PARALLEL=${TENSOR_PARALLEL} exceeds local GPUs (${LOCAL_GPU_COUNT})"
  echo ""
  echo "  Either:"
  echo "    1. Set TENSOR_PARALLEL=${LOCAL_GPU_COUNT} for single-node, OR"
  echo "    2. Configure a worker node:"
  echo "       export WORKER_HOST=\"192.168.x.x\"    # Ethernet IP for SSH"
  echo "       export WORKER_IB_IP=\"169.254.x.x\"   # InfiniBand IP for NCCL"
  echo ""
  PREFLIGHT_FAILED=true
else
  echo "  ✅ WORKER_HOST=${WORKER_HOST} (SSH)"
  if [ -n "${WORKER_IB_IP}" ]; then
    echo "  ✅ WORKER_IB_IP=${WORKER_IB_IP} (NCCL)"
  fi
fi

# Check 2: SSH connectivity to worker
if [ -n "${WORKER_HOST}" ]; then
  echo ""
  echo "Checking SSH connectivity to worker..."
  if ssh -o BatchMode=yes -o ConnectTimeout=5 "${WORKER_USER}@${WORKER_HOST}" "echo 'SSH OK'" >/dev/null 2>&1; then
    echo "  ✅ SSH to ${WORKER_USER}@${WORKER_HOST} successful"
  else
    echo "  ❌ Cannot SSH to ${WORKER_USER}@${WORKER_HOST}"
    echo ""
    echo "  Passwordless SSH must be configured. Run:"
    echo ""
    echo "    ssh-copy-id ${WORKER_USER}@${WORKER_HOST}"
    echo ""
    echo "  Note: You also need passwordless SSH to the worker's InfiniBand IP"
    echo "  for NCCL communication. If your worker has IB IP 169.254.x.x, also run:"
    echo ""
    echo "    ssh-copy-id ${WORKER_USER}@<worker-infiniband-ip>"
    echo ""
    PREFLIGHT_FAILED=true
  fi
fi

# Check 3: Docker available
echo ""
echo "Checking Docker..."
if command -v docker >/dev/null 2>&1; then
  echo "  ✅ Docker is available"
else
  echo "  ❌ Docker is not installed or not in PATH"
  PREFLIGHT_FAILED=true
fi

# Check 4: NVIDIA GPU access
echo ""
echo "Checking NVIDIA GPU access..."
if docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
  echo "  ✅ NVIDIA GPU access via Docker is working"
else
  echo "  ❌ Cannot access NVIDIA GPUs via Docker"
  echo "  Ensure nvidia-container-toolkit is installed"
  PREFLIGHT_FAILED=true
fi

# Check 5: InfiniBand detection (skip in single-node mode)
echo ""
echo "Checking InfiniBand..."
if [ "${SINGLE_NODE_MODE}" = "true" ]; then
  echo "  ℹ️  InfiniBand check skipped (single-node mode)"
  echo "  ✅ Using local communication (NVLink/PCIe for GPU-to-GPU)"
  echo "  ✅ Head IP: ${HEAD_IP}"
elif [ -n "${PRIMARY_IB_IF}" ]; then
  echo "  ✅ InfiniBand interface detected: ${PRIMARY_IB_IF}"
  echo "  ✅ Head IP (auto-detected): ${HEAD_IP}"
else
  echo "  ❌ No active InfiniBand interface detected"
  echo "  Run 'ibdev2netdev' to check interface status"
  PREFLIGHT_FAILED=true
fi

# Check 6: Local HuggingFace cache permissions
echo ""
echo "Checking local HuggingFace cache permissions..."
if [ -d "${HF_CACHE}" ]; then
  if [ -w "${HF_CACHE}" ]; then
    echo "  ✅ Local HF cache is writable (${HF_CACHE})"
  else
    echo "  ⚠️  Local HF cache is not writable: ${HF_CACHE}"
    echo ""
    echo "  Docker containers run as root and may have created files owned by root."
    read -p "  Fix permissions now? (requires sudo) [y/N]: " fix_local_perms
    if [[ "$fix_local_perms" =~ ^[Yy]$ ]]; then
      if sudo chown -R "$USER" "${HF_CACHE}"; then
        echo "  ✅ Local HF cache permissions fixed"
      else
        echo "  ❌ Failed to fix local permissions"
        PREFLIGHT_FAILED=true
      fi
    else
      echo "  ❌ Local HF cache permissions not fixed - rsync may fail"
      PREFLIGHT_FAILED=true
    fi
  fi
else
  echo "  ℹ️  Local HF cache will be created at ${HF_CACHE}"
fi

# Check 7: Remote HuggingFace cache permissions (only if WORKER_HOST is set)
if [ -n "${WORKER_HOST}" ]; then
  echo ""
  echo "Checking remote HuggingFace cache permissions on worker..."
  REMOTE_HF_WRITABLE=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "${WORKER_USER}@${WORKER_HOST}" "
    if [ -d '${WORKER_HF_CACHE}' ]; then
      if [ -w '${WORKER_HF_CACHE}' ]; then
        echo 'writable'
      else
        echo 'not_writable'
      fi
    else
      echo 'not_exists'
    fi
  " 2>/dev/null || echo "ssh_failed")

  case "$REMOTE_HF_WRITABLE" in
    writable)
      echo "  ✅ Remote HF cache is writable (${WORKER_HF_CACHE})"
      ;;
    not_writable)
      echo "  ⚠️  Remote HF cache is not writable: ${WORKER_HF_CACHE}"
      echo ""
      echo "  Docker containers run as root and may have created files owned by root."
      read -p "  Fix permissions on worker now? (requires sudo password on worker) [y/N]: " fix_remote_perms
      if [[ "$fix_remote_perms" =~ ^[Yy]$ ]]; then
        echo "  Running: sudo chown -R ${WORKER_USER} ${WORKER_HF_CACHE} on ${WORKER_HOST}"
        if ssh -t "${WORKER_USER}@${WORKER_HOST}" "sudo chown -R ${WORKER_USER} '${WORKER_HF_CACHE}'"; then
          echo "  ✅ Remote HF cache permissions fixed"
        else
          echo "  ❌ Failed to fix remote permissions"
          PREFLIGHT_FAILED=true
        fi
      else
        echo "  ❌ Remote HF cache permissions not fixed - rsync may fail"
        PREFLIGHT_FAILED=true
      fi
      ;;
    not_exists)
      echo "  ℹ️  Remote HF cache will be created at ${WORKER_HF_CACHE}"
      ;;
    ssh_failed)
      echo "  ⚠️  Could not check remote HF cache (SSH failed)"
      ;;
  esac
fi

echo ""

# Exit if any pre-flight check failed
if [ "$PREFLIGHT_FAILED" = true ]; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "❌ Pre-flight checks failed. Please fix the issues above and re-run."
  echo ""
  echo "To configure required variables, run:"
  echo "  source ./setup-env.sh --head"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All pre-flight checks passed"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log "Starting DGX Spark vLLM Head Node Setup"
log "Configuration:"
log "  Image:           ${IMAGE}"
log "  Head IP:         ${HEAD_IP} (auto-detected)"
log "  Model:           ${MODEL}"
log "  Tensor Parallel: ${TENSOR_PARALLEL}"
log "  Load Format:     ${LOAD_FORMAT}"
log "  Ray Version:     ${RAY_VERSION}"
log ""
if [ -n "${WORKER_HOST}" ]; then
  log "Worker Node Configuration (orchestrated setup enabled):"
  log "  Worker Host:     ${WORKER_HOST} (SSH)"
  if [ -n "${WORKER_IB_IP}" ]; then
    log "  Worker IB IP:    ${WORKER_IB_IP} (NCCL)"
  fi
  log "  Worker User:     ${WORKER_USER}"
  log "  Worker HF Cache: ${WORKER_HF_CACHE}"
  log ""
else
  log "Worker Node Configuration:"
  log "  ⚠️  WORKER_HOST not set - manual worker setup required"
  log "     export WORKER_HOST=<ethernet_ip>  # For SSH"
  log "     export WORKER_IB_IP=<ib_ip>       # For NCCL"
  log ""
fi
if [ "${SINGLE_NODE_MODE}" = "true" ]; then
  log "Network Configuration (single-node mode - InfiniBand not required):"
  log "  Mode:            Single-node (local GPUs only)"
  log "  Local GPUs:      ${LOCAL_GPU_COUNT}"
  log "  Communication:   NVLink/PCIe (no IB needed)"
else
  log "Network Configuration (auto-detected from ibdev2netdev):"
  log "  Primary IB IF:   ${PRIMARY_IB_IF:-<not detected>}"
  log "  GLOO Interface:  ${GLOO_IF}"
  log "  TP Interface:    ${TP_IF}"
  log "  NCCL Interface:  ${NCCL_IF}"
  log "  UCX Device:      ${UCX_DEV}"
  log "  OMPI MCA IF:     ${OMPI_MCA_IF}"
  log "  NCCL IB HCAs:    ${NCCL_IB_HCA}"
fi
log ""
if [ -n "${HF_TOKEN}" ]; then
  log "  HF Auth:        ✅ Token provided"
else
  log "  HF Auth:        ⚠️  No token (gated models will fail)"
fi
log ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Calculate total steps based on whether worker orchestration is enabled
if [ -n "${WORKER_HOST}" ]; then
  TOTAL_STEPS=12
else
  TOTAL_STEPS=9
fi

log "Step 1/${TOTAL_STEPS}: Pulling Docker image"
if ! docker pull "${IMAGE}"; then
  error "Failed to pull image ${IMAGE}"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log "Step 2/${TOTAL_STEPS}: Cleaning old container"
if docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
  log "  Removing existing container: ${NAME}"
  docker rm -f "${NAME}" >/dev/null
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log "Step 3/${TOTAL_STEPS}: Starting head container"

# Build environment variable args for NCCL configuration
# These are passed into the container to configure GPU communication
ENV_ARGS=(
  -e VLLM_HOST_IP="${HEAD_IP}"
  # Debug settings (can be disabled for production by setting NCCL_DEBUG=WARN)
  -e NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
  -e NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"
  # NVIDIA/GPU settings
  -e NVIDIA_VISIBLE_DEVICES=all
  -e NVIDIA_DRIVER_CAPABILITIES=all
  # Ray settings
  -e RAY_memory_usage_threshold=0.998
  -e RAY_GCS_SERVER_PORT="${RAY_PORT}"
  # vLLM timeout settings (large models like 70B+ can take 20+ minutes to load)
  -e VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-1800}"
  # HuggingFace cache
  -e HF_HOME=/root/.cache/huggingface
)

if [ "${SINGLE_NODE_MODE}" = "true" ]; then
  # Single-node mode: disable IB, use NVLink/PCIe
  ENV_ARGS+=(
    -e NCCL_IB_DISABLE=1
    -e NCCL_SOCKET_IFNAME="${NCCL_IF}"
  )
else
  # Multi-node mode: enable IB/RoCE for high-speed communication
  ENV_ARGS+=(
    # IB/RoCE interface settings for NCCL communication
    -e GLOO_SOCKET_IFNAME="${GLOO_IF}"
    -e TP_SOCKET_IFNAME="${TP_IF}"
    -e NCCL_SOCKET_IFNAME="${NCCL_IF}"
    -e UCX_NET_DEVICES="${UCX_DEV}"
    -e OMPI_MCA_btl_tcp_if_include="${OMPI_MCA_IF}"
    # NCCL InfiniBand settings
    -e NCCL_IB_DISABLE=0
    -e NCCL_IB_HCA="${NCCL_IB_HCA}"
    -e NCCL_NET_GDR_LEVEL=5
  )
fi

# Add HuggingFace token if provided
if [ -n "${HF_TOKEN}" ]; then
  ENV_ARGS+=(-e HF_TOKEN="${HF_TOKEN}")
fi

# Offline mode: prevent HF/Transformers from making network calls inside the container
if [ -n "${SKIP_MODEL_DOWNLOAD}" ]; then
  ENV_ARGS+=(
    -e HF_HUB_OFFLINE=1
    -e TRANSFORMERS_OFFLINE=1
  )
fi

# Build Docker run command
# HF cache is mounted to /root/.cache/huggingface
DOCKER_ARGS=(
  --restart unless-stopped
  --name "${NAME}"
  --gpus all
  --network host
  --shm-size="${SHM_SIZE}"
  --ulimit memlock=-1
  --ulimit stack=67108864
  --cap-add=SYS_NICE
  -v "${HF_CACHE}:/root/.cache/huggingface"
)

# Add InfiniBand device only in multi-node mode (and only if device exists)
if [ "${SINGLE_NODE_MODE}" != "true" ] && [ -d "/dev/infiniband" ]; then
  DOCKER_ARGS+=(--device=/dev/infiniband)
fi

# Run container as root (required for NVIDIA/vLLM)
docker run -d \
  "${DOCKER_ARGS[@]}" \
  "${ENV_ARGS[@]}" \
  "${IMAGE}" sleep infinity

if ! docker ps | grep -q "${NAME}"; then
  error "Container failed to start"
fi

log "  Container started successfully"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log "Step 4/${TOTAL_STEPS}: Verifying RDMA/InfiniBand libraries for NCCL"
if [ "${SINGLE_NODE_MODE}" = "true" ]; then
  log "  ℹ️  RDMA check skipped (single-node mode uses NVLink/PCIe)"
else
  log "  These libraries are required for NCCL to use InfiniBand/RoCE instead of Ethernet"
  # NVIDIA vLLM container already includes RDMA libraries (libibverbs, librdmacm, ibverbs-providers)
  # We just verify they're present rather than installing
  if docker exec "${NAME}" bash -lc "ldconfig -p 2>/dev/null | grep -q libibverbs"; then
    log "  ✅ RDMA libraries available (libibverbs, librdmacm)"
  else
    log "  ⚠️  Warning: RDMA libraries may not be properly installed"
    log "     NCCL may fall back to Socket transport (slower)"
  fi
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

log "Step 5/${TOTAL_STEPS}: Verifying container dependencies"
# The nvidia vLLM container already has vLLM and Ray properly built
# Do NOT pip install vllm as it would overwrite the CUDA-enabled version with CPU-only PyPI version

# Verify vLLM is available with CUDA
CUDA_AVAILABLE=$(docker exec "${NAME}" python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
if [ "${CUDA_AVAILABLE}" != "True" ]; then
  error "PyTorch CUDA not available - container may be corrupted. Try: docker pull nvcr.io/nvidia/vllm:26.03-py3"
fi
log "  ✅ PyTorch CUDA available"

INSTALLED_VLLM_VERSION=$(docker exec "${NAME}" python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
if [ "${INSTALLED_VLLM_VERSION}" == "unknown" ]; then
  error "vLLM not found in container"
fi
log "  ✅ vLLM ${INSTALLED_VLLM_VERSION} available"

# Verify Ray version
INSTALLED_RAY_VERSION=$(docker exec "${NAME}" python3 -c "import ray; print(ray.__version__)" 2>/dev/null || echo "unknown")
if [ "${INSTALLED_RAY_VERSION}" == "unknown" ]; then
  error "Ray not found in container"
fi
log "  ✅ Ray ${INSTALLED_RAY_VERSION} available"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [ -n "${SKIP_MODEL_DOWNLOAD}" ]; then
  log "Step 6/${TOTAL_STEPS}: Verifying pre-staged model weights (offline mode)"
  log "  Model: ${MODEL}"
  log "  Skipping download - SKIP_MODEL_DOWNLOAD is set"
else
  log "Step 6/${TOTAL_STEPS}: Pre-downloading model weights"
  log "  Model: ${MODEL}"
  log "  This may take a while for large models on first download..."

  # Build HF token arg if provided
  HF_TOKEN_ARG=""
  if [ -n "${HF_TOKEN}" ]; then
    HF_TOKEN_ARG="--token ${HF_TOKEN}"
  fi

  # Download model with verification
  if ! docker exec "${NAME}" bash -lc "
    export HF_HOME=/root/.cache/huggingface
    echo '  Downloading model files (excluding original/* and metal/* to save space)...'
    hf download ${MODEL} ${HF_TOKEN_ARG} --exclude 'original/*' --exclude 'metal/*' 2>&1 | tail -5
  "; then
    error "Failed to download model ${MODEL}"
  fi
fi

# Verify model is present in the cache (works for both online and offline paths)
if ! docker exec "${NAME}" bash -lc "
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
  if [ -n "${SKIP_MODEL_DOWNLOAD}" ]; then
    error "Model not found in ${HF_CACHE}/hub/ - populate the cache before running with SKIP_MODEL_DOWNLOAD=1, or unset it to download"
  else
    error "Model verification failed - config.json not found"
  fi
fi

if [ -n "${SKIP_MODEL_DOWNLOAD}" ]; then
  log "  Model verified in local cache"
else
  log "  Model download complete and verified"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Worker orchestration steps (only if WORKER_HOST is set)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [ -n "${WORKER_HOST}" ]; then
  if [ -n "${SKIP_MODEL_DOWNLOAD}" ]; then
    log "Step 7/${TOTAL_STEPS}: Skipping model rsync (offline mode)"
    log "  SKIP_MODEL_DOWNLOAD is set - worker is expected to have its own"
    log "  pre-staged HF cache at ${WORKER_HF_CACHE}/hub/."

    # Still need SSH to start the worker, so verify it now
    if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "${WORKER_USER}@${WORKER_HOST}" "echo 'SSH OK'" >/dev/null 2>&1; then
      error "Cannot SSH to ${WORKER_USER}@${WORKER_HOST}. Please ensure:"$'\n'"  1. SSH keys are set up (ssh-copy-id ${WORKER_USER}@${WORKER_HOST})"$'\n'"  2. The worker host is reachable"
    fi
    log "  ✅ SSH connectivity verified"
  else
    log "Step 7/${TOTAL_STEPS}: Syncing model to worker node"
    log "  Worker host: ${WORKER_USER}@${WORKER_HOST}"
    log "  Source: ${HF_CACHE}/"
    log "  Destination: ${WORKER_HF_CACHE}/"
    log ""
    log "  This may take a while for large models..."

    # Test SSH connectivity first
    if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "${WORKER_USER}@${WORKER_HOST}" "echo 'SSH OK'" >/dev/null 2>&1; then
      error "Cannot SSH to ${WORKER_USER}@${WORKER_HOST}. Please ensure:"$'\n'"  1. SSH keys are set up (ssh-copy-id ${WORKER_USER}@${WORKER_HOST})"$'\n'"  2. The worker host is reachable"
    fi
    log "  ✅ SSH connectivity verified"

    # Ensure destination directory exists and fix permissions if needed
    ssh "${WORKER_USER}@${WORKER_HOST}" "
      mkdir -p ${WORKER_HF_CACHE}/hub
      # Fix ownership if directories exist but are owned by root (from Docker)
      if [ -d '${WORKER_HF_CACHE}' ] && [ ! -w '${WORKER_HF_CACHE}' ]; then
        echo 'Fixing permissions on ${WORKER_HF_CACHE} (requires sudo)...'
        sudo chown -R \$(id -u):\$(id -g) '${WORKER_HF_CACHE}' 2>/dev/null || true
      fi
    "

    # Convert model name to HF cache directory format (e.g., openai/gpt-oss-120b -> models--openai--gpt-oss-120b)
    MODEL_CACHE_NAME="models--$(echo "${MODEL}" | sed 's|/|--|g')"
    MODEL_CACHE_PATH="${HF_CACHE}/hub/${MODEL_CACHE_NAME}"

    if [ ! -d "${MODEL_CACHE_PATH}" ]; then
      error "Model cache not found at ${MODEL_CACHE_PATH}. Download the model first."
    fi

    log "  Syncing model: ${MODEL_CACHE_NAME}"
    log "  From: ${MODEL_CACHE_PATH}"
    log "  To:   ${WORKER_USER}@${WORKER_HOST}:${WORKER_HF_CACHE}/hub/"

    # Rsync only the specific model directory (exclude lock files)
    # Use --no-perms --no-owner --no-group to avoid permission issues with mixed ownership
    if ! rsync -a --info=progress2 --human-readable \
      --no-perms --no-owner --no-group \
      --exclude='.locks' \
      --exclude='*.lock' \
      "${MODEL_CACHE_PATH}" \
      "${WORKER_USER}@${WORKER_HOST}:${WORKER_HF_CACHE}/hub/"; then
      error "Failed to rsync model to worker node"
    fi
    log "  ✅ Model synced to worker"
  fi

  # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  log "Step 8/${TOTAL_STEPS}: Copying worker script to worker node"
  WORKER_SCRIPT="${SCRIPT_DIR}/start_worker_vllm.sh"

  if [ ! -f "${WORKER_SCRIPT}" ]; then
    error "Worker script not found at ${WORKER_SCRIPT}"
  fi

  # Copy the worker script to the worker's home directory
  if ! scp "${WORKER_SCRIPT}" "${WORKER_USER}@${WORKER_HOST}:~/start_worker_vllm.sh"; then
    error "Failed to copy worker script to ${WORKER_HOST}"
  fi
  log "  ✅ Worker script copied to ${WORKER_USER}@${WORKER_HOST}:~/start_worker_vllm.sh"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Step number depends on whether worker orchestration steps ran
if [ -n "${WORKER_HOST}" ]; then
  RAY_STEP=9
else
  RAY_STEP=7
fi
log "Step ${RAY_STEP}/${TOTAL_STEPS}: Starting Ray head"
docker exec "${NAME}" bash -lc "
  ray stop --force 2>/dev/null || true
  ray start --head \
    --node-ip-address=${HEAD_IP} \
    --port=${RAY_PORT} \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=${RAY_DASHBOARD_PORT}
" >/dev/null

log "  Ray head started, waiting for readiness..."

# Wait for Ray to become ready
for i in {1..30}; do
  if docker exec "${NAME}" bash -lc "ray status --address='127.0.0.1:${RAY_PORT}' >/dev/null 2>&1"; then
    log "  ✅ Ray head is ready (${i}s)"
    break
  fi
  if [ $i -eq 30 ]; then
    error "Ray head failed to become ready after 30 seconds"
  fi
  sleep 1
done

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [ -n "${WORKER_HOST}" ]; then
  # Orchestrated mode: Start worker via SSH
  log "Step 10/${TOTAL_STEPS}: Starting worker node via SSH"
  log "  Starting worker script on ${WORKER_USER}@${WORKER_HOST}..."
  log ""

  # Build environment variables to pass to worker
  # Worker needs HEAD_IP, MODEL, HF_TOKEN, and HF_CACHE
  WORKER_ENV="HEAD_IP=${HEAD_IP} MODEL=${MODEL} HF_CACHE=${WORKER_HF_CACHE} RAY_VERSION=${RAY_VERSION} RAY_PORT=${RAY_PORT} SKIP_MODEL_DOWNLOAD=1"
  if [ -n "${HF_TOKEN}" ]; then
    WORKER_ENV="${WORKER_ENV} HF_TOKEN=${HF_TOKEN}"
  fi

  # Start the worker script via SSH (run in background)
  # Use ssh -f with nohup to properly detach the process
  # The sleep 1 at the end ensures SSH doesn't exit before nohup takes over
  ssh -f "${WORKER_USER}@${WORKER_HOST}" "nohup bash -c '${WORKER_ENV} bash ~/start_worker_vllm.sh > ~/worker_setup.log 2>&1' </dev/null >/dev/null 2>&1 &"

  # Give SSH a moment to start the remote process
  sleep 2

  log "  Worker script started in background on ${WORKER_HOST}"
  log "  Logs available at: ${WORKER_USER}@${WORKER_HOST}:~/worker_setup.log"
  log ""

  # Wait for worker to join the cluster (check for 2+ nodes)
  log "  Waiting for worker to join Ray cluster..."
  WORKER_JOINED=false
  for i in {1..120}; do
    NODE_COUNT=$(docker exec "${NAME}" bash -lc "ray status --address=127.0.0.1:${RAY_PORT} 2>/dev/null | grep -E '^ [0-9]+ node_' | wc -l" 2>/dev/null || echo "0")
    CURRENT_GPUS=$(docker exec "${NAME}" bash -lc "ray status --address=127.0.0.1:${RAY_PORT} 2>/dev/null | grep 'GPU:' | awk -F'/' '{print \$2}' | awk '{print \$1}'" 2>/dev/null || echo "1")

    if [ "${NODE_COUNT}" -ge 2 ]; then
      echo ""
      log "  ✅ Worker joined cluster (${i}s) - ${NODE_COUNT} nodes, ${CURRENT_GPUS} GPUs"
      WORKER_JOINED=true
      break
    fi

    # Show spinner
    SPINNER="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    SPIN_CHAR="${SPINNER:$((i % 10)):1}"
    printf "\r  %s Waiting for worker... [%ds elapsed, %s nodes]  " "${SPIN_CHAR}" "${i}" "${NODE_COUNT}"

    sleep 1
  done

  if [ "${WORKER_JOINED}" != "true" ]; then
    echo ""
    log "  ⚠️  Worker did not join cluster within 120s"
    log "     Check worker logs: ssh ${WORKER_USER}@${WORKER_HOST} 'cat ~/worker_setup.log'"
    log ""
    log "  Proceeding anyway - vLLM may fail if insufficient GPUs"
  fi

else
  # Manual mode: Wait for user to start workers
  log "Step 8/${TOTAL_STEPS}: Waiting for worker nodes"
  log ""
  log "  ⚠️  IMPORTANT: Set WORKER_HOST to start workers automatically:"
  log "     export WORKER_HOST=<worker_ethernet_ip>  # For SSH"
  log "     export WORKER_IB_IP=<worker_ib_ip>       # For NCCL"
  log "     bash start_cluster.sh"
  log ""
  log "  Checking Ray cluster status..."

  # Show current cluster status
  docker exec "${NAME}" bash -lc "ray status --address=127.0.0.1:${RAY_PORT} 2>/dev/null | head -15" || true

  CURRENT_NODES=$(docker exec "${NAME}" bash -lc "ray status --address=127.0.0.1:${RAY_PORT} 2>/dev/null | grep -E '^ [0-9]+ node' | awk '{print \$1}'" 2>/dev/null || echo "1")
  CURRENT_GPUS=$(docker exec "${NAME}" bash -lc "ray status --address=127.0.0.1:${RAY_PORT} 2>/dev/null | grep 'GPU:' | awk -F'/' '{print \$2}' | awk '{print \$1}'" 2>/dev/null || echo "1")

  log ""
  log "  Current cluster: ${CURRENT_NODES} node(s), ${CURRENT_GPUS} GPU(s)"

  if [ "${TENSOR_PARALLEL}" -gt "${CURRENT_GPUS:-1}" ]; then
    log ""
    log "  ⚠️  Warning: tensor-parallel-size (${TENSOR_PARALLEL}) > available GPUs (${CURRENT_GPUS})"
    log "     Waiting 30 seconds for worker nodes to join..."
    log "     (Press Ctrl+C to abort and add workers manually)"

    for i in {1..30}; do
      CURRENT_GPUS=$(docker exec "${NAME}" bash -lc "ray status --address=127.0.0.1:${RAY_PORT} 2>/dev/null | grep 'GPU:' | awk -F'/' '{print \$2}' | awk '{print \$1}'" 2>/dev/null || echo "1")
      if [ "${CURRENT_GPUS:-1}" -ge "${TENSOR_PARALLEL}" ]; then
        log "  ✅ Sufficient GPUs available: ${CURRENT_GPUS}"
        break
      fi
      if [ $i -eq 30 ]; then
        log "  ⚠️  Proceeding with ${CURRENT_GPUS} GPU(s) - vLLM may fail if insufficient"
      fi
      sleep 1
    done
  fi
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Step number depends on whether worker orchestration was used
if [ -n "${WORKER_HOST}" ]; then
  VLLM_STEP=11
else
  VLLM_STEP=8
fi
log "Step ${VLLM_STEP}/${TOTAL_STEPS}: Starting vLLM server"
log ""

# Kill any existing vLLM processes
docker exec "${NAME}" bash -lc "pkill -f 'vllm serve' 2>/dev/null || true" || true

log "  Starting vLLM in background (this launches the server process)..."

# Build vLLM command arguments
VLLM_ARGS="--distributed-executor-backend ray"
VLLM_ARGS="${VLLM_ARGS} --host 0.0.0.0"
VLLM_ARGS="${VLLM_ARGS} --port ${VLLM_PORT}"
VLLM_ARGS="${VLLM_ARGS} --tensor-parallel-size ${TENSOR_PARALLEL}"
VLLM_ARGS="${VLLM_ARGS} --max-model-len ${MAX_MODEL_LEN}"
VLLM_ARGS="${VLLM_ARGS} --gpu-memory-utilization ${GPU_MEMORY_UTIL}"
VLLM_ARGS="${VLLM_ARGS} --swap-space ${SWAP_SPACE}"
VLLM_ARGS="${VLLM_ARGS} --download-dir /root/.cache/huggingface"
VLLM_ARGS="${VLLM_ARGS} --load-format ${LOAD_FORMAT}"

# Add optional flags
if [ "${ENABLE_EXPERT_PARALLEL}" = "true" ]; then
  VLLM_ARGS="${VLLM_ARGS} --enable-expert-parallel"
fi
if [ "${TRUST_REMOTE_CODE}" = "true" ]; then
  VLLM_ARGS="${VLLM_ARGS} --trust-remote-code"
fi
if [ -n "${EXTRA_ARGS}" ]; then
  VLLM_ARGS="${VLLM_ARGS} ${EXTRA_ARGS}"
fi

# Start vLLM in background using nohup
# Note: HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are set in the container env only
# when SKIP_MODEL_DOWNLOAD=1 (see ENV_ARGS above). In online mode we leave them
# unset so the model name can be resolved against huggingface.co.
docker exec "${NAME}" bash -lc "
  export HF_HOME=/root/.cache/huggingface
  export RAY_ADDRESS=127.0.0.1:${RAY_PORT}
  export PYTHONUNBUFFERED=1
  export VLLM_LOGGING_LEVEL=INFO
  export VLLM_MXFP4_USE_MARLIN=1

  nohup vllm serve ${MODEL} ${VLLM_ARGS} > /var/log/vllm.log 2>&1 &

  sleep 1
" || true

log "  vLLM server process started"
log ""
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "  🔄 MODEL LOADING IN PROGRESS"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log ""
log "  This process typically takes 5-20 minutes depending on model size."
log "  The server will:"
log "    1. Load model weights into GPU memory"
log "    2. Initialize tensor parallelism across GPUs"
log "    3. Compile CUDA graphs for optimized inference"
log ""
log "  Progress updates will appear below..."
log ""

# Wait for vLLM to become ready with detailed progress feedback
VLLM_READY=false
MAX_WAIT=1800  # 30 minutes max for very large models (70B+ can take 15-20+ min)
LAST_STATUS=""
START_TIME=$(date +%s)

for i in $(seq 1 $MAX_WAIT); do
  CURRENT_TIME=$(date +%s)
  ELAPSED=$((CURRENT_TIME - START_TIME))
  MINS=$((ELAPSED / 60))
  SECS=$((ELAPSED % 60))

  # Check if vLLM is ready
  if docker exec "${NAME}" bash -lc "curl -sf http://127.0.0.1:${VLLM_PORT}/health >/dev/null 2>&1"; then
    echo ""
    log ""
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "  ✅ MODEL LOADED SUCCESSFULLY!"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log ""
    log "  vLLM is ready and accepting requests (loaded in ${MINS}m ${SECS}s)"
    log ""
    VLLM_READY=true
    break
  fi

  # Check vLLM process status and extract progress from logs
  VLLM_PID=$(docker exec "${NAME}" bash -lc "pgrep -f 'vllm serve' 2>/dev/null" || echo "")

  if [ -z "${VLLM_PID}" ]; then
    # vLLM process died - check logs for error
    echo ""
    log ""
    log "  ❌ vLLM process exited unexpectedly!"
    log ""
    log "  Last 20 lines of vLLM log:"
    docker exec "${NAME}" tail -20 /var/log/vllm.log 2>/dev/null || true
    log ""
    error "vLLM failed to start. Check logs: docker exec ${NAME} cat /var/log/vllm.log"
  fi

  # Parse last meaningful log line to show progress
  CURRENT_STATUS=$(docker exec "${NAME}" bash -lc "tail -50 /var/log/vllm.log 2>/dev/null | grep -E '(Loading|Loaded|weight|layer|graph|CUDA|tensor|parallel|shard|download|progress|INFO)' | tail -1 | sed 's/.*INFO/INFO/' | cut -c1-80" 2>/dev/null || echo "")

  # Show spinner with elapsed time
  SPINNER="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
  SPIN_CHAR="${SPINNER:$((i % 10)):1}"

  # Update progress display
  printf "\r  %s Loading model... [%dm %02ds elapsed]  " "${SPIN_CHAR}" "${MINS}" "${SECS}"

  # Show status updates when they change (avoid flooding terminal)
  if [ -n "${CURRENT_STATUS}" ] && [ "${CURRENT_STATUS}" != "${LAST_STATUS}" ]; then
    echo ""
    log "     ${CURRENT_STATUS}"
    LAST_STATUS="${CURRENT_STATUS}"
  fi

  # Periodic milestone messages
  if [ $((i % 60)) -eq 0 ]; then
    echo ""
    log "  ⏳ Still loading... (${MINS}m ${SECS}s) - this is normal for large models"
  fi

  sleep 1
done

# Handle timeout
if [ "${VLLM_READY}" != "true" ]; then
  echo ""
  log ""
  log "  ⚠️  vLLM not ready after $((MAX_WAIT / 60)) minutes"
  log ""
  log "  This could mean:"
  log "    - Model is still loading (very large models may need more time)"
  log "    - An error occurred during loading"
  log ""
  log "  Check the logs for details:"
  log "    docker exec ${NAME} tail -100 /var/log/vllm.log"
  log ""
  log "  You can also monitor GPU memory to see if loading is progressing:"
  log "    watch -n 1 nvidia-smi"
  log ""
fi

log ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Final step is always TOTAL_STEPS
log "Step ${TOTAL_STEPS}/${TOTAL_STEPS}: Running health checks"

# Check Ray status
RAY_NODES=$(docker exec "${NAME}" bash -lc "ray status --address=127.0.0.1:${RAY_PORT} 2>/dev/null | grep 'Healthy:' -A1 | tail -1 | awk '{print \$1}'" || echo "0")
log "  Ray cluster: ${RAY_NODES} node(s) healthy"

# Check vLLM models
VLLM_MODEL=$(docker exec "${NAME}" bash -lc "curl -sf http://127.0.0.1:${VLLM_PORT}/v1/models 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin)[\"data\"][0][\"id\"])' 2>/dev/null" || echo "unknown")
log "  vLLM model: ${VLLM_MODEL}"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Detect public-facing IP for user access (exclude loopback and docker bridge)
PUBLIC_IP=$(ip -o addr show | grep "inet " | grep -v "127.0.0.1" | grep -v "172.17" | awk '{print $4}' | cut -d'/' -f1 | head -1)
if [ -z "${PUBLIC_IP}" ]; then
  PUBLIC_IP="${HEAD_IP}"  # Fallback to IB IP if no other found
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -n "${WORKER_HOST}" ]; then
  echo "✅ Cluster is ready! (Head + Worker orchestrated)"
elif [ "${SINGLE_NODE_MODE}" = "true" ]; then
  echo "✅ Single-node vLLM server is ready!"
else
  echo "✅ Head node is ready!"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🌐 Services (accessible from network):"
echo "  Ray Dashboard:  http://${PUBLIC_IP}:${RAY_DASHBOARD_PORT}"
echo "  vLLM API:       http://${PUBLIC_IP}:${VLLM_PORT}"
echo ""

if [ -n "${WORKER_HOST}" ]; then
  echo "🖥️  Cluster Nodes:"
  echo "  Head:   $(hostname) (${HEAD_IP})"
  echo "  Worker: ${WORKER_HOST}"
  echo ""
  echo "🔧 Worker Logs:"
  echo "  ssh ${WORKER_USER}@${WORKER_HOST} 'cat ~/worker_setup.log'"
  echo ""
elif [ "${SINGLE_NODE_MODE}" = "true" ]; then
  echo "🖥️  Single-Node Configuration:"
  echo "  Host: $(hostname)"
  echo "  GPUs: ${TENSOR_PARALLEL} (tensor parallelism)"
  echo ""
  echo "🔗 To Add Worker Nodes Later:"
  echo "    export WORKER_HOST=<worker_ethernet_ip>  # For SSH"
  echo "    export WORKER_IB_IP=<worker_ib_ip>       # For NCCL"
  echo "    bash start_cluster.sh"
  echo ""
else
  echo "🔗 Next Steps - Start Workers Automatically:"
  echo "  Set worker IPs and re-run start_cluster.sh:"
  echo "    export WORKER_HOST=<worker_ethernet_ip>  # For SSH"
  echo "    export WORKER_IB_IP=<worker_ib_ip>       # For NCCL"
  echo "    bash start_cluster.sh"
  echo ""
  echo "  Workers will be started automatically via SSH."
  echo "  Head IP for worker communication: ${HEAD_IP}"
  echo ""
fi

echo "📊 Quick API Tests:"
echo "  # List models"
echo "  curl http://${PUBLIC_IP}:${VLLM_PORT}/v1/models"
echo ""
echo "  # Chat completion"
echo "  curl http://${PUBLIC_IP}:${VLLM_PORT}/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'"
echo ""
echo "🔍 Monitoring Commands:"
echo "  # View vLLM logs"
echo "  docker exec ${NAME} tail -f /var/log/vllm.log"
echo ""
echo "  # Ray cluster status (check for worker nodes)"
echo "  docker exec ${NAME} ray status --address=127.0.0.1:${RAY_PORT}"
echo ""
echo "  # GPU utilization"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "⚙️  Current Configuration:"
echo "  Model:              ${MODEL}"
echo "  Tensor Parallelism: ${TENSOR_PARALLEL} GPUs"
echo "  Max Context:        ${MAX_MODEL_LEN} tokens"
echo "  GPU Memory:         ${GPU_MEMORY_UTIL} utilization"
echo "  CUDA Graphs:        Enabled (optimized for performance)"
echo ""
echo "📊 Expected Performance:"
echo "  Throughput:         50-100 tokens/second"
echo "  First request:      May be slower (graph warmup)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
