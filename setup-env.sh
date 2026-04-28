#!/usr/bin/env bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# vLLM DGX Spark Environment Configuration Script (Simplified)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# With auto-detection enabled in the scripts, you only need to set:
# - HF_TOKEN (for gated models like Llama, Gemma)
# - WORKER_HOST (space-separated list of worker SSH IPs; supports 1-N workers)
# - WORKER_IB_IP (space-separated list of worker NCCL IPs; optional, 1:1 with WORKER_HOST)
#
# For manual worker setup (not orchestrated from head):
# - HEAD_IP (head node's InfiniBand IP)
#
# Usage:
#   source ./setup-env.sh             # Interactive mode (sets env vars)
#   source ./setup-env.sh --head      # Head node mode
#   source ./setup-env.sh --worker    # Worker node mode (manual setup only)
#   ./setup-env.sh --discover         # Multi-Spark mDNS discovery + SSH key push
#                                     #   (wraps NVIDIA's discover-sparks script;
#                                     #    no need to source - it's a one-shot)
#
# NOTE: For interactive/head/worker modes, source this script (not execute) so
# the environment variables propagate to your shell. The --discover mode is a
# one-shot helper and can be run normally.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Check if script is being sourced. The --discover mode is a one-shot helper
# (just runs NVIDIA's discover-sparks script and exits) so it's fine to execute
# directly; the other modes need to be sourced so env vars propagate.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]] && [[ "$1" != "--discover" ]]; then
    echo "❌ Error: This script must be sourced, not executed"
    echo "   Usage:"
    echo "     source ./setup-env.sh                   # interactive / head / worker"
    echo "     ./setup-env.sh --discover               # mDNS discovery (one-shot)"
    exit 1
fi

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Helper function to prompt for input
prompt_input() {
    local var_name="$1"
    local prompt_text="$2"
    local default_value="$3"
    local is_secret="${4:-false}"
    local current_value="${!var_name:-}"

    # If variable is already set, use it
    if [ -n "$current_value" ]; then
        if [ "$is_secret" = true ]; then
            echo -e "${GREEN}✓${NC} $var_name already set (hidden)"
        else
            echo -e "${GREEN}✓${NC} $var_name=$current_value"
        fi
        return
    fi

    # Show prompt
    if [ -n "$default_value" ]; then
        echo -ne "${BLUE}?${NC} $prompt_text [${default_value}]: "
    else
        echo -ne "${YELLOW}!${NC} $prompt_text: "
    fi

    # Read input (with or without echo for secrets)
    if [ "$is_secret" = true ]; then
        read -s user_input
        echo ""  # New line after secret input
    else
        read user_input
    fi

    # Use default if no input provided
    if [ -z "$user_input" ] && [ -n "$default_value" ]; then
        user_input="$default_value"
    fi

    # Export the variable
    if [ -n "$user_input" ]; then
        export "$var_name=$user_input"
        if [ "$is_secret" = true ]; then
            echo -e "${GREEN}✓${NC} $var_name set (hidden)"
        else
            echo -e "${GREEN}✓${NC} $var_name=$user_input"
        fi
    else
        if [ -n "$default_value" ]; then
            echo -e "${YELLOW}⊘${NC} $var_name not set (will use default: $default_value)"
        else
            echo -e "${YELLOW}⊘${NC} $var_name not set (optional)"
        fi
    fi
}

# Detect node type from arguments
NODE_TYPE="interactive"
if [[ "$1" == "--head" ]]; then
    NODE_TYPE="head"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}HEAD NODE CONFIGURATION${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
elif [[ "$1" == "--worker" ]]; then
    NODE_TYPE="worker"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}WORKER NODE CONFIGURATION${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
elif [[ "$1" == "--discover" ]]; then
    # Multi-Spark mDNS discovery + bidirectional SSH key push.
    # Wraps NVIDIA's discover-sparks script (from the official Spark playbooks
    # repo). It scans the local network for dgx-spark-*.local hostnames,
    # prompts once for each Spark's password, and sets up bidirectional SSH.
    DISCOVER_URL="https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/connect-two-sparks/assets/discover-sparks"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Multi-Spark Discovery (NVIDIA discover-sparks wrapper)${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "This will:"
    echo "  1. Download NVIDIA's discover-sparks script to a temp file"
    echo "  2. Run it - the script will prompt you for each Spark's password"
    echo "  3. After completion, you'll have bidirectional passwordless SSH set up"
    echo ""
    echo "Source: ${DISCOVER_URL}"
    echo ""
    echo "Prerequisites: every Spark must already have:"
    echo "  - The same username configured (see README section 3a)"
    echo "  - CX7 IPs assigned via netplan (see README section 3c)"
    echo "  - Avahi/mDNS running (default on the DGX Spark base image)"
    echo ""
    read -p "Proceed? [y/N]: " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        return 0 2>/dev/null || exit 0
    fi
    DISCOVER_TMP="$(mktemp -t discover-sparks.XXXXXX)"
    if ! curl -fsSL "${DISCOVER_URL}" -o "${DISCOVER_TMP}"; then
        echo -e "${RED}✗ Failed to download discover-sparks from ${DISCOVER_URL}${NC}"
        echo "  Falling back to manual instructions:"
        echo "    ssh-copy-id -i ~/.ssh/id_ed25519.pub <user>@<worker-IP>   # repeat per worker"
        rm -f "${DISCOVER_TMP}"
        return 1 2>/dev/null || exit 1
    fi
    echo ""
    echo "Running discover-sparks..."
    echo ""
    bash "${DISCOVER_TMP}"
    DISCOVER_RC=$?
    rm -f "${DISCOVER_TMP}"
    echo ""
    if [ "${DISCOVER_RC}" -eq 0 ]; then
        echo -e "${GREEN}✓ Discovery complete.${NC}"
        echo ""
        echo "Next: re-run setup-env.sh interactively (no --discover flag) to capture"
        echo "the discovered IPs into WORKER_HOST and WORKER_IB_IP. For 3+ Sparks,"
        echo "those vars take space-separated lists, e.g.:"
        echo "    WORKER_HOST=\"192.168.7.111 192.168.7.112 192.168.7.113\""
        echo "    WORKER_IB_IP=\"169.254.216.8 169.254.216.9 169.254.216.10\""
    else
        echo -e "${RED}✗ discover-sparks exited with code ${DISCOVER_RC}.${NC}"
        echo "  See README section 3e for manual ssh-copy-id instructions."
    fi
    return 0 2>/dev/null || exit 0
else
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}vLLM DGX Spark - Environment Setup${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
fi

echo ""
echo "ℹ️  Note: Network configuration (IPs, interfaces, HCAs) is now auto-detected!"
echo "   You only need to provide the essential configuration below."
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fix HuggingFace Cache Permissions (Required)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HF_CACHE="${HF_CACHE:-/raid/hf-cache}"
echo -e "${YELLOW}Checking HuggingFace cache permissions...${NC}"

# Function to check and fix HF cache permissions (checks for root-owned files inside)
check_fix_hf_cache() {
    local cache_dir="$1"
    local location="$2"  # "local" or remote host description

    if [ ! -d "$cache_dir" ]; then
        echo -e "${BLUE}ℹ${NC}  HF cache directory will be created at $cache_dir ($location)"
        return 0
    fi

    # Check for root-owned files/directories (Docker creates these)
    local root_owned=$(find "$cache_dir" -user root 2>/dev/null | head -5)

    if [ -n "$root_owned" ]; then
        echo -e "${RED}⚠${NC}  Found root-owned files in $cache_dir ($location)"
        echo "   Docker containers run as root and create files owned by root."
        echo "   This will cause rsync permission errors during model sync."
        echo ""
        echo -e "${YELLOW}To fix permissions, run:${NC}"
        echo "   sudo chown -R \$USER $cache_dir"
        echo ""
        read -p "Would you like to fix permissions now? (requires sudo) [y/N]: " fix_perms
        if [[ "$fix_perms" =~ ^[Yy]$ ]]; then
            echo "Running: sudo chown -R $USER $cache_dir"
            if sudo chown -R "$USER" "$cache_dir"; then
                echo -e "${GREEN}✓${NC} Permissions fixed successfully ($location)"
            else
                echo -e "${RED}✗${NC} Failed to fix permissions. Please run manually:"
                echo "   sudo chown -R \$USER $cache_dir"
                return 1
            fi
        fi
    elif [ ! -w "$cache_dir" ]; then
        echo -e "${RED}⚠${NC}  HF cache at $cache_dir is not writable ($location)"
        return 1
    else
        echo -e "${GREEN}✓${NC} HF cache permissions OK ($cache_dir) ($location)"
    fi
    return 0
}

# Check local HF cache
check_fix_hf_cache "$HF_CACHE" "local"
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Head Node Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [[ "$NODE_TYPE" == "head" ]] || [[ "$NODE_TYPE" == "interactive" ]]; then
    echo -e "${GREEN}Head Node Settings:${NC}"
    echo ""

    # HuggingFace Token (required for gated models)
    echo "HuggingFace Token (required for gated models like Llama):"
    echo "  Get yours at: https://huggingface.co/settings/tokens"
    prompt_input "HF_TOKEN" "Enter your HuggingFace token" "" true
    echo ""

    # Worker node IPs (required for orchestrated multi-Spark setup).
    # WORKER_HOST is space-separated; supports 1-N workers.
    echo "Worker IP(s) for SSH/rsync (multi-Spark only - skip for single-Spark):"
    echo "  1 worker:  192.168.7.111"
    echo "  3 workers: 192.168.7.111 192.168.7.112 192.168.7.113"
    echo "  (For first-time setup across N Sparks, run './setup-env.sh --discover'"
    echo "   to auto-discover via mDNS and push SSH keys.)"
    prompt_input "WORKER_HOST" "Enter worker IP(s), space-separated" ""
    echo ""

    # Worker InfiniBand IPs (used by NCCL; 1:1 positional with WORKER_HOST).
    # Empty is fine - cluster will use the SSH IPs for NCCL too.
    if [ -n "${WORKER_HOST}" ]; then
        echo "Worker InfiniBand IP(s) for NCCL (optional, 1:1 with WORKER_HOST):"
        echo "  1 worker:  169.254.216.8"
        echo "  3 workers: 169.254.216.8 169.254.216.9 169.254.216.10"
        echo "  Leave blank to use the SSH IPs above for NCCL too."
        prompt_input "WORKER_IB_IP" "Enter IB IP(s), space-separated (or blank)" ""
        echo ""
    fi

    # Worker node username (if different from current user)
    echo "Worker Node Username (must be the same on every Spark):"
    echo "  Leave blank to use current username: $(whoami)"
    prompt_input "WORKER_USER" "Enter worker node username" "$(whoami)"
    echo ""

    # Compute a sensible TENSOR_PARALLEL default from WORKER_HOST cardinality.
    read -ra _WHOSTS <<< "${WORKER_HOST:-}"
    DEFAULT_TP=$((${#_WHOSTS[@]} + 1))

    # Optional model configuration
    echo -e "${BLUE}Optional Configuration (press Enter to use defaults):${NC}"
    prompt_input "MODEL" "Model to serve" "openai/gpt-oss-120b"
    prompt_input "TENSOR_PARALLEL" "Number of GPUs (tensor parallel size)" "${DEFAULT_TP}"
    prompt_input "MAX_MODEL_LEN" "Maximum context length (tokens)" "8192"
    prompt_input "GPU_MEMORY_UTIL" "GPU memory utilization (0.0-1.0)" "0.90"
    echo ""

    # Check worker node(s) HF cache permissions via SSH (loops over array form).
    if [ -n "$WORKER_HOST" ]; then
        echo -e "${YELLOW}Checking worker node HF cache permissions...${NC}"
        WORKER_USER="${WORKER_USER:-$(whoami)}"
        for w in "${_WHOSTS[@]}"; do
            if ssh -o ConnectTimeout=5 -o BatchMode=yes "${WORKER_USER}@${w}" "exit" 2>/dev/null; then
                root_owned=$(ssh "${WORKER_USER}@${w}" "find $HF_CACHE -user root 2>/dev/null | head -5" 2>/dev/null)
                if [ -n "$root_owned" ]; then
                    echo -e "${RED}⚠${NC}  ${w}: found root-owned files in $HF_CACHE"
                    read -p "       Fix permissions on ${w}? (requires sudo on worker) [y/N]: " fix_worker_perms
                    if [[ "$fix_worker_perms" =~ ^[Yy]$ ]]; then
                        if ssh -t "${WORKER_USER}@${w}" "sudo chown -R $WORKER_USER $HF_CACHE" 2>/dev/null; then
                            echo -e "       ${GREEN}✓${NC} Permissions fixed on ${w}"
                        else
                            echo -e "       ${RED}✗${NC} Failed to fix permissions on ${w} - run manually:"
                            echo "         ssh ${WORKER_USER}@${w} 'sudo chown -R \$USER $HF_CACHE'"
                        fi
                    fi
                else
                    echo -e "  ${GREEN}✓${NC} ${w}: HF cache permissions OK ($HF_CACHE)"
                fi
            else
                echo -e "  ${YELLOW}⚠${NC}  ${w}: cannot SSH (skipping permission check)"
                echo "       Run: ./setup-env.sh --discover  # to push SSH keys via mDNS"
            fi
        done
        echo ""
    fi
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Worker Node Configuration (only needed for manual worker setup)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [[ "$NODE_TYPE" == "worker" ]]; then
    echo -e "${BLUE}Worker Node Settings:${NC}"
    echo ""
    echo "Note: If running the worker from the head node (orchestrated setup),"
    echo "      HEAD_IP is automatically passed via SSH. Only set this if"
    echo "      you're starting the worker manually."
    echo ""

    # HEAD_IP (required for manual worker setup)
    echo "Head Node InfiniBand IP (required for manual worker setup):"
    echo "  Run 'ibdev2netdev' on the head node to find its IB interface,"
    echo "  then 'ip addr show <interface>' to get the IP"
    echo "  Example: 169.254.103.56"
    prompt_input "HEAD_IP" "Enter head node InfiniBand IP" ""
    echo ""
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Summary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Configuration Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Variables set in your current shell session:"
echo ""

if [[ "$NODE_TYPE" == "head" ]] || [[ "$NODE_TYPE" == "interactive" ]]; then
    [ -n "${HF_TOKEN:-}" ] && echo "  ✓ HF_TOKEN (hidden)"
    [ -n "${WORKER_HOST:-}" ] && echo "  ✓ WORKER_HOST=$WORKER_HOST"
    [ -n "${WORKER_USER:-}" ] && echo "  ✓ WORKER_USER=$WORKER_USER"
    [ -n "${MODEL:-}" ] && echo "  ✓ MODEL=$MODEL"
    [ -n "${TENSOR_PARALLEL:-}" ] && echo "  ✓ TENSOR_PARALLEL=$TENSOR_PARALLEL"
    [ -n "${MAX_MODEL_LEN:-}" ] && echo "  ✓ MAX_MODEL_LEN=$MAX_MODEL_LEN"
    [ -n "${GPU_MEMORY_UTIL:-}" ] && echo "  ✓ GPU_MEMORY_UTIL=$GPU_MEMORY_UTIL"
fi

if [[ "$NODE_TYPE" == "worker" ]]; then
    [ -n "${HEAD_IP:-}" ] && echo "  ✓ HEAD_IP=$HEAD_IP"
fi

echo ""
echo "Auto-detected by scripts (no configuration needed):"
echo "  ✓ HEAD_IP - detected from InfiniBand on head node"
echo "  ✓ WORKER_IP - detected from InfiniBand on worker node"
echo "  ✓ Network interfaces (GLOO_IF, TP_IF, NCCL_IF, UCX_DEV)"
echo "  ✓ InfiniBand HCAs (NCCL_IB_HCA)"
echo ""
echo "Next steps:"
echo "  Run on head node: bash start_cluster.sh"
echo ""
echo "  Note: Workers are started automatically via SSH from the head node."
echo "  Make sure WORKER_HOST is set (or provided via setup-env.sh)."
echo ""
