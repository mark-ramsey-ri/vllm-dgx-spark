#!/usr/bin/env bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# vLLM DGX Spark Environment Configuration Script (Simplified)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# With auto-detection enabled in the scripts, you only need to set:
# - HF_TOKEN (for gated models like Llama)
# - WORKER_HOST (worker's Ethernet IP for SSH/rsync from head node)
#
# For manual worker setup (not orchestrated from head):
# - HEAD_IP (head node's InfiniBand IP)
#
# Usage:
#   source ./setup-env.sh           # Interactive mode
#   source ./setup-env.sh --head    # Head node mode
#   source ./setup-env.sh --worker  # Worker node mode (manual setup only)
#
# NOTE: This script must be sourced (not executed) to set environment variables
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Check if script is being sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "❌ Error: This script must be sourced, not executed"
    echo "   Usage: source ./setup-env.sh"
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

    # Worker node Ethernet IP (required for orchestrated setup)
    echo "Worker Node IP (required for multi-node setup):"
    echo "  This is the worker's standard Ethernet IP for SSH/rsync"
    echo "  Example: 192.168.7.111"
    prompt_input "WORKER_HOST" "Enter worker node Ethernet IP" ""
    echo ""

    # Worker node username (if different from current user)
    echo "Worker Node Username (if different from current user):"
    echo "  Leave blank to use current username: $(whoami)"
    prompt_input "WORKER_USER" "Enter worker node username" "$(whoami)"
    echo ""

    # Optional model configuration
    echo -e "${BLUE}Optional Configuration (press Enter to use defaults):${NC}"
    prompt_input "MODEL" "Model to serve" "openai/gpt-oss-120b"
    prompt_input "TENSOR_PARALLEL" "Number of GPUs (tensor parallel size)" "2"
    prompt_input "MAX_MODEL_LEN" "Maximum context length (tokens)" "8192"
    prompt_input "GPU_MEMORY_UTIL" "GPU memory utilization (0.0-1.0)" "0.90"
    echo ""

    # Check worker node HF cache permissions via SSH
    if [ -n "$WORKER_HOST" ]; then
        echo -e "${YELLOW}Checking worker node HF cache permissions...${NC}"
        WORKER_USER="${WORKER_USER:-$(whoami)}"

        # Check SSH connectivity first
        if ssh -o ConnectTimeout=5 -o BatchMode=yes "${WORKER_USER}@${WORKER_HOST}" "exit" 2>/dev/null; then
            # Check for root-owned files on worker
            root_owned=$(ssh "${WORKER_USER}@${WORKER_HOST}" "find $HF_CACHE -user root 2>/dev/null | head -5" 2>/dev/null)

            if [ -n "$root_owned" ]; then
                echo -e "${RED}⚠${NC}  Found root-owned files in $HF_CACHE on worker ($WORKER_HOST)"
                echo "   Docker containers run as root and create files owned by root."
                echo "   This will cause rsync permission errors during model sync."
                echo ""
                echo -e "${YELLOW}To fix, run this on the worker node:${NC}"
                echo "   sudo chown -R \$USER $HF_CACHE"
                echo ""
                read -p "Would you like to fix worker permissions now? (requires sudo on worker) [y/N]: " fix_worker_perms
                if [[ "$fix_worker_perms" =~ ^[Yy]$ ]]; then
                    echo "Running on worker: sudo chown -R $WORKER_USER $HF_CACHE"
                    if ssh -t "${WORKER_USER}@${WORKER_HOST}" "sudo chown -R $WORKER_USER $HF_CACHE" 2>/dev/null; then
                        echo -e "${GREEN}✓${NC} Worker permissions fixed successfully"
                    else
                        echo -e "${RED}✗${NC} Failed to fix worker permissions. Please run manually on worker:"
                        echo "   sudo chown -R \$USER $HF_CACHE"
                    fi
                fi
            else
                echo -e "${GREEN}✓${NC} Worker HF cache permissions OK ($HF_CACHE on $WORKER_HOST)"
            fi
        else
            echo -e "${YELLOW}⚠${NC}  Cannot SSH to worker ($WORKER_HOST) - skipping permission check"
            echo "   Make sure SSH keys are set up for passwordless access"
        fi
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
