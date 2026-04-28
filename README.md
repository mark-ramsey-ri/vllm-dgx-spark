# vLLM on DGX Spark Cluster

Deploy [vLLM](https://github.com/vllm-project/vllm) on NVIDIA DGX Spark systems - supports both single-node and dual-node cluster configurations with InfiniBand RDMA for serving large language models.

> **DISCLAIMER**: This project is NOT affiliated with, endorsed by, or officially supported by NVIDIA, vLLM, or any other organization. This is a community-driven effort to run vLLM on DGX Spark hardware. Use at your own risk. The software is provided "AS IS", without warranty of any kind.

> **Upgraded (2026-04-28)**: Components bumped to `nvcr.io/nvidia/vllm:26.04-py3` (vLLM 0.19.0, PyTorch 2.12.0a0, CUDA 13.2.1) with Ray 2.55.1. The 26.04 container no longer ships Ray, so `start_cluster.sh` and `start_worker_vllm.sh` now `pip install ray[default]==${RAY_VERSION}` into the head and worker containers at startup (~30s). Smoke tested on a single Spark; multi-node TP=2 verification pending.
>
> Previous (2026-04-09): `nvcr.io/nvidia/vllm:26.03-py3` (vLLM 0.17.1) with Ray 2.54.0.

## Features

- **Single-node and multi-node support** - Run on one DGX Spark or scale to two
- **Zero-config single-node** - No InfiniBand setup required for single-node deployments
- **Single-command deployment** - Start entire cluster from head node via SSH
- **Auto-detection** of InfiniBand IPs, network interfaces, and HCA devices (multi-node)
- **Generic scripts** that work on any DGX Spark configuration
- **13 model presets** including Llama, Qwen, Mixtral, Gemma
- **InfiniBand RDMA** for high-speed inter-node communication (200Gb/s)
- **Comprehensive benchmarking** with multiple test profiles

## Cluster Architecture

### Single-Node Mode (1x DGX Spark)

```
┌─────────────────────────────────────────────────────────────────┐
│                    DGX Spark Single Node                        │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                      SINGLE NODE                          │  │
│  │                                                           │  │
│  │  GPU: 1x GB10 (Blackwell, sm100) ~120GB VRAM             │  │
│  │  /raid/hf-cache                                          │  │
│  │  Port: 8000 (API)                                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Tensor Parallel (TP=1): Full model on single GPU              │
│  Best for: Models up to ~100GB (GPT-OSS 120B MXFP4, Llama 70B) │
└─────────────────────────────────────────────────────────────────┘
```

### Dual-Node Mode (2x DGX Spark)

```
┌─────────────────────────────────────────────────────────────────┐
│                    DGX Spark 2-Node Cluster                     │
│                                                                 │
│  ┌──────────────────────┐      ┌──────────────────────┐        │
│  │     HEAD NODE        │      │    WORKER NODE       │        │
│  │    (Ray head)        │ SSH  │    (Ray worker)      │        │
│  │                      │─────►│                      │        │
│  │  GPU: 1x GB10        │◄────►│  GPU: 1x GB10        │        │
│  │  (Blackwell, sm100)  │ IB   │  (Blackwell, sm100)  │        │
│  │                      │200Gb │                      │        │
│  │  /raid/hf-cache      │      │  /raid/hf-cache      │        │
│  │  Port: 8000 (API)    │      │                      │        │
│  └──────────────────────┘      └──────────────────────┘        │
│                                                                 │
│  Tensor Parallel (TP=2): Model split across both GPUs          │
│  Best for: Models that exceed a single Spark's ~120GB VRAM     │
└─────────────────────────────────────────────────────────────────┘
```

## Hardware Requirements

### Single-Node
- **Nodes:** 1x DGX Spark system
- **GPUs:** 1x NVIDIA GB10 (Grace Blackwell, sm100), ~120GB VRAM
- **Storage:** Model cache at `/raid/hf-cache` (or configure in `config.env`)

### Dual-Node (for larger models)
- **Nodes:** 2x DGX Spark systems
- **GPUs:** 1x NVIDIA GB10 per node, ~120GB VRAM each
- **Network:** 200Gb/s InfiniBand RoCE between nodes
- **Storage:** Model cache at `/raid/hf-cache` on both nodes
- **SSH:** Passwordless SSH from head to worker node

## Prerequisites

Complete these steps on your server(s) before running `start_cluster.sh`.

**Single-node setups only require steps 1, 2, and 5 (HuggingFace token for gated models).** InfiniBand and SSH configuration are automatically skipped when running in single-node mode.

### 1. NVIDIA GPU Drivers

Ensure NVIDIA drivers are installed and working:
```bash
nvidia-smi
```
You should see your GPU listed with driver version.

### 2. Docker with NVIDIA Container Runtime

Docker must be installed with NVIDIA Container Runtime configured:
```bash
# Verify Docker works with GPU access
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```
If this fails, install/configure the NVIDIA Container Toolkit.

### 3. InfiniBand Network Configuration (Dual-Node Only)

**Note:** Skip this section for single-node setups.

**CRITICAL:** InfiniBand (QSFP) interfaces must be configured and operational for multi-node performance.

```bash
# Install InfiniBand Tools
sudo apt install infiniband-diags

# Check InfiniBand status
ibstatus

# Find InfiniBand interfaces (typically enp1s0f1np1, enP2p1s0f1np1 on DGX Spark)
ip addr show | grep 169.254

# Verify both nodes can reach each other via InfiniBand
ping <infiniband-ip-of-other-node>
```

InfiniBand IPs are typically in the `169.254.x.x` range.

**Performance Warning:** Using standard Ethernet IPs instead of InfiniBand will result in **10-20x slower performance**.

Need help with InfiniBand setup? See NVIDIA's guide: https://build.nvidia.com/spark/nccl/stacked-sparks

### 4. Firewall Configuration

Ensure the following ports are open between both nodes:
- **6379** - Ray GCS
- **8265** - Ray Dashboard
- **8000** - vLLM API

### 5. Hugging Face Authentication (for gated models)

Some models (Llama, Gemma, etc.) require Hugging Face authorization:

```bash
# Install the Hugging Face CLI (run on both nodes)
pip install huggingface_hub

# Login to Hugging Face (run on both nodes)
hf auth login
# Enter your token when prompted

# Accept model licenses
# Visit the model page on huggingface.co and accept the license agreement
# Example: https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct
```

Alternatively, set `HF_TOKEN` in your `config.local.env`:
```bash
HF_TOKEN="hf_your_token_here"
```

## Quick Start

### 1. Clone and Setup

```bash
git clone <this-repo>
cd vllm-dgx-spark
```

### 2. Choose Your Configuration

#### Option A: Single-Node (Simplest)

For running on a single DGX Spark with one GPU:

```bash
# Set tensor parallelism to 1 (single GPU)
export TENSOR_PARALLEL=1

# Choose a model that fits in ~120GB VRAM
export MODEL="openai/gpt-oss-120b"  # ~65GB in native MXFP4, or any model up to ~100GB

# Start the server
./start_cluster.sh
```

That's it! **No InfiniBand, SSH setup, or worker configuration needed.** The script automatically detects single-node mode when `TENSOR_PARALLEL` is less than or equal to the number of local GPUs and no `WORKER_HOST` is configured. In single-node mode:

- InfiniBand detection and configuration is skipped
- NCCL uses NVLink/PCIe for GPU-to-GPU communication
- The setup is simpler and faster

#### Option B: Dual-Node Cluster (For Larger Models)

For running across two DGX Spark systems:

**Setup SSH (one-time):**
```bash
# On head node, generate key if needed:
ssh-keygen -t ed25519  # Press enter for defaults

# Copy to worker (replace with your worker's InfiniBand IP):
ssh-copy-id <username>@<worker-ib-ip>

# Test connection:
ssh <username>@<worker-ib-ip> "hostname"
```

**Configure Environment:**

```bash
# Option 1: Interactive setup (recommended)
source ./setup-env.sh

# Option 2: Edit config file
cp config.env config.local.env
vim config.local.env

# Set at minimum:
# WORKER_HOST="<worker-ethernet-ip>"   # For SSH
# WORKER_IB_IP="<worker-infiniband-ip>" # For NCCL
# WORKER_USER="<ssh-username>"
```

**Start the Cluster:**
```bash
./start_cluster.sh
```

This will:
1. Pull the Docker image on both nodes
2. Download the model (with rsync to worker)
3. SSH to worker and start Ray worker container
4. Start Ray head and vLLM server
5. Wait for the cluster to become ready (~2-5 minutes)

### 5. Verify the Cluster

```bash
# Check health
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Test inference
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"openai/gpt-oss-120b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":50}'
```

### 6. Run Benchmarks

```bash
# Single-request latency test
./benchmark_current.sh --single

# Quick benchmark (20 prompts)
./benchmark_current.sh --quick

# Full benchmark (100 prompts)
./benchmark_current.sh
```

### 7. Stop the Cluster

```bash
./stop_cluster.sh
```

## Scripts Overview

| Script | Description |
|--------|-------------|
| `setup-env.sh` | Interactive environment setup (source this!) |
| `config.env` | Configuration template |
| `start_cluster.sh` | **Main script** - starts head + workers via SSH |
| `stop_cluster.sh` | Stops containers on head + workers |
| `switch_model.sh` | Switch between different models |
| `benchmark_current.sh` | Benchmark current model |
| `benchmark_all.sh` | Benchmark all models and create comparison matrix |
| `checkout_setup.sh` | System diagnostics (InfiniBand, NCCL, GPU) |

## Configuration

Key settings in `config.env` or `config.local.env`:

```bash
# ┌─────────────────────────────────────────────────────────────────┐
# │ Multi-Node Settings (Optional - skip for single-node)          │
# └─────────────────────────────────────────────────────────────────┘
# If these are not set and TENSOR_PARALLEL <= local GPU count,
# the script runs in single-node mode (no InfiniBand required)
WORKER_HOST="<worker-ethernet-ip>" # Worker Ethernet IP for SSH (optional)
WORKER_IB_IP="<worker-ib-ip>"      # Worker InfiniBand IP for NCCL (optional)
WORKER_USER="<username>"           # SSH username for workers

# ┌─────────────────────────────────────────────────────────────────┐
# │ Model Settings                                                  │
# └─────────────────────────────────────────────────────────────────┘
MODEL="openai/gpt-oss-120b"        # Model to serve
TENSOR_PARALLEL="2"                # GPUs: 1 for single-node, 2 for dual-node
                                   # Single-node mode: when TP <= local GPUs and no WORKER_HOST
GPU_MEMORY_UTIL="0.90"             # GPU memory utilization for KV cache

# ┌─────────────────────────────────────────────────────────────────┐
# │ vLLM Options                                                    │
# └─────────────────────────────────────────────────────────────────┘
MAX_MODEL_LEN="8192"               # Max context length
SWAP_SPACE="16"                    # Swap space in GB
ENABLE_EXPERT_PARALLEL="true"      # For MoE models
TRUST_REMOTE_CODE="false"          # For custom model code

# ┌─────────────────────────────────────────────────────────────────┐
# │ Optional                                                        │
# └─────────────────────────────────────────────────────────────────┘
HF_TOKEN="hf_xxx"                  # For gated models (Llama, etc.)
VLLM_IMAGE="nvcr.io/nvidia/vllm:26.04-py3"  # Docker image
```

### Single-Node vs Multi-Node Mode Detection

The script automatically determines which mode to use:

| Condition | Mode | InfiniBand |
|-----------|------|------------|
| `WORKER_HOST` not set AND `TENSOR_PARALLEL` ≤ local GPUs | Single-node | Not required |
| `WORKER_HOST` set OR `TENSOR_PARALLEL` > local GPUs | Multi-node | Required |

In **single-node mode**:
- InfiniBand detection and configuration is skipped
- NCCL uses NVLink/PCIe for local GPU communication
- No SSH or worker setup needed
- The `/dev/infiniband` device is not mounted in the container

In **multi-node mode**:
- InfiniBand interfaces are auto-detected via `ibdev2netdev`
- HEAD_IP is auto-detected from the InfiniBand interface
- NCCL is configured for RDMA communication

### Finding Worker InfiniBand IP

On the **worker node**, run:
```bash
# Find InfiniBand interface name
ibdev2netdev

# Example output: mlx5_0 port 1 ==> enp1s0f1np1 (Up)

# Get IP address for that interface
ip addr show enp1s0f1np1 | grep "inet "

# Example output: inet 169.254.x.x/16 ...
```

## Switching Models

Use `switch_model.sh` to easily switch between models:

```bash
# List available models
./switch_model.sh --list

# Interactive selection
./switch_model.sh

# Direct selection (by number)
./switch_model.sh 3  # Switch to Qwen2.5-7B

# Update config only (don't restart)
./switch_model.sh -s 5

# Download model only
./switch_model.sh -d 1

# Download and sync to worker
./switch_model.sh -r 1
```

## Supported Models

Models can run on single-node (TP=1) or dual-node (TP=2) depending on size.

| # | Model | Size | Single-Node | Notes |
|---|-------|------|-------------|-------|
| 1 | `openai/gpt-oss-120b` | ~65GB | Yes | Default, MoE, native MXFP4 |
| 2 | `openai/gpt-oss-20b` | ~16-20GB | Yes | MoE, fast |
| 3 | `Qwen/Qwen2.5-7B-Instruct` | ~7GB | Yes | Very fast |
| 4 | `Qwen/Qwen2.5-14B-Instruct` | ~14GB | Yes | Fast |
| 5 | `Qwen/Qwen2.5-32B-Instruct` | ~30GB | Yes | Strong mid-size |
| 6 | `Qwen/Qwen2.5-72B-Instruct` | ~70GB | Yes | High quality |
| 7 | `mistralai/Mistral-7B-Instruct-v0.3` | ~7GB | Yes | Very fast |
| 8 | `mistralai/Mistral-Nemo-Instruct-2407` | ~12GB | Yes | 128k context |
| 9 | `mistralai/Mixtral-8x7B-Instruct-v0.1` | ~45GB | Yes | MoE, fast |
| 10 | `meta-llama/Llama-3.1-8B-Instruct` | ~8GB | Yes | Very fast (needs HF token) |
| 11 | `meta-llama/Llama-3.1-70B-Instruct` | ~65GB | Yes | High quality (needs HF token) |
| 12 | `microsoft/phi-4` | ~14-16GB | Yes | Small but smart |
| 13 | `google/gemma-2-27b-it` | ~24-28GB | Yes | Strong mid-size (needs HF token) |
| 14 | `CohereForAI/c4ai-command-r-plus-08-2024` | ~208GB | No | BF16, 104B, requires 2 Sparks (needs HF token) |
| 15 | `nvidia/Llama-3.1-405B-Instruct-FP4` | ~200GB | No | FP4 quantized 405B, requires 2 Sparks (needs HF token) |
| 16 | `meta-llama/Llama-3.3-70B-Instruct` | ~141GB | No | BF16, requires 2 Sparks (needs HF token) |

**Single-Node:** Models up to ~100GB fit on one DGX Spark (~120GB VRAM), including GPT-OSS 120B in its native MXFP4 format
**Dual-Node:** Required for models that exceed a single Spark's VRAM — combined ~240GB across two Sparks supports models up to ~200GB (e.g., Command-R-Plus, FP4-quantized Llama 405B, BF16 Llama 3.3 70B)

## Benchmark Profiles

The `benchmark_current.sh` script supports multiple options:

```bash
# Single-request latency test
./benchmark_current.sh --single

# Quick benchmark (20 prompts)
./benchmark_current.sh --quick

# Full benchmark (100 prompts, default)
./benchmark_current.sh

# Custom options
./benchmark_current.sh -n 50 -c 50 -o results.json
```

| Option | Description |
|--------|-------------|
| `-u, --url URL` | vLLM API URL (default: auto-detect) |
| `-n, --num-prompts N` | Number of prompts to benchmark (default: 100) |
| `-c, --concurrency N` | Max concurrent requests (default: 100) |
| `-d, --dataset PATH` | Path to ShareGPT dataset JSON |
| `-s, --single` | Run single-request benchmark only |
| `-q, --quick` | Quick mode: 20 prompts, lower concurrency |
| `-o, --output FILE` | Output results to JSON file |

### Benchmark All Models

Use `benchmark_all.sh` to automatically benchmark multiple models and create a comparison matrix:

```bash
# Benchmark all models (takes several hours)
./benchmark_all.sh

# Only single-node models (faster)
./benchmark_all.sh --single-node

# Skip models requiring HF token
./benchmark_all.sh --skip-token
```

## API Endpoints

Once running, the API is available on the head node:

| Endpoint | Description |
|----------|-------------|
| `http://<head-ip>:8000/health` | Health check |
| `http://<head-ip>:8000/v1/models` | List models |
| `http://<head-ip>:8000/v1/chat/completions` | Chat API (OpenAI compatible) |
| `http://<head-ip>:8000/v1/completions` | Completions API |

### Example: Chat Completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain quantum computing briefly."}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

### Example: Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)
print(response.choices[0].message.content)
```

## Troubleshooting

### SSH Connection Failed

```bash
# Test SSH connectivity
ssh <username>@<worker-ip> "hostname"

# If it fails, setup passwordless SSH:
ssh-copy-id <username>@<worker-ip>
```

### Worker Not Joining Cluster

```bash
# Check worker logs (from head node)
ssh <username>@<worker-ip> "docker logs ray-worker"

# Check Ray cluster status
docker exec ray-head ray status --address=127.0.0.1:6385
```

### Low Throughput (Using Ethernet instead of InfiniBand)

```bash
# Run NCCL diagnostics
./checkout_setup.sh --nccl

# Check vLLM logs for transport type
docker exec ray-head tail -100 /var/log/vllm.log | grep -E "NCCL|NET"

# Good: "NCCL INFO NET/IB" or "GPU Direct RDMA"
# Bad:  "NCCL INFO NET/Socket" (falling back to Ethernet)
```

### NCCL Communication Issues

```bash
# Full InfiniBand check
./checkout_setup.sh --infiniband

# Check InfiniBand devices
ibv_devinfo

# If IB issues persist, check cables and run:
ibdev2netdev  # Should show "(Up)" status
```

### Out of Memory

```bash
# Reduce memory utilization
export GPU_MEMORY_UTIL=0.80
./start_cluster.sh

# Or reduce context length
export MAX_MODEL_LEN=4096
./start_cluster.sh

# Or try a smaller model
./switch_model.sh --list  # Pick single-node model
```

### vLLM Server Not Starting

```bash
# Check vLLM logs
docker exec ray-head tail -100 /var/log/vllm.log

# Check Ray status
docker exec ray-head ray status --address=127.0.0.1:6385

# Common issues:
# - Insufficient GPUs for tensor-parallel-size
# - Model download failed (check HF_TOKEN for gated models)
# - NCCL timeout (check InfiniBand connectivity)
```

### Model Download Issues

```bash
# Check if HF token is set (for gated models)
echo $HF_TOKEN

# Pre-download model manually
./switch_model.sh -d <model-number>

# Sync to worker
./switch_model.sh -r <model-number>
```

## System Diagnostics

Use `checkout_setup.sh` for comprehensive system checks:

```bash
# Interactive menu
./checkout_setup.sh

# Quick overview
./checkout_setup.sh --quick

# Full InfiniBand check
./checkout_setup.sh --infiniband

# NCCL transport verification
./checkout_setup.sh --nccl

# Everything
./checkout_setup.sh --full
```

## Performance Notes

### Expected Performance (GPT-OSS 120B)

Performance will vary based on single-node vs dual-node deployment, context length, and concurrency. The numbers below are rough guidance from dual-node runs:

| Metric | Value |
|--------|-------|
| Output Throughput | ~50-100 tok/s |
| Time to First Token | ~2-5s |
| Batch Throughput | ~400-700 tok/s |

### Optimization Tips

1. **Use InfiniBand IPs** - Ensure WORKER_IPS uses the 169.254.x.x InfiniBand addresses
2. **Memory Utilization** - Set to 0.90 for max KV cache, reduce if OOM
3. **Expert Parallel** - Enable for MoE models (gpt-oss, Mixtral)
4. **Pre-download Models** - Use `switch_model.sh -d` to avoid download delays

## File Structure

```
vllm-dgx-spark/
├── README.md              # This file
├── config.env             # Configuration template
├── config.local.env       # Your local config (gitignored)
├── .gitignore             # Git ignore patterns
├── setup-env.sh           # Interactive setup script
├── start_cluster.sh       # Main cluster startup script
├── start_worker_vllm.sh   # Worker script (copied to workers by start_cluster.sh)
├── stop_cluster.sh        # Cluster shutdown script
├── switch_model.sh        # Model switching utility
├── benchmark_current.sh   # Single model benchmark tool
├── benchmark_all.sh       # Multi-model comparison benchmark
├── checkout_setup.sh      # System diagnostics (InfiniBand, NCCL, GPU)
└── benchmark_results/     # Benchmark output directory
```

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [NVIDIA vLLM Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm)
- [NVIDIA DGX Spark vLLM Playbook](https://build.nvidia.com/spark/vllm/stacked-sparks)
- [NVIDIA NCCL over InfiniBand](https://build.nvidia.com/spark/nccl/stacked-sparks)

## License

MIT
