# vLLM on DGX Spark Cluster

Deploy [vLLM](https://github.com/vllm-project/vllm) on NVIDIA DGX Spark systems - supports both single-node and dual-node cluster configurations with InfiniBand RDMA for serving large language models.

> **DISCLAIMER**: This project is NOT affiliated with, endorsed by, or officially supported by NVIDIA, vLLM, or any other organization. This is a community-driven effort to run vLLM on DGX Spark hardware. Use at your own risk. The software is provided "AS IS", without warranty of any kind.

> **Upgraded (2026-04-28)**: Components bumped to `nvcr.io/nvidia/vllm:26.04-py3` (vLLM 0.19.0, PyTorch 2.12.0a0, CUDA 13.2.1) with Ray 2.55.1. The 26.04 container no longer ships Ray, so `start_cluster.sh` and `start_worker_vllm.sh` now `pip install ray[default]==${RAY_VERSION}` into the head and worker containers at startup (~30s). Smoke tested on a single Spark; multi-node TP=2 verification pending.
>
> Previous (2026-04-09): `nvcr.io/nvidia/vllm:26.03-py3` (vLLM 0.17.1) with Ray 2.54.0.

## Features

- **1-to-N Spark support** - Single Spark, two Sparks (stacked / direct cable), or 3+ Sparks (switched fabric). `WORKER_HOST` and `WORKER_IB_IP` are space-separated lists; `TENSOR_PARALLEL` defaults to `1 + N` workers.
- **Zero-config single-node** - No InfiniBand setup required for single-Spark deployments
- **Single-command deployment** - Start the entire cluster from the head Spark via SSH
- **Auto-detection** of InfiniBand IPs, network interfaces, and HCA devices (multi-Spark)
- **41 model presets** - 16 community-tested + 25 from NVIDIA's official Spark vLLM matrix (FP8, NVFP4, MXFP4, BF16 quants)
- **InfiniBand RDMA** for high-speed inter-Spark communication (200Gb/s)
- **Comprehensive benchmarking** with multiple test profiles
- **Offline mode** - `SKIP_MODEL_DOWNLOAD=1` for air-gapped Sparks with a pre-staged HF cache

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

### Multi-Spark Mode (1 head + N workers)

The same scripts handle 2 Sparks (directly cabled QSFP) and 3+ Sparks (switched fabric). `TENSOR_PARALLEL` defaults to `1 + N` (one GPU per Spark) but can be overridden.

```
┌────────────────────────────────────────────────────────────────────┐
│            DGX Spark Cluster (1 Head + N Workers)                  │
│                                                                    │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌─────────┐  │
│  │     HEAD NODE        │  │   WORKER NODE 1      │  │  ...    │  │
│  │    (Ray head)        │  │   (Ray worker)       │  │ Worker  │  │
│  │                      │  │                      │  │   N     │  │
│  │  GPU: 1x GB10        │◄►│  GPU: 1x GB10        │◄►│         │  │
│  │  (Blackwell, sm100)  │IB│  (Blackwell, sm100)  │IB│         │  │
│  │                      │  │                      │  │         │  │
│  │  /raid/hf-cache      │  │  /raid/hf-cache      │  │  ...    │  │
│  │  Port: 8000 (API)    │  │                      │  │         │  │
│  └──────────────────────┘  └──────────────────────┘  └─────────┘  │
│           ▲                          ▲                    ▲        │
│           └──────────────────────────┴────────────────────┘        │
│         200Gb/s QSFP - direct cable (2 Sparks) or switch (3+)      │
│                                                                    │
│  Tensor Parallel (TP=N+1): Model split across all GPUs             │
│  Default TP: 1 + WORKER_COUNT (override via TENSOR_PARALLEL=...)   │
└────────────────────────────────────────────────────────────────────┘
```

NVIDIA documents three reference topologies — see their official playbook at <https://build.nvidia.com/spark/vllm/instructions> (single Spark, stacked / direct-cable 2 Sparks, switched 3+ Sparks). Our scripts cover all three with the same code path.

## Hardware Requirements

### Single-Spark
- **Nodes:** 1x DGX Spark system
- **GPUs:** 1x NVIDIA GB10 (Grace Blackwell, sm100), ~120GB unified memory
- **Storage:** Model cache at `/raid/hf-cache` (or configure in `config.env`)

### Multi-Spark (2+ Sparks for larger models)
- **Nodes:** 2 or more DGX Spark systems
- **GPUs:** 1x NVIDIA GB10 per node, ~120GB unified memory each
- **Network:**
  - **2 Sparks**: 200Gb/s QSFP direct cable (no switch needed)
  - **3+ Sparks**: 200Gb/s QSFP through a switch
- **Storage:** Model cache at `/raid/hf-cache` on every Spark (use `${WORKER_HF_CACHE}` to override on workers)
- **SSH:** Passwordless SSH from head to every worker

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

# Two Sparks (head + 1 worker):
# WORKER_HOST="<worker1-ethernet-ip>"
# WORKER_IB_IP="<worker1-infiniband-ip>"
# WORKER_USER="<ssh-username>"
#
# 3+ Sparks (head + N workers; lists are space-separated, 1:1 positional):
# WORKER_HOST="<w1-eth> <w2-eth> <w3-eth>"
# WORKER_IB_IP="<w1-ib> <w2-ib> <w3-ib>"
# WORKER_USER="<ssh-username>"
# # TENSOR_PARALLEL defaults to 1 + N - override only if needed
```

**Start the Cluster:**
```bash
./start_cluster.sh
```

This will:
1. Pull the Docker image on the head node (workers pull theirs on first launch)
2. Download the model on the head and rsync to each worker
3. SSH to every worker and start its Ray worker container
4. Start Ray head and vLLM server
5. Wait for all `1 + N` Sparks to join the Ray cluster (~2-5 minutes for 2 Sparks; +60s budget per additional worker)

#### Option C: 3+ Spark Cluster (NVIDIA's switched-fabric topology)

Same as Option B, just longer lists in `WORKER_HOST` and `WORKER_IB_IP`. Example for a 4-Spark cluster running NVIDIA's MiniMax-M2.5 TP=4 example:

```bash
export WORKER_HOST="192.168.7.111 192.168.7.112 192.168.7.113"
export WORKER_IB_IP="169.254.216.8 169.254.216.9 169.254.216.10"
export WORKER_USER="rispark"
# TENSOR_PARALLEL defaults to 4 (= 1 + 3 workers); override only if needed
./switch_model.sh   # pick "MiniMax-M2.5" entry
```

The same `start_cluster.sh` handles 2 Sparks (direct QSFP cable) and 3+ Sparks (switched fabric); all you change is the length of the lists.

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

The cluster ships with **41 model presets** in `switch_model.sh`. They split into two groups:

- **`[NVIDIA]`** — 25 entries from NVIDIA's official Spark vLLM matrix at <https://build.nvidia.com/spark/vllm/instructions>. Most are FP8 / NVFP4 quantized variants from `nvidia/...` that take maximum advantage of Spark's unified memory.
- **`[community]`** — 16 entries that work but are not on NVIDIA's verified matrix (Mistral family, Qwen2.5 base, Gemma 2, Command-R-Plus, etc.). Kept for backwards compatibility.

Run `./switch_model.sh` for the full menu (it groups by Single-Spark vs Multi-Spark and shows quant labels, gating, and cache status). Some highlights:

| Family | Variant | Quant | Sparks | TP |
|---|---|---|---|---|
| `nvidia/Llama-3.1-8B-Instruct-{FP8,NVFP4}` | 8B | FP8 / NVFP4 | 1 | 1 |
| `nvidia/Llama-3.3-70B-Instruct-NVFP4` | 70B | NVFP4 | 1 | 1 |
| `nvidia/Qwen3-{8B,14B,32B}-{FP8,NVFP4}` | 8-32B | FP8 / NVFP4 | 1 | 1 |
| `nvidia/Phi-4-multimodal-instruct-{FP8,NVFP4}` | multimodal | FP8 / NVFP4 | 1 | 1 |
| `nvidia/Phi-4-reasoning-plus-{FP8,NVFP4}` | reasoning | FP8 / NVFP4 | 1 | 1 |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-{BF16,FP8}` | 30B MoE | BF16 / FP8 | 1 | 1 |
| `google/gemma-4-{E2B,E4B,26B-A4B,31B}-it` | base | BF16 | 1 | 1 |
| `nvidia/Gemma-4-31B-IT-NVFP4` | 31B | NVFP4 | 1 | 1 |
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | 120B MoE | NVFP4 | 2 | 2 |
| `MiniMaxAI/MiniMax-M2.5` | MoE | base | 4 | 4 |
| `openai/gpt-oss-{20b,120b}` | 20B / 120B MoE | MXFP4 | 1 | 1 |
| `meta-llama/Llama-3.3-70B-Instruct` | 70B | BF16 | 2 | 2 |

> **Gemma 4** uses a separate vLLM image (`vllm/vllm-openai:gemma4-cu130`); `switch_model.sh` sets `VLLM_IMAGE` automatically when you pick a Gemma 4 entry.
> **Phi-4-multimodal-instruct** and **MiniMax-M2.5** require `--trust-remote-code` (handled automatically — `switch_model.sh` exports `TRUST_REMOTE_CODE=true` for those entries).
>
> Spark's 120GB unified memory is shared between CPU and GPU; quantized variants (FP8 / NVFP4 / MXFP4) typically give the best throughput at usable batch sizes.

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

### Unified-Memory Pressure (UMA buffer-cache flush)

DGX Spark uses a Unified Memory Architecture (UMA) where CPU and GPU share the same physical DRAM. Linux's page cache can hold onto memory that vLLM/CUDA can't reclaim, leading to apparent OOM well within capacity. NVIDIA recommends flushing the buffer cache when this happens:

```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

Run this on every Spark before launching a large model if you've recently been working with big files.

### NCCL: IB Verbs vs TCP-on-QSFP

Our multi-Spark scripts enable RDMA verbs (`NCCL_IB_HCA`, `NCCL_NET_GDR_LEVEL=5`) for best throughput. NVIDIA's official Spark playbook only sets `NCCL_SOCKET_IFNAME` and routes NCCL over TCP on the same QSFP interface — slower in theory but more compatible.

If you hit NCCL hangs during model load on a multi-Spark setup, the fast workaround is to fall back to NVIDIA's TCP path:

```bash
export NCCL_IB_DISABLE=1
./start_cluster.sh
```

That forces NCCL to use sockets only and matches the official playbook exactly.

### Silent vLLM crashes (used to look like "Loading...")

Older versions of `start_cluster.sh` used `pgrep -f 'vllm serve'` to check whether vLLM was alive during startup. Because the pattern lived in the wrapping shell's argv, pgrep self-matched and our death-detection never fired — a real vLLM crash looked identical to a slow legitimate load until the 30-min timeout. Fixed in commit `462aad5` by writing a pidfile at launch and using `kill -0` instead. If you're on an older revision and seeing 30-min "Loading..." spinners, pull latest.

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

Performance will vary based on Spark count, model quantization, context length, and concurrency. The numbers below are rough guidance from 2-Spark runs:

| Metric | Value |
|--------|-------|
| Output Throughput | ~50-100 tok/s |
| Time to First Token | ~2-5s |
| Batch Throughput | ~400-700 tok/s |

### Optimization Tips

1. **Use InfiniBand IPs** - Ensure `WORKER_IB_IP` uses 169.254.x.x InfiniBand addresses (not the slower Ethernet IPs)
2. **Memory Utilization** - Set `GPU_MEMORY_UTIL=0.90` for max KV cache, reduce if OOM
3. **Expert Parallel** - Enable for MoE models (gpt-oss, Mixtral, Nemotron-3-Nano-A3B, MiniMax). Auto-detected from the model name pattern in `start_cluster.sh`; per-model presets in `switch_model.sh` set it explicitly.
4. **Pre-download Models** - Use `switch_model.sh -d` to avoid first-launch download delays
5. **Quantized variants** - On Spark, NVFP4 / FP8 / MXFP4 generally beat BF16 at usable batch sizes since they free more unified memory for the KV cache

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

## How this repo relates to vLLM's `run_cluster.sh`

NVIDIA's official Spark playbook uses vLLM's upstream helper script `run_cluster.sh` (from `vllm-project/vllm/examples/online_serving`) which boils a multi-node Ray + Docker setup down to ~50 lines. That's the minimal path — no diagnostics, no model presets, no IB tuning, no auto-detection. You set `MN_IF_NAME` and `HEAD_NODE_IP` by hand and run the same command on every box.

This repo is a **superset**:

- Single-command orchestration from the head Spark (no need to SSH to every worker manually)
- Auto-detects InfiniBand HCAs and IPs via `ibdev2netdev`
- 41 model presets with per-model TP / max_model_len / quant / image / trust-remote-code defaults
- Multi-Spark (1-N workers) with array `WORKER_HOST` / `WORKER_IB_IP`
- Rich diagnostics (`checkout_setup.sh`) for SSH, RDMA, NCCL transport, port conflicts, GPU topology
- Offline-mode support (`SKIP_MODEL_DOWNLOAD=1`) for air-gapped Sparks
- Benchmark harness (single-request, quick, full) and per-model results

If you only need the minimal upstream path, the official NVIDIA pages and `run_cluster.sh` are linked below.

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [vLLM `run_cluster.sh`](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/run_cluster.sh) — minimal multi-node helper used by NVIDIA's playbook
- [NVIDIA vLLM Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm)
- [NVIDIA DGX Spark vLLM Playbook](https://build.nvidia.com/spark/vllm/instructions) (single Spark / stacked / switched / troubleshooting tabs)
- [NVIDIA NCCL over InfiniBand](https://build.nvidia.com/spark/nccl/stacked-sparks)

## License

MIT
