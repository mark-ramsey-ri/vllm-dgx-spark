# Multi-Spark RDMA Throughput — Improvement Plan

Status as of 2026-06-22. Cluster: 4× DGX Spark (GB10, 128 GB unified mem,
ConnectX-7 2×200 G), head `spark-6033` + workers `.2/.3/.4` on `10.10.10.0/24`,
through a **MikroTik CRS804-4DDQ** (RouterOS 7.19.6).

## Where we are

| Config | 405B-FP4 TP=4 throughput |
|--------|--------------------------|
| NCCL over TCP (socket) | ~1.74 tok/s |
| NCCL over RoCE/RDMA (current) | ~2.1 tok/s same-request, ~2.4 warm (+20–38%) |

RDMA works (commit `ddc3f9c`) but the fabric delivers only **~2.5 Gb/s of a
200 G link (~1%)**. That is the ceiling we need to lift.

## Root causes of the low fabric bandwidth (measured)

1. **No lossless config — PFC disabled on every priority** (`mlnx_qos` shows
   `enabled 0 0 0 0 0 0 0 0`, all RoCE traffic on priority 0). RoCEv2 on a
   lossy switch drops under load → go-back-N retransmits → throughput collapse.
   **This is the dominant problem.**
2. **MTU 1024** (netdev 1500 → RoCE active_mtu 1024). Switch drops jumbo today
   (8 KB DF ping = 100 % loss).
3. **Only one of two NIC ports used.** Second 200 G port (`roceP2p1s0f1`,
   .11–.14) is idle. (We pinned to one port to fix the dual-HCA NCCL hang.)
4. **GDR not engaged in NCCL rings** (`GDR 0` in the standalone test).
5. **Algorithmic:** TP=4 all-reduces every layer (~252 cross-node collectives
   per token) — the worst traffic pattern for a slow fabric.

---

## Tier 1 — Make the fabric lossless (biggest lever: ~2.5 → tens of Gb/s)

RoCEv2 needs lossless, configured to agree on **both** NICs and switch.

### MikroTik CRS804 reality
RouterOS on the Marvell Prestera chip has **limited/poor PFC + ECN (DCQCN)
support** and **small shared packet buffers** (a few MB). Do **not** expect
full per-priority DCB/DCQCN like an NVIDIA Spectrum switch. The practical
substitute on MikroTik:

- **Enable link-level flow control (802.3x pause) on every QSFP port.** Because
  this fabric carries *only* RoCE traffic, plain global pause makes the link
  effectively lossless without needing per-priority PFC.
  ```
  /interface ethernet set qsfp56-dd-1 tx-flow-control=on rx-flow-control=on
  # repeat for all 4 ports (qsfp56-dd-1..4)
  ```
- **Match flow control on the NICs (all 4 Sparks):**
  ```
  sudo ethtool -A enp1s0f1np1 rx on tx on
  ```
- Keep the bridge **hardware-offloaded** (`hw=yes`), pure L2, no firewall/CPU
  path for fabric traffic (all Sparks already in one /24).
- Check switch port drops after load: RouterOS `/interface ethernet monitor`
  and stats (`rx-drop`, `fcs-error`); NIC side `ethtool -S | grep -i drop`.

If lossless via global pause still isn't enough (small MikroTik buffers can
still overrun), the real fix is a **deep-buffer / RoCE-optimized switch**
(e.g., NVIDIA Spectrum) — that is the ceiling on this MikroTik.

## Tier 1b — Jumbo frames (multiplies large-message BW ~3–4×)

MikroTik supports large MTU well; ConnectX-7 maxmtu is 9978.

- **Switch:** raise `l2mtu` on the QSFP ports / bridge to ≥9216
  ```
  /interface ethernet set qsfp56-dd-1 l2mtu=9216   # all 4 ports
  /interface bridge set <bridge> mtu=9216
  ```
- **Hosts (all 4):** `sudo ip link set enp1s0f1np1 mtu 9000` (persist via netplan)
- Result: RoCE active_mtu 1024 → 4096. Verify: `ping -M do -s 8972 <peer>` must
  succeed, then `ibv_devinfo -d rocep1s0f1 | grep active_mtu` shows 4096.

## Tier 2 — Use both NIC ports (2-rail RDMA, ~2×)

Each Spark has a second idle 200 G port (`roceP2p1s0f1`). The CRS804 has only
4× QSFP56-DD, all used — but each 400 G port can **break out to 2×200 G**
(needs breakout DACs), giving 8×200 G = two rails for four Sparks.
- Put the 2nd port on a separate subnet (e.g. `10.10.11.0/24`).
- Let NCCL use both HCAs (multi-rail). Removing the same-subnet collision is
  what makes the dual-HCA path safe.

---

## Tier 3 — Algorithmic / serving changes (no switch needed — do today)

- **Pipeline parallelism instead of tensor parallelism.** Highest-leverage
  host-only change. `--pipeline-parallel-size 4 --tensor-parallel-size 1`
  passes activations only at stage boundaries (point-to-point, ~once per
  microbatch) instead of all-reducing every layer — ~100× less cross-node
  traffic. Needs concurrency/microbatching to fill the pipeline.
- **Concurrency.** Per-token is comms-bound; the collectives amortize across a
  batch. Many parallel requests raise *aggregate* tok/s far more than
  single-stream (which stays ~2.4).
- **Right-size the model.** Anything that fits on one Spark (70B/120B-class
  quantized in 128 GB) runs at **TP=1 with zero cross-node traffic** — 10–50×
  faster per token. Most drastic practical win if 405B isn't required.

## Tier 4 — NCCL micro-tuning (after lossless; modest)

- `NCCL_IB_QPS_PER_CONNECTION=4`, `NCCL_IB_SPLIT_DATA_ON_QPS=1`
- Once PFC/priority is set on the switch: `NCCL_IB_TC` / `NCCL_IB_SL` to put
  NCCL on the lossless priority.
- Get GDR actually engaged (investigate `GDR 0` in rings; `NCCL_NET_GDR_LEVEL`,
  GPU↔NIC PCIe path; GB10 unified-memory semantics differ).

---

## Current working RDMA config (for reference)

`config.local.env` (host-specific, gitignored):
```
NCCL_IB_DISABLE=0
NCCL_IB_HCA=rocep1s0f1        # pin to the single port carrying 10.10.10.1-.4
GPU_MEMORY_UTIL=0.85          # RDMA pinned/DMABUF buffers OOM 128GB unified at 0.90
```
NCCL auto-selects RoCEv2 GID 3. Revert to TCP fallback:
`NCCL_IB_DISABLE=1 NCCL_NET_PLUGIN=none NCCL_NET=Socket`.

## Priority order

1. **Tier 1b jumbo** + **Tier 1 flow control** on the MikroTik — easy, supported,
   biggest reliable win on this switch.
2. **Tier 3 PP=4 / concurrency / right-size** — host-only, no switch downtime.
3. **Tier 2 two-rail** (needs breakout cables).
4. If still bandwidth-bound: **replace MikroTik with a deep-buffer RoCE switch.**
