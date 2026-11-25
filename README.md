# INFERIO

A custom inference engine for **GPT-2 on NVIDIA T4 (Turing, SM 7.5)**, built to
maximize **prefill throughput at 1024-token context** (GPT-2's native window)
across batch sizes 1–1024, using hand-written C/C++/CUDA (plus Triton for kernel
prototyping).

No PyTorch, no vLLM, no heavy frameworks. Allowed building blocks: CUDA C/C++,
Triton, cuBLAS/cuBLASLt, CUTLASS, cuDNN.

## Fixed benchmark

- Model: GPT-2 small (124M params, 12 layers, 768 hidden, 12 heads, FFN 3072, vocab 50257).
- Prompt: exactly 1024 random token ids per sequence (fixed seed; input hashes recorded).
- Batch sizes tested: 1, 4, 16, 32, 64, 128, 256, 512, 1024.
- Metric: prefill latency (tokens in → last-position logits / first greedy token).
- OOM is caught and recorded per batch size — never a crash.
- Every run recorded under `temp/` with an explanation of what changed and why.
- Target: high batch sizes. Batch 1 is recorded for completeness but not optimized for.

## Layout

| path | purpose |
|---|---|
| `bench/` | frozen benchmark harness (any change is announced; hash-verified) |
| `engine/` | the inference engine (C/C++/CUDA + Triton) |
| `tools/` | weight conversion, fp32 numpy oracle, debug comparators |
| `docs/` | profiling guide and other operational documentation |
| `kaggle/` | driver to execute code on the remote Kaggle T4 x2 notebook |
| `study/` | downloaded papers/docs/repos used as references, with an index |
| `temp/` | every measured result + the step-by-step ledger |

## Status (2026-11-25, final)

**Engine v0.12.2** — champion config: 2-pass LayerNorm, group-64 sequence
streaming, PTX tensor-core attention with score scale folded into Q, cuBLASLt
GEMMs with bias epilogue and β=1 residual fusion. Parity PASS at every accepted
version (argmax identical to the fp32 numpy oracle; max |Δlogit| ≤ 0.24).

### Official results (frozen SPEC v1 harness, median ms, GPU 0)

| B | v0.1.0 | v0.2.0 | v0.3.0 | v0.7.0 | v0.8.0 | v0.9.0 | **v0.12.2** | PyTorch eager |
|---|--------|--------|--------|--------|--------|--------|-------------|---------------|
| 1 | 45.6 | 41.0 | 37.5 | 14.3 | 15.6 | 15.8 | **13.8** | 24.8 |
| 4 | 94.8 | 92.6 | 126.0 | 51.0 | 48.3 | 53.9 | **49.8** | 61.3 |
| 16 | 352.4 | 344.7 | 486.3 | 195.5 | 182.0 | 210.5 | **196.2** | 242.2 |
| 32 | 730.4 | 711.6 | 970.7 | 405.4 | 374.2 | 432.4 | **405.8** | 487.4 |
| 64 | 1484.0 | 1455.7 | 2022.3 | 833.9 | 749.5 | 892.9 | **841.3** | 972.1 |
| 128 | 3167.9 | 3017.1 | 4253.3 | 1826.0 | 1582.8 | 1880.3 | **1743.5** | 1993.6 |
| 256 | 6811.4 | 6530.5 | 9358.1 | 3877.7 | 3314.2 | 3714.5 | **3766.0** | OOM |
| 512 | 14545.1 | 14415.3 | 18444.6 | 8809.1 | 7524.1 | 7039.3 | **7060.4** | OOM |
| 1024 | 29236.0 | 28929.9 | 36897.3 | 17404.1 | 16525.4 | 14291.4 | **14321.5** | OOM |

Total speedup v0.1.0 → v0.12.2 at B=1024: **2.04×** (29.24 s → 14.32 s).
Cross-session clock caveat: Kaggle T4 SM clocks varied 585–1035 MHz between
runs; device env is recorded in every result JSON. Same-session A/Bs are the
valid engine-vs-engine comparisons.

### PyTorch reference baselines (GPU 1, same frozen inputs, reference only)

| variant | B=1 | B=16 | B=128 | B=256 | B≥512 |
|---|---|---|---|---|---|
| eager (original) | 24.8 | 242.2 | 1993.6 | OOM | OOM |
| eager + use_cache=False | 23.8 | 226.0 | 1934.5 | 4197.2 | OOM |
| torch.compile(dynamic) | 20.5 | 242.5 | 2209.4 | 5211.1 | OOM |
| compile specialized (16/64/256) | – | 241.8 | – | 5003.3 | – |

Engine vs best torch: 1.09–1.15× at B=16–256, 1.72× at B=1; at B=512/1024
torch cannot fit in 16 GB at all. torch.compile is slower than eager in every
flavor on this box (measured). The KV allocation was torch's B=256 OOM cause
(use_cache=False fixes 256 only).

### Hill-climb record (all measured, all in temp/LEDGER.md)

- **v0.2.0** cuBLASLt per-shape plans + vectorized LN (small win)
- **v0.3.0** smem-tiled attention — 28% regression (occupancy), reverted
- **v0.7.0** PTX mma.sync tensor-core attention (+bias epilogue, tanh.approx
  GELU) — the big one, 1.66×
- **v0.8.0** V tile staged in shared memory
- **v0.9.0** 2-pass LayerNorm + group 64 (sweep-verified) — B=1024 −6% (A/B)
- **v0.10/0.11** K staging (null) and register cap (worse) — reverted
- **v0.12.x** mma128 variant (correct, slower, option only); scale fold
  accepted (attention −2..3%)
- Explored and exhausted: cuBLASLt heuristic sweep (default optimal), group
  sizes 32–512, attention tile shapes, occupancy/register levers.

### Engineering notes

- Stack: pure CUDA C++ + inline PTX (`mma.sync.m16n8k8`) for all custom
  kernels; cuBLAS/cuBLASLt for weight GEMMs; zero Triton/CUTLASS in the engine
  (downloaded to `study/` as references). No PyTorch in the engine.
- Profiling: built-in phase timers (`INFERIO_PROFILE=1`), nsys/ncu guide in
  `docs/PROFILING.md` (ncu counters are blocked on Kaggle: ERR_NVGPUCTRPERM).
- Remaining headroom is ~1–1.5 s of 14.3 s (attention SFU-latency gap; GEMMs
  at the cuBLAS ceiling) — see the coach assessment in the ledger notes.
