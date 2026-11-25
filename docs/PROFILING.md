# Profiling guide — INFERIO on T4 (SM75)

Two levels of measurement exist, and they answer different questions:

1. **Benchmark harness (frozen)** — whole-prefill latency per batch size.
   This is the only source of the official numbers.
2. **Where is the time going?** — two tools:
   - built-in phase timers (always available, zero setup),
   - `ncu` / `nsys` (kernel-level, preinstalled on the Kaggle box).

Profiling must never be the basis of a ledger number; the frozen harness does
that. Profiling tells us **what to change next**.

## 1. Built-in phase timers (engine)

Set `INFERIO_PROFILE=1` before calling the engine; the engine records CUDA
events around each phase (embed, ln, qkv, attn, proj, fc1, fc2, head, argmax)
and reports summed ms per phase in `inferio_stats::phase_ms`. The frozen
harness passes them through into each result row (`phase_ms` field) when run
with `--profile`.

Run one config:

```bash
INFERIO_PROFILE=1 python bench/bench_prefill.py --single --batch 64 \
  --lib engine/build/libinferio.so --weights /kaggle/working/weights_gpt2
```

This is the first stop for every optimization step: it shows the split between
GEMM (qkv/proj/fc1/fc2) and attention, and between phases at each batch size.
Event overhead is negligible relative to the measured windows.

## 2. Nsight Systems (timeline, kernel durations, gaps)

`nsys` is at `/usr/local/bin/nsys`. Use it to see launch gaps and per-kernel
durations across the whole run:

```bash
nsys profile -t cuda,nvtx --capture-range=none -o temp/prof/nsys_b64 \
  python bench/bench_prefill.py --single --batch 64 \
  --lib engine/build/libinferio.so --weights /kaggle/working/weights_gpt2
nsys stats --report cuda_gpu_kern_sum temp/prof/nsys_b64.nsys-rep
```

`cuda_gpu_kern_sum` gives total time per kernel; compare the top kernels with
the phase table from (1). Gaps in the timeline = launch overhead (relevant at
batch 1, negligible at high batch).

## 3. Nsight Compute (kernel micro-analysis)

`ncu` is at `/usr/local/cuda/bin/ncu`. It replays kernels with counters, so it
is for **single-kernel analysis**, not end-to-end timing. Example — profile the
attention kernel at batch 16:

```bash
ncu --target-processes all -k regex:attn_kernel --launch-count 3 \
  --set detailed -o temp/prof/ncu_attn_b16 \
  python bench/bench_prefill.py --single --batch 16 \
  --lib engine/build/libinferio.so --weights /kaggle/working/weights_gpt2
ncu --import temp/prof/ncu_attn_b16.ncu-rep --page details | head -100
```

Useful metrics on SM75 (Turing):

| metric | meaning |
|---|---|
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | tensor-core utilization |
| `sm__throughput.avg.pct_of_peak_sustained_active` | SM issue utilization |
| `dram__bytes.sum.per_second` | HBM bandwidth achieved (T4 peak 320 GB/s) |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` | shared-memory bank conflicts |
| `smsp__warps_active.avg.pct_of_peak_sustained_active` | achieved occupancy |
| `smsp__average_warps_issue_stalled_*` | warp stall reasons |

The GEMMs are cuBLAS calls, so for them the interesting question is *which
kernel cuBLAS chose* (visible in the nsys kernel table) and how close it is to
the tensor-core roofline — not how to rewrite cuBLAS internals.

Caveats, stated plainly:

- `ncu` serializes and replays kernels; its absolute times are **not**
  comparable to harness times. Only ratios/percentages are useful.
- Kaggle sessions are shared infrastructure; SM clock and temperature are
  recorded in every result row (`device_env`) so slow sessions can be
  identified rather than guessed about.
- The phase timers measure enqueued work on the default stream; there is no
  overlap between phases in v0, so the sum is meaningful. If concurrency is
  introduced later (streams, graphs), revalidate the summation assumption.
