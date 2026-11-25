# Reading log

What was actually read, and how it changed decisions. (Not a wish list —
entries appear here only after the material was read.)

## 2026-11-25

### Turing Tuning Guide (docs.nvidia.com/cuda/turing-tuning-guide, v13.5)
Read in full. Key facts applied:
- SM75: max 32 warps/SM, 64K regs/SM, max 16 blocks/SM, 64 KB smem/SM,
  96 KB unified L1/smem (64/32 or 32/64 carveout).
- FMA latency hidden by as few as 4 warps with 4-way ILP; 16 warps without ILP
  → occupancy directly explains why the 33 KB-smem tiled attention kernel
  regressed (1 CTA/SM) vs the register-only kernel (v0.3 measurement:
  attention 15.7 s → 26.0 s at B=1024).
- Turing Tensor Cores accept the *same* WMMA/mma fragment shapes as Volta but
  a Volta-compiled binary reaches only half of Turing's TC peak → we compile
  `-arch=sm_75` explicitly. Decision: skip `nvcuda::wmma` (broken in the CUDA 13
  headers for this usage) and use inline PTX `mma.sync.m16n8k8` directly.

### Microbenchmark on the actual Kaggle T4 (measured, not read)
- `exp2f`: 68.35 Gexp/s → softmax-exp floor at B=1024 ≈ 1.1 s (77.4e9 exps).
- FP32 FMA: 2052 GFMA/s (≈4.1 TFLOPS) at ~855 MHz SM clock (card runs at
  ~54% of the 1590 MHz boost; clock recorded per run in results).
- Conclusion: attention's per-key fp32 FMA work (~9.9e12 FMA → ~4.8 s floor)
  dominates, not exp → tensor cores for QK^T/PV are the right fix.

### Previous measurements that shaped these reads (recorded in temp/LEDGER.md)
- v0 baseline 29.24 s @ B=1024; v0.2 28.93 s (GEMM default already near the
  clock-adjusted cuBLAS ceiling); v0.3 36.9 s regression from smem-tiled
  attention (occupancy); tanh.approx GELU cut fc1 3.77 s → 2.15 s.

## Pending reads (not yet done — tracked honestly)

- CUDA C++ Programming Guide (downloaded) — targeted sections as needed.
- FlashAttention v1/v2 papers (downloaded) — skimmed only via tutorial prior;
  full read pending.
- CUTLASS docs (vendor clone) — SM75 GEMM tuning pages pending.
- llm.c kernels (vendor clone) — attention/softmax reference reading pending.
- PTX ISA m16n8k8 layout section — consulted for the inline-PTX attention
  kernel; formal cross-check against the doc page pending.
