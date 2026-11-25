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
| `bench/` | frozen benchmark harness (any change is announced; hash-verified) _(planned)_ |
| `engine/` | the inference engine (C/C++/CUDA + Triton) _(planned)_ |
| `kaggle/` | driver to execute code on the remote Kaggle T4 x2 notebook |
| `study/` | downloaded papers/docs/repos used as references, with an index |
| `temp/` | every measured result + the step-by-step ledger _(created with the first run)_ |

## Status

Step 1: design brainstorm (nothing built yet). Progress is recorded factually in
`temp/LEDGER.md` starting with the first measured run.
