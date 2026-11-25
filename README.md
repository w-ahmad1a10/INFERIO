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

## Status (2026-11-25)

- Frozen benchmark SPEC v1 + harness committed; correctness gate is a parity
  check against an fp32 numpy oracle (`tools/oracle_gpt2.py`).
- Engine v0 (correct-first: cuBLAS fp16 GEMMs + naive online-softmax attention,
  sequence-streaming) builds on the Kaggle T4 and passes stage-level checks
  against the numpy reference.
- First full 9-batch measurement is running; results land in `temp/LEDGER.md`.
