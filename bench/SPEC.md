# INFERIO frozen prefill benchmark — SPEC v2 (2026-11-26)

This file and `bench_prefill.py` are **FROZEN**. Any change is announced to the
user first, bumps `SPEC_VERSION`, and changes the harness hash embedded in every
future result.

**SPEC v2 (2026-11-26):** announced change — adds the MFU layer to every result
row: `clock_sm_mhz` (SM clock sampled at that batch's measured window),
`peak_tflops` (= 40 SMs × 1024 fp16 tensor-core flops/cycle × sampled clock),
`achieved_tflops` (= B × 1024 tokens × 207.7 MFLOP/token ÷ median_s), and
`mfu` (= achieved ÷ peak). Workload, metric window, batches, ids, and the ms
metric are unchanged from v1; MFU is the clock-normalized diagnostic and the
primary cross-session comparison metric. Convention note: 207.7 MFLOP/token
counts full attention; the engine's causal attention computes half the scores,
so engine-useful MFU is somewhat higher than the reported number.

## Model

GPT-2 small (124M): 12 layers, d_model 768, 12 heads × 64 head-dim, FFN 3072,
vocab 50257, LayerNorm eps 1e-5, GELU tanh-approx (`gelu_new`), learned absolute
positions 0–1023.

Engine arithmetic: fp16 storage and fp16 GEMM inputs/outputs with fp32
accumulation; LayerNorm, softmax, GELU and the logits cast computed in fp32.

Weights: converted from `openai-community/gpt2` `model.safetensors` (sha256
`248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`) to fp16.

## Inputs (frozen generation rule)

For batch size B:

```python
rng = numpy.random.default_rng(20261125 + B)
ids = rng.integers(0, 50257, size=(B, 1024), dtype=np.int32)
```

SHA-256 of the ids buffer is recorded in every result. Inputs never change.

## Workload and metric

- Batch sizes: **1, 4, 16, 32, 64, 128, 256, 512, 1024**. Each batch size runs
  in its own fresh process; an OOM is recorded as a result row and never aborts
  the run.
- Timed operation: one prefill call:
  host token ids → last-position fp32 logits for every sequence (row count must
  equal B).
- `prefill_ms` is measured with CUDA events inside the engine (start before the
  ids H2D copy; end after the last-position logits cast kernel is enqueued).
  Excluded from the timed region: process start, weight upload, warmup,
  greedy argmax, and logits download.
- Per batch size: 2 warmup prefills + 5 measured prefills; report
  median / min / max and `tokens_per_s = B*1024 / (median_ms/1000)`.
- `peak_bytes` = engine-tracked `cudaMalloc` high-water mark.

## Engine contract

The harness loads `libinferio.so` via ctypes and calls, in order:
`inferio_init(model_dir, 0)`, `inferio_prefill(ids, B, 1024, &stats)` (blocking),
`inferio_last_argmax(out, B)` → exactly B int32 values,
`inferio_copy_last_logits(...)` (optional export), `inferio_shutdown()`.
Error code 1 = OOM; anything else = error.

Internal execution freedom: the engine may schedule sequences in internal groups
(group size is an implementation detail and may be tuned). Semantics must not
change: every one of the B sequences is processed in full through all 12 layers
and exactly B result rows are produced.

## Correctness gate (parity)

Reference: `tools/oracle_gpt2.py`, an fp32 numpy forward using the same frozen
ids and the tied-embedding head. Gate: the argmax of every row must match the
oracle. The max-abs logit difference is recorded for transparency and trended
over engine versions (fp16 pipeline vs fp32 oracle).

## Result schema (`results/<run_id>.json`)

```
run_id, timestamp_utc, gpu, engine_version, engine_commit, harness_sha256,
device_env, rows[]:
  {batch, status: ok|oom|error, error, median_ms, min_ms, max_ms, latencies_ms[],
   tokens_per_s, peak_bytes, weight_bytes, ids_sha256, group_seqs, n_groups,
   phase_ms (only when profiled)}
```

## Recording rules

Every executed run appends a section to `temp/LEDGER.md` containing: what
changed vs the previous run (added/removed), the device/environment snapshot, the
full 9-row table, and the verdict. Numbers that were not measured are marked
"(not measured)" and never written as if measured.
