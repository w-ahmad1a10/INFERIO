# INFERIO B=1 frozen benchmark — SPEC b1-v1 (2026-11-26)

This file and `bench_b1.py` are **FROZEN**. Any change is announced to the user
first, bumps `B1_SPEC_VERSION`, and changes the harness hash embedded in every
future result. The main 9-batch SPEC (`bench/SPEC.md`) is untouched by this
campaign; this sector optimizes **batch size 1 only**.

## Workload

- Model: GPT-2 small (124M), 1024 tokens, **batch size 1 only** — no other
  batch size exists in this test.
- Inputs (frozen rule, identical to the main spec at B=1):
  `rng = numpy.random.default_rng(20261125 + 1); ids = rng.integers(0, 50257,
  size=(1, 1024), dtype=int32)` — ids sha256 `ed44f8e2…` recorded per run.
- Metric: engine `prefill_ms` (CUDA events: ids H2D → last-position fp32
  logits), 2 warmup + 5 measured, median / min / max.
- `tokens_per_s = 1024 / (median_ms/1000)`.
- OOM is recorded as a result row, never a crash.

## Correctness gate

Parity vs the fp32 numpy oracle (`tools/oracle_gpt2.py`, batch 1): argmax of
the single row must match; max-abs logit diff recorded. A version that fails
parity cannot be accepted regardless of speed.

## Engine contract

Same C ABI as the main harness (`inferio_init` / `inferio_prefill(ids, 1,
1024, &stats)` / `inferio_last_argmax` / `inferio_shutdown`). Internal freedom
(group size, graphs, streams) is unrestricted as long as semantics hold.

## Result schema (`b1/results/<run_id>.json`)

```
run_id, timestamp_utc, engine_version, engine_commit, harness_sha256,
device_env, rows[0]: {batch:1, status, median_ms, min_ms, max_ms,
latencies_ms, tokens_per_s, peak_bytes, ids_sha256, argmax_first8, phase_ms}
```

## Recording rules

Every executed run appends to `b1/LEDGER.md`: what changed vs the previous
B=1 run, device snapshot, the row, verdict. Numbers not measured are marked
"(not measured)" and never written as if measured.