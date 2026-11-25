# BATCH 1 OPTIMIZATION — campaign plan (starts 2026-11-26)

Goal: make **batch size 1** prefill as fast as possible. All other batch sizes
are recorded by the frozen harness but are **not targets** — no experiment will
be kept or reverted based on them.

## Baseline (champion to beat)

- Engine v0.12.2, B=1: **13.82 ms = 74,097 tokens/s** (official frozen-harness
  run; note: that run was at 1035 MHz SM clock — re-baseline B=1 on the next
  session's card before the first experiment, same-session A/B).
- Reference: PyTorch eager 24.8 ms (41.3k tok/s); best torch variant 20.5 ms
  (50.0k tok/s).

## Why B=1 is a different problem (measured phase profile, v0.8 B=1)

- Latency-bound, not memory-bound: per-layer working set fits in cache.
- ~1.5–3 ms of the 13.8 ms is kernel-launch gaps (~300 launches × 5–10 µs).
- Attention 5.8 ms at 50% occupancy (register-limited), ~1.2 waves over 40 SMs.
- ~8 ms of small-GEMM serial execution is the compute floor.

## Queued experiments (in order, one per version, A/B measured)

1. **CUDA Graphs** — capture the whole prefill, replay with ~1 launch.
   Biggest single lever (kills the launch gaps). Capture per (batch=1, seq=1024)
   shape; the harness calls prefill repeatedly, so graph replay fits naturally.
2. **Attention retiling for B=1** — fatter blocks / persistent kernel
   (fewer waves, better tail); target the 5.8 ms.
3. **Launch fusion** — merge elementwise chains (bias+GELU already fused;
   look at LN-adjacent ops) to cut the ~300 launches further.
4. If graphs land: re-measure 2+3 on top of the graph version.

## Rules (unchanged)

- Frozen SPEC v1 harness is the scoreboard: the B=1 row of the official run is
  the only number that counts; parity gate must PASS every version.
- Every experiment: same-session A/B vs the current B=1 champion, then an
  official run for the accepted version; ledger entry with mechanism.
- Honesty: null results recorded and reverted with the reason.
- GPU protocol: one process per GPU; B=1 work items on GPU 0 by default.

## Definition of done

A new champion with a measured B=1 speedup over 74,097 tok/s, parity PASS,
official run stored, ledger + README updated, everything pushed.