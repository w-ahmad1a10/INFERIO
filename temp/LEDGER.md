# INFERIO measured ledger (append-only in spirit; rebuilt 2026-11-25)

Every number below was actually run on the Kaggle T4s with the frozen SPEC v1
harness (2 warmup + 5 measured, median). The original file was corrupted by a
notebook-upload artifact (nested JSON-escaping); this rebuild regenerates every
entry verbatim from the result JSONs of record in `temp/results/`.

## Run `20261125-093210Z-gpu0` — engine v0.1.0 (git ac9bfff) — GPU 0
- spec v1, harness sha256 04eb754a522eea33…
- **Change vs previous run:** v0 baseline: first correct engine (cuBLAS fp16 GEMMs + naive online-softmax attention, phase timers on)
- device env: 0, Tesla T4, 580.178.04, 15360 MiB, 885 MHz, 45; 1, Tesla T4, 580.178.04, 15360 MiB, 300 MHz, 41

| B | status | median ms | min ms | max ms | tokens/s | peak GiB | group |
|---|--------|-----------|--------|--------|----------|----------|-------|
| 1 | ok | 45.61 | 29.39 | 45.62 | 22451.7 | 0.25 | 1 |
| 4 | ok | 94.78 | 94.19 | 95.14 | 43214.2 | 0.29 | 4 |
| 16 | ok | 352.40 | 350.84 | 352.86 | 46493.3 | 0.47 | 16 |
| 32 | ok | 730.40 | 711.07 | 732.37 | 44863.2 | 0.71 | 32 |
| 64 | ok | 1484.00 | 1469.73 | 1508.51 | 44161.7 | 1.19 | 64 |
| 128 | ok | 3167.93 | 3138.95 | 3180.56 | 41374.7 | 2.14 | 128 |
| 256 | ok | 6811.39 | 6658.28 | 7113.49 | 38486.1 | 2.18 | 128 |
| 512 | ok | 14545.06 | 14360.69 | 14838.36 | 36045.8 | 2.25 | 128 |
| 1024 | ok | 29235.99 | 29155.56 | 29279.44 | 35865.9 | 2.40 | 128 |

- only the rows above were measured; nothing inferred. full data: `20261125-093210Z-gpu0.json`

## Run `20261125-101245Z-gpu0` — engine v0.2.0 (git c39d5c3) — GPU 0
- spec v1, harness sha256 04eb754a522eea33…
- **Change vs previous run:** v0.2.0: cuBLASLt per-shape algo plans for all GEMMs + vectorized LayerNorm (half2); still naive attention
- device env: 0, Tesla T4, 580.178.04, 15360 MiB, 795 MHz, 39; 1, Tesla T4, 580.178.04, 15360 MiB, 300 MHz, 41

| B | status | median ms | min ms | max ms | tokens/s | peak GiB | group |
|---|--------|-----------|--------|--------|----------|----------|-------|
| 1 | ok | 41.01 | 30.06 | 49.83 | 24969.0 | 0.34 | 1 |
| 4 | ok | 92.56 | 92.50 | 92.93 | 44254.3 | 0.39 | 4 |
| 16 | ok | 344.70 | 342.02 | 347.49 | 47530.8 | 0.56 | 16 |
| 32 | ok | 711.64 | 705.64 | 711.93 | 46045.9 | 0.80 | 32 |
| 64 | ok | 1455.73 | 1441.13 | 1460.85 | 45019.5 | 1.28 | 64 |
| 128 | ok | 3017.08 | 2958.17 | 3048.89 | 43443.3 | 2.24 | 128 |
| 256 | ok | 6530.52 | 6325.73 | 6812.54 | 40141.3 | 2.27 | 128 |
| 512 | ok | 14415.30 | 14225.09 | 14868.50 | 36370.2 | 2.35 | 128 |
| 1024 | ok | 28929.88 | 28847.16 | 29132.26 | 36245.4 | 2.49 | 128 |

- only the rows above were measured; nothing inferred. full data: `20261125-101245Z-gpu0.json`

## Run `20261125-102615Z-gpu0` — engine v0.3.0 (git 02195be) — GPU 0
- spec v1, harness sha256 04eb754a522eea33…
- **Change vs previous run:** v0.3.0: tiled online-softmax attention (exp2, per-tile rescale, smem score staging) + tanh.approx GELU + half2 bias kernels — REGRESSION, reverted as default (33 KB smem -> 1 CTA/SM); kept as INFERIO_ATTN=tiled option
- device env: 0, Tesla T4, 580.178.04, 15360 MiB, 690 MHz, 44; 1, Tesla T4, 580.178.04, 15360 MiB, 300 MHz, 41

| B | status | median ms | min ms | max ms | tokens/s | peak GiB | group |
|---|--------|-----------|--------|--------|----------|----------|-------|
| 1 | ok | 37.53 | 37.15 | 71.56 | 27286.6 | 0.34 | 1 |
| 4 | ok | 125.99 | 123.89 | 126.55 | 32510.5 | 0.39 | 4 |
| 16 | ok | 486.34 | 486.03 | 490.16 | 33688.6 | 0.56 | 16 |
| 32 | ok | 970.70 | 969.39 | 973.04 | 33757.0 | 0.80 | 32 |
| 64 | ok | 2022.26 | 2010.11 | 2036.90 | 32407.3 | 1.28 | 64 |
| 128 | ok | 4253.32 | 4239.09 | 4284.93 | 30816.4 | 2.24 | 128 |
| 256 | ok | 9358.05 | 8959.36 | 9702.50 | 28012.7 | 2.27 | 128 |
| 512 | ok | 18444.62 | 18419.31 | 18545.03 | 28425.0 | 2.35 | 128 |
| 1024 | ok | 36897.33 | 36849.99 | 36909.60 | 28418.7 | 2.49 | 128 |

- only the rows above were measured; nothing inferred. full data: `20261125-102615Z-gpu0.json`

## Run `20261125-111524Z-gpu0` — engine v0.7.0 (git f54a942) — GPU 0
- spec v1, harness sha256 04eb754a522eea33…
- **Change vs previous run:** v0.7.0: PTX mma.sync tensor-core attention (default), bias epilogue in GEMMs, tanh.approx GELU, vectorized LN/bias kernels
- device env: 0, Tesla T4, 580.178.04, 15360 MiB, 585 MHz, 51; 1, Tesla T4, 580.178.04, 15360 MiB, 300 MHz, 48

| B | status | median ms | min ms | max ms | tokens/s | peak GiB | group |
|---|--------|-----------|--------|--------|----------|----------|-------|
| 1 | ok | 14.25 | 14.20 | 14.32 | 71859.4 | 0.34 | 1 |
| 4 | ok | 50.96 | 48.92 | 51.88 | 80382.6 | 0.39 | 4 |
| 16 | ok | 195.47 | 192.15 | 196.79 | 83817.7 | 0.56 | 16 |
| 32 | ok | 405.38 | 404.93 | 407.72 | 80833.4 | 0.80 | 32 |
| 64 | ok | 833.91 | 832.50 | 840.36 | 78588.9 | 1.28 | 64 |
| 128 | ok | 1825.99 | 1792.36 | 1845.03 | 71781.3 | 2.24 | 128 |
| 256 | ok | 3877.67 | 3785.01 | 3899.82 | 67603.6 | 2.27 | 128 |
| 512 | ok | 8809.05 | 8569.33 | 9007.04 | 59517.0 | 2.35 | 128 |
| 1024 | ok | 17404.10 | 17319.71 | 17496.65 | 60248.8 | 2.49 | 128 |

- only the rows above were measured; nothing inferred. full data: `20261125-111524Z-gpu0.json`

## Run `20261125-120134Z-gpu0` — engine v0.8.0 (git 1d16163) — GPU 0
- spec v1, harness sha256 04eb754a522eea33…
- **Change vs previous run:** v0.8.0: MMA attention with V tile staged in shared memory
- device env: 0, Tesla T4, 580.178.04, 15360 MiB, 915 MHz, 39; 1, Tesla T4, 580.178.04, 15360 MiB, 300 MHz, 41

| B | status | median ms | min ms | max ms | tokens/s | peak GiB | group |
|---|--------|-----------|--------|--------|----------|----------|-------|
| 1 | ok | 15.55 | 15.54 | 15.60 | 65841.5 | 0.34 | 1 |
| 4 | ok | 48.27 | 47.09 | 50.92 | 84853.6 | 0.39 | 4 |
| 16 | ok | 181.96 | 179.69 | 185.73 | 90039.9 | 0.56 | 16 |
| 32 | ok | 374.15 | 369.79 | 377.77 | 87579.1 | 0.80 | 32 |
| 64 | ok | 749.49 | 748.70 | 750.61 | 87441.1 | 1.28 | 64 |
| 128 | ok | 1582.77 | 1565.42 | 1593.11 | 82811.8 | 2.24 | 128 |
| 256 | ok | 3314.22 | 3244.49 | 3379.21 | 79096.8 | 2.27 | 128 |
| 512 | ok | 7524.12 | 7144.20 | 7770.05 | 69680.9 | 2.35 | 128 |
| 1024 | ok | 16525.37 | 16227.87 | 17446.72 | 63452.5 | 2.49 | 128 |

- only the rows above were measured; nothing inferred. full data: `20261125-120134Z-gpu0.json`

## Run `20261125-164550Z-gpu0` — engine v0.9.0 (git 3cc672f) — GPU 0
- spec v1, harness sha256 04eb754a522eea33…
- **Change vs previous run:** v0.9.0: two-pass LayerNorm (sum+sqsum in one pass, var=E[x^2]-mean^2), default group 64 (exploratory sweep: +9% B=256, +2.5% B=1024 vs group 128; same-session A/B: -6% at B=1024, wash at B<=256)
- device env: 0, Tesla T4, 580.178.04, 15360 MiB, 900 MHz, 71; 1, Tesla T4, 580.178.04, 15360 MiB, 300 MHz, 33

| B | status | median ms | min ms | max ms | tokens/s | peak GiB | group |
|---|--------|-----------|--------|--------|----------|----------|-------|
| 1 | ok | 15.82 | 15.78 | 15.82 | 64745.6 | 0.34 | 1 |
| 4 | ok | 53.87 | 52.89 | 54.59 | 76034.6 | 0.39 | 4 |
| 16 | ok | 210.50 | 209.78 | 212.37 | 77833.8 | 0.56 | 16 |
| 32 | ok | 432.35 | 427.58 | 434.81 | 75789.9 | 0.80 | 32 |
| 64 | ok | 892.88 | 888.97 | 901.27 | 73398.1 | 1.28 | 64 |
| 128 | ok | 1880.33 | 1831.10 | 1912.76 | 69707.0 | 1.30 | 64 |
| 256 | ok | 3714.45 | 3641.71 | 3792.78 | 70574.1 | 1.34 | 64 |
| 512 | ok | 7039.34 | 7007.68 | 7070.59 | 74479.7 | 1.41 | 64 |
| 1024 | ok | 14291.41 | 14266.45 | 14498.40 | 73371.1 | 1.56 | 64 |

- only the rows above were measured; nothing inferred. full data: `20261125-164550Z-gpu0.json`

## Run `20261125-120222Z-pytorch` — PyTorch eager GPT-2 fp16 (no torch.compile) — GPU 1
- torch 2.11.0+cu130, harness sha256 94c9a178f89810b5…
- **Note:** PyTorch eager GPT-2 fp16 reference (no torch.compile) on identical frozen inputs — reference baseline only, not the engine.
- device env: 0, Tesla T4, 580.178.04, 15360 MiB, 630 MHz, 69; 1, Tesla T4, 580.178.04, 15360 MiB, 1500 MHz, 57

| B | status | median ms | min ms | max ms | tokens/s |
|---|--------|-----------|--------|--------|----------|
| 1 | ok | 24.82 | 24.73 | 24.83 | 41254.7 |
| 4 | ok | 61.29 | 60.74 | 61.57 | 66833.8 |
| 16 | ok | 242.21 | 240.83 | 244.77 | 67644.7 |
| 32 | ok | 487.39 | 486.84 | 488.36 | 67231.7 |
| 64 | ok | 972.09 | 960.56 | 973.40 | 67417.9 |
| 128 | ok | 1993.61 | 1968.54 | 2001.19 | 65746.1 |
| 256 | oom | – | – | – | – |
| 512 | oom | – | – | – | – |
| 1024 | oom | – | – | – | – |

## Exploratory note `2026-11-25 heur sweep` — engine v0.8.0 — GPU 0 + 1
- **What was tried:** cuBLASLt heuristic index 0–7 (INFERIO_HEUR) at B=256, quick probe (1 warmup + 1 measured, not the frozen harness).
- **Result (median ms):** h0 3182.9 | h1 3255.4 | h2 3506.1 | h3 3371.1 | h4 3167.3 | h5 3223.4 | h6 3226.5 | h7 3244.3.
- **Verdict:** spread ≤ 10%, h0 (default) within noise of best (h4 is 0.5% faster). No change made; GEMM heuristic tuning exhausted for now — remaining GEMM headroom is not in algo selection.
- These are exploratory numbers; the official numbers remain the frozen-harness rows above.

## Exploratory note `2026-11-25 group sweep` — engine v0.8.0 — GPU 0 + 1
- **What was tried:** INFERIO_GROUP in {64,128,256,512} at B=256 and B=1024, quick probe; then same-GPU A/B (alternating, 3 iters) for 64 vs 128.
- **Result:** group 64 best. Same-GPU A/B: B=256 ~-9%, B=1024 ~-2.5% vs group 128; first probe (cross-GPU) showed -15/-17% (inflated by session drift).
- **Verdict:** group 64 accepted as new default in v0.9.0; effect concentrated in the QKV GEMM phase.

## Exploratory note `2026-11-25 v0.8 vs v0.9 A/B` — GPU 0
- **What was done:** both binaries built from the same session, alternated v08/v09/v08/v09, quick probe (1 warmup + 3 iters), B=64,256,1024.
- **Result (median ms):** B=1024: v08 15697.5/15173.2 vs v09 14245.6/14318.8 (**-6%**, consistent). B=256: 3399/3855 vs 3567/3603 (wash). B=64: 788/908 vs 906/893 (wash).
- **Verdict:** v0.9.0 accepted: never worse outside noise, -6% at B=1024 (target), parity PASS (max logit diff 0.115, improved from 0.247). The official-run small-batch regression (+12..19%) is NOT reproducible in A/B — session noise, recorded as such.
- Phase effect at B=1024: qkv -26%, fc2 -16%, ln -17%, attn -2%, fc1 +6%, proj +20% (proj regression not investigated yet).
## Run `20261125-180521Z-gpu0` — engine v0.12.2 (git 8172228) — GPU 0
- spec v1, harness sha256 04eb754a522eea33…
- **Change vs previous run:** v0.12.2: attention score scale folded into Q at load (attn phase -2..3%); default config otherwise identical to v0.9.0 (group 64, mma attention)
- device env: 0, Tesla T4, 580.178.04, 15360 MiB, 1035 MHz, 62; 1, Tesla T4, 580.178.04, 15360 MiB, 300 MHz, 34

| B | status | median ms | min ms | max ms | tokens/s | peak GiB | group |
|---|--------|-----------|--------|--------|----------|----------|-------|
| 1 | ok | 13.82 | 13.69 | 13.85 | 74118.0 | 0.34 | 1 |
| 4 | ok | 49.77 | 48.91 | 52.26 | 82295.7 | 0.39 | 4 |
| 16 | ok | 196.20 | 195.50 | 197.37 | 83504.5 | 0.56 | 16 |
| 32 | ok | 405.84 | 404.24 | 407.27 | 80740.3 | 0.80 | 32 |
| 64 | ok | 841.32 | 832.39 | 857.90 | 77896.4 | 1.28 | 64 |
| 128 | ok | 1743.51 | 1718.77 | 1749.30 | 75177.1 | 1.30 | 64 |
| 256 | ok | 3765.97 | 3642.42 | 3879.76 | 69608.6 | 1.34 | 64 |
| 512 | ok | 7060.36 | 6999.94 | 7269.37 | 74258.0 | 1.41 | 64 |
| 1024 | ok | 14321.53 | 14248.13 | 14534.72 | 73216.8 | 1.56 | 64 |

- only the rows above were measured; nothing inferred. full data: `20261125-180521Z-gpu0.json`

## Exploratory note `2026-11-25 v0.10 K staging` — engine v0.10.0 — GPU 0
- **What was tried:** stage the 64-key K tile in shared memory (buffer reused with V between phases, occupancy-neutral) to remove 4x redundant global K reads.
- **Result:** null. Attention phase +2..3% (two extra __syncthreads per tile eat the smem savings); everything else a wash. Reverted (commit 54a0f95).
- **Verdict:** K global reads are L2-cached and cheap; instruction overhead is not the attention bottleneck.

## Exploratory note `2026-11-25 v0.11 launch_bounds` — engine v0.11.0 — GPU 0
- **What was tried:** __launch_bounds__(128,5) on attn_mma_kernel to fit 5 blocks/SM (from 4) and feed the SFU better.
- **Result:** clearly worse — B=1024 +5..9%, attention +5..10% (register cap to 102 causes spills). Reverted (commit 0ea814b).
- **Verdict:** the register/occupancy lever needs a kernel restructure (fewer live registers), not a cap.

## Exploratory note `2026-11-25 mma128 variant` — engine v0.12.0/1 — GPU 0 + 1
- **What was tried:** attn_mma128 (INFERIO_ATTN=mma128 option): 128-key tiles, halving V loads and syncs per key; smem 18 KB -> 3 blocks/SM. First version had a stack overflow (c/pa sized 8 indexed to 16 -> NaN, parity FAIL); fixed in v0.12.1.
- **Result (fixed, parity PASS, max diff 0.166):** attention 25..28% SLOWER (occupancy 4->3 blocks/SM hurts more than the saved loads — the v0.3 lesson repeats); total time a wash. Kept as an option, not default.
- **Verdict:** 64-key tiles are the right size for this kernel shape on SM75.

## Exploratory note `2026-11-25 group tweak + scale fold` — GPU 0
- **Group tweak (128 for B<=256, 64 above):** A/B at B=256 showed group 128 is 5..6% SLOWER than 64 (v12 3774/3788 vs v09 3606/3572) — group 64 wins at 256 too; the official v0.9-run mid-batch regression was session noise. Tweak reverted; group 64 everywhere.
- **Scale fold (v0.12.2):** Q pre-scaled by 1/sqrt(64)*log2(e) at load, removing 32 multiplies per thread per tile. A/B at B=1024: attention phase -2..3% consistently, total a wash. Accepted (parity PASS, max diff 0.237 vs 0.115 — extra fp16 rounding, argmax identical).
- **Official v0.12.2 run:** within noise of v0.9.0 at every batch (B=1024: 14321.5 vs 14291.4 ms). Caveat: this run was at 1035 MHz SM clock vs 900-915 MHz for v0.9.0's official run, so cross-session deltas are clock-inflated; the same-session A/B (wash overall, attention -2..3%) is the valid comparison. Champions: v0.9.0/v0.12.2 config.

## Run `20261125-182650Z-pytorch-eager-nocache` — PyTorch eager-nocache GPT-2 fp16 — GPU 1
- torch 2.11.0+cu130, harness sha256 659b0ef99014289a…
- **Note:** PyTorch eager fp16, use_cache=False (no KV allocation), identical frozen inputs — reference baseline only, not the engine.
- device env: 0, Tesla T4, 580.178.04, 15360 MiB, 300 MHz, 39; 1, Tesla T4, 580.178.04, 15360 MiB, 1275 MHz, 67

| B | status | median ms | min ms | max ms | tokens/s |
|---|--------|-----------|--------|--------|----------|
| 1 | ok | 23.75 | 22.90 | 23.80 | 43114.6 |
| 4 | ok | 57.49 | 56.97 | 58.21 | 71241.2 |
| 16 | ok | 225.99 | 225.28 | 230.37 | 72497.9 |
| 32 | ok | 454.32 | 452.93 | 460.02 | 72125.7 |
| 64 | ok | 916.64 | 912.09 | 923.78 | 71495.6 |
| 128 | ok | 1934.46 | 1887.37 | 1950.26 | 67756.3 |
| 256 | ok | 4197.23 | 4162.53 | 4334.41 | 62456.5 |
| 512 | oom | – | – | – | – |
| 1024 | oom | – | – | – | – |

## Run `20261125-182800Z-pytorch-compile` — PyTorch compile GPT-2 fp16 — GPU 1
- torch 2.11.0+cu130, harness sha256 659b0ef99014289a…
- **Note:** PyTorch torch.compile(transformer, dynamic=True) fp16, use_cache=False — reference baseline only, not the engine.
- device env: 0, Tesla T4, 580.178.04, 15360 MiB, 300 MHz, 37; 1, Tesla T4, 580.178.04, 15360 MiB, 1200 MHz, 71

| B | status | median ms | min ms | max ms | tokens/s |
|---|--------|-----------|--------|--------|----------|
| 1 | ok | 20.55 | 16.63 | 20.67 | 49839.3 |
| 4 | ok | 57.90 | 56.70 | 60.23 | 70741.4 |
| 16 | ok | 242.54 | 241.68 | 242.70 | 67552.7 |
| 32 | ok | 486.52 | 485.02 | 489.65 | 67352.2 |
| 64 | ok | 1006.96 | 995.27 | 1018.92 | 65083.3 |
| 128 | ok | 2209.43 | 2179.42 | 2263.69 | 59323.9 |
| 256 | ok | 5211.07 | 5054.00 | 5544.67 | 50305.2 |
| 512 | oom | – | – | – | – |
| 1024 | oom | – | – | – | – |

## Run `20261125-183113Z-pytorch-compile` — PyTorch compile GPT-2 fp16 — GPU 1
- torch 2.11.0+cu130, harness sha256 dc5707627d5a8f3a…
- **Note:** torch.compile specialized (no dynamic), screening subset — reference baseline only, not the engine.
- device env: 0, Tesla T4, 580.178.04, 15360 MiB, 300 MHz, 38; 1, Tesla T4, 580.178.04, 15360 MiB, 1080 MHz, 80

| B | status | median ms | min ms | max ms | tokens/s |
|---|--------|-----------|--------|--------|----------|
| 16 | ok | 241.82 | 241.21 | 242.21 | 67753.9 |
| 64 | ok | 997.38 | 988.01 | 1010.77 | 65707.9 |
| 256 | ok | 5003.26 | 4726.01 | 5659.44 | 52394.6 |

