/usr/bin/bash: line 20: .venv/Scripts/python.exe: No such file or directory

## Exploratory note `2026-11-26 graph A/B vs official` — clock caveat
- Same-session A/B (alternating, 5 iters): v0.13.1 graph 12.1-12.3 ms vs v0.12.2 16.9-17.4 ms = **-27.5%**, consistent 4/4 pairs. Graph capture confirmed via INFERIO_GRAPH print.
- Official runs are cross-session and clock-confounded: v0.13.1 official 16.30 ms @ 825 MHz vs v0.14.1 official 14.52 ms @ 1410 MHz (clocks.sm snapshot). The A/B is the valid engine-vs-engine comparison (same card, same session); official numbers are recorded for the schema with this caveat.

## Exploratory note `2026-11-26 mma256 A/B` — B=1
- mma256 (256 threads, 128 queries/block, half the blocks) vs mma (128 threads, 64 queries/block), graph on, alternating 5 iters: paired deltas -6.9/-3.1/-3.7/+2.5 ms -> median **-22%** (3/4 pairs). Parity PASS (argmax [308] = golden). Accepted as default (v0.14.1); INFERIO_ATTN=mma restores the old default. Large-batch behavior of mma256 is unmeasured (B=1 sector only).
- Combined campaign so far (same-session A/B chain): v0.12.2 16.9 -> graph 12.25 -> mma256 ~10-12 ms at B=1 (session-clock dependent), i.e. roughly -30..-40% total at B=1.
