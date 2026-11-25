#!/usr/bin/env python3
"""Fast exploratory probe (NOT the frozen benchmark): a few prefills per batch
size, printing medians and phase times. Used to screen variants cheaply before
spending a full frozen-harness run."""
import argparse
import ctypes
import json
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
from bench_prefill import SEQ, InferioStats, err_text, ids_for_batch, load_lib  # noqa: E402

PHASES = ["embed", "ln", "qkv", "attn", "proj", "fc1", "fc2", "head", "argmax"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--batches", default="64,256")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=2)
    a = ap.parse_args()

    lib = load_lib(a.lib)
    rc = lib.inferio_init(a.weights.encode(), 0)
    if rc:
        print("init failed", rc, err_text(lib))
        return 1
    out = {"engine": lib.inferio_version().decode(), "results": {}}
    st = InferioStats()
    for b in [int(x) for x in a.batches.split(",")]:
        ids = ids_for_batch(b)
        p = ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        for _ in range(a.warmup):
            if lib.inferio_prefill(p, b, SEQ, ctypes.byref(st)):
                print("warmup failed", err_text(lib))
                return 1
        ts = []
        for _ in range(a.iters):
            if lib.inferio_prefill(p, b, SEQ, ctypes.byref(st)):
                print("prefill failed", err_text(lib))
                return 1
            ts.append(st.prefill_ms)
        med = statistics.median(ts)
        phases = {name: st.phase_ms[i] for i, name in enumerate(PHASES)}
        out["results"][b] = {"median_ms": med, "min_ms": min(ts),
                             "phases": phases, "group": st.group_seqs}
        phase_str = " ".join(f"{k}={v:.0f}" for k, v in phases.items() if v > 0.5)
        print(f"B={b} median={med:.1f}ms group={st.group_seqs} {phase_str}")
    lib.inferio_shutdown()
    print("QUICK_JSON:" + json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
