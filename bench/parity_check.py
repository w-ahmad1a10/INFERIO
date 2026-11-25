#!/usr/bin/env python3
"""Parity check: engine last-position logits vs the fp32 numpy oracle.

Uses the frozen ids rule (same seed formula) and the golden logits produced by
tools/oracle_gpt2.py. PASS = every row's argmax matches the oracle and the ids
match the frozen rule.
"""
import argparse
import ctypes
import json
import os

import numpy as np

from bench_prefill import InferioStats, VOCAB, SEQ, err_text, ids_for_batch, load_lib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--golden", default="/kaggle/working/golden")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--out", default="temp/results/parity.json")
    args = ap.parse_args()

    ids = ids_for_batch(args.batch)
    golden_ids = np.load(f"{args.golden}/ids_b{args.batch}.npy")
    golden = np.load(f"{args.golden}/last_logits_b{args.batch}.npy")
    ids_match = bool(np.array_equal(ids, golden_ids))

    lib = load_lib(args.lib)
    rc = lib.inferio_init(args.weights.encode(), 0)
    if rc:
        print("INIT FAILED", rc, err_text(lib))
        return 1
    st = InferioStats()
    rc = lib.inferio_prefill(
        ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), args.batch, SEQ,
        ctypes.byref(st))
    if rc:
        print("PREFILL FAILED", rc, err_text(lib))
        return 1
    got = np.zeros((args.batch, VOCAB), dtype=np.float32)
    rc = lib.inferio_copy_last_logits(
        got.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), args.batch, VOCAB)
    if rc:
        print("COPY FAILED", rc, err_text(lib))
        return 1

    arg_got = got.argmax(-1)
    arg_ref = golden.argmax(-1)
    diff = np.abs(got - golden)
    res = {
        "batch": args.batch,
        "ids_match_frozen_rule": ids_match,
        "argmax_equal": bool(np.array_equal(arg_got, arg_ref)),
        "argmax_engine": arg_got.tolist(),
        "argmax_oracle": arg_ref.tolist(),
        "max_abs_logit_diff": float(diff.max()),
        "mean_abs_logit_diff": float(diff.mean()),
        "prefill_ms": st.prefill_ms,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    ok = res["argmax_equal"] and res["ids_match_frozen_rule"]
    print("PARITY:", "PASS" if ok else "FAIL")
    lib.inferio_shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
