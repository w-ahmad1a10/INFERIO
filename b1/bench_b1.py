#!/usr/bin/env python3
"""FROZEN INFERIO B=1-only benchmark harness (SPEC b1-v1).

Batch size 1 only — no other batch size exists here. Any change is announced
to the user and bumps B1_SPEC_VERSION; the sha256 of this file is embedded in
every result it writes.
"""
import argparse
import ctypes
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np

B1_SPEC_VERSION = "b1-v2"
BATCH = 1
SEQ = 1024
VOCAB = 50257
SEED_BASE = 20261125
WARMUP = 2
ITERS = 5
PHASES = ["embed", "ln", "qkv", "attn", "proj", "fc1", "fc2", "head", "argmax"]
# b1-v2: MFU layer (same convention as SPEC v2).
FLOPS_PER_TOKEN = 207.7e6
SM_COUNT = 40
FLOPS_PER_CYCLE = 1024


def sample_clock_mhz():
    try:
        # nvidia-smi filters its listing by CUDA_VISIBLE_DEVICES, so the
        # child's device is always the first visible line.
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,clocks.sm", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        lines = r.stdout.strip().splitlines()
        return int(lines[0].split(",")[1].strip().split()[0])
    except Exception as e:
        import sys as _s
        print(f"INFERIO_CLOCK sampling failed: {e}", file=_s.stderr)
        return 0


class InferioStats(ctypes.Structure):
    _fields_ = [
        ("prefill_ms", ctypes.c_double),
        ("peak_bytes", ctypes.c_longlong),
        ("weight_bytes", ctypes.c_longlong),
        ("batch", ctypes.c_int64),
        ("seqlen", ctypes.c_int64),
        ("group_seqs", ctypes.c_int64),
        ("n_groups", ctypes.c_int64),
        ("rc", ctypes.c_int),
        ("phase_ms", ctypes.c_double * len(PHASES)),
    ]


def harness_sha256():
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def ids_for_batch1():
    rng = np.random.default_rng(SEED_BASE + 1)
    return np.ascontiguousarray(rng.integers(0, VOCAB, size=(1, SEQ), dtype=np.int32))


def load_lib(path):
    lib = ctypes.CDLL(path)
    lib.inferio_version.restype = ctypes.c_char_p
    lib.inferio_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.inferio_prefill.argtypes = [
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(InferioStats)]
    lib.inferio_last_argmax.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    lib.inferio_last_error.restype = ctypes.c_char_p
    return lib


def err_text(lib):
    try:
        return lib.inferio_last_error().decode()
    except Exception:
        return ""


def nvidia_snapshot():
    try:
        r = subprocess.run(
            ["bash", "-lc",
             "nvidia-smi --query-gpu=index,name,driver_version,memory.total,clocks.sm,temperature.gpu "
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        return [l for l in r.stdout.strip().splitlines() if l] or [r.stderr.strip()]
    except Exception as e:
        return [f"nvidia-smi failed: {e}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=os.environ.get("INFERIO_LIB",
                                                    "engine/build/libinferio.so"))
    ap.add_argument("--weights", default=os.environ.get("INFERIO_WEIGHTS",
                                                        "/kaggle/working/weights_gpt2"))
    ap.add_argument("--results-root", default="b1")
    ap.add_argument("--engine-commit", default=os.environ.get("INFERIO_COMMIT", "unknown"))
    ap.add_argument("--note", default="")
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()

    run_id = time.strftime("%Y%m%d-%H%M%SZ", time.gmtime()) + "-b1"
    ids = ids_for_batch1()
    out = {"spec": B1_SPEC_VERSION, "batch": BATCH, "seq": SEQ,
           "harness_sha256": harness_sha256(), "status": "ok", "error": None,
           "ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest()}
    lib = load_lib(args.lib)
    rc = lib.inferio_init(args.weights.encode(), 0)
    if rc != 0:
        out["status"] = "oom" if rc == 1 else "error"
        out["error"] = f"init rc={rc} {err_text(lib)}"
    else:
        out["engine_version"] = lib.inferio_version().decode()
        st = InferioStats()
        ids_p = ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        ok = True
        for _ in range(WARMUP):
            rc = lib.inferio_prefill(ids_p, BATCH, SEQ, ctypes.byref(st))
            if rc != 0:
                out["status"] = "oom" if rc == 1 else "error"
                out["error"] = f"warmup rc={rc} {err_text(lib)}"
                ok = False
                break
        lat = []
        if ok:
            for _ in range(ITERS):
                rc = lib.inferio_prefill(ids_p, BATCH, SEQ, ctypes.byref(st))
                if rc != 0:
                    out["status"] = "oom" if rc == 1 else "error"
                    out["error"] = f"prefill rc={rc} {err_text(lib)}"
                    ok = False
                    break
                lat.append(st.prefill_ms)
        if ok:
            clk = sample_clock_mhz()
            peak_tflops = SM_COUNT * FLOPS_PER_CYCLE * (clk * 1e6) / 1e12 if clk else 0.0
            achieved_tflops = (BATCH * SEQ * FLOPS_PER_TOKEN) / (statistics.median(lat) / 1000.0) / 1e12
            mfu = (achieved_tflops / peak_tflops) if peak_tflops else 0.0
            arg = np.zeros(BATCH, dtype=np.int32)
            rc = lib.inferio_last_argmax(
                arg.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), BATCH)
            if rc != 0:
                out["status"] = "error"
                out["error"] = f"argmax rc={rc} {err_text(lib)}"
            else:
                out.update(
                    median_ms=statistics.median(lat),
                    min_ms=min(lat),
                    max_ms=max(lat),
                    latencies_ms=lat,
                    tokens_per_s=BATCH * SEQ / (statistics.median(lat) / 1000.0),
                    peak_bytes=st.peak_bytes,
                    weight_bytes=st.weight_bytes,
                    group_seqs=st.group_seqs,
                    argmax_first8=arg[:8].tolist(),
                    clock_sm_mhz=clk,
                    peak_tflops=peak_tflops,
                    achieved_tflops=achieved_tflops,
                    mfu=mfu,
                )
                if args.profile or os.environ.get("INFERIO_PROFILE") == "1":
                    out["phase_ms"] = {name: st.phase_ms[i] for i, name in enumerate(PHASES)}
        lib.inferio_shutdown()

    results = {
        "run_id": run_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine_commit": args.engine_commit,
        "engine_version": out.get("engine_version"),
        "harness_sha256": harness_sha256(),
        "weights_dir": args.weights,
        "device_env": nvidia_snapshot(),
        "note": args.note,
        "rows": [out],
    }
    os.makedirs(os.path.join(args.results_root, "results"), exist_ok=True)
    path = os.path.join(args.results_root, "results", run_id + ".json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    r = out
    lines = [
        f"\n## B=1 run `{run_id}` — engine {results['engine_version']} "
        f"(git {results['engine_commit']})",
        f"- spec {B1_SPEC_VERSION}, harness sha256 {results['harness_sha256'][:16]}…",
        f"- **Change vs previous B=1 run:** {args.note or '(not specified)'}",
        f"- device env: {'; '.join(results['device_env'])}",
        "",
    ]
    if r["status"] == "ok":
        lines.append(f"- B=1: **{r['median_ms']:.2f} ms = {r['tokens_per_s']:.0f} tok/s** "
                     f"(min {r['min_ms']:.2f} / max {r['max_ms']:.2f})")
        if "mfu" in r:
            lines.append(f"- clock {r['clock_sm_mhz']} MHz | peak {r['peak_tflops']:.1f} TF/s | "
                         f"achieved {r['achieved_tflops']:.2f} TF/s | **MFU {r['mfu']*100:.1f}%**")
    else:
        lines.append(f"- B=1: {r['status']} — {r['error']}")
    lines.append(f"- only the row above was measured; nothing inferred. "
                 f"full data: `{run_id}.json`")
    with open(os.path.join(args.results_root, "LEDGER.md"), "a") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", path)
    print("appended", os.path.join(args.results_root, "LEDGER.md"))
    return 0 if out["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())