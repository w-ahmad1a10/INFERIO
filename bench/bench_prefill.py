#!/usr/bin/env python3
"""FROZEN INFERIO prefill benchmark harness (SPEC v1).

Any change to this file is announced to the user and bumps SPEC_VERSION; the
sha256 of this file is embedded in every result it writes.
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

SPEC_VERSION = "v1"
BATCHES = [1, 4, 16, 32, 64, 128, 256, 512, 1024]
SEQ = 1024
VOCAB = 50257
SEED_BASE = 20261125
WARMUP = 2
ITERS = 5
PHASES = ["embed", "ln", "qkv", "attn", "proj", "fc1", "fc2", "head", "argmax"]


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


def ids_for_batch(batch):
    rng = np.random.default_rng(SEED_BASE + batch)
    return np.ascontiguousarray(rng.integers(0, VOCAB, size=(batch, SEQ), dtype=np.int32))


def load_lib(path):
    lib = ctypes.CDLL(path)
    lib.inferio_version.restype = ctypes.c_char_p
    lib.inferio_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.inferio_prefill.argtypes = [
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(InferioStats)]
    lib.inferio_last_argmax.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    lib.inferio_copy_last_logits.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
    lib.inferio_last_error.restype = ctypes.c_char_p
    return lib


def err_text(lib):
    try:
        return lib.inferio_last_error().decode()
    except Exception:
        return ""


def single_run(args):
    out = {"spec": SPEC_VERSION, "batch": args.batch, "seq": SEQ,
           "harness_sha256": harness_sha256(), "status": "ok", "error": None}
    ids = ids_for_batch(args.batch)
    out["ids_sha256"] = hashlib.sha256(ids.tobytes()).hexdigest()
    lib = load_lib(args.lib)
    rc = lib.inferio_init(args.weights.encode(), 0)
    if rc != 0:
        out["status"] = "oom" if rc == 1 else "error"
        out["error"] = f"init rc={rc} {err_text(lib)}"
        return out
    out["engine_version"] = lib.inferio_version().decode()
    st = InferioStats()
    ids_p = ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    for _ in range(WARMUP):
        rc = lib.inferio_prefill(ids_p, args.batch, SEQ, ctypes.byref(st))
        if rc != 0:
            out["status"] = "oom" if rc == 1 else "error"
            out["error"] = f"warmup rc={rc} {err_text(lib)}"
            lib.inferio_shutdown()
            return out
    lat = []
    for _ in range(ITERS):
        rc = lib.inferio_prefill(ids_p, args.batch, SEQ, ctypes.byref(st))
        if rc != 0:
            out["status"] = "oom" if rc == 1 else "error"
            out["error"] = f"prefill rc={rc} {err_text(lib)}"
            lib.inferio_shutdown()
            return out
        lat.append(st.prefill_ms)
    arg = np.zeros(args.batch, dtype=np.int32)
    rc = lib.inferio_last_argmax(arg.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), args.batch)
    if rc != 0:
        out["status"] = "error"
        out["error"] = f"argmax rc={rc} {err_text(lib)}"
        lib.inferio_shutdown()
        return out
    if args.export_logits:
        lg = np.zeros((args.batch, VOCAB), dtype=np.float32)
        rc = lib.inferio_copy_last_logits(
            lg.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), args.batch, VOCAB)
        if rc == 0:
            os.makedirs(os.path.dirname(os.path.abspath(args.export_logits)), exist_ok=True)
            np.save(args.export_logits, lg)
            if not np.array_equal(lg.argmax(-1), arg):
                out["status"] = "error"
                out["error"] = "argmax/logits mismatch"
    out.update(
        median_ms=statistics.median(lat),
        min_ms=min(lat),
        max_ms=max(lat),
        latencies_ms=lat,
        tokens_per_s=args.batch * SEQ / (statistics.median(lat) / 1000.0),
        peak_bytes=st.peak_bytes,
        weight_bytes=st.weight_bytes,
        group_seqs=st.group_seqs,
        n_groups=st.n_groups,
        argmax_first8=arg[:8].tolist(),
    )
    if os.environ.get("INFERIO_PROFILE") == "1":
        out["phase_ms"] = {name: st.phase_ms[i] for i, name in enumerate(PHASES)}
    lib.inferio_shutdown()
    return out


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


def write_ledger_entry(path, results, note):
    lines = []
    ver = results.get("engine_version") or "?"
    lines.append(f"\n## Run `{results['run_id']}` — engine {ver} "
                 f"(git {results['engine_commit']}) — GPU {results['gpu']}")
    lines.append(f"- spec {SPEC_VERSION}, harness sha256 {results['harness_sha256'][:16]}…")
    lines.append(f"- **Change vs previous run:** {note or '(not specified)'}")
    lines.append(f"- device env: {'; '.join(results['device_env'])}")
    lines.append("")
    lines.append("| B | status | median ms | min ms | max ms | tokens/s | peak GiB | group |")
    lines.append("|---|--------|-----------|--------|--------|----------|----------|-------|")
    for r in results["rows"]:
        if r.get("status") == "ok":
            lines.append(f"| {r['batch']} | ok | {r['median_ms']:.2f} | {r['min_ms']:.2f} | "
                         f"{r['max_ms']:.2f} | {r['tokens_per_s']:.1f} | "
                         f"{r['peak_bytes'] / 2**30:.2f} | {r['group_seqs']} |")
        else:
            lines.append(f"| {r['batch']} | {r['status']} | – | – | – | – | – | – |")
    lines.append("")
    lines.append(f"- only the rows above were measured; nothing inferred. "
                 f"full data: `{results['run_id']}.json`")
    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n")


def run_all(args):
    run_id = time.strftime("%Y%m%d-%H%M%SZ", time.gmtime()) + f"-gpu{args.gpu}"
    results = {
        "spec": SPEC_VERSION,
        "run_id": run_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": args.gpu,
        "engine_commit": args.engine_commit,
        "engine_version": None,
        "harness_sha256": harness_sha256(),
        "weights_dir": args.weights,
        "device_env": nvidia_snapshot(),
        "rows": [],
    }
    for b in BATCHES:
        cmd = [sys.executable, os.path.abspath(__file__), "--single", "--batch", str(b),
               "--lib", args.lib, "--weights", args.weights]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        if args.profile:
            env["INFERIO_PROFILE"] = "1"
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, env=env,
                           timeout=args.child_timeout)
        row = None
        for line in p.stdout.splitlines():
            if line.startswith("RESULT_JSON:"):
                row = json.loads(line[len("RESULT_JSON:"):])
        if row is None:
            row = {"batch": b, "status": "error",
                   "error": f"child failed rc={p.returncode}: {p.stderr[-800:]}"}
        row["wall_s"] = round(time.time() - t0, 2)
        if row.get("engine_version") and not results["engine_version"]:
            results["engine_version"] = row["engine_version"]
        results["rows"].append(row)
        med = row.get("median_ms")
        print(f"B={b:5d} status={row['status']:5s} "
              f"median={med:.2f} ms " if med else f"B={b:5d} status={row['status']} ")
        sys.stdout.flush()

    json_path = os.path.join(args.results_root, "results", run_id + ".json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    ledger_path = os.path.join(args.results_root, "LEDGER.md")
    os.makedirs(args.results_root, exist_ok=True)
    write_ledger_entry(ledger_path, results, args.note)
    print("wrote", json_path)
    print("appended", ledger_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--batch", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--lib", default=os.environ.get("INFERIO_LIB",
                                                    "engine/build/libinferio.so"))
    ap.add_argument("--weights", default=os.environ.get("INFERIO_WEIGHTS",
                                                        "/kaggle/working/weights_gpt2"))
    ap.add_argument("--results-root", default=os.environ.get("INFERIO_RESULTS", "temp"))
    ap.add_argument("--engine-commit", default=os.environ.get("INFERIO_COMMIT", "unknown"))
    ap.add_argument("--note", default="")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--export-logits", default=None)
    ap.add_argument("--child-timeout", type=int, default=1800)
    args = ap.parse_args()

    if args.single:
        res = single_run(args)
        print("RESULT_JSON:" + json.dumps(res))
        return 0 if res["status"] == "ok" else 1
    if args.all:
        run_all(args)
        return 0
    ap.error("one of --single or --all is required")


if __name__ == "__main__":
    raise SystemExit(main())
