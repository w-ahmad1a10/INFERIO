#!/usr/bin/env python3
"""PyTorch eager reference baseline for the frozen INFERIO benchmark (SPEC v1).

Plain HuggingFace GPT-2 in fp16 on one GPU — NO torch.compile, NO custom kernels.
This is a reference point for speed comparisons, not part of the engine.
Same frozen inputs, batch sizes, warmup/measure counts and metric window as
bench/bench_prefill.py.
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
from bench_prefill import (BATCHES, ITERS, SEQ, WARMUP, ids_for_batch,  # noqa: E402
                           nvidia_snapshot)

import torch  # noqa: E402

# MFU layer (SPEC v2 convention): 207.7 MFLOP/token, T4 40 SMs x 1024 flops/cycle.
FLOPS_PER_TOKEN = 207.7e6
SM_COUNT = 40
FLOPS_PER_CYCLE = 1024


def sample_clock_mhz():
    try:
        phys = int((os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0").strip())
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,clocks.sm", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        lines = r.stdout.strip().splitlines()
        return int(lines[phys].split(",")[1].strip().split()[0])
    except Exception:
        return 0


def sha_self():
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="temp")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    from transformers import GPT2LMHeadModel

    dev = torch.device("cuda")
    print("torch", torch.__version__, "| cuda", torch.version.cuda, "|",
          torch.cuda.get_device_name(0))
    model = GPT2LMHeadModel.from_pretrained("gpt2").half().to(dev).eval()
    torch.set_grad_enabled(False)
    wte_t = model.transformer.wte.weight.t().contiguous()  # [768, 50257]

    run_id = time.strftime("%Y%m%d-%H%M%SZ", time.gmtime()) + "-pytorch"
    rows = []
    for b in BATCHES:
        ids = ids_for_batch(b)
        ids_sha = hashlib.sha256(ids.tobytes()).hexdigest()

        def one():
            st = torch.cuda.Event(enable_timing=True)
            en = torch.cuda.Event(enable_timing=True)
            st.record()
            ids_t = torch.from_numpy(ids).to(dev)
            h = model.transformer(ids_t).last_hidden_state
            logits = h[:, -1, :] @ wte_t
            en.record()
            en.synchronize()
            return st.elapsed_time(en), logits

        try:
            for _ in range(WARMUP):
                one()
            ts = []
            for _ in range(ITERS):
                ms, logits = one()
                ts.append(ms)
            arg = logits.argmax(-1).cpu().numpy()
            clk = sample_clock_mhz()
            peak_tflops = SM_COUNT * FLOPS_PER_CYCLE * (clk * 1e6) / 1e12 if clk else 0.0
            achieved_tflops = (b * SEQ * FLOPS_PER_TOKEN) / (statistics.median(ts) / 1000.0) / 1e12
            mfu = (achieved_tflops / peak_tflops) if peak_tflops else 0.0
            rows.append({"batch": b, "status": "ok", "median_ms": statistics.median(ts),
                         "min_ms": min(ts), "max_ms": max(ts), "latencies_ms": ts,
                         "tokens_per_s": b * SEQ / (statistics.median(ts) / 1000.0),
                         "ids_sha256": ids_sha, "argmax_first8": arg[:8].tolist(),
                         "clock_sm_mhz": clk, "peak_tflops": peak_tflops,
                         "achieved_tflops": achieved_tflops, "mfu": mfu})
            print(f"B={b} ok median={statistics.median(ts):.1f} ms")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            rows.append({"batch": b, "status": "oom", "ids_sha256": ids_sha})
            print(f"B={b} oom")
        del ids
        torch.cuda.empty_cache()

    results = {
        "run_id": run_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine": f"pytorch-eager-{torch.__version__} (HF GPT-2, fp16, no torch.compile)",
        "harness_sha256": sha_self(),
        "device_env": nvidia_snapshot(),
        "note": args.note,
        "rows": rows,
    }
    os.makedirs(os.path.join(args.results_root, "results"), exist_ok=True)
    path = os.path.join(args.results_root, "results", run_id + ".json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    lines = [
        f"\n## Run `{run_id}` — PyTorch eager GPT-2 fp16 (no torch.compile) — GPU 1",
        f"- torch {torch.__version__}, harness sha256 {results['harness_sha256'][:16]}…",
        f"- **Note:** {args.note or '(not specified)'} — reference baseline only, not the engine.",
        f"- device env: {'; '.join(results['device_env'])}",
        "",
        "| B | status | median ms | min ms | max ms | tokens/s | clock MHz | peak TF/s | achieved TF/s | MFU |",
        "|---|--------|-----------|--------|--------|----------|-----------|-----------|---------------|-----|",
    ]
    for r in rows:
        if r["status"] == "ok":
            lines.append(f"| {r['batch']} | ok | {r['median_ms']:.2f} | {r['min_ms']:.2f} | "
                         f"{r['max_ms']:.2f} | {r['tokens_per_s']:.1f} | "
                         f"{r.get('clock_sm_mhz',0)} | {r.get('peak_tflops',0):.1f} | "
                         f"{r.get('achieved_tflops',0):.2f} | {r.get('mfu',0)*100:.1f}% |")
        else:
            lines.append(f"| {r['batch']} | {r['status']} | – | – | – | – | – | – | – | – |")
    with open(os.path.join(args.results_root, "LEDGER.md"), "a") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", path, "and appended LEDGER.md")


if __name__ == "__main__":
    main()
