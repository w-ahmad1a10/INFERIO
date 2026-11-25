#!/usr/bin/env python3
"""Side-by-side comparison of stored benchmark runs (engine versions + baselines).

Usage: python3 tools/compare_runs.py temp/results/<run1>.json [run2.json ...]
"""
import json
import sys


def main():
    runs = []
    for path in sys.argv[1:]:
        with open(path) as f:
            runs.append(json.load(f))
    labels = []
    for r in runs:
        eng = r.get("engine") or r.get("engine_version") or "?"
        if len(eng) > 24:
            eng = eng[:24]
        labels.append(eng)

    header = f"{'B':>5} | " + " | ".join(f"{l:>24}" for l in labels)
    print(header)
    print("-" * len(header))
    batches = sorted({row["batch"] for r in runs for row in r["rows"]})
    for b in batches:
        cells = []
        for r in runs:
            row = next((x for x in r["rows"] if x["batch"] == b), None)
            if row is None:
                cells.append(f"{'-':>24}")
            elif row["status"] == "ok":
                cells.append(f"{row['median_ms']:>18.1f} ms     ")
            else:
                cells.append(f"{row['status']:>24}")
        print(f"{b:>5} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
