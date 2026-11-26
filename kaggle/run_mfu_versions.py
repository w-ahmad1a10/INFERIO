import subprocess
import sys

# MFU re-run (SPEC v2) of engine versions. Reads <commit> <tag> lines from
# /kaggle/working/mfu_versions.txt. For each: clone at that commit, build the
# OLD engine .so, then run the NEW (main) v2 harness against it on GPU 0.
cmd = (
    'cd /kaggle/working && rm -rf INFERIO && '
    'git clone -q https://github.com/w-ahmad1a10/INFERIO.git && cd INFERIO && '
    'while read commit tag; do '
    '[ -z "$commit" ] && continue; '
    'echo "==== $tag ($commit) ===="; '
    'git checkout -q "$commit" && make -C engine 2>&1 | grep -E "error" | head -5; '
    'cp engine/build/libinferio.so /kaggle/working/lib_$tag.so || { echo BUILD_FAIL; continue; }; '
    'git checkout -q main; '
    'CUDA_VISIBLE_DEVICES=0 INFERIO_COMMIT=$commit INFERIO_RESULTS=/kaggle/working/mfu_results '
    'python3 bench/bench_prefill.py --all '
    '--gpu 0 --profile --lib /kaggle/working/lib_$tag.so '
    '--note "MFU re-run (SPEC v2): $tag, old engine binary with v2 harness" '
    '| tail -11; '
    'done < /kaggle/working/mfu_versions.txt'
)
r = subprocess.run(['bash', '-lc', cmd], capture_output=True, text=True, timeout=1750)
print(r.stdout[-12000:])
if r.stderr:
    print(r.stderr[-2000:], file=sys.stderr)
print('EXIT', r.returncode)
if r.returncode:
    raise SystemExit(r.returncode)