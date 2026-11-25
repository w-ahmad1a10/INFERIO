import subprocess
import sys

# Exploratory group-size sweep (not the frozen harness): INFERIO_GROUP in
# {64,128,256,512} at B=256 and B=1024. GPU0: 64,128 | GPU1: 256,512.
sweep = (
    'for g in {groups}; do '
    'echo "---- group $g ----"; '
    'CUDA_VISIBLE_DEVICES={gpu} INFERIO_GROUP=$g INFERIO_PROFILE=1 '
    'python3 tools/quick_compare.py --lib engine/build/libinferio.so '
    '--weights /kaggle/working/weights_gpt2 --batches 256,1024 --iters 2 --warmup 1 '
    '2>&1 | grep -E "^B=|failed" | head -3; '
    'done'
)

cmd = (
    'cd /kaggle/working/INFERIO && '
    'make -C engine 2>&1 | grep -E "error" | head -10; '
    'ls -la engine/build/libinferio.so; '
    'echo "==== PARITY (GPU1) ====" && '
    'CUDA_VISIBLE_DEVICES=1 python3 bench/parity_check.py '
    '--lib engine/build/libinferio.so --weights /kaggle/working/weights_gpt2 '
    '--golden /kaggle/working/golden --out temp/results/parity_group.json && '
    'echo "==== GROUP SWEEP (GPU0: 64,128 | GPU1: 256,512; B=256,1024) ====" && '
    '(' + sweep.format(gpu=0, groups='64 128') + ') > /kaggle/working/grp_gpu0.txt 2>&1 & W=$!; '
    '(' + sweep.format(gpu=1, groups='256 512') + ') > /kaggle/working/grp_gpu1.txt 2>&1; '
    'wait $W; '
    'echo ---- GPU0 ----; cat /kaggle/working/grp_gpu0.txt; '
    'echo ---- GPU1 ----; cat /kaggle/working/grp_gpu1.txt'
)
r = subprocess.run(['bash', '-lc', cmd], capture_output=True, text=True, timeout=1700)
print(r.stdout[-9000:])
if r.stderr:
    print(r.stderr[-2000:], file=sys.stderr)
print('EXIT', r.returncode)
if r.returncode:
    raise SystemExit(r.returncode)