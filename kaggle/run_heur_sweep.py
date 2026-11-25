import subprocess
import sys

sweep = (
    'for h in {heurs}; do '
    'echo "---- heur $h ----"; '
    'CUDA_VISIBLE_DEVICES={gpu} INFERIO_HEUR=$h INFERIO_PROFILE=1 python3 tools/quick_compare.py '
    '--lib engine/build/libinferio.so --weights /kaggle/working/weights_gpt2 '
    '--batches 256 --iters 1 --warmup 1 2>&1 | grep -E "^B=|failed" | head -2; '
    'done'
)

cmd = (
    'cd /kaggle/working && rm -rf INFERIO && '
    'git clone -q https://github.com/w-ahmad1a10/INFERIO.git && cd INFERIO && '
    'make -C engine 2>&1 | grep -E "error" | head -10; '
    'echo "==== HEUR SWEEP (GPU0: 0-3 | GPU1: 4-7, B=256) ====" && '
    '(' + sweep.format(gpu=0, heurs='0 1 2 3') + ') > /kaggle/working/heur_gpu0.txt 2>&1 & W=$!; '
    '(' + sweep.format(gpu=1, heurs='4 5 6 7') + ') > /kaggle/working/heur_gpu1.txt 2>&1; '
    'wait $W; '
    'echo ---- GPU0 ----; cat /kaggle/working/heur_gpu0.txt; '
    'echo ---- GPU1 ----; cat /kaggle/working/heur_gpu1.txt'
)
r = subprocess.run(['bash', '-lc', cmd], capture_output=True, text=True, timeout=1700)
print(r.stdout[-9000:])
if r.stderr:
    print(r.stderr[-2000:], file=sys.stderr)
print('EXIT', r.returncode)
if r.returncode:
    raise SystemExit(r.returncode)
