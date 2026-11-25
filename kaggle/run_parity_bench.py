import subprocess
import sys

cmd = (
    'cd /kaggle/working && rm -rf INFERIO && '
    'git clone -q https://github.com/w-ahmad1a10/INFERIO.git && '
    'cd INFERIO && make -C engine -j 2>&1 | tail -3 && '
    'python3 bench/parity_check.py --lib engine/build/libinferio.so '
    '--weights /kaggle/working/weights_gpt2 --golden /kaggle/working/golden '
    '--out temp/results/parity.json && echo ==== PARITY_PASS ==== && '
    'INFERIO_COMMIT=$(git rev-parse --short HEAD) python3 bench/bench_prefill.py --all '
    '--gpu 0 --profile --note "v0.3.0: tiled online-softmax attention (exp2, per-tile '
    'rescale, smem score staging) + tanh.approx GELU + half2 bias kernels"'
)
r = subprocess.run(['bash', '-lc', cmd], capture_output=True, text=True, timeout=1750)
print(r.stdout[-12000:])
if r.stderr:
    print(r.stderr[-3000:], file=sys.stderr)
print('EXIT', r.returncode)
if r.returncode:
    raise SystemExit(r.returncode)
