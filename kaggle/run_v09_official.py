import subprocess
import sys

cmd = (
    'cd /kaggle/working && rm -rf INFERIO && '
    'git clone -q https://github.com/w-ahmad1a10/INFERIO.git && cd INFERIO && '
    'make -C engine 2>&1 | grep -E "error" | head -10; '
    'ls -la engine/build/libinferio.so; '
    'echo "==== PARITY (GPU0, v0.9) ====" && '
    'CUDA_VISIBLE_DEVICES=0 python3 bench/parity_check.py '
    '--lib engine/build/libinferio.so --weights /kaggle/working/weights_gpt2 '
    '--golden /kaggle/working/golden --out temp/results/parity_v090.json && '
    'echo "==== OFFICIAL v0.9.0 (GPU0) ====" && '
    'CUDA_VISIBLE_DEVICES=0 INFERIO_COMMIT=$(git rev-parse --short HEAD) '
    'python3 bench/bench_prefill.py --all --gpu 0 --profile '
    '--note "v0.9.0: two-pass LayerNorm (sum+sqsum in one pass, var=E[x2]-mean^2), '
    'default group 64 (exploratory sweep: +9% B=256, +2.5% B=1024 vs group 128)"'
)
r = subprocess.run(['bash', '-lc', cmd], capture_output=True, text=True, timeout=1750)
print(r.stdout[-12000:])
if r.stderr:
    print(r.stderr[-3000:], file=sys.stderr)
print('EXIT', r.returncode)
if r.returncode:
    raise SystemExit(r.returncode)