import subprocess
import sys

cmd = (
    'cd /kaggle/working && rm -rf INFERIO && '
    'git clone -q https://github.com/w-ahmad1a10/INFERIO.git && cd INFERIO && '
    'python3 tools/convert_gpt2.py && '
    'python3 tools/oracle_gpt2.py /kaggle/working/model.safetensors /kaggle/working/golden 1 && '
    'make -C engine 2>&1 | grep -E "error" | head -10; '
    'ls -la engine/build/libinferio.so; '
    'echo "==== B=1 BASELINE v0.12.2 (GPU0) ====" && '
    'CUDA_VISIBLE_DEVICES=0 INFERIO_COMMIT=$(git rev-parse --short HEAD) '
    'python3 b1/bench_b1.py --lib engine/build/libinferio.so '
    '--weights /kaggle/working/weights_gpt2 --profile '
    '--note "v0.12.2 re-baseline on fresh card: B=1 campaign starting line"'
)
r = subprocess.run(['bash', '-lc', cmd], capture_output=True, text=True, timeout=1700)
print(r.stdout[-8000:])
if r.stderr:
    print(r.stderr[-2000:], file=sys.stderr)
print('EXIT', r.returncode)
if r.returncode:
    raise SystemExit(r.returncode)