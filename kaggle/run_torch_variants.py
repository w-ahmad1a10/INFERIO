import subprocess
import sys

# PyTorch reference variants on GPU1 (engine card GPU0 untouched):
#   eager-nocache (all 9 batches) then compile (all batches, dynamic).
cmd = (
    'cd /kaggle/working && rm -rf INFERIO && '
    'git clone -q https://github.com/w-ahmad1a10/INFERIO.git && cd INFERIO && '
    'python3 -c "import torch; print(torch.__version__)" 2>/dev/null || '
    'pip install -q torch --index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -2; '
    'echo "==== eager-nocache (GPU1) ====" && '
    'CUDA_VISIBLE_DEVICES=1 python3 tools/bench_torch_variants.py --variant eager-nocache '
    '--note "PyTorch eager fp16, use_cache=False (no KV allocation), identical frozen inputs" '
    '2>&1 | grep -vE "Loading|it/s|^$" | tail -12; '
    'echo "==== compile (GPU1) ====" && '
    'CUDA_VISIBLE_DEVICES=1 python3 tools/bench_torch_variants.py --variant compile '
    '--note "PyTorch torch.compile(transformer, dynamic=True) fp16, use_cache=False" '
    '2>&1 | grep -vE "Loading|it/s|^$" | tail -12'
)
r = subprocess.run(['bash', '-lc', cmd], capture_output=True, text=True, timeout=1750)
print(r.stdout[-9000:])
if r.stderr:
    print(r.stderr[-2000:], file=sys.stderr)
print('EXIT', r.returncode)
if r.returncode:
    raise SystemExit(r.returncode)