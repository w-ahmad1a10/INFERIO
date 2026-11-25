import subprocess
import sys

cmd = (
    'cd /kaggle/working && rm -rf INFERIO && '
    'git clone -q https://github.com/w-ahmad1a10/INFERIO.git && '
    'cd INFERIO && nvcc -O3 -arch=sm_75 tools/microbench_sfu.cu -o /kaggle/working/sfu '
    '&& /kaggle/working/sfu'
)
r = subprocess.run(['bash', '-lc', cmd], capture_output=True, text=True, timeout=600)
print(r.stdout[-4000:])
if r.stderr:
    print(r.stderr[-2000:], file=sys.stderr)
print('EXIT', r.returncode)
if r.returncode:
    raise SystemExit(r.returncode)
