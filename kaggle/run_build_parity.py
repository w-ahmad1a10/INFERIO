import subprocess
import sys

cmd = (
    'cd /kaggle/working && rm -rf INFERIO && '
    'git clone -q https://github.com/w-ahmad1a10/INFERIO.git && '
    'cd INFERIO && make -C engine -j 2>&1 | tail -25 && '
    'echo ==== BUILT ==== && ls -la engine/build && '
    'echo ==== PARITY ==== && '
    'python3 bench/parity_check.py --lib engine/build/libinferio.so '
    '--weights /kaggle/working/weights_gpt2 --golden /kaggle/working/golden'
)
r = subprocess.run(['bash', '-lc', cmd], capture_output=True, text=True, timeout=1400)
print(r.stdout[-9000:])
if r.stderr:
    print(r.stderr[-4000:], file=sys.stderr)
print('EXIT', r.returncode)
if r.returncode:
    raise SystemExit(r.returncode)
