import subprocess
import sys

cmd = (
    'cd /kaggle/working && rm -rf dump && mkdir -p dump && cd INFERIO && '
    'git pull -q origin main && make -C engine -j 2>&1 | tail -5 && '
    'INFERIO_DUMP_DIR=/kaggle/working/dump python3 bench/bench_prefill.py --single --batch 2 '
    '--lib engine/build/libinferio.so --weights /kaggle/working/weights_gpt2 '
    '> /kaggle/working/dump/bench_stdout.txt 2>&1; '
    'echo ==== STAGES ==== && '
    'python3 tools/debug_stagecheck.py /kaggle/working/model.safetensors '
    '/kaggle/working/dump /kaggle/working/golden/ids_b2.npy'
)
r = subprocess.run(['bash', '-lc', cmd], capture_output=True, text=True, timeout=1300)
print(r.stdout[-9000:])
if r.stderr:
    print(r.stderr[-3000:], file=sys.stderr)
print('EXIT', r.returncode)
if r.returncode:
    raise SystemExit(r.returncode)
