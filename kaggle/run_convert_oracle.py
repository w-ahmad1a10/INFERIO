import subprocess
import sys

r = subprocess.run(
    ['bash', '-lc',
     'cd /kaggle/working && python3 tools/convert_gpt2.py && python3 tools/oracle_gpt2.py'],
    capture_output=True, text=True, timeout=1400)
print(r.stdout[-5000:])
if r.stderr:
    print(r.stderr[-3000:], file=sys.stderr)
print('EXIT', r.returncode)
if r.returncode:
    raise SystemExit(r.returncode)
