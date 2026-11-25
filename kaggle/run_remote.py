import argparse
import base64
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
import requests
import websocket
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

def load_conf():
    conf = {}
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kaggle.env')
    if os.path.exists(p):
        for line in open(p, encoding='utf-8'):
            line = line.strip()
            if line and '=' in line and (not line.startswith('#')):
                k, v = line.split('=', 1)
                conf[k] = v
    base = conf.get('KAGGLE_BASE') or os.environ['KAGGLE_BASE']
    kernel = conf.get('KAGGLE_KERNEL') or os.environ['KAGGLE_KERNEL']
    return (base.rstrip('/'), kernel)
BASE, KERNEL = load_conf()

def api(method, path, **kw):
    r = requests.request(method, BASE + path, timeout=60, **kw)
    r.raise_for_status()
    return r

def upload(local, remote):
    data = open(local, 'rb').read()
    try:
        body = {'type': 'file', 'format': 'text', 'content': data.decode('utf-8')}
    except UnicodeDecodeError:
        body = {'type': 'file', 'format': 'base64', 'content': base64.b64encode(data).decode()}
    api('PUT', '/api/contents/' + remote, json=body)
    print(f'uploaded {local} -> {remote} ({len(data)} bytes)')

def download(remote, local):
    r = api('GET', '/api/contents/' + remote)
    c = r.json()['content']
    if isinstance(c, str):
        data = c.encode()
    elif isinstance(c, dict) and 'content' in c:
        data = base64.b64decode(c['content']) if c.get('format') == 'base64' else c['content'].encode()
    else:
        data = str(c).encode()
    os.makedirs(os.path.dirname(os.path.abspath(local)), exist_ok=True)
    open(local, 'wb').write(data)
    print(f'downloaded {remote} -> {local} ({len(data)} bytes)')

def execute(code, timeout=600, silent=False):
    sid = str(uuid.uuid4())
    http_url = f'{BASE}/api/kernels/{KERNEL}/channels?session_id={sid}'
    ws_url = http_url.replace('https://', 'wss://', 1).replace('http://', 'ws://', 1)
    ws = websocket.create_connection(ws_url, timeout=30)
    ws.settimeout(1.0)
    msg_id = str(uuid.uuid4())
    msg = {'header': {'msg_id': msg_id, 'username': 'moq-driver', 'session': sid, 'date': datetime.now(timezone.utc).isoformat(), 'msg_type': 'execute_request', 'version': '5.3'}, 'parent_header': {}, 'metadata': {}, 'content': {'code': code, 'silent': silent, 'store_history': not silent, 'user_expressions': {}, 'allow_stdin': False, 'stop_on_error': True}, 'channel': 'shell', 'buffers': []}
    ws.send(json.dumps(msg))
    deadline = time.time() + timeout
    status = None
    while time.time() < deadline:
        try:
            raw = ws.recv()
        except websocket.WebSocketTimeoutException:
            continue
        try:
            m = json.loads(raw)
        except Exception:
            continue
        if m.get('parent_header', {}).get('msg_id') != msg_id:
            continue
        t = m.get('header', {}).get('msg_type')
        c = m.get('content', {})
        if t == 'stream':
            sys.stdout.write(c.get('text', ''))
            sys.stdout.flush()
        elif t in ('execute_result', 'display_data'):
            s = c.get('data', {}).get('text/plain', '')
            if s:
                sys.stdout.write(s + '\n')
                sys.stdout.flush()
        elif t == 'error':
            sys.stdout.write('\n'.join(c.get('traceback', [])) + '\n')
            sys.stdout.flush()
        elif t == 'execute_reply':
            status = c.get('status')
            break
    ws.close()
    if status is None:
        print(f'\n[driver] TIMEOUT after {timeout}s (kernel may still be running)')
        return 1
    return 0 if status == 'ok' else 1

def probe():
    code = '\nimport platform, os, shutil, subprocess, sys\nprint("python:", sys.version.split()[0])\nprint("platform:", platform.platform())\nprint("cpus:", os.cpu_count())\nprint("disk:", shutil.disk_usage("/"))\nfor c in ["git", "gcc", "g++", "cmake", "ninja", "make", "uv", "apt-get", "curl"]:\n    print(f"{c}: {shutil.which(c)}")\nprint("--- mem ---")\nprint(subprocess.run(["bash","-lc","free -h | head -3; nproc; df -h / | tail -1"], capture_output=True, text=True).stdout)\nprint("--- net ---")\nprint(subprocess.run(["bash","-lc","curl -sI -m 10 https://github.com | head -1"], capture_output=True, text=True).stdout)\nprint("--- cwd ---")\nprint(os.getcwd())\nprint(subprocess.run(["bash","-lc","ls -la /kaggle/working 2>/dev/null | head; ls -la /kaggle/input 2>/dev/null | head"], capture_output=True, text=True).stdout)\n'
    return execute(code, timeout=120)

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('probe')
    e = sub.add_parser('exec')
    e.add_argument('--code')
    e.add_argument('--code-file')
    e.add_argument('--timeout', type=int, default=600)
    u = sub.add_parser('upload')
    u.add_argument('--local', required=True)
    u.add_argument('--remote', required=True)
    d = sub.add_parser('download')
    d.add_argument('--remote', required=True)
    d.add_argument('--local', required=True)
    a = ap.parse_args()
    if a.cmd == 'probe':
        return probe()
    if a.cmd == 'exec':
        code = a.code if a.code is not None else open(a.code_file, encoding='utf-8').read()
        return execute(code, timeout=a.timeout)
    if a.cmd == 'upload':
        return upload(a.local, a.remote)
    if a.cmd == 'download':
        return download(a.remote, a.local)
    return 2
if __name__ == '__main__':
    raise SystemExit(main())
