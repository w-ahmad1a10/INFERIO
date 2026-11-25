# Remote Kaggle Notebook — Driver & Guide

This folder lets an agent (or a human) **execute arbitrary code on a running Kaggle
notebook** from the command line, without SSH. It was used to build and run the
llama.cpp quantization unit tests in a real Linux environment.

---

## 1. How the connection works

A running Kaggle notebook is a normal Jupyter server behind a signed proxy URL:

```
https://kkb-production.jupyter-proxy.kaggle.net/k/<notebook-id>/<JWT>/proxy
```

- `/lab/tree/...` is the JupyterLab UI.
- `/api/...` is the standard **Jupyter Server REST API** (status, kernels, contents).
- `/api/kernels/<kernel-id>/channels` is the **websocket** used to execute code.

The JWT in the path authenticates the session; no cookies or Kaggle credentials are
needed from this machine (verified: `/api/status` returns `200` with no cookie jar).
The URL is a live credential for the notebook — treat it like a password.

Find the kernel to drive:

```bash
curl -s "$KAGGLE_BASE/api/kernels"
```

If the notebook was restarted, the kernel id (and possibly the whole URL) changes;
update `kaggle.env` with the fresh values from the notebook page.

## 2. Files

| file | purpose |
|---|---|
| `run_remote.py` | driver: `probe`, `exec`, `upload`, `download` |
| `kaggle.env` | `KAGGLE_BASE=...`, `KAGGLE_KERNEL=...` (**gitignored — never commit**) |
| `pyproject.toml`, `uv.lock` | `uv`-managed deps (`requests`, `websocket-client`) |

## 3. Setup

```bash
cd temp/kaggle
uv sync
```

`uv sync` creates `.venv/` from `uv.lock`, so the environment is reproducible.

## 4. Usage

```bash
# health check + environment report (Python, CPUs, RAM, compilers, internet)
uv run python run_remote.py probe

# run a code cell in the notebook kernel
uv run python run_remote.py exec --code "print('hello from Kaggle')"

# run a cell from a file
uv run python run_remote.py exec --code-file temp/tests/kaggle_setup_probe.py --timeout 900

# push a local file into the notebook working dir
uv run python run_remote.py upload --local temp/tests/test.py --remote temp/tests/test.py

# pull a result back
uv run python run_remote.py download --remote temp/tests/out.txt --local temp/tests/out.txt
```

`exec` streams stdout/stderr and cell results until the kernel reports
`execute_reply`; it exits non-zero on error or timeout.

## 5. Notes for future agents

- **The filesystem is ephemeral.** Anything worth keeping must be committed/pushed
  to the repo or downloaded before the session ends.
- Long builds are fine (llama.cpp CPU build ≈ minutes on 4 vCPU), but a single
  `exec` call has a configurable timeout (`--timeout`, seconds).
- Code runs as `root` in `/kaggle/working`; `apt-get`, `pip`, `uv`, `git` all work.
- Python version at verification time: **3.13**, glibc 2.39, 4 vCPU, 31 GiB RAM.
- The websocket URL must use `wss://` (the driver converts the `https://` base).
- If the kernel is busy, `execute_request` is queued by Jupyter; the driver waits.
- Do **not** upload `kaggle.env` anywhere. It contains a live session token.

## 6. Verified environment (2026-11-23)

```
platform : Linux-6.18.55+-x86_64-with-glibc2.39
cpus     : 4
mem      : 31 GiB total (~29 GiB available)
disk     : overlay 8.0T (2.5T free)
tools    : git, gcc, g++, cmake, ninja, make, uv, apt-get, curl
network  : github.com reachable (HTTP/2 200)
cwd      : /kaggle/working
```
