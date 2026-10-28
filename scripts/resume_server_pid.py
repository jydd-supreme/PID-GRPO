#!/usr/bin/env python3
"""Start or resume the server's DeepMath PID run after a container reboot."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.utils.config import load_config
from src.utils.experiment import runtime_config

CONFIG = REPO / "configs/experiments/deepmath_pid.yaml"
PROXY_PORT = 7890
SESSION_LIMIT = 10800
PLATFORM_LIMIT = 14400
SHUTDOWN_MARGIN = 3600
MIN_SESSION = 1800
DATA_PROBE_PATH = (
    "/datasets/zwhe99/DeepMath-103K/resolve/"
    "5cf055d1fe3d7a2eb19719ac020211469736ae44/data/train-00000-of-00010.parquet"
)
MODEL_PROBE_PATH = (
    "/Qwen/Qwen2.5-1.5B-Instruct/resolve/"
    "989aa7980e4cf806f80c7fef2b1adb7bc71aa306/config.json"
)
DATASET_API_PATHS = (
    "/api/datasets/cais/mmlu/revision/c30699e8356da336a370243923dbaf21066bb9fe",
    "/api/datasets/allenai/ai2_arc/revision/210d026faf9955653af8916fad021475a3f00453",
    "/api/datasets/Rowan/hellaswag/revision/218ec52e09a7e7462a5400043bb9a69a41d06b76",
)
ENDPOINTS = ("https://huggingface.co", "https://hf-mirror.com")


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def run_state(output: Path, expected_config: dict) -> tuple[str, int | None]:
    saved = read_json(output / "run_config.json")
    if saved is None:
        raise RuntimeError("formal PID run does not exist; this script only resumes an existing run")
    if saved != expected_config:
        raise RuntimeError("saved run configuration differs from this checkout; do not resume")
    result = read_json(output / "run_result.json") or {}
    progress = read_json(output / "progress.json") or {}
    if result.get("status") in {"complete", "stopped_hard"}:
        return result["status"], result.get("global_step")
    if progress.get("status") in {"complete", "stopped_hard"}:
        return progress["status"], progress.get("global_step")
    if progress.get("status") == "paused" and progress.get("pending_evaluation"):
        # The trainer understands pending evaluations, but surface the fact here.
        print("Pending evaluation will be finished before the next optimizer update.")
    latest = read_json(output / "restart_checkpoints/latest.json")
    return "resume", latest.get("global_step") if latest else progress.get("global_step")


def session_budget(container_uptime: int) -> int:
    return min(SESSION_LIMIT, PLATFORM_LIMIT - container_uptime - SHUTDOWN_MARGIN)


def process_command(pid: int) -> bytes:
    try:
        return (Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ")
    except OSError:
        return b""


def pid_is_training(pid: int) -> bool:
    command = process_command(pid)
    return b"scripts/run_experiment.py" in command and b"deepmath_pid.yaml" in command


def proxy_responds() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", PROXY_PORT), timeout=2):
            return True
    except OSError:
        return False


def start_proxy(root: Path) -> None:
    proxy_dir = root / "proxy"
    binary = proxy_dir / "mihomo"
    config = proxy_dir / "config.yaml"
    pid_path = proxy_dir / "mihomo.pid"
    if not binary.is_file() or not config.is_file():
        raise RuntimeError(f"Mihomo or its configuration is missing in {proxy_dir}")
    if proxy_responds():
        print("Local proxy port is ready.")
        return
    if pid_path.is_file():
        try:
            pid = int(pid_path.read_text().strip())
            if str(binary).encode() in process_command(pid):
                raise RuntimeError("saved Mihomo PID is alive but port 7890 is unavailable; inspect proxy")
        except ValueError:
            pass
    with (proxy_dir / "mihomo.log").open("ab") as log:
        process = subprocess.Popen(
            [str(binary), "-d", str(proxy_dir)], stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    pid_path.write_text(f"{process.pid}\n", encoding="ascii")
    for _ in range(10):
        if proxy_responds():
            print(f"Mihomo started (PID {process.pid}).")
            return
        if process.poll() is not None:
            break
        time.sleep(1)
    raise RuntimeError(f"Mihomo did not open port 7890; inspect {proxy_dir / 'mihomo.log'}")


def probe_url(url: str, *, byte_range: bool) -> bool:
    command = [
        "curl", "-4", "-fsSL", "--retry", "2", "--retry-all-errors",
        "--retry-delay", "2", "--connect-timeout", "10",
        "--max-time", "45", "-x", f"http://127.0.0.1:{PROXY_PORT}",
    ]
    if byte_range:
        command.extend(["--range", "0-1023"])
    command.extend(["-o", os.devnull, "-w", "%{http_code}", url])
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return result.returncode == 0 and result.stdout.strip().endswith("206" if byte_range else "200")


def probe_endpoint(preferred: str | None = None) -> str:
    candidates = (preferred,) if preferred else ENDPOINTS
    for endpoint in candidates:
        if (probe_url(endpoint + DATA_PROBE_PATH, byte_range=True)
            and probe_url(endpoint + MODEL_PROBE_PATH, byte_range=False)
            and all(probe_url(endpoint + path, byte_range=False) for path in DATASET_API_PATHS)):
            print(f"Data and model probes passed through {endpoint}.")
            return endpoint
    raise RuntimeError("No selected Hugging Face endpoint passed all data and model probes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-endpoint", choices=ENDPOINTS,
                        help="force the official or mirror endpoint instead of automatic selection")
    args = parser.parse_args()
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    root = args.root_dir.resolve()
    python = root / "venvs/pid-grpo-cu124/bin/python"
    config = runtime_config(load_config(CONFIG), seed=args.seed, root_dir=str(root))
    output = Path(config["project"]["output_dir"])
    if not output.is_dir():
        raise SystemExit(f"No existing formal PID run: {output}")
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    launch_lock = output / ".launch.lock"
    with launch_lock.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        status, step = run_state(output, config)
        if status == "complete":
            print(f"PID run is complete at step {step}; nothing to launch.")
            return
        if status == "stopped_hard":
            raise SystemExit("PID run stopped for safety; automatic resume is forbidden.")
        pid_path = logs / f"deepmath-pid500-seed{args.seed}.pid"
        if pid_path.is_file():
            try:
                old_pid = int(pid_path.read_text().strip())
            except ValueError:
                old_pid = 0
            if pid_is_training(old_pid):
                print(f"PID training is already running (PID {old_pid}).")
                return
        with (output / ".run.lock").open("a") as run_lock:
            try:
                fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit("Another process already owns this training run.")
        uptime = int(subprocess.check_output(["ps", "-p", "1", "-o", "etimes="], text=True).strip())
        budget = session_budget(uptime)
        if budget < MIN_SESSION:
            raise SystemExit("Too little time remains before the 4-hour platform restart; start in the next session.")
        print(f"Latest announced checkpoint: step {step}; session budget: {budget // 60} minutes.")
        subprocess.run([str(python), str(REPO / "scripts/verify_environment.py")],
                       cwd=REPO, check=True)
        if not (REPO / config["evaluation"]["manifest_file"]).is_file():
            raise RuntimeError("frozen benchmark manifest is missing")
        subprocess.run(
            [str(python), str(REPO / "scripts/run_experiment.py"), "preflight",
             "--config", str(CONFIG), "--root-dir", str(root), "--seed", str(args.seed)],
            cwd=REPO, stdout=subprocess.DEVNULL, check=True,
        )
        subprocess.run(
            [str(python), "-c", "import torch; x=torch.ones(1, device='cuda'); "
             "print('CUDA ready:', torch.cuda.get_device_name(0), x.item())"],
            cwd=REPO, check=True,
        )
        start_proxy(root)
        endpoint = probe_endpoint(args.hf_endpoint)
        environment = os.environ.copy()
        environment.update({
            "HTTP_PROXY": f"http://127.0.0.1:{PROXY_PORT}",
            "HTTPS_PROXY": f"http://127.0.0.1:{PROXY_PORT}",
            "ALL_PROXY": f"http://127.0.0.1:{PROXY_PORT}",
            "NO_PROXY": "localhost,127.0.0.1",
            "HF_ENDPOINT": endpoint,
            "HF_HUB_DOWNLOAD_TIMEOUT": "60",
            "HF_HUB_ETAG_TIMEOUT": "30",
        })
        command = [
            str(python), "-u", str(REPO / "scripts/run_experiment.py"), "train",
            "--config", str(CONFIG), "--root-dir", str(root), "--seed", str(args.seed),
            "--enable-real-training", "--resume", "--session-max-seconds", str(budget),
        ]
        log_path = logs / f"deepmath-pid500-seed{args.seed}.log"
        with log_path.open("ab") as log:
            process = subprocess.Popen(command, cwd=REPO, env=environment,
                                       stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        pid_path.write_text(f"{process.pid}\n", encoding="ascii")
        time.sleep(3)
        if process.poll() is not None:
            raise RuntimeError(f"training exited with code {process.returncode}; inspect {log_path}")
        print(f"PID training launched (PID {process.pid}). Log: {log_path}")
        print("Data/model loading can take several minutes; watch sessions.jsonl for 'restored'.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from None
