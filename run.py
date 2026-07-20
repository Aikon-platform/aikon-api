#!/usr/bin/env python3
"""
Start / stop the AIKON API.

    python run.py [up|down|logs|build]

local/prod : manages the docker container (build = rebuild image + restart;
             needed after any .env change, since the image bakes api/.env)
dev        : runs flask + dramatiq on the host until Ctrl+C
"""

import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

API = Path(__file__).resolve().parent
WIN = os.name == "nt"

# commands from the legacy api/run.sh (uv run works here, unlike the front celery)
DEV_PROCS = [
    (
        "flask",
        [
            "uv",
            "run",
            "flask",
            "--app",
            "app.main",
            "run",
            "--debug",
            "-p",
            "{API_PORT}",
        ],
        API,
    ),
    ("dramatiq", ["uv", "run", "dramatiq", "app.main", "-t", "1", "-p", "1"], API),
]


def read_env() -> dict:
    env_file = API / ".env"
    if not env_file.exists():
        sys.exit("no api/.env found: run `python install.py` first")
    return dict(
        line.split("=", 1)
        for line in env_file.read_text().splitlines()
        if "=" in line and not line.startswith("#")
    )


def sh(cmd: list, check: bool = True) -> int:
    return subprocess.run(cmd, cwd=API, check=check).returncode


def kill_stale(*patterns: str) -> None:
    if WIN:
        return
    for p in patterns:
        if not subprocess.run(["pkill", "-f", p], capture_output=True).returncode:
            print(f"killed stale '{p}'")
    time.sleep(1)


def docker_run() -> None:
    name = ENV["CONTAINER_NAME"]
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "--restart",
        "unless-stopped",
        "--env-file",
        str(API / ".env"),
        "-v",
        f"{ENV['DATA_FOLDER']}:/data",
    ]
    if ENV.get("DEVICE_NB"):
        cmd += [
            "--gpus",
            ENV["DEVICE_NB"],
            "--ipc=host",
            "-v",
            f"{ENV['CUDA_HOME']}:/cuda",
        ]
    if ENV.get("BUNDLED") == "True":
        cmd += ["--network", "aikon_aikon", "--network-alias", "api"]
    else:
        cmd += ["-p", f"{ENV['CONTAINER_HOST']}:{ENV['API_PORT']}:{ENV['API_PORT']}"]
    sh(cmd + [name])
    print(f"→ api container '{name}' started")


def docker_build() -> None:
    userid = os.getuid() if hasattr(os, "getuid") else 1000
    args = [
        f"--build-arg={k}={v}"
        for k, v in {
            "USERID": userid,
            "API_PORT": ENV["API_PORT"],
            "HTTP_PROXY": ENV.get("HTTP_PROXY", ""),
            "HTTPS_PROXY": ENV.get("HTTPS_PROXY", ""),
            "HUGGING_FACE_HUB_TOKEN": ENV.get("HUGGING_FACE_HUB_TOKEN", ""),
        }.items()
    ]
    sh(
        ["docker", "build", "-t", ENV["CONTAINER_NAME"], "-f", "docker/Dockerfile", "."]
        + args
    )


def spawn(name: str, cmd: list, cwd: Path) -> subprocess.Popen:
    cmd = [c.format(**ENV) for c in cmd]
    cmd[0] = shutil.which(cmd[0]) or sys.exit(
        f"'{cmd[0]}' not found, run `python install.py` first"
    )
    kwargs = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if WIN
        else {"start_new_session": True}
    )
    print(f"starting {name}: {' '.join(cmd)}")
    return subprocess.Popen(cmd, cwd=cwd, **kwargs)


def stop(name: str, proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.CTRL_BREAK_EVENT) if WIN else os.killpg(
            proc.pid, signal.SIGTERM
        )
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill() if WIN else os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    except ProcessLookupError:
        pass
    print(f"stopped {name}")


def redis_up() -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", int(ENV.get("REDIS_PORT", 6379)))) == 0


def run_dev() -> None:
    kill_stale("dramatiq app.main", "flask --app app.main")
    if not redis_up():
        sys.exit(
            f"redis not reachable on :{ENV.get('REDIS_PORT', 6379)} — start the services first (python run.py from the root)")
    os.environ["CUDA_VISIBLE_DEVICES"] = ENV.get("DEVICE_NB") or "0"
    procs = {name: spawn(name, cmd, cwd) for name, cmd, cwd in DEV_PROCS}
    print(f"\n→ api at http://localhost:{ENV['API_PORT']}  (Ctrl+C to stop)\n")
    try:
        while True:
            for name, p in procs.items():
                if p.poll() not in (None, 0):
                    print(f"\n'{name}' exited with code {p.returncode}, shutting down")
                    raise KeyboardInterrupt
            time.sleep(2)
    except KeyboardInterrupt:
        print()
        for name, p in procs.items():
            stop(name, p)


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "up"
    ENV = read_env()
    dev = ENV.get("MODE") == "dev"

    if action == "down":
        if dev:
            print("dev mode: Ctrl+C in the `run.py up` terminal stops the api")
        else:
            sh(["docker", "rm", "-f", ENV["CONTAINER_NAME"]], check=False)
    elif action == "logs" and not dev:
        sh(["docker", "logs", "-f", ENV["CONTAINER_NAME"]], check=False)
    elif action == "build" and not dev:
        docker_build()
        docker_run()
    elif action == "up":
        run_dev() if dev else docker_run()
    else:
        sys.exit(__doc__)
