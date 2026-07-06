#!/usr/bin/env python3
"""
AIKON API installer — works standalone (no front files needed) or delegated
from the root install.py (which passes --root-env to share its configuration).

    python install.py [--mode local|dev|prod] [--root-env PATH] [--defaults]

local/prod = build and start the docker container
dev        = create the venv on the host, then `python run.py`
"""

import argparse
import subprocess
import sys
from pathlib import Path

API = Path(__file__).resolve().parent
TEMPLATE = API / ".env.template"
ENV_FILE = API / ".env"
DOCKER_DIR = API / "docker"

MODES = ("local", "dev", "prod")
API_APPS = (
    "dticlustering",
    "watermarks",
    "similarity",
    "region_extraction",
    "vectorization",
    "search",
)
PROMPTED = {
    "local": (),
    "dev": ("INSTALLED_APPS",),
    "prod": (
        "INSTALLED_APPS",
        "DATA_FOLDER",
        "PROD_URL",
        "CONTAINER_HOST",
        "DEVICE_NB",
        "CUDA_HOME",
    ),
}
# root .env key → api key, applied when --root-env is given (bundle install)
ROOT_MAP = {
    "MODE": "MODE",
    "INSTALLED_APPS": "INSTALLED_APPS",
    "API_PORT": "API_PORT",
    "REDIS_PORT": "REDIS_PORT",
    "HTTP_PROXY": "HTTP_PROXY",
    "HTTPS_PROXY": "HTTPS_PROXY",
}


def parse_env(path: Path) -> dict:
    entries, desc = {}, ""
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("#"):
            desc = line.lstrip("# ").strip()
        elif "=" in line:
            key, _, val = line.partition("=")
            entries[key.strip()] = (val.strip().strip('"'), desc)
            desc = ""
    return entries


def sh(cmd: list, cwd: Path = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def render(template: Path, out: Path, mapping: dict) -> None:
    text = template.read_text()
    for key, val in mapping.items():
        text = text.replace(key, str(val))
    out.write_text(text)


def resolve(mode: str, root_env: Path, use_defaults: bool) -> dict:
    current = (
        {k: v for k, (v, _) in parse_env(ENV_FILE).items()} if ENV_FILE.exists() else {}
    )
    root = {k: v for k, (v, _) in parse_env(root_env).items()} if root_env else {}
    v = {}
    for key, (default, desc) in parse_env(TEMPLATE).items():
        val = root.get(
            next((rk for rk, ak in ROOT_MAP.items() if ak == key), ""),
            current.get(key, default),
        )
        if key in PROMPTED[mode] and key not in ROOT_MAP.values() and not use_defaults:
            user = input(f"{key} — {desc}\n  [{val or 'empty'}]: ").strip()
            val = user or val
        v[key] = val

    v["MODE"] = mode
    v["TARGET"] = mode  # legacy alias, in case the api code still reads TARGET
    docker = mode != "dev"
    v["DOCKER"] = str(docker)
    # in a bundle install the root .env is the source of truth: the api data folder
    # derives from its DATA_DIR (standalone customizations are overwritten)
    data_folder = str(
        Path(
            Path(root["DATA_DIR"]) / "api" if root else v["DATA_FOLDER"] or API / "data"
        ).resolve()
    )
    v["DATA_FOLDER"] = data_folder  # host path mounted at /data
    v["API_DATA_FOLDER"] = "/data/" if docker else data_folder  # path read by base.py
    v["YOLO_CONFIG_DIR"] = v["YOLO_CONFIG_DIR"] or str(
        Path(v["API_DATA_FOLDER"]) / "yolotmp"
    )
    v["REDIS_HOST"] = (
        "host.docker.internal"
        if docker and not root
        else "redis"
        if docker
        else "localhost"
    )
    if root:
        v["PROD_URL"] = root.get("PROD_API_URL", "").split("://")[-1] or v["PROD_URL"]
        v["BUNDLED"] = "True"  # api container joins the front compose network

    invalid = [a for a in v["INSTALLED_APPS"].split(",") if a and a not in API_APPS]
    if invalid:
        sys.exit(f"Invalid INSTALLED_APPS {invalid}, allowed: {list(API_APPS)}")

    ENV_FILE.write_text("\n".join(f"{k}={val}" for k, val in v.items()) + "\n")
    print(f"wrote {ENV_FILE}")
    (Path(v["DATA_FOLDER"])).mkdir(parents=True, exist_ok=True)
    return v


def render_confs(v: dict) -> None:
    nb_procs = len([a for a in v["INSTALLED_APPS"].split(",") if a]) or 1
    render(
        DOCKER_DIR / "nginx.conf.template",
        DOCKER_DIR / "nginx.conf",
        {"API_PORT": v["API_PORT"]},
    )
    render(
        DOCKER_DIR / "supervisord.conf.template",
        DOCKER_DIR / "supervisord.conf",
        {"NB_PROCS": nb_procs},
    )


def setup_dev(v: dict) -> None:
    import shutil

    uv = shutil.which("uv") or sys.exit(
        "uv is required in dev mode (https://docs.astral.sh/uv/)"
    )
    sh([uv, "sync", "--group=dev"], cwd=API)
    sh([uv, "tool", "install", "pre-commit", "--with", "pre-commit-uv"], cwd=API)
    subprocess.run(["git", "submodule", "update", "--init"], cwd=API)
    print("\n✅ api dev setup complete. Start it with:  python run.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument(
        "--root-env",
        type=Path,
        help="root .env when installed as part of the full aikon bundle",
    )
    parser.add_argument("--defaults", action="store_true")
    args = parser.parse_args()

    mode = args.mode or (
        input(f"Install mode ({' / '.join(MODES)}) [local]: ").strip() or "local"
    )
    if mode not in MODES:
        sys.exit(f"Invalid mode '{mode}'")

    v = resolve(mode, args.root_env, args.defaults or mode == "local")
    if mode == "dev":
        setup_dev(v)
    else:
        import shutil

        shutil.which("docker") or sys.exit(
            "docker is required (https://docs.docker.com/engine/install/)"
        )
        render_confs(v)
        sh([sys.executable, str(API / "run.py"), "build"], cwd=API)
