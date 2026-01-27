from __future__ import annotations

import argparse
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _csv_list(s: str) -> list[str]:
    return [p.strip() for p in (s or "").split(",") if p.strip()]


def _is_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("", int(port)))
        except OSError:
            return False
        return True


def _port_range_is_free(base_port: int, n: int) -> bool:
    return all(_is_port_free(int(base_port) + i) for i in range(int(n)))


def _detect_gpu_count() -> int:
    for cand in ("nvidia-smi", "/usr/bin/nvidia-smi", "/usr/local/nvidia/bin/nvidia-smi"):
        if cand == "nvidia-smi" and shutil_which("nvidia-smi") is None:
            continue
        if cand != "nvidia-smi" and not Path(cand).is_file():
            continue
        try:
            out = subprocess.check_output([cand, "--list-gpus"], stderr=subprocess.DEVNULL, text=True)
        except Exception:
            continue
        lines = [ln for ln in out.splitlines() if ln.strip()]
        if lines:
            return len(lines)
    return 1


def shutil_which(cmd: str) -> str | None:
    paths = (os.getenv("PATH") or "").split(os.pathsep)
    for p in paths:
        cand = Path(p) / cmd
        if cand.is_file() and os.access(cand.as_posix(), os.X_OK):
            return cand.as_posix()
    return None


def parse_gpu_ids(cuda_visible_devices: str) -> list[str]:
    ids = [x.strip() for x in _csv_list(cuda_visible_devices) if x.strip()]
    return ids


def render_meta_env(
    *,
    host: str,
    base_port: int,
    num_servers: int,
    model: str,
    served_model_name: str,
) -> str:
    return "\n".join(
        [
            f"VLLM_HOST={shlex.quote(host)}",
            f"VLLM_BASE_PORT={shlex.quote(str(base_port))}",
            f"VLLM_NUM_SERVERS={shlex.quote(str(num_servers))}",
            f"VLLM_MODEL={shlex.quote(model)}",
            f"VLLM_SERVED_MODEL_NAME={shlex.quote(served_model_name)}",
            "",
        ]
    )


def main() -> int:
    repo_root = _repo_root()

    ap = argparse.ArgumentParser(description="Launch multiple OpenAI-compatible vLLM servers (one per GPU).")
    ap.add_argument("--model", default=os.getenv("VLLM_MODEL", "Qwen/Qwen3-8B"))
    ap.add_argument("--served-model-name", default=os.getenv("VLLM_SERVED_MODEL_NAME", ""))
    ap.add_argument("--host", default=os.getenv("VLLM_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.getenv("VLLM_PORT", "18080")))
    ap.add_argument("--tensor-parallel-size", type=int, default=int(os.getenv("VLLM_TENSOR_PARALLEL", "1")))
    ap.add_argument("--gpu-memory-utilization", type=float, default=float(os.getenv("VLLM_GPU_MEM_UTIL", "0.9")))
    ap.add_argument("--max-servers", type=int, default=int(os.getenv("MAX_VLLM_SERVERS", "4")))
    ns = ap.parse_args()

    model = (ns.model or "").strip()
    if not model:
        raise SystemExit("VLLM model is required. Set --model or VLLM_MODEL.")
    served = (ns.served_model_name or "").strip() or os.path.basename(model.rstrip("/")) or "local-vllm"

    cuda_ids = parse_gpu_ids(os.getenv("CUDA_VISIBLE_DEVICES", ""))
    if not cuda_ids:
        count = _detect_gpu_count()
        cuda_ids = [str(i) for i in range(max(1, int(count)))]

    if ns.max_servers > 0:
        cuda_ids = cuda_ids[: int(ns.max_servers)]
    if not cuda_ids:
        cuda_ids = ["0"]

    base_port = int(ns.port)
    if not _port_range_is_free(base_port, len(cuda_ids)):
        orig = base_port
        step = 100
        tries = 50
        for _ in range(tries):
            base_port += step
            if _port_range_is_free(base_port, len(cuda_ids)):
                break
        if not _port_range_is_free(base_port, len(cuda_ids)):
            raise SystemExit(
                f"Ports {orig}..{orig + len(cuda_ids) - 1} are busy, and failed to find a free range. Set VLLM_PORT/--port."
            )
        print(f"[vLLM] WARN: ports {orig}..{orig + len(cuda_ids) - 1} busy; using base port {base_port}.", file=sys.stderr)

    meta_dir = repo_root / "serve"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_file = meta_dir / "vllm_servers.env"
    meta_file.write_text(
        render_meta_env(
            host=str(ns.host),
            base_port=int(base_port),
            num_servers=len(cuda_ids),
            model=model,
            served_model_name=served,
        ),
        encoding="utf-8",
    )

    print(f"[vLLM] Starting {len(cuda_ids)} OpenAI servers (tp={ns.tensor_parallel_size})", file=sys.stderr)
    print(f"  model             = {model}", file=sys.stderr)
    print(f"  served-model-name = {served}", file=sys.stderr)
    print(f"  host              = {ns.host}", file=sys.stderr)
    print(f"  base port         = {base_port}", file=sys.stderr)
    print(f"  meta file         = {meta_file.as_posix()}", file=sys.stderr)

    procs: list[subprocess.Popen] = []
    for idx, gid in enumerate(cuda_ids):
        port = base_port + idx
        print(f"  - gpu={gid}, port={port}", file=sys.stderr)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gid
        pp = env.get("PYTHONPATH", "")
        repo = repo_root.as_posix()
        env["PYTHONPATH"] = (pp + (":" if pp else "") + repo) if repo not in pp.split(":") else pp

        shm_name = (env.get("VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME") or "").strip()
        if not shm_name:
            env["VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME"] = f"VLLM_OBJECT_STORAGE_SHM_BUFFER_{port}_{gid}_{int(time.time() * 1000)}"
        elif len(cuda_ids) > 1:
            env["VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME"] = f"{shm_name}_{port}_{gid}_{int(time.time() * 1000)}"

        cmd = [
            sys.executable,
            "-m",
            "videoseal.serve.vllm_openai",
            "--model",
            model,
            "--served-model-name",
            served,
            "--host",
            str(ns.host),
            "--port",
            str(port),
            "--tensor-parallel-size",
            str(ns.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(ns.gpu_memory_utilization),
        ]
        procs.append(subprocess.Popen(cmd, cwd=repo_root.as_posix(), env=env))

    # Wait: if any server exits, terminate the rest and surface the error code.
    while procs:
        for p in list(procs):
            ret = p.poll()
            if ret is None:
                continue
            procs.remove(p)
            if ret != 0:
                for other in procs:
                    other.terminate()
                for other in procs:
                    other.wait()
                raise SystemExit(ret)
        time.sleep(0.2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

