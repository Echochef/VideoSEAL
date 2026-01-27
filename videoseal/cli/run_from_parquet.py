from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _csv_list(s: str) -> list[str]:
    return [p.strip() for p in (s or "").split(",") if p.strip()]


def _require_int(name: str, value: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _require_positive_int(name: str, value: str) -> int:
    x = _require_int(name, value)
    if x < 1:
        raise ValueError(f"{name} must be a positive integer, got {x}")
    return x


def _require_non_negative_int(name: str, value: str) -> int:
    x = _require_int(name, value)
    if x < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {x}")
    return x


def _require_float(name: str, value: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a float, got {value!r}") from exc


def _env_flag(env: Mapping[str, str], name: str, *, default: bool = False) -> bool:
    v = (env.get(name) or "").strip().lower()
    if not v:
        return default
    return v in {"1", "true", "yes", "on"}


def _normalize_host_for_client(host: str) -> str:
    host = (host or "").strip() or "127.0.0.1"
    if host == "0.0.0.0":
        return "127.0.0.1"
    return host


def parse_shell_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens = shlex.split(line, posix=True)
        if len(tokens) != 1:
            raise ValueError(f"Invalid env line (expected 1 assignment): {raw!r}")
        key, sep, val = tokens[0].partition("=")
        if not sep or not key:
            raise ValueError(f"Invalid env assignment: {raw!r}")
        out[key] = val
    return out


@dataclass(frozen=True)
class VllmDiscovery:
    host: str
    base_port: int
    num_servers: int
    served_model_name: str
    meta_used: bool

    @property
    def local_openai_base(self) -> str:
        return f"http://{self.host}:{self.base_port}/v1"

    def openai_base_for_shard(self, shard_index: int) -> str:
        port = self.base_port + (int(shard_index) % int(self.num_servers))
        return f"http://{self.host}:{port}/v1"


def resolve_vllm_discovery(env: Mapping[str, str], *, repo_root: Path) -> VllmDiscovery:
    host_was_set = "VLLM_HOST" in env
    base_port_was_set = "VLLM_BASE_PORT" in env
    num_servers_was_set = "VLLM_NUM_SERVERS" in env

    host = (env.get("VLLM_HOST") or "").strip() or "127.0.0.1"
    port = (env.get("VLLM_PORT") or "").strip() or "18080"
    base_port = (env.get("VLLM_BASE_PORT") or "").strip() or port
    num_servers = (env.get("VLLM_NUM_SERVERS") or "").strip() or "1"

    model = (env.get("VLLM_MODEL") or "").strip() or "Qwen/Qwen3-8B"
    served_model_name = (env.get("VLLM_SERVED_MODEL_NAME") or "").strip() or os.path.basename(model.rstrip("/")) or "local-vllm"
    desired_model_id = served_model_name

    meta_used = False
    use_meta = (env.get("USE_VLLM_META_FILE") or "").strip() or "1"
    meta_file = (env.get("VLLM_META_FILE") or "").strip() or str(repo_root / "serve" / "vllm_servers.env")
    require_match = (env.get("VLLM_META_REQUIRE_MODEL_MATCH") or "").strip() or "1"

    if use_meta == "1":
        p = Path(meta_file)
        if p.is_file():
            meta = parse_shell_env_file(p)
            meta_host = (meta.get("VLLM_HOST") or "").strip()
            meta_base_port = (meta.get("VLLM_BASE_PORT") or "").strip()
            meta_num_servers = (meta.get("VLLM_NUM_SERVERS") or "").strip()
            meta_served = (meta.get("VLLM_SERVED_MODEL_NAME") or "").strip()

            meta_ok = True
            if require_match == "1" and desired_model_id and not meta_served:
                meta_ok = False
            if require_match == "1" and desired_model_id and meta_served and meta_served != desired_model_id:
                meta_ok = False

            if meta_ok:
                if (not host_was_set) and meta_host:
                    host = meta_host
                if (not base_port_was_set) and meta_base_port:
                    base_port = meta_base_port
                if (not num_servers_was_set) and meta_num_servers:
                    num_servers = meta_num_servers
                meta_used = True

    host_client = _normalize_host_for_client(host)
    base_port_i = _require_positive_int("VLLM_BASE_PORT", base_port)
    num_servers_i = _require_positive_int("VLLM_NUM_SERVERS", num_servers)
    return VllmDiscovery(
        host=host_client,
        base_port=base_port_i,
        num_servers=num_servers_i,
        served_model_name=served_model_name,
        meta_used=meta_used,
    )


def resolve_visual_inspect_base_pool(env: Mapping[str, str]) -> list[str]:
    explicit = _csv_list(env.get("VISUAL_INSPECT_API_BASES", "") or "")
    if explicit:
        return explicit

    tool_bases = _csv_list(env.get("TOOL_VLM_API_BASES", "") or "")
    if tool_bases:
        return tool_bases

    tool_host = _normalize_host_for_client((env.get("TOOL_VLM_HOST") or "").strip() or "127.0.0.1")
    tool_ports = _csv_list(env.get("TOOL_VLM_PORTS", "") or "")
    if tool_ports:
        ports = [_require_positive_int("TOOL_VLM_PORTS", p) for p in tool_ports]
        return [f"http://{tool_host}:{p}/v1" for p in ports]

    base_port_raw = (env.get("TOOL_VLM_BASE_PORT") or "").strip()
    if base_port_raw:
        base_port = _require_positive_int("TOOL_VLM_BASE_PORT", base_port_raw)
        n_raw = (env.get("TOOL_VLM_NUM_SERVERS") or "").strip() or "4"
        n = _require_positive_int("TOOL_VLM_NUM_SERVERS", n_raw)
        return [f"http://{tool_host}:{base_port + i}/v1" for i in range(n)]

    return []


def resolve_run_num_shards(*, vllm_num_servers: int, tool_pool_size: int, env: Mapping[str, str]) -> int:
    run_num_shards = int(max(1, vllm_num_servers))
    if tool_pool_size > run_num_shards:
        run_num_shards = int(tool_pool_size)

    force = (env.get("FORCE_NUM_SHARDS") or "").strip()
    if force:
        run_num_shards = _require_positive_int("FORCE_NUM_SHARDS", force)
    return int(run_num_shards)


@dataclass(frozen=True)
class ConcurrencyPlan:
    per_shard: int
    total: int


def resolve_concurrency(*, env: Mapping[str, str], num_shards: int) -> ConcurrencyPlan:
    conc_raw = (env.get("CONCURRENCY") or "").strip() or "64"
    conc = _require_non_negative_int("CONCURRENCY", conc_raw)

    per_raw = (env.get("CONCURRENCY_PER_SHARD") or "").strip()
    per = _require_non_negative_int("CONCURRENCY_PER_SHARD", per_raw) if per_raw else None

    if num_shards > 1 and per is None:
        total = conc
        per_shard = (total + num_shards - 1) // num_shards
    else:
        per_shard = per if per is not None else conc
        total = per_shard * num_shards

    per_shard = max(1, int(per_shard))
    total = max(1, int(per_shard) * int(num_shards))
    return ConcurrencyPlan(per_shard=per_shard, total=total)


def _child_env_base(repo_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    pp = env.get("PYTHONPATH", "")
    repo = repo_root.as_posix()
    env["PYTHONPATH"] = (pp + (":" if pp else "") + repo) if repo not in pp.split(":") else pp
    return env


def run_shards(
    *,
    repo_root: Path,
    runner_module: str,
    parquet: Path,
    save_runs: Path,
    video_id: str | None,
    uids_file: str | None,
    max_steps: int,
    task_timeout_sec: float,
    vllm: VllmDiscovery,
    run_num_shards: int,
    conc: ConcurrencyPlan,
    visual_inspect_pool: list[str],
) -> None:
    base_env = _child_env_base(repo_root)

    allow_tool_local = _env_flag(base_env, "ALLOW_TOOL_LOCAL_MLLM", default=False)
    if not allow_tool_local:
        local_bases = {vllm.openai_base_for_shard(i) for i in range(vllm.num_servers)}
        bad = [b for b in visual_inspect_pool if b.rstrip("/") in {x.rstrip("/") for x in local_bases}]
        if bad:
            raise SystemExit(
                "VISUAL_INSPECT API base points to agent vLLM; set ALLOW_TOOL_LOCAL_MLLM=1 to allow. "
                f"bad={bad[0]}"
            )

    agent_backend = (base_env.get("AGENT_LLM_BACKEND") or "").strip() or "api"
    agent_api_base = (base_env.get("AGENT_LLM_API_BASE") or "").strip()
    use_agent_local_ports = agent_backend == "api" and agent_api_base.rstrip("/") == vllm.local_openai_base.rstrip("/")

    procs: list[subprocess.Popen] = []
    for shard_index in range(run_num_shards):
        child_env = dict(base_env)

        # Tool-side VLM pool routing (per shard).
        if visual_inspect_pool:
            child_env["VISUAL_INSPECT_API_BASE"] = visual_inspect_pool[shard_index % len(visual_inspect_pool)]
        for src, dst in (
            ("TOOL_VLM_API_KEY", "VISUAL_INSPECT_API_KEY"),
            ("TOOL_VLM_MODEL", "VISUAL_INSPECT_MODEL"),
            ("TOOL_VLM_BACKEND", "VISUAL_INSPECT_BACKEND"),
        ):
            v = (child_env.get(src) or "").strip()
            if v:
                child_env[dst] = v

        # Agent-side API base routing (per shard) only when the agent points to the local vLLM base.
        if use_agent_local_ports:
            child_env["AGENT_LLM_API_BASE"] = vllm.openai_base_for_shard(shard_index)

        cmd = [
            sys.executable,
            "-m",
            runner_module,
            "--parquet",
            parquet.as_posix(),
            "--save-runs",
            save_runs.as_posix(),
            "--concurrency",
            str(conc.per_shard),
            "--max-steps",
            str(max_steps),
            "--task-timeout-sec",
            str(task_timeout_sec),
        ]
        if video_id:
            cmd += ["--video-id", video_id]
        if uids_file:
            cmd += ["--uids-file", uids_file]
        if run_num_shards > 1:
            cmd += ["--shard-index", str(shard_index), "--num-shards", str(run_num_shards), "--no-aggregate"]

        procs.append(subprocess.Popen(cmd, cwd=repo_root.as_posix(), env=child_env))

    for p in procs:
        ret = p.wait()
        if ret != 0:
            raise SystemExit(ret)


def main() -> int:
    repo_root = _repo_root()

    ap = argparse.ArgumentParser(description="Batch runner for agentrllm-style parquet (API/vLLM backends).")
    ap.add_argument("--parquet", default=os.getenv("PARQUET", ""))
    ap.add_argument("--save-runs", default=os.getenv("SAVE_RUNS", str(repo_root / "data" / "runs")))
    ap.add_argument("--video-id", default=os.getenv("VIDEO_ID", "") or None)
    ap.add_argument("--uids-file", default=os.getenv("UIDS_FILE", "") or None)
    ap.add_argument("--max-steps", type=int, default=int((os.getenv("MAX_STEPS") or "").strip() or "16"))
    ap.add_argument("--task-timeout-sec", type=float, default=float((os.getenv("TASK_TIMEOUT_SEC") or "").strip() or "1000"))
    ap.add_argument("--dry-run", action="store_true", default=_env_flag(os.environ, "DRY_RUN", default=False))
    ns = ap.parse_args()

    parquet_arg = str(ns.parquet or "").strip()
    if not parquet_arg:
        raise SystemExit("Missing parquet path. Set --parquet or export PARQUET.")
    parquet = Path(parquet_arg).expanduser()
    if not parquet.is_file():
        raise SystemExit(f"PARQUET not found: {parquet}")

    save_runs = Path(str(ns.save_runs)).expanduser()
    save_runs.mkdir(parents=True, exist_ok=True)

    vllm = resolve_vllm_discovery(os.environ, repo_root=repo_root)
    visual_inspect_pool = resolve_visual_inspect_base_pool(os.environ)
    agent_backend = (os.getenv("AGENT_LLM_BACKEND") or "").strip() or "api"
    runner_module = "videoseal.runner.per_question_runner" if agent_backend == "api" else "videoseal.runner.per_question_runner_vllm"

    run_num_shards = resolve_run_num_shards(vllm_num_servers=vllm.num_servers, tool_pool_size=len(visual_inspect_pool), env=os.environ)
    if agent_backend != "api":
        run_num_shards = 1
    conc = resolve_concurrency(env=os.environ, num_shards=run_num_shards)

    if ns.dry_run:
        print(
            f"[DRY_RUN] runner={runner_module} shards={run_num_shards} per_shard_concurrency={conc.per_shard} "
            f"vllm_host={vllm.host} vllm_base_port={vllm.base_port} vllm_num_servers={vllm.num_servers} meta_used={int(vllm.meta_used)}"
        )
        print(f"[DRY_RUN] parquet={parquet} save_runs={save_runs} video_id={ns.video_id or ''}")
        return 0

    run_shards(
        repo_root=repo_root,
        runner_module=runner_module,
        parquet=parquet,
        save_runs=save_runs,
        video_id=ns.video_id,
        uids_file=ns.uids_file,
        max_steps=int(ns.max_steps),
        task_timeout_sec=float(ns.task_timeout_sec),
        vllm=vllm,
        run_num_shards=run_num_shards,
        conc=conc,
        visual_inspect_pool=visual_inspect_pool,
    )

    cmd = [sys.executable, "-m", "videoseal.runner.backfill_preds", "--runs-root", save_runs.as_posix()]
    if ns.video_id:
        cmd += ["--video-id", ns.video_id]
    subprocess.run(cmd, cwd=repo_root.as_posix(), env=_child_env_base(repo_root), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
