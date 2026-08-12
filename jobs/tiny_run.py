# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "trl @ git+https://github.com/huggingface/trl.git",
#   "trackio",
#   "datasets>=4.7.0",
#   "transformers>=4.56.2",
#   "accelerate>=1.4.0",
#   "huggingface_hub>=0.22",
#   "vllm>=0.17.0,<=0.25.1",
#   "torch",
#   "openenv @ git+https://github.com/huggingface/OpenEnv.git",
#   "openenv-opencode-env @ git+https://github.com/huggingface/OpenEnv.git#subdirectory=envs/opencode_env",
# ]
# ///
"""Tiny RailroadHarness smoke job for HF Jobs (2 GPUs: vLLM + AsyncGRPO)."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = "https://github.com/HarleyCoops/RailroadHarness.git"
MODEL = os.environ.get("RAILROAD_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
N_PROMPTS = int(os.environ.get("RAILROAD_N_PROMPTS", "8"))
NUM_GENERATIONS = int(os.environ.get("RAILROAD_NUM_GENERATIONS", "2"))
MAX_STEPS = int(os.environ.get("RAILROAD_MAX_STEPS", "5"))
MAX_INFLIGHT = int(os.environ.get("RAILROAD_MAX_INFLIGHT", "2"))
VLLM_PORT = int(os.environ.get("RAILROAD_VLLM_PORT", "8000"))
VLLM_URL = f"http://127.0.0.1:{VLLM_PORT}"
WORKDIR = Path("/tmp/railroad-harness-job")
VLLM_LOG = WORKDIR / "vllm.log"
TRAIN_LOG = WORKDIR / "trainer.log"


def log(msg: str) -> None:
    print(msg, flush=True)


def tail_file(path: Path, n: int = 80) -> str:
    if not path.exists():
        return f"(no log file: {path})"
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])


def wait_for_vllm(url: str, proc: subprocess.Popen, timeout_s: int = 900) -> None:
    health = f"{url}/health"
    models = f"{url}/v1/models"
    deadline = time.time() + timeout_s
    last_heartbeat = 0.0
    while time.time() < deadline:
        rc = proc.poll()
        if rc is not None:
            raise RuntimeError(
                f"vLLM exited early with code {rc}. Tail of {VLLM_LOG}:\n{tail_file(VLLM_LOG, 120)}"
            )
        for endpoint in (health, models):
            try:
                with urllib.request.urlopen(endpoint, timeout=5) as resp:
                    if 200 <= resp.status < 300:
                        log(f"vLLM ready via {endpoint}")
                        return
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                pass
        now = time.time()
        if now - last_heartbeat > 30:
            log(f"still waiting for vLLM... ({int(deadline - now)}s left)")
            last_heartbeat = now
        time.sleep(5)
    raise RuntimeError(
        f"vLLM did not become ready within {timeout_s}s at {url}. Tail of {VLLM_LOG}:\n{tail_file(VLLM_LOG, 120)}"
    )


def preflight() -> None:
    try:
        import trl  # noqa: F401
        from trl.experimental.async_grpo.openenv_harness import HarnessRolloutWorker  # noqa: F401

        log(f"preflight ok: trl={trl.__version__} has openenv_harness")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"preflight failed (need TRL from git with openenv_harness): {exc}") from exc


def main() -> int:
    os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
    os.environ.setdefault("HF_DEBUG", "1")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    log(f"Model={MODEL} n_prompts={N_PROMPTS} gens={NUM_GENERATIONS} steps={MAX_STEPS}")
    log(f"HF_TOKEN present: {bool(os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN'))}")
    preflight()
    WORKDIR.mkdir(parents=True, exist_ok=True)
    repo_dir = WORKDIR / "RailroadHarness"
    if not repo_dir.exists():
        log(f"Cloning {REPO}")
        subprocess.check_call(["git", "clone", "--depth", "1", REPO, str(repo_dir)])
    else:
        log("Repo already present")

    vllm_cmd = [
        "vllm",
        "serve",
        MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        "--logprobs-mode",
        "processed_logprobs",
        "--return-tokens-as-token-ids",
        "--weight-transfer-config",
        '{"backend":"nccl"}',
        "--gpu-memory-utilization",
        os.environ.get("RAILROAD_GPU_MEM_UTIL", "0.85"),
        "--max-model-len",
        os.environ.get("RAILROAD_MAX_MODEL_LEN", "16384"),
    ]
    vllm_env = os.environ.copy()
    vllm_env["CUDA_VISIBLE_DEVICES"] = "0"
    vllm_env["VLLM_SERVER_DEV_MODE"] = "1"

    log(f"Starting vLLM on CUDA:0 (logs -> {VLLM_LOG}) ...")
    log_f = open(VLLM_LOG, "w", encoding="utf-8")
    vllm_proc = subprocess.Popen(
        vllm_cmd,
        cwd=str(repo_dir),
        env=vllm_env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
    )

    def _shutdown(*_args: object) -> None:
        log("Shutting down vLLM")
        if vllm_proc.poll() is None:
            vllm_proc.send_signal(signal.SIGTERM)
            try:
                vllm_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                vllm_proc.kill()
        try:
            log_f.flush()
            log_f.close()
        except Exception:
            pass

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        wait_for_vllm(VLLM_URL, vllm_proc)
        train_env = os.environ.copy()
        train_env["CUDA_VISIBLE_DEVICES"] = "1"
        train_env["TRL_EXPERIMENTAL_SILENCE"] = "1"
        train_env["HF_DEBUG"] = "1"
        train_env["PYTHONUNBUFFERED"] = "1"
        # Ensure hub auth reaches the trainer child
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            train_env["HF_TOKEN"] = token
            train_env["HUGGING_FACE_HUB_TOKEN"] = token
        train_cmd = [
            sys.executable,
            "-u",
            "train_railroad_local.py",
            "--model",
            MODEL,
            "--vllm-url",
            VLLM_URL,
            "--n-prompts",
            str(N_PROMPTS),
            "--num-generations",
            str(NUM_GENERATIONS),
            "--max-steps",
            str(MAX_STEPS),
            "--max-inflight",
            str(MAX_INFLIGHT),
            "--output-dir",
            str(WORKDIR / "outputs"),
            "--project",
            "railroad-harness-tiny",
        ]
        log(f"Starting trainer on CUDA:1 (logs -> {TRAIN_LOG}): {' '.join(train_cmd)}")
        with open(TRAIN_LOG, "w", encoding="utf-8") as train_f:
            train = subprocess.run(
                train_cmd,
                cwd=str(repo_dir),
                env=train_env,
                check=False,
                stdout=train_f,
                stderr=subprocess.STDOUT,
                text=True,
            )
        log(f"Trainer exited with code {train.returncode}")
        log(f"--- trainer.log tail ---\n{tail_file(TRAIN_LOG, 120)}")
        if train.returncode != 0:
            log(f"--- vllm.log tail ---\n{tail_file(VLLM_LOG, 40)}")
        return train.returncode
    except Exception as exc:  # noqa: BLE001
        log(f"FATAL: {exc}")
        return 1
    finally:
        _shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
