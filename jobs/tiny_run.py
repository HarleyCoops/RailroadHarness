# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "trl",
#   "trackio",
#   "datasets",
#   "transformers",
#   "accelerate",
#   "huggingface_hub>=0.22",
#   "vllm",
#   "torch",
#   "openenv @ git+https://github.com/huggingface/OpenEnv.git",
#   "openenv-opencode-env @ git+https://github.com/huggingface/OpenEnv.git#subdirectory=envs/opencode_env",
# ]
# ///
"""Tiny RailroadHarness smoke job for HF Jobs (2 GPUs: vLLM + AsyncGRPO).

Launches vLLM on CUDA:0, then train_railroad_local.py on CUDA:1 against
HarleyCooper/volume2gym-railroad-1959 with a small prompt/step budget.
"""

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


def log(msg: str) -> None:
    print(msg, flush=True)


def wait_for_vllm(url: str, timeout_s: int = 900) -> None:
    health = f"{url}/health"
    models = f"{url}/v1/models"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        for endpoint in (health, models):
            try:
                with urllib.request.urlopen(endpoint, timeout=5) as resp:
                    if 200 <= resp.status < 300:
                        log(f"vLLM ready via {endpoint}")
                        return
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                pass
        time.sleep(5)
    raise RuntimeError(f"vLLM did not become ready within {timeout_s}s at {url}")


def main() -> int:
    log(f"Model={MODEL} n_prompts={N_PROMPTS} gens={NUM_GENERATIONS} steps={MAX_STEPS}")
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
    ]
    vllm_env = os.environ.copy()
    vllm_env["CUDA_VISIBLE_DEVICES"] = "0"
    vllm_env["VLLM_SERVER_DEV_MODE"] = "1"

    log("Starting vLLM on CUDA:0 ...")
    vllm_proc = subprocess.Popen(
        vllm_cmd,
        cwd=str(repo_dir),
        env=vllm_env,
        stdout=subprocess.PIPE,
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

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        wait_for_vllm(VLLM_URL)
        train_env = os.environ.copy()
        train_env["CUDA_VISIBLE_DEVICES"] = "1"
        train_cmd = [
            sys.executable,
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
        log(f"Starting trainer on CUDA:1: {' '.join(train_cmd)}")
        train = subprocess.run(train_cmd, cwd=str(repo_dir), env=train_env, check=False)
        log(f"Trainer exited with code {train.returncode}")
        return train.returncode
    finally:
        _shutdown()
        if vllm_proc.stdout is not None:
            # Drain a bit of vLLM log for debugging if it failed early
            try:
                leftover = vllm_proc.stdout.read()
                if leftover:
                    log("--- vLLM log (tail) ---")
                    log(leftover[-4000:])
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
