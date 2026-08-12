# Copyright 2024 HarleyCoops. All rights reserved.
# Adapted from TRL's examples/scripts/openenv/opencode.py
# Original copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl @ git+https://github.com/huggingface/trl.git",
#     "wandb",
#     "trackio",
#     "datasets>=4.7.0",
#     "transformers>=4.56.2",
#     "accelerate>=1.4.0",
#     "vllm>=0.17.0,<=0.25.1",
#     "kernels==0.16.0",
#     "openenv @ git+https://github.com/huggingface/OpenEnv.git",
#     "openenv-opencode-env @ git+https://github.com/huggingface/OpenEnv.git#subdirectory=envs/opencode_env",
# ]
# ///

"""
AsyncGRPO training of the OpenCode agent on 1959 railroad operating-rule scenarios (local subprocess).

This script adapts TRL's opencode.py (DeepCoder coding tasks) to the railroad domain:
- Dataset: HarleyCooper/volume2gym-railroad-1959 (2,708 synthetic scenarios)
- Agent task: Write operating-rule response to `answer.txt`
- Verifier: Deterministic similarity scoring (exact match / token F1)
- Reward: Verifier score + degeneracy penalties

The agent owns its tool loop; TRL reads the proxy trace, rebuilds training rows,
scores via the held-out verifier, and trains with GRPO.

Architecture (loop-owning):
    Trainer ←→ vLLM (local, NCCL weight-sync)
                ↑
    Sandbox subprocess ← proxy captures (token_ids, logprobs)

Run (2 GPUs: vLLM on one, trainer on the other):

```sh
# Terminal 1 - serve the policy
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-4B-Instruct-2507 \\
    --host 0.0.0.0 --port 8000 \\
    --enable-auto-tool-choice --tool-call-parser hermes \\
    --logprobs-mode processed_logprobs \\
    --return-tokens-as-token-ids \\
    --weight-transfer-config '{"backend":"nccl"}'

# Terminal 2 - train
CUDA_VISIBLE_DEVICES=1 python train_railroad_local.py \\
    --model Qwen/Qwen3-4B-Instruct-2507 --vllm-url http://localhost:8000
```

IMPORTANT: These are 1959 historical rules. Do NOT use for real railroad operations.
"""

from __future__ import annotations

import argparse
import os
import random
import shlex
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from opencode_env import harness as oc_harness
from opencode_env.config import OpenCodeConfig
from opencode_env.harness import OpenCodeSessionFactory
from opencode_env.sandbox.base import ExecResult, SandboxHandle
from opencode_env.task import OpenCodeTask
from openenv.core.harness import ResourceSession, ResourceSessionFactory, VerifyResult
from transformers import AutoTokenizer

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.experimental.async_grpo.openenv_harness import (
    HarnessRolloutOutcome,
    HarnessRolloutWorker,
    TraceEntry,
    has_tool_call,
)

from railroad_verifier import (
    RailroadScenarioVerifier,
    instruction_id,
    compute_similarity_reward,
)

# Dataset configuration
DATASET = "HarleyCooper/volume2gym-railroad-1959"
DATASET_SPLIT_TRAIN = "train"

# Sandbox paths (remapped by LocalSandboxHandle)
SANDBOX_HOME = "/home/user"
WORKDIR = f"{SANDBOX_HOME}/workdir"
ANSWER_FILE = "answer.txt"

# ============================================================================================================
# Local subprocess sandbox backend (adapted from TRL opencode.py)
# ============================================================================================================

_OPENCODE_INSTALL = "curl -fsSL https://opencode.ai/install | bash -s -- --no-modify-path"


class LocalBgJob:
    """A background process (the opencode agent or its proxy) running directly on the node."""

    def __init__(self, popen: subprocess.Popen):
        self._p = popen

    @property
    def pid(self) -> int:
        return self._p.pid

    def wait(self, timeout: float | None = None) -> int:
        try:
            return self._p.wait(timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(str(e)) from e

    def kill(self) -> None:
        if self._p.poll() is not None:
            return
        try:
            pgid = os.getpgid(self._p.pid)
        except ProcessLookupError:
            return
        try:
            os.killpg(pgid, signal.SIGTERM)
            self._p.wait(timeout=5)
        except (subprocess.TimeoutExpired, Exception):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass


class LocalSandboxHandle:
    """One local 'sandbox' = a real directory on the node."""

    def __init__(
        self,
        root: str,
        *,
        home_alias: str = "/home/user",
        base_env: dict[str, str] | None = None,
        cleanup: bool = False,
    ):
        self._root = root
        self._alias = home_alias
        self._cleanup = cleanup
        self._env = {**os.environ, "HOME": root, **(base_env or {})}
        self._bg: list[LocalBgJob] = []

    @property
    def sandbox_id(self) -> str:
        return self._root

    def _remap(self, s: str | None) -> str | None:
        return s if s is None else s.replace(self._alias, self._root)

    def _run_env(self, envs: dict[str, str] | None) -> dict[str, str]:
        return {**self._env, **(envs or {})}

    def exec(self, cmd: str, *, envs=None, cwd=None, timeout: float | None = 60) -> ExecResult:
        try:
            p = subprocess.run(
                ["bash", "-lc", self._remap(cmd)],
                cwd=self._remap(cwd) or self._root,
                env=self._run_env(envs),
                capture_output=True,
                text=True,
                timeout=timeout,
                stdin=subprocess.DEVNULL,
            )
            return ExecResult(exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr)
        except subprocess.TimeoutExpired as e:
            return ExecResult(exit_code=124, stdout=e.stdout or "", stderr=f"timeout after {timeout}s")

    def start_bg(self, cmd: str, *, envs=None, cwd=None) -> LocalBgJob:
        p = subprocess.Popen(
            ["bash", "-lc", self._remap(cmd)],
            cwd=self._remap(cwd) or self._root,
            env=self._run_env(envs),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        job = LocalBgJob(p)
        self._bg.append(job)
        return job

    def write_text(self, path: str, content: str) -> None:
        path = self._remap(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content)

    def read_text(self, path: str) -> str:
        return Path(self._remap(path)).read_text()

    def exists(self, path: str) -> bool:
        return Path(self._remap(path)).exists()

    def kill(self) -> None:
        for job in self._bg:
            try:
                job.kill()
            except Exception:
                pass
        self._bg.clear()
        if self._cleanup:
            shutil.rmtree(self._root, ignore_errors=True)


class LocalSubprocessSandboxBackend:
    """Produces per-rollout LocalSandboxHandles with pre-installed opencode."""

    def __init__(self, root: str, *, home_alias: str = "/home/user"):
        self._root = root
        self._alias = home_alias
        self._template = os.path.join(root, "_template")

    def warmup(self) -> None:
        """Install opencode ONCE into the template dir."""
        marker = os.path.join(self._template, ".opencode", "bin", "opencode")
        if os.path.exists(marker):
            return
        os.makedirs(self._template, exist_ok=True)
        subprocess.run(
            ["bash", "-lc", _OPENCODE_INSTALL],
            env={**os.environ, "HOME": self._template},
            check=True,
            timeout=400,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def create(self, *, timeout_s: int = 900, envs=None, metadata=None) -> LocalSandboxHandle:
        name = (metadata or {}).get("episode_id") or uuid.uuid4().hex
        sdir = os.path.join(self._root, name)
        shutil.rmtree(sdir, ignore_errors=True)
        os.makedirs(sdir, exist_ok=True)
        if os.path.isdir(self._template):
            subprocess.run(["cp", "-al", f"{self._template}/.", f"{sdir}/"], check=True)
        for sub in ("workdir", "task", "logs/agent", "logs/verifier", ".config/opencode"):
            d = os.path.join(sdir, sub)
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)
        return LocalSandboxHandle(sdir, home_alias=self._alias, base_env=envs, cleanup=True)


# ============================================================================================================
# Dataset loading + instruction formatting
# ============================================================================================================


def _railroad_instruction(scenario: str, task_id: str) -> str:
    """
    Format a railroad scenario as an instruction for the OpenCode agent.
    
    The agent must write its final answer to answer.txt using the write tool.
    No reference_response is visible - that's held out for the verifier.
    """
    return (
        "You are responding to a railroad operating-rule scenario from the 1959 "
        "Consolidated Code of Operating Rules. Read the scenario carefully and write "
        "your response to `answer.txt` in the current working directory using the `write` tool.\n\n"
        "Your response should explain the correct operating procedure based on the 1959 rules. "
        "Be specific about which rules apply and what actions should be taken.\n\n"
        "IMPORTANT: This is a HISTORICAL training scenario from 1959. These rules are outdated "
        "and must NOT be used for actual railroad operations.\n\n"
        f"SCENARIO (Task {task_id}):\n{scenario}\n\n"
        "Write your complete response to `answer.txt` now."
    )


def build_dataset(
    n_prompts: int,
    seed: int,
    split: str = DATASET_SPLIT_TRAIN,
) -> tuple[list[dict], dict[str, str]]:
    """
    Load the railroad dataset and prepare prompts + held-out references.
    
    Returns:
        (rows, references_by_id): Prompt rows for training and reference map for verifier
    """
    ds = load_dataset(DATASET, split=split)
    rows_list = list(ds)
    random.Random(seed).shuffle(rows_list)
    
    out: list[dict] = []
    references_by_id: dict[str, str] = {}
    
    for r in rows_list:
        if len(out) >= n_prompts:
            break
        
        scenario = r.get("scenario", "")
        reference = r.get("reference_response", "")
        task_id = r.get("task_id", "")
        
        if not scenario or not reference:
            continue
        
        instruction = _railroad_instruction(scenario, task_id)
        inst_id = instruction_id(instruction)
        
        references_by_id[inst_id] = reference
        out.append({"prompt": [{"role": "user", "content": instruction}]})
    
    return out, references_by_id


# ============================================================================================================
# OpenCode session factory (local sandbox + per-session proxy port)
# ============================================================================================================


class RailroadTaskFactory(ResourceSessionFactory):
    """Adapts the worker's create() onto OpenCodeSessionFactory."""

    def __init__(self, inner: OpenCodeSessionFactory):
        self._inner = inner

    def create(self, task: Any, seed: int | None = None, episode_id: str | None = None) -> ResourceSession:
        instruction = task[-1]["content"] if isinstance(task, list) and task else str(task)
        return self._inner.create(instruction, seed=seed, episode_id=episode_id)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FreePortOpenCodeSessionFactory(OpenCodeSessionFactory):
    """OpenCodeSessionFactory that binds proxy to a free port per session."""

    def _start_proxy(self, sandbox):
        port = _free_port()
        trace_path = oc_harness._PROXY_TRACE_PATH
        log_path = oc_harness._PROXY_LOG_PATH
        if not sandbox.exists("/home/user/proxy/interception.py"):
            self._exec_with_retry(
                sandbox,
                "pip install --quiet 'fastapi>=0.104' 'uvicorn[standard]>=0.24' 'httpx>=0.27' 2>&1 | tail -20",
                timeout=180,
                attempts=3,
                backoff_s=2.0,
                label="proxy deps install",
            )
            sandbox.write_text("/home/user/proxy/interception.py", oc_harness._PROXY_SOURCE_PATH.read_text())
            sandbox.write_text("/home/user/proxy/__init__.py", "")

        proxy_args = [
            "python", "interception.py", "--upstream-url", self._config.base_url,
            "--trace", trace_path, "--port", str(port),
            "--top-logprobs", str(self._config.proxy_top_logprobs),
        ]
        if self._config.proxy_max_tokens_cap is not None:
            proxy_args += ["--max-tokens-cap", str(self._config.proxy_max_tokens_cap)]
        if self._config.proxy_disable_thinking:
            proxy_args.append("--disable-thinking")
        if self._config.model:
            proxy_args += ["--model-override", self._config.model]

        quoted = " ".join(shlex.quote(a) for a in proxy_args)
        proxy_cmd = f"cd /home/user/proxy && {quoted} > {shlex.quote(log_path)} 2>&1"
        proxy_job = sandbox.start_bg(proxy_cmd, envs={"OPENCODE_UPSTREAM_API_KEY": self._config.api_key})

        for _ in range(120):
            if sandbox.exec(f"curl -sf http://127.0.0.1:{port}/healthz", timeout=5).exit_code == 0:
                break
            time.sleep(0.5)
        else:
            log = ""
            try:
                log = sandbox.read_text(log_path)
            except Exception:
                pass
            proxy_job.kill()
            raise RuntimeError(f"proxy did not start on :{port}\n{log[-2000:]}")

        return proxy_job, f"http://127.0.0.1:{port}/v1", trace_path


def build_factory(
    sandbox_root: str,
    vllm_url: str,
    model: str,
    references_by_id: dict[str, str],
) -> RailroadTaskFactory:
    """Build the harness session factory with railroad verifier."""
    config = OpenCodeConfig(
        provider="openai_compatible",
        base_url=f"{vllm_url}/v1",
        model=model,
        sandbox_home=SANDBOX_HOME,
        agent_timeout_s=180.0,
        disabled_tools=["webfetch", "question", "task"],
        run_format="json",
    )
    backend = LocalSubprocessSandboxBackend(sandbox_root)
    backend.warmup()
    
    verifier = RailroadScenarioVerifier(references_by_id, workdir=WORKDIR)
    
    inner = FreePortOpenCodeSessionFactory(
        config=config,
        sandbox_backend=backend,
        mode="transparent_proxy",
        verifier=verifier,
    )
    return RailroadTaskFactory(inner)


# ============================================================================================================
# Reward + turn-selection policy
# ============================================================================================================


def railroad_reward(outcome: HarnessRolloutOutcome) -> float | None:
    """
    Reward function for railroad scenarios.
    
    - Unscorable rollout → None (dropped from group baseline)
    - Never wrote answer file (no `write` tool calls) → -0.1
    - Otherwise: verifier score (0-1) minus step penalty for excessive tool calls
    
    Unlike DeepCoder, we don't require `bash` since the task is text writing,
    not code execution. We penalize not using `write` instead.
    """
    step_budget, step_penalty, step_penalty_cap = 15, 0.03, 0.5
    
    frac = outcome.env_reward
    if frac is None:
        return None
    
    # Penalize if agent never attempted to write
    write_calls = outcome.tool_calls_by_name.get("write", 0)
    edit_calls = outcome.tool_calls_by_name.get("edit", 0)
    if write_calls == 0 and edit_calls == 0:
        return -0.1
    
    base = 0.0 if outcome.timed_out else frac
    over = max(0, outcome.tool_call_count - step_budget)
    return base - min(step_penalty_cap, step_penalty * over)


def railroad_agent_turns(trace: list[TraceEntry]) -> list[TraceEntry]:
    """
    Keep only the REAL agent turns (drop opencode's title/summarizer aux calls).
    
    Anchors on the first tool-enabled turn's system prompt and keeps only matching entries.
    """
    def system_of(messages):
        return next((m.get("content") for m in messages if m.get("role") == "system"), None)

    primary = None
    for entry in trace:
        request = entry.get("request") or {}
        if request.get("messages") and request.get("tools"):
            primary = system_of(request["messages"])
            break
    
    return [
        entry
        for entry in trace
        if (request := entry.get("request") or {}).get("messages")
        and request.get("tools")
        and system_of(request["messages"]) == primary
    ]


# ============================================================================================================
# Training
# ============================================================================================================


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--vllm-url", default="http://localhost:8000")
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--max-inflight", type=int, default=8)
    p.add_argument("--max-completion-length", type=int, default=16384)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--n-prompts", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-staleness", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="async_grpo_railroad")
    p.add_argument("--project", default="railroad-harness")
    p.add_argument(
        "--report-to",
        default="wandb",
        choices=["wandb", "trackio", "none"],
        help="Experiment logger (default: wandb). Use trackio if WANDB_API_KEY is unavailable.",
    )
    p.add_argument("--trackio-space-id", default=None)
    p.add_argument("--sandbox-root", default=None)
    args = p.parse_args()

    sandbox_root = args.sandbox_root or tempfile.mkdtemp(prefix="trl_railroad_")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows, references_by_id = build_dataset(n_prompts=args.n_prompts, seed=args.seed)
    dataset = Dataset.from_list(rows)

    print(f"[RailroadHarness] Loaded {len(rows)} scenarios from {DATASET}")
    print(f"[RailroadHarness] Sandbox root: {sandbox_root}")
    print(f"[RailroadHarness] Model: {args.model}")
    print(f"[RailroadHarness] report_to={args.report_to} project={args.project}")

    config = AsyncGRPOConfig(
        output_dir=args.output_dir,
        save_strategy="no",
        per_device_train_batch_size=4,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        max_staleness=args.max_staleness,
        vllm_server_base_url=args.vllm_url,
        report_to=None if args.report_to == "none" else args.report_to,
        project=args.project,
        trackio_space_id=args.trackio_space_id if args.report_to == "trackio" else None,
        log_completions=True,
    )

    worker = HarnessRolloutWorker(
        harness_session_factory=build_factory(
            sandbox_root, args.vllm_url, args.model, references_by_id
        ),
        harness_adapter=None,  # loop-owning
        rollout_reward_fn=railroad_reward,
        train_turn_fn=has_tool_call,
        agent_turn_fn=railroad_agent_turns,
        model_name=args.model,
        dataset=dataset,
        reward_funcs=[],
        processing_class=tokenizer,
        num_generations=args.num_generations,
        max_inflight_tasks=args.max_inflight,
        vllm_server_url=args.vllm_url,
        max_tokens=args.max_completion_length,
        temperature=args.temperature,
        fork_threshold_tokens=1024,
        log_completions=True,
        num_completions_to_print=2,
    )

    trainer = AsyncGRPOTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        rollout_worker=worker,
    )
    trainer.train()


if __name__ == "__main__":
    main()
