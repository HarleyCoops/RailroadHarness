# Copyright 2024 HarleyCoops. All rights reserved.
# Adapted from TRL's examples/scripts/openenv/opencode_hf_sandbox.py
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
#     "huggingface_hub>=1.22",
#     "vllm>=0.17.0,<=0.25.1",
#     "kernels==0.16.0",
#     "openenv @ git+https://github.com/huggingface/OpenEnv.git",
#     "openenv-opencode-env @ git+https://github.com/huggingface/OpenEnv.git#subdirectory=envs/opencode_env",
# ]
# ///

"""
AsyncGRPO training of the OpenCode agent on 1959 railroad operating-rule scenarios (REMOTE HF sandboxes).

This script adapts TRL's opencode_hf_sandbox.py (DeepCoder coding tasks) to the railroad domain:
- Dataset: HarleyCooper/volume2gym-railroad-1959 (2,708 synthetic scenarios)
- Agent task: Write operating-rule response to `answer.txt`
- Verifier: Deterministic similarity scoring (exact match / token F1)
- Reward: Verifier score + degeneracy penalties

Two vLLM URLs, on purpose:
- `--vllm-url` (default `http://localhost:8000`): the TRAINER <-> vLLM link. Stays local for NCCL weight-sync.
- `--sandbox-vllm-url`: a url the remote sandboxes use to reach that same vLLM (the in-sandbox proxy forwards
  there). Remote sandboxes cannot see `localhost`, so this must be reachable from outside: a public vLLM
  endpoint, or a tunnel to your local one.

The default sandbox image `ghcr.io/huggingface/openenv-opencode-sandbox:latest` pre-bakes the opencode CLI
+ the proxy under `/root`, so the harness skips the cold install.

Requirements:
- An OpenAI-compatible vLLM server reachable locally by the trainer and publicly by the sandboxes.
- An HF token with Jobs + Sandbox access in the environment (`HF_TOKEN`); each rollout is one HF sandbox.

Run (2 GPUs: vLLM on one, trainer on the other; a tunnel exposes vLLM to the sandboxes):

```sh
# Terminal 1 - serve the policy
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-8B-Instruct-2507 \\
    --host 0.0.0.0 --port 8000 \\
    --enable-auto-tool-choice --tool-call-parser hermes \\
    --logprobs-mode processed_logprobs \\
    --return-tokens-as-token-ids \\
    --max-model-len 98304 \\
    --weight-transfer-config '{"backend":"nccl"}'

# Terminal 2 - expose vLLM publicly for remote sandboxes
cloudflared tunnel --no-autoupdate --url http://localhost:8000   # prints https://<name>.trycloudflare.com

# Terminal 3 - train
CUDA_VISIBLE_DEVICES=1 python train_railroad_hf_sandbox.py \\
    --model Qwen/Qwen3-8B-Instruct-2507 \\
    --vllm-url http://localhost:8000 \\
    --sandbox-vllm-url https://<name>.trycloudflare.com
```

IMPORTANT: These are 1959 historical rules. Do NOT use for real railroad operations.
"""

from __future__ import annotations

import argparse
import random
from typing import Any

from datasets import Dataset, load_dataset
from opencode_env.config import OpenCodeConfig
from opencode_env.harness import OpenCodeSessionFactory
from opencode_env.sandbox import HFSandboxBackend, SandboxHandle
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

# The HF sandbox image bakes opencode + the proxy under `/root`
SANDBOX_IMAGE = "ghcr.io/huggingface/openenv-opencode-sandbox:latest"
SANDBOX_HOME = "/root"
WORKDIR = f"{SANDBOX_HOME}/workdir"
ANSWER_FILE = "answer.txt"

# Dataset configuration
DATASET = "HarleyCooper/volume2gym-railroad-1959"
DATASET_SPLIT_TRAIN = "train"

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
# Railroad verifier for HF sandbox (uses SANDBOX_HOME paths)
# ============================================================================================================


class HFSandboxRailroadVerifier:
    """
    Railroad verifier configured for HF sandbox paths (/root instead of /home/user).
    
    Holds reference responses keyed by instruction hash; called as verifier(sandbox, task).
    """
    
    def __init__(self, references_by_id: dict[str, str]):
        self._references_by_id = references_by_id
    
    def __call__(self, sandbox: SandboxHandle, task: OpenCodeTask) -> VerifyResult:
        inst_id = instruction_id(task.instruction)
        reference = self._references_by_id.get(inst_id)
        
        if reference is None:
            return VerifyResult(env_reward=0.0, done=True)
        
        answer_path = f"{WORKDIR}/{ANSWER_FILE}"
        if not sandbox.exists(answer_path):
            return VerifyResult(env_reward=0.0, done=True)
        
        try:
            candidate = sandbox.read_text(answer_path)
        except Exception:
            return VerifyResult(env_reward=0.0, done=True)
        
        reward = compute_similarity_reward(candidate, reference)
        return VerifyResult(env_reward=reward, done=True)


# ============================================================================================================
# OpenCode session factory (remote HF sandbox + in-sandbox proxy)
# ============================================================================================================


class RailroadTaskFactory(ResourceSessionFactory):
    """Adapts the worker's create() onto OpenCodeSessionFactory."""

    def __init__(self, inner: OpenCodeSessionFactory):
        self._inner = inner

    def create(self, task: Any, seed: int | None = None, episode_id: str | None = None) -> ResourceSession:
        instruction = task[-1]["content"] if isinstance(task, list) and task else str(task)
        return self._inner.create(instruction, seed=seed, episode_id=episode_id)


def build_factory(
    sandbox_vllm_url: str,
    model: str,
    references_by_id: dict[str, str],
    image: str,
    flavor: str,
) -> RailroadTaskFactory:
    """Build the harness session factory with railroad verifier for HF sandboxes."""
    config = OpenCodeConfig(
        provider="openai_compatible",
        base_url=f"{sandbox_vllm_url}/v1",  # proxy forwards here; must be public
        model=model,
        sandbox_home=SANDBOX_HOME,  # HF sandbox execs as root
        agent_timeout_s=600.0,  # remote hop adds latency
        disabled_tools=["webfetch", "question", "task"],
        run_format="json",
        proxy_max_tokens_cap=8192,
    )
    
    inner = OpenCodeSessionFactory(
        config=config,
        sandbox_backend=HFSandboxBackend(image=image, flavor=flavor),
        mode="transparent_proxy",
        verifier=HFSandboxRailroadVerifier(references_by_id),
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
    p.add_argument("--model", default="Qwen/Qwen3-8B-Instruct-2507")
    p.add_argument("--vllm-url", default="http://localhost:8000")
    p.add_argument("--sandbox-vllm-url", required=True)
    p.add_argument("--sandbox-image", default=SANDBOX_IMAGE)
    p.add_argument("--sandbox-flavor", default="cpu-basic")
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--max-inflight", type=int, default=8)
    p.add_argument("--max-completion-length", type=int, default=16384)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--n-prompts", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-staleness", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="async_grpo_railroad_hf_sandbox")
    p.add_argument("--project", default="railroad-hf-sandbox")
    p.add_argument(
        "--report-to",
        default="wandb",
        choices=["wandb", "trackio", "none"],
        help="Experiment logger (default: wandb).",
    )
    p.add_argument("--trackio-space-id", default=None)
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--hub-model-id", default=None)
    p.add_argument("--optim", default="adamw_torch")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows, references_by_id = build_dataset(n_prompts=args.n_prompts, seed=args.seed)
    dataset = Dataset.from_list(rows)

    print(f"[RailroadHarness] Loaded {len(rows)} scenarios from {DATASET}")
    print(f"[RailroadHarness] Model: {args.model}")
    print(f"[RailroadHarness] Sandbox vLLM URL: {args.sandbox_vllm_url}")

    config = AsyncGRPOConfig(
        output_dir=args.output_dir,
        save_strategy="no",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
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
        optim=args.optim,
        gradient_checkpointing=args.gradient_checkpointing,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    worker = HarnessRolloutWorker(
        harness_session_factory=build_factory(
            args.sandbox_vllm_url, args.model, references_by_id,
            args.sandbox_image, args.sandbox_flavor
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
    
    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
