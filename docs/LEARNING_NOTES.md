# Learning Notes: OpenCode Harness Training with Railroad Scenarios

This guide explains the key concepts in RailroadHarness, adapted from TRL's OpenEnv harness training.

## Table of Contents

1. [Loop-Owning Architecture](#loop-owning-architecture)
2. [Two vLLM URLs](#two-vllm-urls)
3. [Proxy Token Capture](#proxy-token-capture)
4. [Reward Flow](#reward-flow)
5. [DeepCoder → Railroad Mapping](#deepcoder--railroad-mapping)
6. [Source2Agent Hook](#source2agent-hook)
7. [Suggested First Run](#suggested-first-run)
8. [Troubleshooting](#troubleshooting)

---

## Loop-Owning Architecture

**What does "loop-owning" mean?**

In standard RL training, the trainer controls the agent's action loop:
```
Trainer → Generate action → Environment step → Get reward → Repeat
```

In **loop-owning** mode, the agent (OpenCode) controls its own loop:
```
Agent runs its own tool loop internally
    ↓
Proxy captures (token_ids, logprobs) for each turn
    ↓
TRL reads proxy trace after episode ends
    ↓
Verifier scores final workspace state
    ↓
Training happens on reconstructed trajectory
```

**Why loop-owning?**

OpenCode is a real coding agent with its own sophisticated tool loop (write, edit, bash, read). Rather than reimplementing that loop, we let OpenCode run naturally and capture what we need for training.

**Key setting:**
```python
worker = HarnessRolloutWorker(
    harness_adapter=None,  # None = loop-owning mode
    ...
)
```

---

## Two vLLM URLs

The training setup requires two separate URLs to the same vLLM server:

### `--vllm-url` (Trainer ↔ vLLM)
- **Purpose**: Local connection for NCCL weight synchronization
- **Default**: `http://localhost:8000`
- **Who uses it**: The trainer process on your GPU node
- **Why local**: NCCL weight transfers require low latency

### `--sandbox-vllm-url` (Sandbox ↔ vLLM)
- **Purpose**: URL the remote sandboxes use to reach vLLM
- **Required for**: Remote HF sandbox mode
- **Who uses it**: The OpenCode agent running in the sandbox
- **Why different**: Remote sandboxes can't see `localhost`

```
┌─────────────────────────────────────────────────────────────┐
│ Your GPU Node                                               │
│  ┌─────────┐    localhost:8000    ┌──────────────────────┐  │
│  │ Trainer │ ◄──────────────────► │ vLLM Server          │  │
│  └─────────┘    (NCCL sync)       │ (policy weights)     │  │
│                                   └──────────────────────┘  │
│                                            ▲                │
│                                            │ tunnel         │
└────────────────────────────────────────────│────────────────┘
                                             │
                   https://<name>.trycloudflare.com
                                             │
┌────────────────────────────────────────────│────────────────┐
│ HF Sandbox (remote)                        ▼                │
│  ┌─────────┐         ┌───────────────────────────────────┐  │
│  │OpenCode │ ◄─────► │ Proxy (captures tokens/logprobs)  │  │
│  │ Agent   │         │ forwards to sandbox-vllm-url      │  │
│  └─────────┘         └───────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Local mode**: Only needs `--vllm-url` (sandbox runs on same machine).

---

## Proxy Token Capture

The in-sandbox proxy is the secret sauce that makes training possible:

1. **Interception**: Sits between OpenCode and vLLM
2. **Forwarding**: Passes requests to the real vLLM server
3. **Capture**: Records `completion_token_ids` and `per_token_logps` for each turn
4. **Trace**: Writes a JSON trace file that TRL reads after the episode

**What gets captured:**
```json
{
  "request": {
    "messages": [...],
    "tools": [...]
  },
  "response": {
    "completion_token_ids": [1234, 5678, ...],
    "per_token_logps": [-0.5, -0.3, ...]
  }
}
```

**Why this matters for training:**
- GRPO needs token-level log probabilities
- Can't get these without capturing at generation time
- The trace lets TRL reconstruct the training signal

**Mode setting:**
```python
OpenCodeSessionFactory(
    mode="transparent_proxy",  # Enable proxy capture
    ...
)
```

---

## Reward Flow

The reward computation follows this path:

```
┌──────────────────────────────────────────────────────────────────┐
│ 1. Agent Episode                                                  │
│    Agent reads scenario → uses tools → writes answer.txt         │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ 2. Verifier (held-out reference)                                  │
│    Read answer.txt → Compare to reference_response               │
│    → compute_similarity_reward() → env_reward ∈ [0, 1]           │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ 3. Reward Function (degeneracy penalties)                         │
│    railroad_reward(outcome):                                     │
│      - No write calls? → -0.1 (penalize giving up)              │
│      - Timed out? → 0.0                                         │
│      - Otherwise: env_reward - step_penalty                     │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ 4. GRPO Training                                                  │
│    Group-relative advantage across num_generations rollouts      │
│    → Update policy weights via NCCL sync to vLLM                 │
└──────────────────────────────────────────────────────────────────┘
```

**Scoring detail (from volume2gym dataset card):**
1. Normalize both texts (lowercase, collapse whitespace)
2. Check exact match → if yes, reward = 1.0
3. Otherwise, compute whitespace-token multiset F1

---

## DeepCoder → Railroad Mapping

| Aspect | DeepCoder | Railroad |
|--------|-----------|----------|
| **Task** | Solve competitive coding problem | Respond to operating-rule scenario |
| **Agent output** | `solution.py` | `answer.txt` |
| **Verification** | Run code on stdin/stdout tests | Text similarity to reference |
| **Reward signal** | Fraction of tests passed (dense) | Exact match or token F1 (dense) |
| **Required tool** | `bash` (must run code) | `write` or `edit` (must write answer) |
| **Penalty: no action** | -0.1 if no bash | -0.1 if no write/edit |
| **Step budget** | 20 tool calls | 15 tool calls |
| **Self-check loop** | Run on example cases in problem | Read scenario, reason about rules |

**Key insight**: Both use dense rewards (partial credit) rather than sparse binary rewards, which helps GRPO learn from unsuccessful attempts.

---

## Source2Agent Hook

RailroadHarness is part of the **Source2Agent / volume2gym** pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│ volume2gym Pipeline (upstream)                                   │
│                                                                  │
│ 1. Source scans (117 pages) → OCR/extraction                    │
│ 2. Structured rules (536 canonical rules)                        │
│ 3. Task generation (2,708 scenarios with references)            │
│ 4. Dataset release: HarleyCooper/volume2gym-railroad-1959       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ RailroadHarness (this repo)                                      │
│                                                                  │
│ 5. Load dataset at runtime (no redistribution)                  │
│ 6. Format scenarios as agent instructions                        │
│ 7. Run harness training (AsyncGRPO + OpenCode)                  │
│ 8. Score against held-out references                            │
└─────────────────────────────────────────────────────────────────┘
```

**Design principle**: RailroadHarness doesn't redistribute the dataset or source materials. It loads from Hugging Face Hub at runtime, keeping the harness clean and the dataset authoritative.

---

## Suggested First Run

### Minimal Local Test (no GPU training)

Verify the verifier works:
```bash
cd /path/to/RailroadHarness
pip install -e ".[dev]"
pytest tests/test_verifier.py -v
```

### Tiny Training Run (2 GPUs)

```bash
# Terminal 1: Start vLLM (use smaller model for testing)
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-4B-Instruct-2507 \
    --host 0.0.0.0 --port 8000 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --logprobs-mode processed_logprobs \
    --return-tokens-as-token-ids \
    --weight-transfer-config '{"backend":"nccl"}'

# Terminal 2: Run training with minimal settings
CUDA_VISIBLE_DEVICES=1 python train_railroad_local.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --vllm-url http://localhost:8000 \
    --n-prompts 8 \           # Just 8 scenarios
    --num-generations 2 \     # 2 rollouts per prompt
    --max-steps 5 \           # Stop after 5 steps
    --max-inflight 2          # 2 concurrent sandboxes
```

**Expected**: Should see rollouts completing, rewards being computed, and a few training steps.

### Remote HF Sandbox Test

```bash
# Terminal 1: vLLM (same as above)

# Terminal 2: Tunnel
cloudflared tunnel --no-autoupdate --url http://localhost:8000
# Note the https://<name>.trycloudflare.com URL

# Terminal 3: Train with remote sandboxes
CUDA_VISIBLE_DEVICES=1 python train_railroad_hf_sandbox.py \
    --model Qwen/Qwen3-8B-Instruct-2507 \
    --vllm-url http://localhost:8000 \
    --sandbox-vllm-url https://<name>.trycloudflare.com \
    --n-prompts 8 \
    --num-generations 2 \
    --max-steps 5
```

---

## Troubleshooting

### "Proxy did not start"
- Check if the port is free
- Look at proxy logs in the sandbox

### "No HF_TOKEN" or sandbox auth errors
- Set `HF_TOKEN` environment variable with Jobs+Sandbox access
- Get token from https://huggingface.co/settings/tokens

### Low/zero rewards
- Check if `answer.txt` is being created
- Verify the instruction format matches what the agent expects
- Look at logged completions for agent behavior

### NCCL errors
- Ensure `--weight-transfer-config '{"backend":"nccl"}'` is set on vLLM
- Check CUDA_VISIBLE_DEVICES don't overlap between vLLM and trainer

### Rollouts timing out
- Increase `agent_timeout_s` in OpenCodeConfig
- For remote sandboxes, increase further due to network latency

---

## Further Reading

- [TRL OpenEnv Harness Blog](https://huggingface.co/blog/sergiopaniego/trl-openenv-harness-training)
- [OpenEnv Documentation](https://huggingface.co/docs/openenv)
- [volume2gym Dataset Card](https://huggingface.co/datasets/HarleyCooper/volume2gym-railroad-1959)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [AsyncGRPO Paper](https://arxiv.org/abs/2402.03300)
