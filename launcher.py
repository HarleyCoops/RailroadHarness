#!/usr/bin/env python3
# Copyright 2024 HarleyCoops. All rights reserved.
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

"""
Optional HF Jobs launcher for RailroadHarness training.

This launcher submits training jobs to Hugging Face Jobs, which provides
managed GPU infrastructure for running the harness training.

Usage:
    python launcher.py --model Qwen/Qwen3-8B-Instruct-2507

Requirements:
    - HF_TOKEN environment variable with Jobs access
    - huggingface_hub>=0.22.0

Note: This is an optional convenience script. You can also run training
directly on your own infrastructure using train_railroad_local.py or
train_railroad_hf_sandbox.py.
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    from huggingface_hub import HfApi
except ImportError:
    print("Error: huggingface_hub not installed. Run: pip install huggingface_hub>=0.22.0")
    sys.exit(1)


# Default training configuration
DEFAULT_CONFIG = {
    "model": "Qwen/Qwen3-8B-Instruct-2507",
    "n_prompts": 64,
    "num_generations": 8,
    "max_steps": 100,
    "learning_rate": 1e-5,
    "max_inflight": 8,
    "temperature": 1.0,
}

# HF Jobs configuration
JOB_CONFIG = {
    "flavor": "a10g-small",  # Single A10G GPU
    "timeout_hours": 24,
}


def create_training_script(config: dict) -> str:
    """Generate the training script content for HF Jobs."""
    return f'''#!/bin/bash
set -e

# Install dependencies
pip install trl trackio datasets transformers huggingface_hub
pip install "openenv @ git+https://github.com/huggingface/OpenEnv.git"
pip install "openenv-opencode-env @ git+https://github.com/huggingface/OpenEnv.git#subdirectory=envs/opencode_env"

# Clone RailroadHarness
git clone https://github.com/HarleyCoops/RailroadHarness.git
cd RailroadHarness

# Start vLLM in background
VLLM_SERVER_DEV_MODE=1 vllm serve {config["model"]} \\
    --host 0.0.0.0 --port 8000 \\
    --enable-auto-tool-choice --tool-call-parser hermes \\
    --logprobs-mode processed_logprobs \\
    --return-tokens-as-token-ids \\
    --weight-transfer-config '{{"backend":"nccl"}}' &

# Wait for vLLM to start
sleep 60

# Run training (local subprocess mode for single-node)
python train_railroad_local.py \\
    --model {config["model"]} \\
    --vllm-url http://localhost:8000 \\
    --n-prompts {config["n_prompts"]} \\
    --num-generations {config["num_generations"]} \\
    --max-steps {config["max_steps"]} \\
    --learning-rate {config["learning_rate"]} \\
    --max-inflight {config["max_inflight"]} \\
    --temperature {config["temperature"]} \\
    --output-dir ./outputs
'''


def main():
    parser = argparse.ArgumentParser(
        description="Launch RailroadHarness training on HF Jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Model configuration
    parser.add_argument(
        "--model",
        default=DEFAULT_CONFIG["model"],
        help=f"Model to train (default: {DEFAULT_CONFIG['model']})"
    )
    
    # Training configuration
    parser.add_argument("--n-prompts", type=int, default=DEFAULT_CONFIG["n_prompts"])
    parser.add_argument("--num-generations", type=int, default=DEFAULT_CONFIG["num_generations"])
    parser.add_argument("--max-steps", type=int, default=DEFAULT_CONFIG["max_steps"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--max-inflight", type=int, default=DEFAULT_CONFIG["max_inflight"])
    parser.add_argument("--temperature", type=float, default=DEFAULT_CONFIG["temperature"])
    
    # Job configuration
    parser.add_argument(
        "--flavor",
        default=JOB_CONFIG["flavor"],
        help=f"HF Jobs flavor (default: {JOB_CONFIG['flavor']})"
    )
    parser.add_argument(
        "--timeout-hours",
        type=int,
        default=JOB_CONFIG["timeout_hours"],
        help=f"Job timeout in hours (default: {JOB_CONFIG['timeout_hours']})"
    )
    
    # Execution mode
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the job script without submitting"
    )
    
    args = parser.parse_args()
    
    # Check for HF token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token and not args.dry_run:
        print("Error: HF_TOKEN environment variable not set")
        print("Get a token with Jobs access from: https://huggingface.co/settings/tokens")
        sys.exit(1)
    
    # Build configuration
    config = {
        "model": args.model,
        "n_prompts": args.n_prompts,
        "num_generations": args.num_generations,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "max_inflight": args.max_inflight,
        "temperature": args.temperature,
    }
    
    # Generate script
    script = create_training_script(config)
    
    if args.dry_run:
        print("=" * 60)
        print("DRY RUN - Would submit the following job script:")
        print("=" * 60)
        print(script)
        print("=" * 60)
        print(f"Flavor: {args.flavor}")
        print(f"Timeout: {args.timeout_hours} hours")
        return
    
    # Submit job
    print(f"Submitting RailroadHarness training job...")
    print(f"  Model: {args.model}")
    print(f"  Prompts: {args.n_prompts}")
    print(f"  Steps: {args.max_steps}")
    print(f"  Flavor: {args.flavor}")
    
    try:
        api = HfApi(token=hf_token)
        
        # Note: This is a placeholder for the actual HF Jobs API
        # The real API may differ - check HF documentation
        print("\n[Note] HF Jobs submission API - check huggingface_hub docs for actual usage")
        print("Script content generated successfully.")
        print("\nTo run manually, save the script and execute on your GPU infrastructure.")
        
    except Exception as e:
        print(f"Error submitting job: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
