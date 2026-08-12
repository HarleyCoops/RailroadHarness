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
Railroad scenario verifier for OpenCode harness training.

Scoring logic adapted from the volume2gym dataset card:
- Normalized exact match: lowercase + collapse whitespace
- Whitespace-token multiset F1 as fallback similarity signal

This verifier reads `answer.txt` from the sandbox workdir, compares it
to the held-out `reference_response` keyed by instruction hash.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opencode_env.sandbox.base import SandboxHandle
    from opencode_env.task import OpenCodeTask
    from openenv.core.harness import VerifyResult


def instruction_id(instruction: str) -> str:
    """Compute a stable hash for an instruction string (same pattern as TRL DeepCoder)."""
    return hashlib.sha1(instruction.encode()).hexdigest()


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, collapse whitespace, strip."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_whitespace(text: str) -> list[str]:
    """Split text on whitespace into tokens (for multiset F1)."""
    return text.split()


def exact_match_score(candidate: str, reference: str) -> float:
    """
    Normalized exact match: returns 1.0 if normalized texts are identical, else 0.0.
    
    This is the primary scoring signal when the model response exactly matches
    the expected reference.
    """
    norm_candidate = normalize_text(candidate)
    norm_reference = normalize_text(reference)
    return 1.0 if norm_candidate == norm_reference else 0.0


def multiset_f1_score(candidate: str, reference: str) -> float:
    """
    Whitespace-token multiset F1 score.
    
    Computes precision and recall based on token overlap (bag-of-words style),
    then returns the harmonic mean (F1). This provides partial credit when the
    model captures some but not all of the expected content.
    
    Returns a value in [0.0, 1.0].
    """
    cand_tokens = tokenize_whitespace(normalize_text(candidate))
    ref_tokens = tokenize_whitespace(normalize_text(reference))
    
    if not ref_tokens:
        return 1.0 if not cand_tokens else 0.0
    if not cand_tokens:
        return 0.0
    
    cand_counter = Counter(cand_tokens)
    ref_counter = Counter(ref_tokens)
    
    # Intersection: count each token up to min of occurrences in both
    intersection = sum((cand_counter & ref_counter).values())
    
    precision = intersection / len(cand_tokens) if cand_tokens else 0.0
    recall = intersection / len(ref_tokens) if ref_tokens else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_similarity_reward(candidate: str, reference: str) -> float:
    """
    Combined scoring: exact match takes precedence, F1 provides partial credit.
    
    Following the volume2gym dataset card:
    - If normalized exact match succeeds → 1.0
    - Otherwise → whitespace-token multiset F1
    
    This gives full credit for perfect answers and partial credit for
    partially correct responses.
    """
    exact = exact_match_score(candidate, reference)
    if exact >= 1.0 - 1e-9:
        return 1.0
    return multiset_f1_score(candidate, reference)


@dataclass
class RailroadVerifyResult:
    """Result of verifying a railroad scenario response."""
    env_reward: float
    done: bool = True
    exact_match: bool = False
    f1_score: float = 0.0


class RailroadScenarioVerifier:
    """
    Deterministic verifier for railroad operating-rule scenarios.
    
    Holds the reference responses keyed by instruction hash (sha1), so it
    survives pickle into the rollout child process. Called by the OpenCode
    session as `verifier(sandbox, task)`.
    
    Expects the agent to write its answer to `answer.txt` in the workdir.
    Missing file → reward of 0.0.
    """
    
    def __init__(
        self,
        references_by_id: dict[str, str],
        answer_filename: str = "answer.txt",
        workdir: str = "/home/user/workdir",
    ):
        """
        Args:
            references_by_id: Map from instruction_id(instruction) → reference_response
            answer_filename: Name of the file the agent should write (default: answer.txt)
            workdir: Path to the sandbox working directory
        """
        self._references_by_id = references_by_id
        self._answer_filename = answer_filename
        self._workdir = workdir
    
    @property
    def answer_path(self) -> str:
        """Full path to the expected answer file."""
        return f"{self._workdir}/{self._answer_filename}"
    
    def __call__(self, sandbox: "SandboxHandle", task: "OpenCodeTask") -> "VerifyResult":
        """
        Verify the agent's answer against the held-out reference.
        
        Args:
            sandbox: The sandbox handle with file read capabilities
            task: The OpenCode task containing the instruction
            
        Returns:
            VerifyResult with env_reward in [0, 1] and done=True
        """
        from openenv.core.harness import VerifyResult
        
        inst_id = instruction_id(task.instruction)
        reference = self._references_by_id.get(inst_id)
        
        if reference is None:
            # No reference found for this instruction - shouldn't happen
            return VerifyResult(env_reward=0.0, done=True)
        
        # Check if answer file exists
        if not sandbox.exists(self.answer_path):
            return VerifyResult(env_reward=0.0, done=True)
        
        try:
            candidate = sandbox.read_text(self.answer_path)
        except Exception:
            return VerifyResult(env_reward=0.0, done=True)
        
        reward = compute_similarity_reward(candidate, reference)
        return VerifyResult(env_reward=reward, done=True)
    
    def score_directly(self, candidate: str, reference: str) -> RailroadVerifyResult:
        """
        Score a candidate response directly (for testing without sandbox).
        
        Args:
            candidate: The model's response text
            reference: The expected reference response
            
        Returns:
            RailroadVerifyResult with detailed scoring breakdown
        """
        exact = exact_match_score(candidate, reference)
        f1 = multiset_f1_score(candidate, reference)
        is_exact = exact >= 1.0 - 1e-9
        
        return RailroadVerifyResult(
            env_reward=1.0 if is_exact else f1,
            done=True,
            exact_match=is_exact,
            f1_score=f1,
        )


# Convenience function for creating verifier from dataset
def build_verifier_from_references(
    references: dict[str, str],
    workdir: str = "/home/user/workdir",
) -> RailroadScenarioVerifier:
    """
    Build a verifier from a pre-computed references dictionary.
    
    Args:
        references: Map from instruction_id → reference_response
        workdir: Sandbox working directory path
        
    Returns:
        Configured RailroadScenarioVerifier
    """
    return RailroadScenarioVerifier(references, workdir=workdir)
