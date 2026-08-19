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

Held-out rows come from HarleyCooper/volume2gym-railroad-1959, config
`default`, split `test` (270 rows). Tests and offline use load the
committed first-8-row fixture of that split.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

try:
    from datasets import load_dataset as hf_load_dataset
except ImportError:
    hf_load_dataset = None

try:
    from openenv.core.harness import VerifyResult as OpenEnvVerifyResult
except ImportError:
    OpenEnvVerifyResult = None

if TYPE_CHECKING:
    from opencode_env.sandbox.base import SandboxHandle
    from opencode_env.task import OpenCodeTask
    from openenv.core.harness import VerifyResult


DATASET_ID = "HarleyCooper/volume2gym-railroad-1959"
DATASET_CONFIG = "default"
HELDOUT_SPLIT = "test"
HELDOUT_FIXTURE_LIMIT = 8
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
HELDOUT_FIXTURE_PATH = FIXTURES_DIR / "volume2gym-railroad-1959.test.heldout.jsonl"
HELDOUT_FIXTURE_META_PATH = FIXTURES_DIR / "volume2gym-railroad-1959.test.heldout.meta.json"

HeldOutSource = Literal["hub", "fixture"]


def instruction_id(instruction: str) -> str:
    """Compute a stable hash for an instruction string (same pattern as TRL DeepCoder)."""
    return hashlib.sha1(instruction.encode()).hexdigest()


def format_railroad_instruction(scenario: str, task_id: str) -> str:
    """
    Format a railroad scenario as an instruction for the OpenCode agent.

    The agent must write its final answer to answer.txt using the write tool.
    No reference_response is visible — that's held out for the verifier.
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


def load_held_out_meta(path: Path | None = None) -> dict[str, Any]:
    """Load provenance for the committed held-out fixture."""
    meta_path = path or HELDOUT_FIXTURE_META_PATH
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _coerce_rules(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    return [str(item) for item in raw]


def _held_out_row_from_mapping(raw: dict[str, Any], source_index: int) -> HeldOutRow:
    task_id = str(raw.get("task_id") or "").strip()
    scenario = str(raw.get("scenario") or "")
    reference = str(raw.get("reference_response") or "")
    prompt = str(raw.get("prompt") or "")
    if not task_id or not scenario.strip() or not reference.strip():
        raise ValueError(f"Incomplete held-out row at index {source_index}")
    instruction = format_railroad_instruction(scenario, task_id)
    return HeldOutRow(
        task_id=task_id,
        scenario=scenario,
        reference_response=reference,
        prompt=prompt,
        applicable_rules=_coerce_rules(raw.get("applicable_rules")),
        source_index=source_index,
        instruction=instruction,
    )


def _load_raw_rows_from_fixture(path: Path, max_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= max_rows:
                break
            stripped = line.strip()
            if not stripped:
                continue
            parsed = json.loads(stripped)
            if not isinstance(parsed, dict):
                raise ValueError(f"Fixture row {len(rows)} is not a JSON object")
            rows.append(parsed)
    if not rows:
        raise RuntimeError(f"Held-out fixture {path} contained no rows")
    return rows


def _load_raw_rows_from_hub(max_rows: int) -> list[dict[str, Any]]:
    if hf_load_dataset is None:
        raise RuntimeError("the datasets package is not installed")

    ds = hf_load_dataset(DATASET_ID, name=DATASET_CONFIG, split=HELDOUT_SPLIT)
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(ds):
        if index >= max_rows:
            break
        rows.append(dict(raw))
    if not rows:
        raise RuntimeError(f"Hub dataset {DATASET_ID} split={HELDOUT_SPLIT} returned no rows")
    return rows


def load_held_out_subset(
    *,
    max_rows: int = HELDOUT_FIXTURE_LIMIT,
    prefer_hub: bool = True,
    fixture_path: Path | None = None,
) -> HeldOutSubset:
    """
    Load held-out railroad scenarios.

    Prefers the official Hub `test` split. If that download fails (or
    `prefer_hub=False`), uses the committed fixture subset of the same split.
    """
    path = fixture_path or HELDOUT_FIXTURE_PATH
    source: HeldOutSource = "fixture"
    raw_rows: list[dict[str, Any]] | None = None
    if prefer_hub:
        try:
            raw_rows = _load_raw_rows_from_hub(max_rows)
            source = "hub"
        except Exception:
            raw_rows = None
    if raw_rows is None:
        raw_rows = _load_raw_rows_from_fixture(path, max_rows)
        source = "fixture"
    rows = [_held_out_row_from_mapping(raw, index) for index, raw in enumerate(raw_rows)]
    return HeldOutSubset(
        dataset_id=DATASET_ID,
        config=DATASET_CONFIG,
        split=HELDOUT_SPLIT,
        source=source,
        rows=rows,
    )


@dataclass
class HeldOutRow:
    """One held-out volume2gym railroad scenario plus the verifier instruction."""

    task_id: str
    scenario: str
    reference_response: str
    prompt: str
    applicable_rules: list[str]
    source_index: int
    instruction: str

    @property
    def instruction_hash(self) -> str:
        return instruction_id(self.instruction)


@dataclass
class HeldOutSubset:
    """A loaded held-out slice of HarleyCooper/volume2gym-railroad-1959."""

    dataset_id: str
    config: str
    split: str
    source: HeldOutSource
    rows: list[HeldOutRow]

    def references_by_id(self) -> dict[str, str]:
        return {row.instruction_hash: row.reference_response for row in self.rows}


@dataclass
class RailroadVerifyResult:
    """Result of verifying a railroad scenario response."""

    env_reward: float
    done: bool = True
    exact_match: bool = False
    f1_score: float = 0.0


def _as_harness_result(result: RailroadVerifyResult) -> Any:
    if OpenEnvVerifyResult is not None:
        return OpenEnvVerifyResult(env_reward=result.env_reward, done=result.done)
    return result


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

    @classmethod
    def from_held_out(
        cls,
        subset: HeldOutSubset | None = None,
        *,
        prefer_hub: bool = True,
        answer_filename: str = "answer.txt",
        workdir: str = "/home/user/workdir",
    ) -> RailroadScenarioVerifier:
        """Build a verifier from the official held-out test split (or fixture)."""
        loaded = subset if subset is not None else load_held_out_subset(prefer_hub=prefer_hub)
        return cls(
            loaded.references_by_id(),
            answer_filename=answer_filename,
            workdir=workdir,
        )

    @property
    def answer_path(self) -> str:
        """Full path to the expected answer file."""
        return f"{self._workdir}/{self._answer_filename}"

    def verify_candidate(self, instruction: str, candidate: str | None) -> RailroadVerifyResult:
        """
        Score a candidate string against the held-out reference for `instruction`.

        `candidate=None` means answer.txt was missing. Empty or invalid text is
        still scored (typically 0.0 against a non-empty reference).
        """
        reference = self._references_by_id.get(instruction_id(instruction))
        if reference is None or candidate is None:
            return RailroadVerifyResult(
                env_reward=0.0,
                done=True,
                exact_match=False,
                f1_score=0.0,
            )
        return self.score_directly(candidate, reference)

    def verify_answer_file(self, instruction: str, workdir: str | Path) -> RailroadVerifyResult:
        """Read `answer.txt` from a local workdir and score it."""
        path = Path(workdir) / self._answer_filename
        if not path.is_file():
            return self.verify_candidate(instruction, None)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return self.verify_candidate(instruction, None)
        return self.verify_candidate(instruction, text)

    def __call__(self, sandbox: "SandboxHandle", task: "OpenCodeTask") -> "VerifyResult":
        """
        Verify the agent's answer against the held-out reference.

        Args:
            sandbox: The sandbox handle with file read capabilities
            task: The OpenCode task containing the instruction

        Returns:
            VerifyResult with env_reward in [0, 1] and done=True
        """
        instruction = task.instruction
        if not sandbox.exists(self.answer_path):
            return _as_harness_result(self.verify_candidate(instruction, None))
        try:
            candidate = sandbox.read_text(self.answer_path)
        except Exception:
            return _as_harness_result(self.verify_candidate(instruction, None))
        return _as_harness_result(self.verify_candidate(instruction, candidate))

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


def build_verifier_from_held_out(
    *,
    prefer_hub: bool = True,
    workdir: str = "/home/user/workdir",
) -> tuple[RailroadScenarioVerifier, HeldOutSubset]:
    """Load the held-out subset and return `(verifier, subset)`."""
    subset = load_held_out_subset(prefer_hub=prefer_hub)
    verifier = RailroadScenarioVerifier.from_held_out(subset, workdir=workdir)
    return verifier, subset
