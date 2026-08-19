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

"""Load the official held-out test-split fixture and score dummy answers."""

from __future__ import annotations

from pathlib import Path

import pytest

from railroad_verifier import (
    HELDOUT_FIXTURE_LIMIT,
    HELDOUT_SPLIT,
    RailroadScenarioVerifier,
    format_railroad_instruction,
    instruction_id,
    load_held_out_meta,
    load_held_out_subset,
    tokenize_whitespace,
)


EXPECTED_TASK_IDS = [
    "812-001",
    "S-H(1)-002",
    "307-004",
    "819-002",
    "958-001",
    "95-002",
    "23-005",
    "606-004",
]


class FakeSandbox:
    """Minimal sandbox stand-in for RailroadScenarioVerifier.__call__."""

    def __init__(self, files: dict[str, str] | None = None):
        self.files = files or {}

    def exists(self, path: str) -> bool:
        return path in self.files

    def read_text(self, path: str) -> str:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]


class FakeTask:
    def __init__(self, instruction: str):
        self.instruction = instruction


@pytest.fixture(scope="module")
def held_out():
    return load_held_out_subset(prefer_hub=False)


@pytest.fixture(scope="module")
def verifier(held_out):
    return RailroadScenarioVerifier.from_held_out(held_out, prefer_hub=False)


@pytest.fixture
def row(held_out):
    return held_out.rows[0]


def test_fixture_provenance_is_official_test_split():
    meta = load_held_out_meta()
    assert meta["dataset_id"] == "HarleyCooper/volume2gym-railroad-1959"
    assert meta["config"] == "default"
    assert meta["split"] == HELDOUT_SPLIT == "test"
    assert meta["hub_path"] == "data/test.jsonl"
    assert meta["row_offset"] == 0
    assert meta["row_count"] == HELDOUT_FIXTURE_LIMIT
    assert meta["task_ids"] == EXPECTED_TASK_IDS
    assert "Do NOT treat them as operational advice" in meta["historical_warning"]


def test_loads_committed_fixture_rows(held_out):
    assert held_out.source == "fixture"
    assert held_out.dataset_id == "HarleyCooper/volume2gym-railroad-1959"
    assert held_out.config == "default"
    assert held_out.split == "test"
    assert [row.task_id for row in held_out.rows] == EXPECTED_TASK_IDS
    assert len(held_out.references_by_id()) == len(EXPECTED_TASK_IDS)


def test_fixture_rows_have_hub_schema(held_out):
    for row in held_out.rows:
        assert row.task_id
        assert row.scenario
        assert row.reference_response
        assert row.prompt
        assert row.applicable_rules
        assert row.instruction_hash == instruction_id(
            format_railroad_instruction(row.scenario, row.task_id)
        )
        assert "HISTORICAL" in row.instruction
        assert row.task_id in row.instruction
        assert row.scenario in row.instruction
        assert row.reference_response not in row.instruction


def test_every_fixture_row_exact_matches_its_reference(verifier, held_out):
    for row in held_out.rows:
        result = verifier.score_directly(row.reference_response, row.reference_response)
        assert result.env_reward == 1.0
        assert result.exact_match is True


def test_exact_match_dummy_answer_txt(tmp_path: Path, verifier, row):
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "answer.txt").write_text(row.reference_response, encoding="utf-8")

    result = verifier.verify_answer_file(row.instruction, workdir)
    assert result.exact_match is True
    assert result.env_reward == 1.0
    assert result.f1_score == 1.0
    assert result.done is True


def test_normalized_exact_match_on_held_out_row(tmp_path: Path, verifier, row):
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "answer.txt").write_text(
        f"  {row.reference_response.upper()}  \n\n",
        encoding="utf-8",
    )

    result = verifier.verify_answer_file(row.instruction, workdir)
    assert result.exact_match is True
    assert result.env_reward == 1.0


def test_token_f1_partial_dummy_answer(tmp_path: Path, verifier, row):
    tokens = tokenize_whitespace(row.reference_response)
    assert len(tokens) >= 8
    dummy = " ".join(tokens[: max(3, len(tokens) // 3)])
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "answer.txt").write_text(dummy, encoding="utf-8")

    result = verifier.verify_answer_file(row.instruction, workdir)
    assert result.exact_match is False
    assert 0.0 < result.env_reward < 1.0
    assert result.env_reward == result.f1_score


def test_empty_payload_scores_zero(tmp_path: Path, verifier, row):
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "answer.txt").write_text("", encoding="utf-8")

    result = verifier.verify_answer_file(row.instruction, workdir)
    assert result.exact_match is False
    assert result.env_reward == 0.0
    assert result.f1_score == 0.0


def test_invalid_payload_scores_zero(tmp_path: Path, verifier, row):
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "answer.txt").write_text('{"not": "a railroad answer"}', encoding="utf-8")

    result = verifier.verify_answer_file(row.instruction, workdir)
    assert result.exact_match is False
    assert result.env_reward == 0.0
    assert result.f1_score == 0.0


def test_missing_answer_txt_scores_zero(tmp_path: Path, verifier, row):
    workdir = tmp_path / "workdir"
    workdir.mkdir()

    result = verifier.verify_answer_file(row.instruction, workdir)
    assert result.env_reward == 0.0
    assert result.exact_match is False
    assert result.f1_score == 0.0


def test_call_missing_answer_txt(verifier, row):
    sandbox = FakeSandbox()
    result = verifier(sandbox, FakeTask(row.instruction))
    assert result.env_reward == 0.0
    assert result.done is True


def test_call_exact_match_answer_txt(verifier, row):
    sandbox = FakeSandbox({verifier.answer_path: row.reference_response})
    result = verifier(sandbox, FakeTask(row.instruction))
    assert result.env_reward == 1.0
    assert result.done is True


def test_unknown_instruction_scores_zero(verifier):
    result = verifier.verify_candidate("not a held-out instruction", "anything")
    assert result.env_reward == 0.0
