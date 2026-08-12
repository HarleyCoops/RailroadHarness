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

"""Unit tests for the railroad scenario verifier."""

import pytest

from railroad_verifier import (
    normalize_text,
    tokenize_whitespace,
    exact_match_score,
    multiset_f1_score,
    compute_similarity_reward,
    instruction_id,
    RailroadScenarioVerifier,
)


class TestNormalizeText:
    """Tests for text normalization."""
    
    def test_lowercase(self):
        assert normalize_text("HELLO WORLD") == "hello world"
    
    def test_collapse_whitespace(self):
        assert normalize_text("hello   world") == "hello world"
        assert normalize_text("hello\n\nworld") == "hello world"
        assert normalize_text("hello\t\tworld") == "hello world"
    
    def test_strip(self):
        assert normalize_text("  hello world  ") == "hello world"
    
    def test_combined(self):
        assert normalize_text("  HELLO   WORLD\n\n") == "hello world"
    
    def test_empty(self):
        assert normalize_text("") == ""
        assert normalize_text("   ") == ""


class TestTokenizeWhitespace:
    """Tests for whitespace tokenization."""
    
    def test_basic(self):
        assert tokenize_whitespace("hello world") == ["hello", "world"]
    
    def test_multiple_spaces(self):
        assert tokenize_whitespace("hello   world") == ["hello", "world"]
    
    def test_empty(self):
        assert tokenize_whitespace("") == []
    
    def test_single_token(self):
        assert tokenize_whitespace("hello") == ["hello"]


class TestExactMatchScore:
    """Tests for exact match scoring."""
    
    def test_exact_match(self):
        assert exact_match_score("hello world", "hello world") == 1.0
    
    def test_case_insensitive(self):
        assert exact_match_score("HELLO WORLD", "hello world") == 1.0
    
    def test_whitespace_normalized(self):
        assert exact_match_score("hello   world", "hello world") == 1.0
    
    def test_no_match(self):
        assert exact_match_score("hello", "world") == 0.0
    
    def test_partial_no_match(self):
        assert exact_match_score("hello world", "hello") == 0.0
    
    def test_empty_both(self):
        assert exact_match_score("", "") == 1.0
    
    def test_empty_one(self):
        assert exact_match_score("hello", "") == 0.0
        assert exact_match_score("", "hello") == 0.0


class TestMultisetF1Score:
    """Tests for multiset F1 scoring."""
    
    def test_exact_match(self):
        assert multiset_f1_score("hello world", "hello world") == 1.0
    
    def test_no_overlap(self):
        assert multiset_f1_score("hello world", "foo bar") == 0.0
    
    def test_partial_overlap(self):
        # "hello world" vs "hello there"
        # Intersection: {hello: 1}
        # Precision: 1/2 = 0.5
        # Recall: 1/2 = 0.5
        # F1: 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5
        assert multiset_f1_score("hello world", "hello there") == 0.5
    
    def test_subset(self):
        # "hello" vs "hello world"
        # Intersection: {hello: 1}
        # Precision: 1/1 = 1.0
        # Recall: 1/2 = 0.5
        # F1: 2 * 1.0 * 0.5 / (1.0 + 0.5) = 2/3
        score = multiset_f1_score("hello", "hello world")
        assert abs(score - 2/3) < 1e-9
    
    def test_superset(self):
        # "hello world there" vs "hello world"
        # Intersection: {hello: 1, world: 1}
        # Precision: 2/3
        # Recall: 2/2 = 1.0
        # F1: 2 * (2/3) * 1.0 / (2/3 + 1.0) = (4/3) / (5/3) = 4/5 = 0.8
        score = multiset_f1_score("hello world there", "hello world")
        assert abs(score - 0.8) < 1e-9
    
    def test_empty_reference(self):
        assert multiset_f1_score("hello", "") == 0.0
        assert multiset_f1_score("", "") == 1.0
    
    def test_empty_candidate(self):
        assert multiset_f1_score("", "hello") == 0.0
    
    def test_duplicate_tokens(self):
        # "hello hello" vs "hello"
        # Intersection: {hello: 1}
        # Precision: 1/2 = 0.5
        # Recall: 1/1 = 1.0
        # F1: 2 * 0.5 * 1.0 / (0.5 + 1.0) = 1.0 / 1.5 = 2/3
        score = multiset_f1_score("hello hello", "hello")
        assert abs(score - 2/3) < 1e-9


class TestComputeSimilarityReward:
    """Tests for combined similarity reward."""
    
    def test_exact_match_returns_one(self):
        assert compute_similarity_reward("hello world", "hello world") == 1.0
    
    def test_normalized_exact_match(self):
        assert compute_similarity_reward("HELLO   WORLD", "hello world") == 1.0
    
    def test_partial_returns_f1(self):
        # Not exact match, so returns F1
        score = compute_similarity_reward("hello there", "hello world")
        assert score == 0.5  # F1 from partial overlap
    
    def test_no_match_returns_zero(self):
        assert compute_similarity_reward("foo bar", "hello world") == 0.0


class TestInstructionId:
    """Tests for instruction ID hashing."""
    
    def test_deterministic(self):
        inst = "Test instruction"
        id1 = instruction_id(inst)
        id2 = instruction_id(inst)
        assert id1 == id2
    
    def test_different_instructions(self):
        id1 = instruction_id("Instruction A")
        id2 = instruction_id("Instruction B")
        assert id1 != id2
    
    def test_sha1_format(self):
        inst_id = instruction_id("test")
        assert len(inst_id) == 40  # SHA1 hex digest length
        assert all(c in "0123456789abcdef" for c in inst_id)


class TestRailroadScenarioVerifier:
    """Tests for the verifier class (without sandbox)."""
    
    def test_score_directly_exact(self):
        verifier = RailroadScenarioVerifier({})
        result = verifier.score_directly("hello world", "hello world")
        assert result.env_reward == 1.0
        assert result.exact_match is True
        assert result.f1_score == 1.0
    
    def test_score_directly_partial(self):
        verifier = RailroadScenarioVerifier({})
        result = verifier.score_directly("hello there", "hello world")
        assert result.env_reward == 0.5
        assert result.exact_match is False
        assert result.f1_score == 0.5
    
    def test_score_directly_no_match(self):
        verifier = RailroadScenarioVerifier({})
        result = verifier.score_directly("foo bar", "hello world")
        assert result.env_reward == 0.0
        assert result.exact_match is False
        assert result.f1_score == 0.0


class TestRailroadScoringExamples:
    """Integration tests with railroad-like content."""
    
    def test_rule_citation_exact(self):
        reference = "Under Rule 99, the train must stop and the conductor must flag."
        candidate = "Under Rule 99, the train must stop and the conductor must flag."
        assert compute_similarity_reward(candidate, reference) == 1.0
    
    def test_rule_citation_case_insensitive(self):
        reference = "Under Rule 99, the train must stop."
        candidate = "UNDER RULE 99, THE TRAIN MUST STOP."
        assert compute_similarity_reward(candidate, reference) == 1.0
    
    def test_rule_partial_match(self):
        reference = "The engineer must apply brakes and sound whistle signal per Rule 14."
        candidate = "The engineer must apply brakes per Rule 14."
        # Partial match - should get F1 score
        score = compute_similarity_reward(candidate, reference)
        assert 0.0 < score < 1.0
    
    def test_completely_wrong(self):
        reference = "Stop immediately and protect the train."
        candidate = "Continue at normal speed."
        # No meaningful overlap
        score = compute_similarity_reward(candidate, reference)
        assert score < 0.5


class TestEdgeCases:
    """Edge case tests."""
    
    def test_unicode(self):
        assert normalize_text("héllo wörld") == "héllo wörld"
    
    def test_punctuation_preserved(self):
        # Punctuation is preserved in normalization
        assert normalize_text("Hello, World!") == "hello, world!"
    
    def test_long_text(self):
        long_ref = " ".join(["word"] * 1000)
        long_cand = " ".join(["word"] * 1000)
        assert exact_match_score(long_cand, long_ref) == 1.0
    
    def test_whitespace_only_difference(self):
        ref = "The train must stop at the next station."
        cand = "The  train   must  stop  at  the  next  station."
        assert exact_match_score(cand, ref) == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
