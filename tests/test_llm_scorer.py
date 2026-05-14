"""
Tests for LLMScorer - LLM-as-Judge scorer using project AI credentials.

Testing patterns:
- Mock HTTP client, test parsing separately from LLM calls
- Validate graceful degradation (scoring_failed=True, never throw)
- Test all response parsing modes (JSON, text, choice scores, multi-score)
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from brokle.scorers.llm_scorer import (
    LLMScorer,
    _infer_provider,
    _render_template,
)
from brokle.scores.types import ScoreResult, ScoreType

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_client():
    """Create a mock Brokle client with mocked HTTP client."""
    client = MagicMock()
    client._http = MagicMock()
    return client


@pytest.fixture
def mock_async_client():
    """Create a mock async Brokle client."""
    client = MagicMock()
    client._http = AsyncMock()
    return client


def make_success_response(content: str) -> dict:
    """Playground success response — raw body, no envelope."""
    return {"response": {"content": content}}


def make_error_response(message: str = "Unknown error") -> Exception:
    """Error path is now an exception raised by the HTTP client, not
    a response body. Tests that used make_error_response as a mock
    return_value should switch to mock.side_effect.
    """
    from brokle._http.errors import BrokleError

    return BrokleError(message, hint="Mock playground error for tests.")


# =============================================================================
# Provider Inference Tests
# =============================================================================


class TestProviderInference:
    """Tests for _infer_provider function."""

    def test_openai_gpt4(self):
        assert _infer_provider("gpt-4") == "openai"

    def test_openai_gpt4o(self):
        assert _infer_provider("gpt-4o") == "openai"

    def test_openai_gpt4o_mini(self):
        assert _infer_provider("gpt-4o-mini") == "openai"

    def test_openai_gpt35(self):
        assert _infer_provider("gpt-3.5-turbo") == "openai"

    def test_openai_o1(self):
        assert _infer_provider("o1-preview") == "openai"

    def test_anthropic_claude3(self):
        assert _infer_provider("claude-3-opus") == "anthropic"

    def test_anthropic_claude35(self):
        assert _infer_provider("claude-3-5-sonnet") == "anthropic"
        assert _infer_provider("claude-3.5-sonnet") == "anthropic"

    def test_google_gemini(self):
        assert _infer_provider("gemini-pro") == "google"

    def test_google_gemini_15(self):
        assert _infer_provider("gemini-1.5-pro") == "google"

    def test_case_insensitive(self):
        assert _infer_provider("GPT-4O") == "openai"
        assert _infer_provider("Claude-3-Opus") == "anthropic"

    def test_unknown_model_defaults_to_openai(self):
        assert _infer_provider("unknown-model") == "openai"
        assert _infer_provider("my-custom-model") == "openai"


# =============================================================================
# Template Rendering Tests
# =============================================================================


class TestTemplateRendering:
    """Tests for _render_template function."""

    def test_basic_substitution(self):
        template = "Input: {{input}}, Output: {{output}}"
        result = _render_template(template, {"input": "hello", "output": "world"})
        assert result == "Input: hello, Output: world"

    def test_expected_variable(self):
        template = "Expected: {{expected}}"
        result = _render_template(template, {"expected": "correct answer"})
        assert result == "Expected: correct answer"

    def test_all_three_variables(self):
        template = "Q: {{input}}\nA: {{output}}\nRef: {{expected}}"
        result = _render_template(
            template, {"input": "What is 2+2?", "output": "4", "expected": "4"}
        )
        assert "Q: What is 2+2?" in result
        assert "A: 4" in result
        assert "Ref: 4" in result

    def test_with_dict_value(self):
        template = "Data: {{input}}"
        result = _render_template(template, {"input": {"key": "value", "num": 42}})
        assert '"key": "value"' in result
        assert '"num": 42' in result

    def test_with_list_value(self):
        template = "Items: {{output}}"
        result = _render_template(template, {"output": [1, 2, 3]})
        assert "[" in result and "]" in result

    def test_with_none_value(self):
        template = "Output: {{output}}"
        result = _render_template(template, {"output": None})
        assert result == "Output: "

    def test_whitespace_in_braces(self):
        template = "Value: {{ input }}"
        result = _render_template(template, {"input": "test"})
        assert result == "Value: test"

    def test_missing_variable_stays_unchanged(self):
        template = "Known: {{input}}, Unknown: {{foo}}"
        result = _render_template(template, {"input": "hello"})
        assert "Known: hello" in result
        assert "{{foo}}" in result

    def test_special_characters_in_value(self):
        template = "Code: {{output}}"
        result = _render_template(template, {"output": "def foo(): return 'bar'"})
        assert "def foo(): return 'bar'" in result


# =============================================================================
# LLMScorer Initialization Tests
# =============================================================================


class TestLLMScorerInit:
    """Tests for LLMScorer initialization."""

    def test_default_values(self, mock_client):
        scorer = LLMScorer(
            client=mock_client, name="test_scorer", prompt="Rate this: {{output}}"
        )
        assert scorer.name == "test_scorer"
        assert scorer.model == "gpt-4o"
        assert scorer.temperature == 0.0
        assert scorer.use_cot is False
        assert scorer.multi_score is False
        assert scorer.choice_scores is None
        assert scorer.credential_id is None
        assert scorer.max_tokens is None
        assert scorer._provider == "openai"

    def test_custom_options(self, mock_client):
        scorer = LLMScorer(
            client=mock_client,
            name="custom",
            prompt="{{input}} -> {{output}}",
            model="claude-3-opus",
            credential_id="cred_123",
            multi_score=True,
            temperature=0.7,
            max_tokens=1000,
            use_cot=True,
            choice_scores={"A": 1.0, "B": 0.0},
        )
        assert scorer.model == "claude-3-opus"
        assert scorer._provider == "anthropic"
        assert scorer.credential_id == "cred_123"
        assert scorer.multi_score is True
        assert scorer.temperature == 0.7
        assert scorer.max_tokens == 1000
        assert scorer.use_cot is True
        assert scorer.choice_scores == {"A": 1.0, "B": 0.0}


# =============================================================================
# Request Payload Building Tests
# =============================================================================


class TestRequestPayloadBuilding:
    """Tests for _build_request_payload method."""

    def test_basic_payload(self, mock_client):
        scorer = LLMScorer(client=mock_client, name="test", prompt="Rate: {{output}}")
        payload = scorer._build_request_payload("Rate: hello")

        assert payload["template"] == "Rate: hello"
        assert payload["prompt_type"] == "text"
        assert payload["variables"] == {}
        assert payload["config_overrides"]["model"] == "gpt-4o"
        assert payload["config_overrides"]["provider"] == "openai"
        assert payload["config_overrides"]["temperature"] == 0.0

    def test_payload_with_credential_id(self, mock_client):
        scorer = LLMScorer(
            client=mock_client,
            name="test",
            prompt="{{output}}",
            credential_id="cred_abc",
        )
        payload = scorer._build_request_payload("test")

        assert payload["config_overrides"]["credential_id"] == "cred_abc"

    def test_payload_with_max_tokens(self, mock_client):
        scorer = LLMScorer(
            client=mock_client, name="test", prompt="{{output}}", max_tokens=500
        )
        payload = scorer._build_request_payload("test")

        assert payload["config_overrides"]["max_tokens"] == 500

    def test_payload_without_optional_fields(self, mock_client):
        scorer = LLMScorer(client=mock_client, name="test", prompt="{{output}}")
        payload = scorer._build_request_payload("test")

        assert "credential_id" not in payload["config_overrides"]
        assert "max_tokens" not in payload["config_overrides"]


# =============================================================================
# JSON Response Parsing Tests
# =============================================================================


class TestJsonResponseParsing:
    """Tests for JSON response parsing."""

    def test_numeric_score_with_reason(self, mock_client):
        scorer = LLMScorer(client=mock_client, name="quality", prompt="{{output}}")
        mock_client._http.post.return_value = make_success_response(
            '{"score": 0.85, "reason": "Good response"}'
        )

        result = scorer(output="test output")

        assert isinstance(result, ScoreResult)
        assert result.name == "quality"
        assert result.value == 0.85
        assert result.type == ScoreType.NUMERIC
        assert result.reason == "Good response"
        assert result.scoring_failed is not True

    def test_value_key_instead_of_score(self, mock_client):
        scorer = LLMScorer(client=mock_client, name="test", prompt="{{output}}")
        mock_client._http.post.return_value = make_success_response('{"value": 0.7}')

        result = scorer(output="test")
        assert result.value == 0.7

    def test_rating_key(self, mock_client):
        scorer = LLMScorer(client=mock_client, name="test", prompt="{{output}}")
        mock_client._http.post.return_value = make_success_response('{"rating": 8}')

        result = scorer(output="test")
        assert result.value == 8.0  # Raw value, no normalization in JSON

    def test_explanation_key_as_reason(self, mock_client):
        scorer = LLMScorer(client=mock_client, name="test", prompt="{{output}}")
        mock_client._http.post.return_value = make_success_response(
            '{"score": 0.5, "explanation": "Mediocre"}'
        )

        result = scorer(output="test")
        assert result.reason == "Mediocre"

    def test_choice_scores_mapping(self, mock_client):
        scorer = LLMScorer(
            client=mock_client,
            name="factuality",
            prompt="{{output}}",
            choice_scores={"A": 1.0, "B": 0.5, "C": 0.0},
        )
        mock_client._http.post.return_value = make_success_response(
            '{"choice": "B", "reason": "Partially correct"}'
        )

        result = scorer(output="test")

        assert result.value == 0.5
        assert result.type == ScoreType.CATEGORICAL
        assert result.string_value == "B"
        assert result.reason == "Partially correct"
        assert result.metadata["choice"] == "B"

    def test_choice_scores_case_insensitive(self, mock_client):
        scorer = LLMScorer(
            client=mock_client,
            name="test",
            prompt="{{output}}",
            choice_scores={"A": 1.0, "B": 0.0},
        )
        mock_client._http.post.return_value = make_success_response('{"choice": "a"}')

        result = scorer(output="test")
        assert result.value == 1.0
        assert result.string_value == "A"

    def test_choice_first_char_fallback(self, mock_client):
        scorer = LLMScorer(
            client=mock_client,
            name="test",
            prompt="{{output}}",
            choice_scores={"A": 1.0, "B": 0.0},
        )
        mock_client._http.post.return_value = make_success_response(
            '{"choice": "Answer A is correct"}'
        )

        result = scorer(output="test")
        assert result.value == 1.0  # Falls back to first char "A"

    def test_multi_score_mode(self, mock_client):
        scorer = LLMScorer(
            client=mock_client, name="multi", prompt="{{output}}", multi_score=True
        )
        mock_client._http.post.return_value = make_success_response(
            '{"accuracy": 0.9, "fluency": 0.8, "coherence": 0.7, "reason": "Good overall"}'
        )

        results = scorer(output="test")

        assert isinstance(results, list)
        assert len(results) == 3
        names = [r.name for r in results]
        assert "accuracy" in names
        assert "fluency" in names
        assert "coherence" in names
        # All should have the same reason
        for r in results:
            assert r.reason == "Good overall"

    def test_boolean_result_true(self, mock_client):
        scorer = LLMScorer(client=mock_client, name="test", prompt="{{output}}")
        mock_client._http.post.return_value = make_success_response(
            '{"result": true, "reason": "Correct"}'
        )

        result = scorer(output="test")
        assert result.value == 1.0
        assert result.type == ScoreType.BOOLEAN

    def test_boolean_result_false(self, mock_client):
        scorer = LLMScorer(client=mock_client, name="test", prompt="{{output}}")
        mock_client._http.post.return_value = make_success_response('{"result": false}')

        result = scorer(output="test")
        assert result.value == 0.0
        assert result.type == ScoreType.BOOLEAN

    def test_reasoning_in_metadata(self, mock_client):
        scorer = LLMScorer(client=mock_client, name="test", prompt="{{output}}")
        mock_client._http.post.return_value = make_success_response(
            '{"score": 0.9, "reasoning": "Step 1: Check grammar..."}'
        )

        result = scorer(output="test")
        assert result.metadata is not None
        assert result.metadata["reasoning"] == "Step 1: Check grammar..."

    def test_json_in_markdown_block(self, mock_client):
        scorer = LLMScorer(client=mock_client, name="test", prompt="{{output}}")
        mock_client._http.post.return_value = make_success_response(
            '```json\n{"score": 0.75}\n```'
        )

        result = scorer(output="test")
        assert result.value == 0.75

    def test_unparseable_json_returns_failed(self, mock_client):
        scorer = LLMScorer(client=mock_client, name="test", prompt="{{output}}")
        mock_client._http.post.return_value = make_success_response(
            '{"invalid": "no score field"}'
        )

        result = scorer(output="test")
        assert result.scoring_failed is True
        assert result.value == 0.0
        assert "Could not parse" in result.reason


# =============================================================================
# Text Response Parsing Tests
# =============================================================================


class TestTextResponseParsing:
    """Tests for text (non-JSON) response parsing."""

    def test_extract_number_from_text(self, mock_client):
        scorer = LLMScorer(client=mock_client, name="test", prompt="{{output}}")
        mock_client._http.post.return_value = make_success_response(
            "I would rate this response 8 out of 10."
        )

        result = scorer(output="test")
        assert result.value == 0.8  # 8/10 normalized
        assert result.type == ScoreType.NUMERIC

    def test_extract_decimal_number(self, mock_client):
        scorer = LLMScorer(client=mock_client, name="test", prompt="{{output}}")
        mock_client._http.post.return_value = make_success_response("Score: 0.75")

        result = scorer(output="test")
        assert result.value == 0.75

    def test_yes_detection(self, mock_client):
        scorer = LLMScorer(client=mock_client, name="test", prompt="{{output}}")
        mock_client._http.post.return_value = make_success_response(
            "Yes, this response is correct and well-formatted."
        )

        result = scorer(output="test")
        assert result.value == 1.0
        assert result.type == ScoreType.BOOLEAN

    def test_no_detection(self, mock_client):
        scorer = LLMScorer(client=mock_client, name="test", prompt="{{output}}")
        mock_client._http.post.return_value = make_success_response(
            "No, this is incorrect."
        )

        result = scorer(output="test")
        assert result.value == 0.0
        assert result.type == ScoreType.BOOLEAN

    def test_unparseable_text_returns_failed(self, mock_client):
        scorer = LLMScorer(client=mock_client, name="test", prompt="{{output}}")
        # Text without numbers and without "yes"/"no" keywords
        mock_client._http.post.return_value = make_success_response(
            "This response is interesting but lacks a clear rating."
        )

        result = scorer(output="test")
        assert result.scoring_failed is True
        assert result.value == 0.0

    def test_clamp_to_0_1_range(self, mock_client):
        scorer = LLMScorer(client=mock_client, name="test", prompt="{{output}}")
        mock_client._http.post.return_value = make_success_response(
            "Score: 15"  # Out of range, should clamp
        )

        result = scorer(output="test")
        # 15 > 10 so no normalization, but clamped to 1.0
        assert result.value == 1.0


# =============================================================================
# Error Handling Tests (Graceful Degradation)
# =============================================================================


class TestErrorHandling:
    """Tests for graceful degradation - errors return scoring_failed, not exceptions."""

    def test_http_error_returns_failed(self, mock_client):
        scorer = LLMScorer(client=mock_client, name="test", prompt="{{output}}")
        # HTTP errors now raise typed exceptions from the HTTP client;
        # the scorer catches them and returns scoring_failed instead of
        # re-raising.
        mock_client._http.post.side_effect = make_error_response("Server error")

        result = scorer(output="test")

        assert result.scoring_failed is True
        assert result.value == 0.0
        assert "Server error" in result.reason

    def test_empty_response_returns_failed(self, mock_client):
        scorer = LLMScorer(client=mock_client, name="test", prompt="{{output}}")
        mock_client._http.post.return_value = {"response": {"content": ""}}

        result = scorer(output="test")

        assert result.scoring_failed is True
        assert "Empty LLM response" in result.reason

    def test_missing_content_returns_failed(self, mock_client):
        scorer = LLMScorer(client=mock_client, name="test", prompt="{{output}}")
        mock_client._http.post.return_value = {"response": {}}

        result = scorer(output="test")

        assert result.scoring_failed is True

    def test_network_exception_returns_failed(self, mock_client):
        scorer = LLMScorer(client=mock_client, name="test", prompt="{{output}}")
        mock_client._http.post.side_effect = Exception("Network timeout")

        result = scorer(output="test")

        assert result.scoring_failed is True
        assert result.value == 0.0
        assert "Network timeout" in result.reason
        assert result.metadata["error"] == "Network timeout"

    def test_json_decode_error_falls_back_to_text(self, mock_client):
        scorer = LLMScorer(client=mock_client, name="test", prompt="{{output}}")
        mock_client._http.post.return_value = make_success_response(
            "Not valid JSON but score is 7"
        )

        result = scorer(output="test")

        # Should fall back to text parsing and find "7"
        assert result.value == 0.7
        assert result.scoring_failed is not True


# =============================================================================
# Chain-of-Thought Tests
# =============================================================================


class TestChainOfThought:
    """Tests for use_cot option."""

    def test_cot_appends_instruction(self, mock_client):
        scorer = LLMScorer(
            client=mock_client,
            name="test",
            prompt="Rate this: {{output}}",
            use_cot=True,
        )
        mock_client._http.post.return_value = make_success_response('{"score": 0.8}')

        scorer(output="test")

        # Check the payload that was sent
        call_args = mock_client._http.post.call_args
        payload = call_args[1]["json"]
        assert "Think step-by-step" in payload["template"]

    def test_without_cot_no_instruction(self, mock_client):
        scorer = LLMScorer(
            client=mock_client,
            name="test",
            prompt="Rate this: {{output}}",
            use_cot=False,
        )
        mock_client._http.post.return_value = make_success_response('{"score": 0.8}')

        scorer(output="test")

        call_args = mock_client._http.post.call_args
        payload = call_args[1]["json"]
        assert "Think step-by-step" not in payload["template"]


# =============================================================================
# Async Support Tests
# =============================================================================


class TestAsyncSupport:
    """Tests for async execution path."""

    @pytest.mark.asyncio
    async def test_async_call_success(self, mock_async_client):
        scorer = LLMScorer(
            client=mock_async_client, name="async_test", prompt="{{output}}"
        )
        mock_async_client._http.post.return_value = make_success_response(
            '{"score": 0.9, "reason": "Excellent"}'
        )

        result = await scorer.__call_async__(output="test")

        assert isinstance(result, ScoreResult)
        assert result.value == 0.9
        assert result.reason == "Excellent"

    @pytest.mark.asyncio
    async def test_async_error_handling(self, mock_async_client):
        scorer = LLMScorer(
            client=mock_async_client, name="async_test", prompt="{{output}}"
        )
        mock_async_client._http.post.side_effect = Exception("Async network error")

        result = await scorer.__call_async__(output="test")

        assert result.scoring_failed is True
        assert result.value == 0.0
        assert "Async network error" in result.reason

    @pytest.mark.asyncio
    async def test_async_with_multi_score(self, mock_async_client):
        scorer = LLMScorer(
            client=mock_async_client,
            name="multi",
            prompt="{{output}}",
            multi_score=True,
        )
        mock_async_client._http.post.return_value = make_success_response(
            '{"accuracy": 0.9, "fluency": 0.85}'
        )

        results = await scorer.__call_async__(output="test")

        assert isinstance(results, list)
        assert len(results) == 2


# =============================================================================
# End-to-End Integration Tests
# =============================================================================


class TestEndToEnd:
    """End-to-end tests combining multiple features."""

    def test_full_workflow_with_template(self, mock_client):
        scorer = LLMScorer(
            client=mock_client,
            name="relevance",
            prompt="Question: {{input}}\nAnswer: {{output}}\n\nRate relevance 0-10.",
            model="gpt-4o",
        )
        mock_client._http.post.return_value = make_success_response(
            '{"score": 8, "reason": "Answer addresses the question well"}'
        )

        result = scorer(
            input="What is Python?",
            output="Python is a programming language.",
            expected=None,
        )

        assert result.value == 8.0
        assert result.reason == "Answer addresses the question well"

        # Verify the template was rendered correctly
        call_args = mock_client._http.post.call_args
        payload = call_args[1]["json"]
        assert "What is Python?" in payload["template"]
        assert "Python is a programming language" in payload["template"]

    def test_classification_workflow(self, mock_client):
        scorer = LLMScorer(
            client=mock_client,
            name="sentiment",
            prompt="Classify sentiment of: {{output}}\n(A) Positive (B) Neutral (C) Negative",
            choice_scores={"A": 1.0, "B": 0.5, "C": 0.0},
        )
        mock_client._http.post.return_value = make_success_response(
            '{"choice": "A", "reason": "Clearly positive language"}'
        )

        result = scorer(output="I love this product!")

        assert result.value == 1.0
        assert result.type == ScoreType.CATEGORICAL
        assert result.string_value == "A"
