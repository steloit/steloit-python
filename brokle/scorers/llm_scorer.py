"""
LLM-as-Judge Scorer for Brokle Evaluations

Provides LLMScorer for evaluating outputs using LLM models as judges.
Uses project AI credentials via the Brokle backend.

Features:
- Mustache-style template variables: {{input}}, {{output}}, {{expected}}
- Choice scores mapping for classification tasks
- Chain-of-thought (CoT) reasoning option
- Multi-score support (single LLM call → multiple ScoreResults)
- Graceful error handling with scoring_failed flag

Usage:
    >>> from brokle import Brokle
    >>> from brokle.scorers import LLMScorer
    >>>
    >>> client = Brokle(api_key="bk_...")
    >>>
    >>> # Basic numeric scorer
    >>> relevance = LLMScorer(
    ...     client=client,
    ...     name="relevance",
    ...     prompt='''
    ...     Rate the relevance of the response (0-10).
    ...
    ...     Question: {{input}}
    ...     Response: {{output}}
    ...
    ...     Return JSON: {"score": <0-10>, "reason": "<explanation>"}
    ...     ''',
    ...     model="gpt-4o",
    ... )
    >>>
    >>> # Classification with choice scores
    >>> factuality = LLMScorer(
    ...     client=client,
    ...     name="factuality",
    ...     prompt='''
    ...     Compare the factual content:
    ...     Expert: {{expected}}
    ...     Submission: {{output}}
    ...
    ...     (A) Subset (B) Superset (C) Exact (D) Contradicts
    ...     Return JSON: {"choice": "<A|B|C|D>", "reason": "..."}
    ...     ''',
    ...     choice_scores={"A": 0.4, "B": 0.6, "C": 1.0, "D": 0.0},
    ... )
"""

import json
import re
from typing import Any, Dict, List, Optional, Union

from ..scores.types import ScoreResult, ScoreType

# Model to provider mapping
MODEL_PROVIDER_MAP = {
    "gpt-4": "openai",
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "gpt-4-turbo": "openai",
    "gpt-3.5": "openai",
    "o1": "openai",
    "o3": "openai",
    "claude-3": "anthropic",
    "claude-3-5": "anthropic",
    "claude-3.5": "anthropic",
    "claude-4": "anthropic",
    "gemini": "google",
    "gemini-pro": "google",
    "gemini-1.5": "google",
    "gemini-2": "google",
}


def _infer_provider(model: str) -> str:
    """Infer provider from model name."""
    model_lower = model.lower()
    for prefix, provider in MODEL_PROVIDER_MAP.items():
        if model_lower.startswith(prefix):
            return provider
    return "openai"  # default


def _render_template(template: str, variables: Dict[str, Any]) -> str:
    """
    Render Mustache-style template with variables.

    Supports {{variable}} syntax for template variables.

    Args:
        template: Template string with {{variable}} placeholders
        variables: Dict of variable name -> value

    Returns:
        Rendered template string
    """
    result = template
    for key, value in variables.items():
        # Convert value to string representation
        if isinstance(value, (dict, list)):
            str_value = json.dumps(value, indent=2)
        elif value is None:
            str_value = ""
        else:
            str_value = str(value)

        # Replace {{key}} with value (Mustache syntax)
        pattern = r"\{\{\s*" + re.escape(key) + r"\s*\}\}"
        result = re.sub(pattern, str_value, result)

    return result


class LLMScorer:
    """
    LLM-as-Judge scorer using project AI credentials.

    Calls the Brokle backend to execute LLM prompts for evaluation.
    Uses LLM models as judges for evaluation scoring.

    Args:
        client: Brokle client instance (sync or async)
        name: Score name for results
        prompt: Prompt template with {{input}}, {{output}}, {{expected}} variables
        model: LLM model to use (default: "gpt-4o")
        credential_id: Specific credential ID (optional, uses project default)
        multi_score: If True, parse response as multiple scores
        temperature: Sampling temperature (default: 0.0)
        max_tokens: Maximum response tokens (optional)
        use_cot: Enable chain-of-thought reasoning (default: False)
        choice_scores: Map of choice labels to scores for classification

    Example:
        >>> # Basic scorer
        >>> scorer = LLMScorer(
        ...     client=client,
        ...     name="quality",
        ...     prompt="Rate quality 0-10: {{output}}\\n\\nJSON: {score, reason}",
        ... )
        >>>
        >>> # Classification with choice mapping
        >>> scorer = LLMScorer(
        ...     client=client,
        ...     name="factuality",
        ...     prompt="Is this correct? (A) Yes (B) No\\n{{output}}",
        ...     choice_scores={"A": 1.0, "B": 0.0},
        ... )
    """

    def __init__(
        self,
        client: Any,
        name: str,
        prompt: str,
        model: str = "gpt-4o",
        credential_id: Optional[str] = None,
        multi_score: bool = False,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        use_cot: bool = False,
        choice_scores: Optional[Dict[str, float]] = None,
    ):
        self.client = client
        self.name = name
        self.prompt = prompt
        self.model = model
        self.credential_id = credential_id
        self.multi_score = multi_score
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_cot = use_cot
        self.choice_scores = choice_scores

        # Infer provider from model name
        self._provider = _infer_provider(model)

    def __call__(
        self,
        output: Any,
        expected: Any = None,
        input: Any = None,
        **kwargs: Any,
    ) -> Union[ScoreResult, List[ScoreResult]]:
        """
        Execute LLM scorer on the given output.

        This method is synchronous and calls the backend API.
        For async usage, use __call__ with await.

        Args:
            output: The actual output to evaluate
            expected: The expected/reference output (optional)
            input: The input data (optional)
            **kwargs: Additional arguments (ignored)

        Returns:
            ScoreResult or List[ScoreResult] (if multi_score=True)
        """
        try:
            variables = {
                "input": input,
                "output": output,
                "expected": expected,
            }

            rendered_prompt = _render_template(self.prompt, variables)

            if self.use_cot:
                rendered_prompt += (
                    "\n\nThink step-by-step before providing your final answer."
                )

            payload = self._build_request_payload(rendered_prompt)
            response = self._execute_llm(payload)
            return self._parse_response(response)

        except Exception as e:
            # Graceful degradation (Optik pattern)
            return ScoreResult(
                name=self.name,
                value=0.0,
                type=ScoreType.NUMERIC,
                reason=f"LLM scoring failed: {str(e)}",
                scoring_failed=True,
                metadata={"error": str(e)},
            )

    async def __call_async__(
        self,
        output: Any,
        expected: Any = None,
        input: Any = None,
        **kwargs: Any,
    ) -> Union[ScoreResult, List[ScoreResult]]:
        """
        Async version of the scorer.

        Args:
            output: The actual output to evaluate
            expected: The expected/reference output (optional)
            input: The input data (optional)
            **kwargs: Additional arguments (ignored)

        Returns:
            ScoreResult or List[ScoreResult] (if multi_score=True)
        """
        try:
            variables = {
                "input": input,
                "output": output,
                "expected": expected,
            }

            rendered_prompt = _render_template(self.prompt, variables)

            if self.use_cot:
                rendered_prompt += (
                    "\n\nThink step-by-step before providing your final answer."
                )

            payload = self._build_request_payload(rendered_prompt)
            response = await self._execute_llm_async(payload)
            return self._parse_response(response)

        except Exception as e:
            # Graceful degradation (Optik pattern)
            return ScoreResult(
                name=self.name,
                value=0.0,
                type=ScoreType.NUMERIC,
                reason=f"LLM scoring failed: {str(e)}",
                scoring_failed=True,
                metadata={"error": str(e)},
            )

    def _build_request_payload(self, rendered_prompt: str) -> Dict[str, Any]:
        """Build the request payload for the playground execute endpoint."""
        config_overrides: Dict[str, Any] = {
            "model": self.model,
            "provider": self._provider,
            "temperature": self.temperature,
        }

        if self.credential_id:
            config_overrides["credential_id"] = self.credential_id

        if self.max_tokens:
            config_overrides["max_tokens"] = self.max_tokens

        # Note: project_id is derived from API key authentication on the backend
        return {
            "template": rendered_prompt,
            "prompt_type": "text",
            "variables": {},
            "config_overrides": config_overrides,
        }

    def _execute_llm(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute LLM call via backend (sync).

        Uses the SDK route with API-key auth (not the dashboard JWT route).
        The HTTP client raises typed exceptions on 4xx/5xx before we get
        here, so a successful return means we have the raw execution
        result body.
        """
        http_client = self.client._http
        return http_client.post("/v1/playground/execute", json=payload)

    async def _execute_llm_async(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute LLM call via backend (async).

        See ``_execute_llm`` for wire-contract details.
        """
        http_client = self.client._http
        return await http_client.post("/v1/playground/execute", json=payload)

    def _parse_response(
        self, response: Dict[str, Any]
    ) -> Union[ScoreResult, List[ScoreResult]]:
        """
        Parse LLM response into ScoreResult(s).

        Supports:
        - Single numeric score: {"score": 0.8, "reason": "..."}
        - Choice classification: {"choice": "A", "reason": "..."}
        - Multi-score: {"accuracy": 0.9, "fluency": 0.8}
        - Chain-of-thought: {"reasoning": "...", "score": 0.8}
        """
        llm_response = response.get("response", {})
        content = llm_response.get("content", "")

        if not content:
            error_msg = response.get("error", "No content in response")
            return ScoreResult(
                name=self.name,
                value=0.0,
                type=ScoreType.NUMERIC,
                reason=f"Empty LLM response: {error_msg}",
                scoring_failed=True,
            )

        # Try to parse as JSON
        try:
            # Find JSON in content (may be wrapped in markdown code blocks)
            json_match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                parsed = json.loads(content)
        except json.JSONDecodeError:
            # Text response - try to extract a score
            return self._parse_text_response(content)

        return self._parse_json_response(parsed, content)

    def _parse_json_response(
        self, parsed: Dict[str, Any], raw_content: str
    ) -> Union[ScoreResult, List[ScoreResult]]:
        """Parse JSON response into ScoreResult(s)."""
        reason = parsed.get("reason", parsed.get("explanation", None))
        reasoning = parsed.get("reasoning", None)  # CoT support

        metadata: Dict[str, Any] = {}
        if reasoning:
            metadata["reasoning"] = reasoning

        # 1. Check for choice scores (classification mode)
        if self.choice_scores:
            choice = parsed.get("choice", parsed.get("label", parsed.get("category")))
            if choice:
                choice_str = str(choice).strip().upper()
                score_value = self.choice_scores.get(
                    choice_str,
                    self.choice_scores.get(choice_str[0], 0.0),  # Try first char
                )
                metadata["choice"] = choice_str
                return ScoreResult(
                    name=self.name,
                    value=float(score_value),
                    type=ScoreType.CATEGORICAL,
                    string_value=choice_str,
                    reason=reason,
                    metadata=metadata if metadata else None,
                )

        # 2. Check for multi-score
        if self.multi_score:
            results: List[ScoreResult] = []
            for key, value in parsed.items():
                if key in ("reason", "reasoning", "explanation"):
                    continue
                if isinstance(value, (int, float)):
                    results.append(
                        ScoreResult(
                            name=key,
                            value=float(value),
                            type=ScoreType.NUMERIC,
                            reason=reason,
                            metadata=metadata if metadata else None,
                        )
                    )
            if results:
                return results

        # 3. Single score
        score_value = parsed.get("score", parsed.get("value", parsed.get("rating")))
        if score_value is not None:
            try:
                return ScoreResult(
                    name=self.name,
                    value=float(score_value),
                    type=ScoreType.NUMERIC,
                    reason=reason,
                    metadata=metadata if metadata else None,
                )
            except (TypeError, ValueError):
                pass

        # 4. Boolean result
        if "result" in parsed and isinstance(parsed["result"], bool):
            return ScoreResult(
                name=self.name,
                value=1.0 if parsed["result"] else 0.0,
                type=ScoreType.BOOLEAN,
                reason=reason,
                metadata=metadata if metadata else None,
            )

        # Fallback: couldn't parse
        return ScoreResult(
            name=self.name,
            value=0.0,
            type=ScoreType.NUMERIC,
            reason=f"Could not parse score from JSON: {raw_content[:200]}",
            scoring_failed=True,
            metadata={"raw_response": raw_content[:500]},
        )

    def _parse_text_response(self, content: str) -> ScoreResult:
        """Parse text response (non-JSON) to extract score."""
        # Try to extract a number
        numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", content)
        if numbers:
            # Take the first number, normalize if needed
            value = float(numbers[0])
            # If it looks like a 0-10 scale, normalize to 0-1
            if value > 1.0 and value <= 10.0:
                value = value / 10.0
            return ScoreResult(
                name=self.name,
                value=min(1.0, max(0.0, value)),  # Clamp to 0-1
                type=ScoreType.NUMERIC,
                reason=content[:500],
                metadata={"raw_response": content[:500]},
            )

        # Check for yes/no
        content_lower = content.lower()
        if "yes" in content_lower:
            return ScoreResult(
                name=self.name,
                value=1.0,
                type=ScoreType.BOOLEAN,
                reason=content[:500],
            )
        if "no" in content_lower:
            return ScoreResult(
                name=self.name,
                value=0.0,
                type=ScoreType.BOOLEAN,
                reason=content[:500],
            )

        # Fallback
        return ScoreResult(
            name=self.name,
            value=0.0,
            type=ScoreType.NUMERIC,
            reason=f"Could not parse score from text: {content[:200]}",
            scoring_failed=True,
            metadata={"raw_response": content[:500]},
        )
