"""
Scores Manager

Provides both synchronous and asynchronous score submission for Brokle.

Supports two scoring modes:
1. Direct score: Pass name + value directly
2. Scorer function: Pass a scorer callable with output/expected

Sync Usage:
    >>> from brokle import Brokle
    >>> from brokle.scorers import ExactMatch
    >>>
    >>> client = Brokle(api_key="bk_...")
    >>>
    >>> # Direct score
    >>> client.scores.submit(
    ...     trace_id="abc123",
    ...     name="quality",
    ...     value=0.9,
    ... )
    >>>
    >>> # With scorer
    >>> exact = ExactMatch()
    >>> client.scores.submit(
    ...     trace_id="abc123",
    ...     scorer=exact,
    ...     output="Paris",
    ...     expected="Paris",
    ... )

Async Usage:
    >>> async with AsyncBrokle(api_key="bk_...") as client:
    ...     await client.scores.submit(
    ...         trace_id="abc123",
    ...         name="quality",
    ...         value=0.9,
    ...     )
"""

from typing import Any, Dict, List, Optional, Union

from .._http import AsyncHTTPClient, SyncHTTPClient
from ..config import BrokleConfig
from .exceptions import ScoreError
from .types import ScoreResult, ScorerProtocol, ScoreSource, ScoreType, ScoreValue


class _BaseScoresManagerMixin:
    """
    Shared functionality for both sync and async scores managers.

    Contains utility methods that don't depend on HTTP client type.
    """

    _config: BrokleConfig

    def _log(self, message: str, *args: Any) -> None:
        """Log debug messages."""
        if self._config.debug:
            print(f"[Brokle Scores] {message}", *args)

    def _normalize_score_result(
        self, result: ScoreValue, scorer: ScorerProtocol
    ) -> List[ScoreResult]:
        """Normalize any scorer return type to List[ScoreResult]."""
        # Handle None (skip scoring)
        if result is None:
            return []

        scorer_name = getattr(scorer, "name", None) or getattr(
            scorer, "__name__", "scorer"
        )

        if isinstance(result, list):
            return result
        elif isinstance(result, ScoreResult):
            return [result]
        elif isinstance(result, bool):
            return [
                ScoreResult(
                    name=scorer_name,
                    value=1.0 if result else 0.0,
                    type=ScoreType.BOOLEAN,
                )
            ]
        elif isinstance(result, (int, float)):
            return [ScoreResult(name=scorer_name, value=float(result))]
        else:
            raise TypeError(
                f"Scorer must return ScoreResult, List[ScoreResult], float, or bool, "
                f"got {type(result).__name__}"
            )


class ScoresManager(_BaseScoresManagerMixin):
    """
    Sync scores manager for Brokle.

    All methods are synchronous. Uses SyncHTTPClient (httpx.Client) internally -
    no event loop involvement.

    Example:
        >>> from brokle import Brokle
        >>> from brokle.scorers import ExactMatch
        >>>
        >>> client = Brokle(api_key="bk_...")
        >>>
        >>> # Direct score
        >>> client.scores.submit(
        ...     trace_id="abc123",
        ...     name="accuracy",
        ...     value=0.95,
        ... )
        >>>
        >>> # With built-in scorer
        >>> exact = ExactMatch(name="answer_match")
        >>> client.scores.submit(
        ...     trace_id="abc123",
        ...     scorer=exact,
        ...     output="4",
        ...     expected="4",
        ... )
    """

    def __init__(
        self,
        http_client: SyncHTTPClient,
        config: BrokleConfig,
    ):
        """
        Initialize sync scores manager.

        Args:
            http_client: Sync HTTP client
            config: Brokle configuration
        """
        self._http = http_client
        self._config = config

    def submit(
        self,
        trace_id: str,
        scorer: Optional[ScorerProtocol] = None,
        output: Any = None,
        expected: Any = None,
        name: Optional[str] = None,
        value: Optional[float] = None,
        type: ScoreType = ScoreType.NUMERIC,
        source: ScoreSource = ScoreSource.CODE,
        span_id: Optional[str] = None,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Submit a score for a trace or span.

        Two modes:
        1. With scorer: Pass scorer callable + output/expected
        2. Direct: Pass name + value directly

        Args:
            trace_id: Trace ID to score
            scorer: Optional scorer callable (ExactMatch, Contains, custom @scorer)
            output: The actual output to evaluate (for scorer mode)
            expected: The expected/reference output (for scorer mode)
            name: Score name (required for direct mode)
            value: Score value (required for direct mode)
            type: Score type (NUMERIC, CATEGORICAL, BOOLEAN)
            source: Score source (code, llm, human)
            span_id: Optional span ID for span-level scoring
            reason: Human-readable explanation
            metadata: Additional metadata
            **kwargs: Additional arguments passed to scorer

        Returns:
            Single score dict or list of score dicts (if scorer returns List[ScoreResult])

        Raises:
            ValueError: If neither scorer nor (name + value) are provided
            ScoreError: If the API request fails

        Example:
            >>> # Direct score
            >>> client.scores.submit(
            ...     trace_id="abc123",
            ...     name="quality",
            ...     value=0.9,
            ...     reason="High quality response",
            ... )
            >>>
            >>> # With scorer
            >>> from brokle.scorers import ExactMatch
            >>> exact = ExactMatch()
            >>> client.scores.submit(
            ...     trace_id="abc123",
            ...     scorer=exact,
            ...     output="Paris",
            ...     expected="Paris",
            ... )
        """
        if scorer is not None:
            try:
                result = scorer(output=output, expected=expected, **kwargs)
            except Exception as e:
                scorer_name = getattr(scorer, "name", None) or getattr(
                    scorer, "__name__", "unknown"
                )
                return self._submit_score(
                    trace_id=trace_id,
                    name=scorer_name,
                    value=0.0,
                    type="NUMERIC",
                    source=source.value,
                    span_id=span_id,
                    reason=f"Scorer failed: {str(e)}",
                    metadata={"scoring_failed": True, "error": str(e)},
                )

            results = self._normalize_score_result(result, scorer)

            if len(results) == 0:
                self._log("Scorer returned None, no score submitted")
                return []

            responses: List[Dict[str, Any]] = []
            for score_result in results:
                resp = self._submit_score(
                    trace_id=trace_id,
                    name=score_result.name,
                    value=score_result.value,
                    type=score_result.type.value,
                    source=source.value,
                    span_id=span_id,
                    string_value=score_result.string_value,
                    reason=score_result.reason or reason,
                    metadata=score_result.metadata or metadata,
                )
                responses.append(resp)

            return responses[0] if len(responses) == 1 else responses
        else:
            if name is None or value is None:
                raise ValueError("name and value required when not using scorer")
            return self._submit_score(
                trace_id=trace_id,
                name=name,
                value=value,
                type=type.value,
                source=source.value,
                span_id=span_id,
                reason=reason,
                metadata=metadata,
            )

    def _submit_score(
        self,
        trace_id: str,
        name: str,
        value: float,
        type: str = "NUMERIC",
        source: str = "code",
        span_id: Optional[str] = None,
        string_value: Optional[str] = None,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Submit a score to the API (sync).

        Args:
            trace_id: Trace ID to score
            name: Score name
            value: Score value
            type: Score type
            source: Score source
            span_id: Optional span ID
            string_value: String value for CATEGORICAL
            reason: Explanation
            metadata: Additional metadata

        Returns:
            Score submission result

        Raises:
            ScoreError: If the API request fails
        """
        self._log(f"Submitting score: {trace_id} - {name}={value}")

        payload: Dict[str, Any] = {
            "trace_id": trace_id,
            "name": name,
            "value": value,
            "type": type,
            "source": source,
        }
        if span_id:
            payload["span_id"] = span_id
        if string_value:
            payload["string_value"] = string_value
        if reason:
            payload["reason"] = reason
        if metadata:
            payload["metadata"] = metadata

        try:
            raw_response = self._http.post("/v1/scores", json=payload)
            return raw_response
        except ValueError as e:
            raise ScoreError(f"Failed to submit score: {e}")
        except Exception as e:
            raise ScoreError(f"Failed to submit score: {e}")

    def batch(
        self,
        scores: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Submit multiple scores to the API.

        Args:
            scores: List of score dictionaries with keys:
                - trace_id: Trace ID (required)
                - name: Score name (required)
                - value: Score value (required)
                - type: Score type (optional, default: "NUMERIC")
                - source: Score source (optional, default: "code")
                - span_id: Span ID (optional)
                - string_value: String value (optional)
                - reason: Reason (optional)
                - metadata: Metadata (optional)

        Returns:
            Batch submission result

        Raises:
            ScoreError: If the API request fails

        Example:
            >>> client.scores.batch([
            ...     {"trace_id": "abc123", "name": "accuracy", "value": 0.9},
            ...     {"trace_id": "abc123", "name": "relevance", "value": 0.85},
            ... ])
        """
        self._log(f"Submitting batch of {len(scores)} scores")

        try:
            raw_response = self._http.post("/v1/scores/batch", json={"scores": scores})
            return raw_response
        except ValueError as e:
            raise ScoreError(f"Failed to submit scores batch: {e}")
        except Exception as e:
            raise ScoreError(f"Failed to submit scores batch: {e}")


class AsyncScoresManager(_BaseScoresManagerMixin):
    """
    Async scores manager for AsyncBrokle.

    All methods are async and return coroutines that must be awaited.
    Uses AsyncHTTPClient (httpx.AsyncClient) internally.

    Example:
        >>> async with AsyncBrokle(api_key="bk_...") as client:
        ...     await client.scores.submit(
        ...         trace_id="abc123",
        ...         name="quality",
        ...         value=0.9,
        ...     )
    """

    def __init__(
        self,
        http_client: AsyncHTTPClient,
        config: BrokleConfig,
    ):
        """
        Initialize async scores manager.

        Args:
            http_client: Async HTTP client
            config: Brokle configuration
        """
        self._http = http_client
        self._config = config

    async def submit(
        self,
        trace_id: str,
        scorer: Optional[ScorerProtocol] = None,
        output: Any = None,
        expected: Any = None,
        name: Optional[str] = None,
        value: Optional[float] = None,
        type: ScoreType = ScoreType.NUMERIC,
        source: ScoreSource = ScoreSource.CODE,
        span_id: Optional[str] = None,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Submit a score for a trace or span (async).

        Two modes:
        1. With scorer: Pass scorer callable + output/expected
        2. Direct: Pass name + value directly

        Args:
            trace_id: Trace ID to score
            scorer: Optional scorer callable (ExactMatch, Contains, custom @scorer)
            output: The actual output to evaluate (for scorer mode)
            expected: The expected/reference output (for scorer mode)
            name: Score name (required for direct mode)
            value: Score value (required for direct mode)
            type: Score type (NUMERIC, CATEGORICAL, BOOLEAN)
            source: Score source (code, llm, human)
            span_id: Optional span ID for span-level scoring
            reason: Human-readable explanation
            metadata: Additional metadata
            **kwargs: Additional arguments passed to scorer

        Returns:
            Single score dict or list of score dicts (if scorer returns List[ScoreResult])

        Raises:
            ValueError: If neither scorer nor (name + value) are provided
            ScoreError: If the API request fails
        """
        if scorer is not None:
            try:
                result = scorer(output=output, expected=expected, **kwargs)
            except Exception as e:
                scorer_name = getattr(scorer, "name", None) or getattr(
                    scorer, "__name__", "unknown"
                )
                return await self._submit_score(
                    trace_id=trace_id,
                    name=scorer_name,
                    value=0.0,
                    type="NUMERIC",
                    source=source.value,
                    span_id=span_id,
                    reason=f"Scorer failed: {str(e)}",
                    metadata={"scoring_failed": True, "error": str(e)},
                )

            results = self._normalize_score_result(result, scorer)

            if len(results) == 0:
                self._log("Scorer returned None, no score submitted")
                return []

            responses: List[Dict[str, Any]] = []
            for score_result in results:
                resp = await self._submit_score(
                    trace_id=trace_id,
                    name=score_result.name,
                    value=score_result.value,
                    type=score_result.type.value,
                    source=source.value,
                    span_id=span_id,
                    string_value=score_result.string_value,
                    reason=score_result.reason or reason,
                    metadata=score_result.metadata or metadata,
                )
                responses.append(resp)

            return responses[0] if len(responses) == 1 else responses
        else:
            if name is None or value is None:
                raise ValueError("name and value required when not using scorer")
            return await self._submit_score(
                trace_id=trace_id,
                name=name,
                value=value,
                type=type.value,
                source=source.value,
                span_id=span_id,
                reason=reason,
                metadata=metadata,
            )

    async def _submit_score(
        self,
        trace_id: str,
        name: str,
        value: float,
        type: str = "NUMERIC",
        source: str = "code",
        span_id: Optional[str] = None,
        string_value: Optional[str] = None,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Submit a score to the API (async).

        Args:
            trace_id: Trace ID to score
            name: Score name
            value: Score value
            type: Score type
            source: Score source
            span_id: Optional span ID
            string_value: String value for CATEGORICAL
            reason: Explanation
            metadata: Additional metadata

        Returns:
            Score submission result

        Raises:
            ScoreError: If the API request fails
        """
        self._log(f"Submitting score: {trace_id} - {name}={value}")

        payload: Dict[str, Any] = {
            "trace_id": trace_id,
            "name": name,
            "value": value,
            "type": type,
            "source": source,
        }
        if span_id:
            payload["span_id"] = span_id
        if string_value:
            payload["string_value"] = string_value
        if reason:
            payload["reason"] = reason
        if metadata:
            payload["metadata"] = metadata

        try:
            raw_response = await self._http.post("/v1/scores", json=payload)
            return raw_response
        except ValueError as e:
            raise ScoreError(f"Failed to submit score: {e}")
        except Exception as e:
            raise ScoreError(f"Failed to submit score: {e}")

    async def batch(
        self,
        scores: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Submit multiple scores to the API (async).

        Args:
            scores: List of score dictionaries with keys:
                - trace_id: Trace ID (required)
                - name: Score name (required)
                - value: Score value (required)
                - type: Score type (optional, default: "NUMERIC")
                - source: Score source (optional, default: "code")
                - span_id: Span ID (optional)
                - string_value: String value (optional)
                - reason: Reason (optional)
                - metadata: Metadata (optional)

        Returns:
            Batch submission result

        Raises:
            ScoreError: If the API request fails
        """
        self._log(f"Submitting batch of {len(scores)} scores")

        try:
            raw_response = await self._http.post(
                "/v1/scores/batch", json={"scores": scores}
            )
            return raw_response
        except ValueError as e:
            raise ScoreError(f"Failed to submit scores batch: {e}")
        except Exception as e:
            raise ScoreError(f"Failed to submit scores batch: {e}")
