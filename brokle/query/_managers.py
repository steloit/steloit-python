"""
Query Managers (THE WEDGE)

Provides both synchronous and asynchronous query managers for querying
production spans. This is Brokle's key differentiator - evaluating existing
production telemetry without re-instrumenting applications.

Sync Usage:
    >>> from brokle import Brokle
    >>>
    >>> client = Brokle(api_key="bk_...")
    >>>
    >>> result = client.query.query(
    ...     filter="service.name=chatbot AND gen_ai.provider.name=openai",
    ...     start_time=datetime.now() - timedelta(days=7),
    ... )
    >>> for span in result.spans:
    ...     print(span.name, span.model)

Async Usage:
    >>> async with AsyncBrokle(api_key="bk_...") as client:
    ...     async for span in client.query.query_iter(
    ...         filter="gen_ai.provider.name=openai",
    ...     ):
    ...         print(span.input, span.output)
"""

from datetime import datetime
from typing import Any, AsyncIterator, Dict, Iterator, Optional

from .._http import AsyncHTTPClient, SyncHTTPClient
from .._http.errors import BrokleError, ValidationError as HTTPValidationError
from ..config import BrokleConfig
from .exceptions import InvalidFilterError
from .types import QueriedSpan, QueryResult, ValidationResult


def _ensure_dict(raw: Any, *, resource: str) -> Dict[str, Any]:
    """
    Guarantee `raw` is a dict before handing it to a `from_dict`
    classmethod that calls `.get()` on it. A 2xx body that parses
    as JSON but is a list/string/number/bool/null otherwise raises
    `AttributeError` from inside `from_dict`, which escapes the
    typed-error contract.

    Pydantic v2's canonical idiom for the same guard is a
    `@model_validator(mode='before')` with `if not isinstance(data,
    dict): raise ValueError(...)`. Brokle's response parsers are
    hand-rolled rather than Pydantic-driven, so we replicate that
    guard as a one-liner here and reuse it across the SDK.
    """
    if not isinstance(raw, dict):
        raise BrokleError(
            f"Failed to parse {resource}: expected JSON object, got "
            f"{type(raw).__name__}",
            details={"response": raw},
        )
    return raw


# Backend error code for filter-parser rejection (see
# `pkg/errors/codes.go` → CodeInvalidFilterExpression). The SDK
# discriminates this 422 sub-kind from generic input-validation 422s
# via the `error.code` field on the response body — matches
# Stripe/OpenAI/JSON:API/RFC 9457 §3.1.3 convention.
_INVALID_FILTER_CODE = "invalid_filter_expression"


def _is_invalid_filter_error(e: HTTPValidationError) -> bool:
    """
    Is this 422 specifically the filter parser's rejection, or a
    generic input-validation failure (invalid limit, page, timestamp,
    etc.)?

    Returns False when the body is missing/malformed so non-filter
    422s still surface as the full structured `ValidationError` — the
    caller's `except ValidationError` catches them and they can read
    `.details["response"]["error"]["errors"]` for per-field diagnostics.
    """
    response = e.details.get("response") if e.details else None
    if not isinstance(response, dict):
        return False
    err = response.get("error")
    if not isinstance(err, dict):
        return False
    return err.get("code") == _INVALID_FILTER_CODE


def _parse_query_result(raw: Any) -> QueryResult:
    """
    Convert a 2xx query response body into a typed `QueryResult`.

    HTTP-level failures are already handled by the shared client and
    surface as `BrokleError` subclasses. The only remaining failure
    class is a backend contract violation — a 2xx body that doesn't
    match the documented shape. We surface that as a `BrokleError`
    (not a module-local wrapper) so users catching `except
    BrokleError` still see every query failure with one clause.
    """
    data = _ensure_dict(raw, resource="query response")
    try:
        return QueryResult.from_dict(data)
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        raise BrokleError(
            f"Failed to parse query response: {e}",
            original_error=e,
            details={"response": raw},
        ) from e


def _parse_validation_result(raw: Any) -> ValidationResult:
    """Same parse-failure contract as `_parse_query_result`."""
    data = _ensure_dict(raw, resource="validation response")
    try:
        return ValidationResult.from_dict(data)
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        raise BrokleError(
            f"Failed to parse validation response: {e}",
            original_error=e,
            details={"response": raw},
        ) from e


class _BaseQueryManagerMixin:
    """
    Shared functionality for both sync and async query managers.
    """

    _config: BrokleConfig

    def _log(self, message: str, *args: Any) -> None:
        """Log debug messages."""
        if self._config.debug:
            print(f"[Brokle Query] {message}", *args)

    def _build_query_body(
        self,
        filter: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
        page: int = 1,
    ) -> Dict[str, Any]:
        """Build request body for query API."""
        body: Dict[str, Any] = {
            "filter": filter,
            "limit": limit,
            "page": page,
        }
        if start_time:
            body["start_time"] = (
                start_time.isoformat() + "Z"
                if start_time.tzinfo is None
                else start_time.isoformat()
            )
        if end_time:
            body["end_time"] = (
                end_time.isoformat() + "Z"
                if end_time.tzinfo is None
                else end_time.isoformat()
            )
        return body


class QueryManager(_BaseQueryManagerMixin):
    """
    Sync query manager for Brokle.

    All methods are synchronous. Uses SyncHTTPClient (httpx.Client) internally.

    Example:
        >>> from brokle import Brokle
        >>> from datetime import datetime, timedelta
        >>>
        >>> client = Brokle(api_key="bk_...")
        >>>
        >>> # Query spans from the last 7 days
        >>> result = client.query.query(
        ...     filter="service.name=chatbot AND gen_ai.provider.name=openai",
        ...     start_time=datetime.now() - timedelta(days=7),
        ... )
        >>>
        >>> # Access spans and their convenience fields
        >>> for span in result.spans:
        ...     print(f"{span.model}: {span.input[:50]}...")
    """

    def __init__(
        self,
        http_client: SyncHTTPClient,
        config: BrokleConfig,
    ):
        """
        Initialize sync query manager.

        Args:
            http_client: Sync HTTP client
            config: Brokle configuration
        """
        self._http = http_client
        self._config = config

    def query(
        self,
        filter: str,
        *,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
        page: int = 1,
    ) -> QueryResult:
        """
        Query spans using filter expression.

        Args:
            filter: Filter expression (e.g., "service.name=chatbot AND gen_ai.provider.name=openai")
            start_time: Start of time range (optional)
            end_time: End of time range (optional)
            limit: Maximum number of spans to return (default: 1000)
            page: Page number for pagination, 1-indexed (default: 1)

        Returns:
            QueryResult with spans and pagination metadata

        Raises:
            InvalidFilterError: If filter syntax is invalid
            BrokleError: If API request fails (AuthenticationError, ServerError,
                ConnectionError, etc. — shared hierarchy)

        Example:
            >>> result = client.query.query(
            ...     filter="gen_ai.response.model=gpt-4",
            ...     limit=100,
            ... )
            >>> print(f"Found {result.total} spans")
        """
        self._log(f"Querying spans: filter={filter}, limit={limit}, page={page}")

        body = self._build_query_body(filter, start_time, end_time, limit, page)

        # Shared client raises typed BrokleError subclasses on 4xx/5xx;
        # those propagate unchanged. A 422 is promoted to
        # InvalidFilterError only when the backend signals the
        # filter-parser rejection via `error.code =
        # "invalid_filter_expression"`. Generic input 422s (invalid
        # limit/page/timestamp) propagate as the shared ValidationError
        # so callers can inspect `.details["response"]["error"]["errors"]`
        # for per-field diagnostics.
        try:
            raw_response = self._http.post("/v1/spans/query", json=body)
        except HTTPValidationError as e:
            if _is_invalid_filter_error(e):
                raise InvalidFilterError(filter, e.message) from e
            raise

        return _parse_query_result(raw_response)

    def query_iter(
        self,
        filter: str,
        *,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        batch_size: int = 100,
    ) -> Iterator[QueriedSpan]:
        """
        Iterate spans with auto-pagination.

        Yields spans one at a time, automatically fetching next pages.
        More memory-efficient than query() for large result sets.

        Args:
            filter: Filter expression
            start_time: Start of time range (optional)
            end_time: End of time range (optional)
            batch_size: Number of spans per API request (default: 100)

        Yields:
            QueriedSpan objects

        Raises:
            InvalidFilterError: If filter syntax is invalid
            BrokleError: If API request fails (AuthenticationError, ServerError,
                ConnectionError, etc. — shared hierarchy)

        Example:
            >>> for span in client.query.query_iter("gen_ai.provider.name=openai"):
            ...     process_span(span)
        """
        page = 1
        while True:
            result = self.query(
                filter,
                start_time=start_time,
                end_time=end_time,
                limit=batch_size,
                page=page,
            )
            for span in result.spans:
                yield span

            if not result.has_more:
                break

            page = result.next_page if result.next_page is not None else (page + 1)

    def validate(self, filter: str) -> ValidationResult:
        """
        Validate filter syntax.

        Check if a filter expression is syntactically valid without executing a query.

        Args:
            filter: Filter expression to validate

        Returns:
            ValidationResult with valid flag and message/error

        Raises:
            BrokleError: If API request fails (AuthenticationError, ServerError,
                ConnectionError, etc. — shared hierarchy)

        Example:
            >>> result = client.query.validate("service.name=chatbot")
            >>> if result.valid:
            ...     print("Filter is valid!")
        """
        self._log(f"Validating filter: {filter}")

        # validate() is a preflight check — 422 returns a
        # ValidationResult(valid=False) rather than raising so callers
        # can inspect `.error` without a try/except. All other
        # BrokleError subclasses (auth, network, 5xx) propagate
        # unchanged.
        try:
            raw_response = self._http.post(
                "/v1/spans/query/validate", json={"filter": filter}
            )
        except HTTPValidationError as e:
            return ValidationResult(valid=False, error=e.message)

        return _parse_validation_result(raw_response)

    def validate_or_raise(self, filter: str) -> None:
        """
        Validate filter and raise if invalid.

        Convenience method that raises InvalidFilterError if the filter is invalid.

        Args:
            filter: Filter expression to validate

        Raises:
            InvalidFilterError: If filter is invalid
            BrokleError: If API request fails (AuthenticationError, ServerError,
                ConnectionError, etc. — shared hierarchy)

        Example:
            >>> client.query.validate_or_raise("service.name=chatbot")  # Returns None if valid
            >>> client.query.validate_or_raise("invalid syntax")  # Raises InvalidFilterError
        """
        result = self.validate(filter)
        if not result.valid:
            raise InvalidFilterError(filter, result.error)


class AsyncQueryManager(_BaseQueryManagerMixin):
    """
    Async query manager for Brokle.

    All methods are asynchronous. Uses AsyncHTTPClient (httpx.AsyncClient) internally.

    Example:
        >>> from brokle import AsyncBrokle
        >>> from datetime import datetime, timedelta
        >>>
        >>> async with AsyncBrokle(api_key="bk_...") as client:
        ...     result = await client.query.query(
        ...         filter="service.name=chatbot",
        ...         start_time=datetime.now() - timedelta(days=7),
        ...     )
        ...     for span in result.spans:
        ...         print(span.model)
    """

    def __init__(
        self,
        http_client: AsyncHTTPClient,
        config: BrokleConfig,
    ):
        """
        Initialize async query manager.

        Args:
            http_client: Async HTTP client
            config: Brokle configuration
        """
        self._http = http_client
        self._config = config

    async def query(
        self,
        filter: str,
        *,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
        page: int = 1,
    ) -> QueryResult:
        """
        Query spans using filter expression (async).

        Args:
            filter: Filter expression (e.g., "service.name=chatbot AND gen_ai.provider.name=openai")
            start_time: Start of time range (optional)
            end_time: End of time range (optional)
            limit: Maximum number of spans to return (default: 1000)
            page: Page number for pagination, 1-indexed (default: 1)

        Returns:
            QueryResult with spans and pagination metadata

        Raises:
            InvalidFilterError: If filter syntax is invalid
            BrokleError: If API request fails (AuthenticationError, ServerError,
                ConnectionError, etc. — shared hierarchy)

        Example:
            >>> result = await client.query.query(
            ...     filter="gen_ai.response.model=gpt-4",
            ...     limit=100,
            ... )
            >>> print(f"Found {result.total} spans")
        """
        self._log(f"Querying spans: filter={filter}, limit={limit}, page={page}")

        body = self._build_query_body(filter, start_time, end_time, limit, page)

        # Same promotion rule as sync QueryManager.query() — discriminate
        # filter-parser rejection via `error.code`, propagate all other
        # 422 flavours unchanged.
        try:
            raw_response = await self._http.post("/v1/spans/query", json=body)
        except HTTPValidationError as e:
            if _is_invalid_filter_error(e):
                raise InvalidFilterError(filter, e.message) from e
            raise

        return _parse_query_result(raw_response)

    async def query_iter(
        self,
        filter: str,
        *,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        batch_size: int = 100,
    ) -> AsyncIterator[QueriedSpan]:
        """
        Iterate spans with auto-pagination (async).

        Yields spans one at a time, automatically fetching next pages.
        More memory-efficient than query() for large result sets.

        Args:
            filter: Filter expression
            start_time: Start of time range (optional)
            end_time: End of time range (optional)
            batch_size: Number of spans per API request (default: 100)

        Yields:
            QueriedSpan objects

        Raises:
            InvalidFilterError: If filter syntax is invalid
            BrokleError: If API request fails (AuthenticationError, ServerError,
                ConnectionError, etc. — shared hierarchy)

        Example:
            >>> async for span in client.query.query_iter("gen_ai.provider.name=openai"):
            ...     await process_span(span)
        """
        page = 1
        while True:
            result = await self.query(
                filter,
                start_time=start_time,
                end_time=end_time,
                limit=batch_size,
                page=page,
            )
            for span in result.spans:
                yield span

            if not result.has_more:
                break

            page = result.next_page if result.next_page is not None else (page + 1)

    async def validate(self, filter: str) -> ValidationResult:
        """
        Validate filter syntax (async).

        Check if a filter expression is syntactically valid without executing a query.

        Args:
            filter: Filter expression to validate

        Returns:
            ValidationResult with valid flag and message/error

        Raises:
            BrokleError: If API request fails (AuthenticationError, ServerError,
                ConnectionError, etc. — shared hierarchy)

        Example:
            >>> result = await client.query.validate("service.name=chatbot")
            >>> if result.valid:
            ...     print("Filter is valid!")
        """
        self._log(f"Validating filter: {filter}")

        try:
            raw_response = await self._http.post(
                "/v1/spans/query/validate", json={"filter": filter}
            )
        except HTTPValidationError as e:
            return ValidationResult(valid=False, error=e.message)

        return _parse_validation_result(raw_response)

    async def validate_or_raise(self, filter: str) -> None:
        """
        Validate filter and raise if invalid (async).

        Convenience method that raises InvalidFilterError if the filter is invalid.

        Args:
            filter: Filter expression to validate

        Raises:
            InvalidFilterError: If filter is invalid
            BrokleError: If API request fails (AuthenticationError, ServerError,
                ConnectionError, etc. — shared hierarchy)

        Example:
            >>> await client.query.validate_or_raise("service.name=chatbot")  # Returns None if valid
            >>> await client.query.validate_or_raise("invalid syntax")  # Raises InvalidFilterError
        """
        result = await self.validate(filter)
        if not result.valid:
            raise InvalidFilterError(filter, result.error)
