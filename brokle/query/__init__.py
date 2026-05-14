"""
Query Module (THE WEDGE)

Provides span querying capabilities for production telemetry.
This is Brokle's key differentiator - evaluating existing production
spans without re-instrumenting applications.

Usage:
    >>> from brokle import Brokle
    >>> from datetime import datetime, timedelta
    >>>
    >>> client = Brokle(api_key="bk_...")
    >>>
    >>> # Query spans
    >>> result = client.query.query(
    ...     filter="service.name=chatbot AND gen_ai.provider.name=openai",
    ...     start_time=datetime.now() - timedelta(days=7),
    ... )
    >>>
    >>> # Iterate with auto-pagination
    >>> for span in client.query.query_iter("gen_ai.provider.name=openai"):
    ...     print(span.input, span.output)
    >>>
    >>> # Validate filter syntax
    >>> validation = client.query.validate("service.name=chatbot")
    >>> if validation.valid:
    ...     print("Filter is valid!")

Types:
    - QueriedSpan: Span from query results with OTEL GenAI conventions
    - QueryResult: Paginated query result
    - ValidationResult: Filter validation result
    - TokenUsage: Token usage extracted from span
    - SpanEvent: Event attached to a span

Errors:
    - InvalidFilterError: Filter-parser rejected the expression (422).
      Subclass of the shared-client `ValidationError` so `except
      ValidationError` catches it too. Every other failure (auth,
      network, 5xx, rate limit, not found) propagates as the shared
      `BrokleError` subclass the HTTP client raised — `except
      BrokleError` catches them all.
"""

from ._managers import AsyncQueryManager, QueryManager
from .exceptions import InvalidFilterError
from .types import QueriedSpan, QueryResult, SpanEvent, TokenUsage, ValidationResult

__all__ = [
    # Types
    "QueriedSpan",
    "QueryResult",
    "ValidationResult",
    "TokenUsage",
    "SpanEvent",
    # Managers
    "QueryManager",
    "AsyncQueryManager",
    # Errors
    "InvalidFilterError",
]
