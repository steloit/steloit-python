"""
Exceptions for the query module.

The query module exposes ONE axis-2 domain error:

    InvalidFilterError  – the backend parser rejected a filter expression.

It inherits from the shared-client `ValidationError` (422) so callers can
catch either `except ValidationError` (the HTTP-family view) or
`except InvalidFilterError` (the filter-syntax-specific view) and both
compose. Every other failure (auth, network, 5xx, rate limit, not found)
propagates as the shared `BrokleError` subclass the HTTP client raised —
see `sdk/python/brokle/_http/errors.py`.

This follows Stripe/OpenAI/Anthropic/Azure/Google/Octokit/Twilio: one
shared error hierarchy per SDK; module-local types exist only for
semantics HTTP status cannot express, and they extend the shared family
rather than wrapping it.
"""

from typing import Optional

from .._http.errors import ValidationError


class InvalidFilterError(ValidationError):
    """
    The backend filter parser rejected a filter expression.

    Raised by `QueryManager.query()` and `AsyncQueryManager.query()` on
    HTTP 422 responses from `/v1/spans/query`. The preflight
    `validate()` variant returns `ValidationResult(valid=False, error=...)`
    instead of raising so callers can branch without a try/except.

    Because `InvalidFilterError` extends the shared `ValidationError`,
    `except ValidationError` (HTTP-family catch) AND `except
    InvalidFilterError` (filter-specific catch) both match.

    Attributes:
        filter: The invalid filter expression.
    """

    def __init__(self, filter_expr: str, message: Optional[str] = None):
        self.filter = filter_expr
        full = f"Invalid filter '{filter_expr}'"
        if message:
            full = f"{full}: {message}"
        # ValidationError.__init__ takes (message, *, hint, original_error,
        # details). Pass the filter as details so tooling can surface it.
        super().__init__(full, details={"filter": filter_expr})
