"""
HTTP Client Module

Provides sync and async HTTP clients for Brokle API communication.
"""

from .client import AsyncHTTPClient, SyncHTTPClient
from .errors import (
    AuthenticationError,
    BrokleError,
    ConnectionError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
    raise_for_status,
)

__all__ = [
    "AsyncHTTPClient",
    "SyncHTTPClient",
    # Errors (Langfuse pattern - no prefix)
    "BrokleError",
    "AuthenticationError",
    "ConnectionError",
    "ValidationError",
    "RateLimitError",
    "NotFoundError",
    "ServerError",
    "raise_for_status",
]
