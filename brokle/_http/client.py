"""
HTTP Clients

Provides both synchronous and asynchronous HTTP clients for Brokle API communication.

Architecture:
- SyncHTTPClient: Uses httpx.Client for sync operations (no event loop)
- AsyncHTTPClient: Uses httpx.AsyncClient for async operations

This design eliminates event loop lifecycle issues that occur when trying to
bridge sync code to async code via asyncio.run().
"""

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Optional

import httpx

from .errors import (
    AuthenticationError,
    BrokleError,
    ConnectionError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
    _safe_extract_error_message,
)


def _parse_retry_after(header_value: Optional[str]) -> Optional[int]:
    """
    Parse Retry-After header, handling both formats per RFC 7231.

    The Retry-After header can be either:
    - Numeric seconds: "120"
    - HTTP-date: "Wed, 21 Oct 2015 07:28:00 GMT"

    Args:
        header_value: Raw Retry-After header value

    Returns:
        Seconds to wait, or None if parsing fails or header is empty
    """
    if not header_value:
        return None

    # Try numeric format first (most common)
    try:
        return int(header_value)
    except ValueError:
        pass

    # Try HTTP-date format: "Wed, 21 Oct 2015 07:28:00 GMT"
    try:
        retry_dt = parsedate_to_datetime(header_value)
        now = datetime.now(timezone.utc)
        delta = retry_dt - now
        seconds = int(delta.total_seconds())
        return max(0, seconds)  # Don't return negative values
    except (ValueError, TypeError):
        pass

    return None  # Unparseable - let caller handle


def _check_response_status(
    response: httpx.Response,
    resource_type: Optional[str] = None,
    identifier: Optional[str] = None,
) -> None:
    """
    Check HTTP response status and raise appropriate Brokle errors.

    Args:
        response: httpx Response object
        resource_type: Optional resource type for error messages
        identifier: Optional identifier for error messages

    Raises:
        AuthenticationError: For 401/403 responses
        NotFoundError: For 404 responses
        ValidationError: For 422 responses
        RateLimitError: For 429 responses
        ServerError: For 5xx responses
        BrokleError: For any other non-2xx responses (400, 405, 409, etc.)
    """
    status = response.status_code

    if 200 <= status < 300:
        return  # Success

    try:
        body = response.json()
    except Exception:
        body = None

    if status in (401, 403):
        raise AuthenticationError.from_response(status, body)

    if status == 404:
        if resource_type and identifier:
            raise NotFoundError.for_resource(resource_type, identifier)
        raise NotFoundError(
            f"Resource not found (HTTP {status})",
            hint="Check the resource identifier and project context.",
            details={"status_code": status, "response": body},
        )

    if status == 422:
        raise ValidationError.from_response(body or {})

    if status == 429:
        retry_after = response.headers.get("Retry-After")
        retry_seconds = _parse_retry_after(retry_after)
        raise RateLimitError.from_response(body, retry_seconds)

    if status >= 500:
        raise ServerError.from_response(status, body)

    # Catch-all for any other non-2xx status codes (400, 405, 409, etc.)
    error_msg = _safe_extract_error_message(
        body,
        default="Request failed",
    )

    raise BrokleError(
        f"HTTP {status}: {error_msg}",
        hint="Check the request parameters and API documentation.",
        details={"status_code": status, "response": body},
    )


class SyncHTTPClient:
    """
    Synchronous HTTP client for Brokle API.

    Uses httpx.Client (sync) - no event loop involvement.
    This is the correct approach for sync operations.
    """

    def __init__(self, config):
        """
        Initialize sync HTTP client.

        Args:
            config: BrokleConfig instance
        """
        self._config = config
        self._client: Optional[httpx.Client] = None

    def _get_client(self) -> httpx.Client:
        """Get or create httpx sync client."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self._config.base_url,
                timeout=self._config.timeout,
                headers={
                    "X-API-Key": self._config.api_key,
                    "Content-Type": "application/json",
                },
            )
        return self._client

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send sync GET request.

        Args:
            path: API path (e.g., "/v1/prompts/greeting")
            params: Optional query parameters

        Returns:
            Response JSON

        Raises:
            AuthenticationError: For 401/403 responses
            NotFoundError: For 404 responses
            ConnectionError: For connection failures
        """
        try:
            response = self._get_client().get(path, params=params)
        except httpx.TransportError as e:
            raise ConnectionError.from_exception(e, self._config.base_url)
        _check_response_status(response)
        return response.json()

    def post(self, path: str, json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send sync POST request.

        Args:
            path: API path
            json: Request body

        Returns:
            Response JSON

        Raises:
            AuthenticationError: For 401/403 responses
            ValidationError: For 422 responses
            ConnectionError: For connection failures
        """
        try:
            response = self._get_client().post(path, json=json)
        except httpx.TransportError as e:
            raise ConnectionError.from_exception(e, self._config.base_url)
        _check_response_status(response)
        return response.json()

    def patch(self, path: str, json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send sync PATCH request.

        Args:
            path: API path
            json: Request body

        Returns:
            Response JSON

        Raises:
            AuthenticationError: For 401/403 responses
            ValidationError: For 422 responses
            ConnectionError: For connection failures
        """
        try:
            response = self._get_client().patch(path, json=json)
        except httpx.TransportError as e:
            raise ConnectionError.from_exception(e, self._config.base_url)
        _check_response_status(response)
        return response.json()

    def delete(self, path: str) -> Optional[Dict[str, Any]]:
        """
        Send sync DELETE request.

        Args:
            path: API path

        Returns:
            Response JSON, or None for 204 No Content responses

        Raises:
            AuthenticationError: For 401/403 responses
            NotFoundError: For 404 responses
            ConnectionError: For connection failures
        """
        try:
            response = self._get_client().delete(path)
        except httpx.TransportError as e:
            raise ConnectionError.from_exception(e, self._config.base_url)
        _check_response_status(response)
        if response.status_code == 204:
            return None
        return response.json()

    def close(self):
        """Close sync HTTP client."""
        if self._client:
            self._client.close()
            self._client = None


class AsyncHTTPClient:
    """
    Asynchronous HTTP client for Brokle API.

    Uses httpx.AsyncClient - requires async context.
    Uses the caller's event loop, never creates its own.
    """

    def __init__(self, config):
        """
        Initialize async HTTP client.

        Args:
            config: BrokleConfig instance
        """
        self._config = config
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create httpx async client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._config.base_url,
                timeout=self._config.timeout,
                headers={
                    "X-API-Key": self._config.api_key,
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def get(
        self, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send async GET request.

        Args:
            path: API path (e.g., "/v1/prompts/greeting")
            params: Optional query parameters

        Returns:
            Response JSON

        Raises:
            AuthenticationError: For 401/403 responses
            NotFoundError: For 404 responses
            ConnectionError: For connection failures
        """
        try:
            response = await self._get_client().get(path, params=params)
        except httpx.TransportError as e:
            raise ConnectionError.from_exception(e, self._config.base_url)
        _check_response_status(response)
        return response.json()

    async def post(
        self, path: str, json: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send async POST request.

        Args:
            path: API path
            json: Request body

        Returns:
            Response JSON

        Raises:
            AuthenticationError: For 401/403 responses
            ValidationError: For 422 responses
            ConnectionError: For connection failures
        """
        try:
            response = await self._get_client().post(path, json=json)
        except httpx.TransportError as e:
            raise ConnectionError.from_exception(e, self._config.base_url)
        _check_response_status(response)
        return response.json()

    async def patch(
        self, path: str, json: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send async PATCH request.

        Args:
            path: API path
            json: Request body

        Returns:
            Response JSON

        Raises:
            AuthenticationError: For 401/403 responses
            ValidationError: For 422 responses
            ConnectionError: For connection failures
        """
        try:
            response = await self._get_client().patch(path, json=json)
        except httpx.TransportError as e:
            raise ConnectionError.from_exception(e, self._config.base_url)
        _check_response_status(response)
        return response.json()

    async def delete(self, path: str) -> Optional[Dict[str, Any]]:
        """
        Send async DELETE request.

        Args:
            path: API path

        Returns:
            Response JSON, or None for 204 No Content responses

        Raises:
            AuthenticationError: For 401/403 responses
            NotFoundError: For 404 responses
            ConnectionError: For connection failures
        """
        try:
            response = await self._get_client().delete(path)
        except httpx.TransportError as e:
            raise ConnectionError.from_exception(e, self._config.base_url)
        _check_response_status(response)
        if response.status_code == 204:
            return None
        return response.json()

    async def close(self):
        """Close async HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
