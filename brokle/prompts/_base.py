"""
Base Prompts Managers

Provides separate sync and async base implementations for prompt operations.

Architecture:
- BaseSyncPromptsManager: Uses SyncHTTPClient (no event loop)
- BaseAsyncPromptsManager: Uses AsyncHTTPClient (async/await)

This design eliminates event loop lifecycle issues.
"""

import asyncio
import threading
from typing import Any, Dict, Optional

from .._http import AsyncHTTPClient, SyncHTTPClient
from ..config import BrokleConfig
from .cache import CacheOptions, PromptCache
from .exceptions import PromptFetchError, PromptNotFoundError
from .prompt import Prompt
from .types import (
    Fallback,
    GetPromptOptions,
    PaginatedResponse,
    Pagination,
    PromptConfig,
    PromptData,
    PromptSummary,
    UpsertPromptRequest,
)


class _BasePromptsManagerMixin:
    """
    Shared functionality for both sync and async prompts managers.

    Contains cache management and utility methods that don't depend on HTTP client type.
    """

    _config: BrokleConfig
    _prompt_config: PromptConfig
    _cache: PromptCache

    def _init_cache(self, config: BrokleConfig, prompt_config: Optional[PromptConfig]):
        """Initialize cache with configuration."""
        self._config = config
        self._prompt_config = prompt_config or PromptConfig()

        if self._prompt_config.cache_enabled:
            cache_opts = CacheOptions(
                max_size=self._prompt_config.cache_max_size,
                default_ttl=self._prompt_config.cache_ttl_seconds,
            )
            self._cache = PromptCache(cache_opts)
        else:
            self._cache = PromptCache(CacheOptions(max_size=0))

    def _log(self, message: str, *args: Any) -> None:
        """Log debug messages."""
        if self._config.debug:
            print(f"[Brokle PromptManager] {message}", *args)

    def invalidate(self, name: str) -> None:
        """
        Invalidate all cached entries for a prompt.

        Removes all cached entries for the prompt name, regardless of
        label or version.

        Args:
            name: Prompt name
        """
        count = self._cache.delete_by_prompt(name)
        self._log(f"Invalidated {count} cache entries for: {name}")

    def clear_cache(self) -> None:
        """Clear the entire cache."""
        self._cache.clear()
        self._log("Cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return self._cache.get_stats()


class BaseSyncPromptsManager(_BasePromptsManagerMixin):
    """
    Sync base class for prompts manager.

    Uses SyncHTTPClient (httpx.Client) - no event loop involvement.
    All methods are synchronous.
    """

    def __init__(
        self,
        http_client: SyncHTTPClient,
        config: BrokleConfig,
        prompt_config: Optional[PromptConfig] = None,
    ):
        """
        Initialize sync prompts manager.

        Args:
            http_client: Sync HTTP client
            config: Brokle configuration
            prompt_config: Optional prompt-specific configuration
        """
        self._http = http_client
        self._init_cache(config, prompt_config)

    def _fetch_prompt(
        self, name: str, options: Optional[GetPromptOptions] = None
    ) -> PromptData:
        """
        Fetch a single prompt from the API (sync).

        Args:
            name: Prompt name
            options: Optional fetch options

        Returns:
            PromptData

        Raises:
            PromptNotFoundError: If prompt is not found
            PromptFetchError: If request fails
        """
        params: Dict[str, Any] = {}
        if options:
            if options.label:
                params["label"] = options.label
            if options.version is not None:
                params["version"] = options.version

        try:
            raw_response = self._http.get(f"/v1/prompts/{name}", params)
            return PromptData.from_dict(raw_response)
        except ValueError as e:
            if "not found" in str(e).lower():
                raise PromptNotFoundError(
                    name,
                    version=options.version if options else None,
                    label=options.label if options else None,
                )
            raise PromptFetchError(str(e))
        except Exception as e:
            raise PromptFetchError(f"Failed to fetch prompt: {e}")

    def _get(
        self,
        name: str,
        label: Optional[str] = None,
        version: Optional[int] = None,
        cache_ttl: Optional[int] = None,
        force_refresh: bool = False,
        fallback: Optional[Fallback] = None,
    ) -> Prompt:
        """
        Get a prompt with caching, SWR support, and fallback (sync).

        Priority order:
        1. Fresh cache - return immediately
        2. Fetch from API - cache and return
        3. Stale cache - return stale, background refresh
        4. Fallback - if provided, create fallback prompt
        5. Raise - if nothing available

        Args:
            name: Prompt name
            label: Optional label filter
            version: Optional version filter
            cache_ttl: Optional TTL override
            force_refresh: Skip cache and fetch fresh
            fallback: Fallback content - string for text, list of messages for chat

        Returns:
            Prompt instance
        """
        options = GetPromptOptions(label=label, version=version)
        cache_key = PromptCache.generate_key(name, label, version)
        ttl = (
            cache_ttl
            if cache_ttl is not None
            else self._prompt_config.cache_ttl_seconds
        )

        # Force refresh - skip cache, but still use fallback on failure
        if force_refresh:
            self._log(f"Force refresh: {cache_key}")
            try:
                data = self._fetch_prompt(name, options)
                self._cache.set(cache_key, data, ttl)
                return Prompt.from_data(data)
            except Exception as fetch_error:
                if fallback is not None:
                    self._log(f"Force refresh failed, using fallback: {name}")
                    return Prompt.create_fallback(name, fallback)
                raise fetch_error

        # Fresh cache - return immediately
        cached = self._cache.get(cache_key)
        if cached and self._cache.is_fresh(cache_key):
            self._log(f"Cache hit (fresh): {cache_key}")
            return Prompt.from_data(cached)

        # Try to fetch from API
        try:
            self._log(f"Cache miss: {cache_key}")
            data = self._fetch_prompt(name, options)
            self._cache.set(cache_key, data, ttl)
            return Prompt.from_data(data)
        except Exception as fetch_error:
            # Stale cache - return stale and refresh in background
            if cached:
                self._log(f"Fetch failed, using stale cache: {cache_key}")

                # Trigger background refresh if not already in progress
                if not self._cache.is_refreshing(cache_key):
                    self._cache.start_refresh(cache_key)
                    self._start_background_refresh(name, options, cache_key, ttl)

                return Prompt.from_data(cached)

            # Fallback - if provided, create fallback prompt
            if fallback is not None:
                self._log(f"Fetch failed, using fallback: {name}")
                return Prompt.create_fallback(name, fallback)

            # No cache, no fallback - raise
            raise fetch_error

    def _start_background_refresh(
        self,
        name: str,
        options: GetPromptOptions,
        cache_key: str,
        ttl: int,
    ) -> None:
        """
        Start background refresh in a separate thread.

        Uses thread-local AsyncHTTPClient with its own event loop.
        This is safe because each thread has its own isolated event loop.
        """

        def _thread_refresh():
            """Run refresh in dedicated thread with thread-local HTTP client."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def refresh_with_thread_local_client():
                # Create thread-local HTTP client bound to this thread's event loop
                thread_http = AsyncHTTPClient(self._config)

                try:
                    params: Dict[str, Any] = {}
                    if options:
                        if options.label:
                            params["label"] = options.label
                        if options.version is not None:
                            params["version"] = options.version

                    raw_response = await thread_http.get(f"/v1/prompts/{name}", params)
                    prompt_data = PromptData.from_dict(raw_response)

                    # Update cache with fresh data
                    self._cache.set(cache_key, prompt_data, ttl)
                    self._log(f"Background refresh complete: {cache_key}")
                except Exception as e:
                    self._log(f"Background refresh failed: {e}")
                finally:
                    self._cache.end_refresh(cache_key)
                    await thread_http.close()

            try:
                loop.run_until_complete(refresh_with_thread_local_client())
            finally:
                loop.close()

        thread = threading.Thread(target=_thread_refresh, daemon=True)
        thread.start()

    def _list(
        self,
        type: Optional[str] = None,
        limit: int = 20,
        page: int = 1,
    ) -> PaginatedResponse:
        """
        List prompts from the API (sync).

        Args:
            type: Optional prompt type filter
            limit: Maximum number of prompts to return
            page: Page number (1-indexed)

        Returns:
            Paginated response with prompts and pagination info
        """
        params: Dict[str, Any] = {
            "limit": limit,
            "page": page,
        }
        if type:
            params["type"] = type

        try:
            raw_response = self._http.get("/v1/prompts", params)

            # _http.get raises typed exceptions on 4xx/5xx; success
            # means raw_response is `{"data": [...], "pagination": {...}}`.
            data = [PromptSummary.from_dict(p) for p in raw_response.get("data", [])]
            pagination_data = raw_response.get("pagination", {})
            pagination = Pagination(
                total=pagination_data.get("total", 0),
                page=pagination_data.get("page", page),
                limit=pagination_data.get("limit", limit),
                pages=pagination_data.get("total_pages", 1),
            )

            return PaginatedResponse(
                data=data,
                pagination=pagination,
            )
        except PromptFetchError:
            raise
        except Exception as e:
            raise PromptFetchError(f"Failed to list prompts: {e}")

    def _upsert(self, request: UpsertPromptRequest) -> Prompt:
        """
        Create or update a prompt (sync).

        Args:
            request: Upsert request

        Returns:
            Created/updated prompt
        """
        try:
            raw_response = self._http.post("/v1/prompts", json=request.to_dict())
            # No-op: raw_response is the created prompt body. We don't
            # return it (caller is the `create()` void API); just let
            # it fall out of scope. Errors from the POST would have
            # raised before we got here.
            _ = raw_response

            # Invalidate cache for this prompt
            self.invalidate(request.name)

            # Fetch the full prompt data
            return self._get(request.name, force_refresh=True)
        except PromptFetchError:
            raise
        except Exception as e:
            raise PromptFetchError(f"Failed to upsert prompt: {e}")

    def _shutdown(self) -> None:
        """
        Internal cleanup method (sync).

        Called by parent client during shutdown.
        """
        pass  # Cache cleanup handled by cache itself


class BaseAsyncPromptsManager(_BasePromptsManagerMixin):
    """
    Async base class for prompts manager.

    Uses AsyncHTTPClient (httpx.AsyncClient) - requires async context.
    All methods are async.
    """

    def __init__(
        self,
        http_client: AsyncHTTPClient,
        config: BrokleConfig,
        prompt_config: Optional[PromptConfig] = None,
    ):
        """
        Initialize async prompts manager.

        Args:
            http_client: Async HTTP client
            config: Brokle configuration
            prompt_config: Optional prompt-specific configuration
        """
        self._http = http_client
        self._init_cache(config, prompt_config)

    async def _fetch_prompt(
        self, name: str, options: Optional[GetPromptOptions] = None
    ) -> PromptData:
        """
        Fetch a single prompt from the API (async).

        Args:
            name: Prompt name
            options: Optional fetch options

        Returns:
            PromptData

        Raises:
            PromptNotFoundError: If prompt is not found
            PromptFetchError: If request fails
        """
        params: Dict[str, Any] = {}
        if options:
            if options.label:
                params["label"] = options.label
            if options.version is not None:
                params["version"] = options.version

        try:
            raw_response = await self._http.get(f"/v1/prompts/{name}", params)
            return PromptData.from_dict(raw_response)
        except ValueError as e:
            if "not found" in str(e).lower():
                raise PromptNotFoundError(
                    name,
                    version=options.version if options else None,
                    label=options.label if options else None,
                )
            raise PromptFetchError(str(e))
        except Exception as e:
            raise PromptFetchError(f"Failed to fetch prompt: {e}")

    async def _get(
        self,
        name: str,
        label: Optional[str] = None,
        version: Optional[int] = None,
        cache_ttl: Optional[int] = None,
        force_refresh: bool = False,
        fallback: Optional[Fallback] = None,
    ) -> Prompt:
        """
        Get a prompt with caching, SWR support, and fallback (async).

        Priority order:
        1. Fresh cache - return immediately
        2. Fetch from API - cache and return
        3. Stale cache - return stale, background refresh
        4. Fallback - if provided, create fallback prompt
        5. Raise - if nothing available

        Args:
            name: Prompt name
            label: Optional label filter
            version: Optional version filter
            cache_ttl: Optional TTL override
            force_refresh: Skip cache and fetch fresh
            fallback: Fallback content - string for text, list of messages for chat

        Returns:
            Prompt instance
        """
        options = GetPromptOptions(label=label, version=version)
        cache_key = PromptCache.generate_key(name, label, version)
        ttl = (
            cache_ttl
            if cache_ttl is not None
            else self._prompt_config.cache_ttl_seconds
        )

        # Force refresh - skip cache, but still use fallback on failure
        if force_refresh:
            self._log(f"Force refresh: {cache_key}")
            try:
                data = await self._fetch_prompt(name, options)
                self._cache.set(cache_key, data, ttl)
                return Prompt.from_data(data)
            except Exception as fetch_error:
                if fallback is not None:
                    self._log(f"Force refresh failed, using fallback: {name}")
                    return Prompt.create_fallback(name, fallback)
                raise fetch_error

        # Fresh cache - return immediately
        cached = self._cache.get(cache_key)
        if cached and self._cache.is_fresh(cache_key):
            self._log(f"Cache hit (fresh): {cache_key}")
            return Prompt.from_data(cached)

        # Try to fetch from API
        try:
            self._log(f"Cache miss: {cache_key}")
            data = await self._fetch_prompt(name, options)
            self._cache.set(cache_key, data, ttl)
            return Prompt.from_data(data)
        except Exception as fetch_error:
            # Stale cache - return stale and refresh in background
            if cached:
                self._log(f"Fetch failed, using stale cache: {cache_key}")

                # Trigger background refresh if not already in progress
                if not self._cache.is_refreshing(cache_key):
                    self._cache.start_refresh(cache_key)
                    self._start_background_refresh(name, options, cache_key, ttl)

                return Prompt.from_data(cached)

            # Fallback - if provided, create fallback prompt
            if fallback is not None:
                self._log(f"Fetch failed, using fallback: {name}")
                return Prompt.create_fallback(name, fallback)

            # No cache, no fallback - raise
            raise fetch_error

    def _start_background_refresh(
        self,
        name: str,
        options: GetPromptOptions,
        cache_key: str,
        ttl: int,
    ) -> None:
        """
        Start background refresh in a separate thread.

        Uses thread-local AsyncHTTPClient with its own event loop.
        This is safe because each thread has its own isolated event loop.
        """

        def _thread_refresh():
            """Run refresh in dedicated thread with thread-local HTTP client."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def refresh_with_thread_local_client():
                # Create thread-local HTTP client bound to this thread's event loop
                thread_http = AsyncHTTPClient(self._config)

                try:
                    params: Dict[str, Any] = {}
                    if options:
                        if options.label:
                            params["label"] = options.label
                        if options.version is not None:
                            params["version"] = options.version

                    raw_response = await thread_http.get(f"/v1/prompts/{name}", params)
                    prompt_data = PromptData.from_dict(raw_response)

                    # Update cache with fresh data
                    self._cache.set(cache_key, prompt_data, ttl)
                    self._log(f"Background refresh complete: {cache_key}")
                except Exception as e:
                    self._log(f"Background refresh failed: {e}")
                finally:
                    self._cache.end_refresh(cache_key)
                    await thread_http.close()

            try:
                loop.run_until_complete(refresh_with_thread_local_client())
            finally:
                loop.close()

        thread = threading.Thread(target=_thread_refresh, daemon=True)
        thread.start()

    async def _list(
        self,
        type: Optional[str] = None,
        limit: int = 20,
        page: int = 1,
    ) -> PaginatedResponse:
        """
        List prompts from the API (async).

        Args:
            type: Optional prompt type filter
            limit: Maximum number of prompts to return
            page: Page number (1-indexed)

        Returns:
            Paginated response with prompts and pagination info
        """
        params: Dict[str, Any] = {
            "limit": limit,
            "page": page,
        }
        if type:
            params["type"] = type

        try:
            raw_response = await self._http.get("/v1/prompts", params)

            # _http.get raises typed exceptions on 4xx/5xx; success
            # means raw_response is `{"data": [...], "pagination": {...}}`.
            data = [PromptSummary.from_dict(p) for p in raw_response.get("data", [])]
            pagination_data = raw_response.get("pagination", {})
            pagination = Pagination(
                total=pagination_data.get("total", 0),
                page=pagination_data.get("page", page),
                limit=pagination_data.get("limit", limit),
                pages=pagination_data.get("total_pages", 1),
            )

            return PaginatedResponse(
                data=data,
                pagination=pagination,
            )
        except PromptFetchError:
            raise
        except Exception as e:
            raise PromptFetchError(f"Failed to list prompts: {e}")

    async def _upsert(self, request: UpsertPromptRequest) -> Prompt:
        """
        Create or update a prompt (async).

        Args:
            request: Upsert request

        Returns:
            Created/updated prompt
        """
        try:
            raw_response = await self._http.post("/v1/prompts", json=request.to_dict())
            # No-op: raw_response is the created prompt body. We don't
            # return it (caller is the `create()` void API); just let
            # it fall out of scope. Errors from the POST would have
            # raised before we got here.
            _ = raw_response

            # Invalidate cache for this prompt
            self.invalidate(request.name)

            # Fetch the full prompt data
            return await self._get(request.name, force_refresh=True)
        except PromptFetchError:
            raise
        except Exception as e:
            raise PromptFetchError(f"Failed to upsert prompt: {e}")

    async def _shutdown(self) -> None:
        """
        Internal cleanup method (async).

        Called by parent client during shutdown.
        """
        pass  # Cache cleanup handled by cache itself
