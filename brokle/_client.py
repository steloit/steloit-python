"""
Brokle Client

Provides both synchronous and asynchronous clients for Brokle's
OpenTelemetry-native LLM observability platform.

Core: Telemetry (traces, spans, metrics, logs)
Features: Prompts, datasets, and scores APIs

Sync Usage:
    >>> from brokle import Brokle
    >>> with Brokle(api_key="bk_...") as client:
    ...     # Core: Create traces and spans
    ...     with client.start_as_current_span("my-operation") as span:
    ...         span.set_attribute("output", "Hello, world!")
    ...
    ...     # Feature: Prompt management (optional)
    ...     prompt = client.prompts.get("greeting", label="production")

Async Usage:
    >>> from brokle import AsyncBrokle
    >>> async with AsyncBrokle(api_key="bk_...") as client:
    ...     # Core: Create traces and spans
    ...     with client.start_as_current_span("process") as span:
    ...         result = await do_work()
    ...
    ...     # Feature: Prompt management (optional)
    ...     prompt = await client.prompts.get("greeting")

Singleton Pattern:
    >>> from brokle import get_client
    >>> client = get_client()  # Reads from BROKLE_* env vars
"""

from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import Optional

from ._base_client import BaseBrokleClient
from ._http import AsyncHTTPClient, SyncHTTPClient
from .annotations import AnnotationQueuesManager, AsyncAnnotationQueuesManager
from .config import BrokleConfig
from .datasets import AsyncDatasetsManager, DatasetsManager
from .experiments import AsyncExperimentsManager, ExperimentsManager
from .prompts import AsyncPromptManager, PromptManager
from .query import AsyncQueryManager, QueryManager
from .scores import AsyncScoresManager, ScoresManager


class Brokle(BaseBrokleClient):
    """
    Synchronous Brokle client for OpenTelemetry-native LLM observability.

    Core responsibility: Telemetry (traces, spans, metrics, logs)
    Feature APIs: Prompts, datasets, and scores

    This client provides synchronous methods for all operations.
    Uses SyncHTTPClient (httpx.Client) internally - no event loop involvement.

    Example:
        >>> from brokle import Brokle
        >>>
        >>> # Context manager (recommended)
        >>> with Brokle(api_key="bk_...") as client:
        ...     # Core: Telemetry - traces and spans
        ...     with client.start_as_current_span("process") as span:
        ...         result = do_work()
        ...         span.set_attribute("result", result)
        ...
        ...     # Feature: Prompt management (optional)
        ...     prompt = client.prompts.get("greeting")
        ...     messages = prompt.to_openai_messages({"name": "Alice"})
    """

    def __init__(self, *args, **kwargs):
        """Initialize sync Brokle client with SyncHTTPClient."""
        super().__init__(*args, **kwargs)
        self._http_client: Optional[SyncHTTPClient] = None
        # Auto-register as global client (first-write-wins)
        if _client_context.get() is None:
            _client_context.set(self)

    @property
    def _http(self) -> SyncHTTPClient:
        """Lazy-init sync HTTP client."""
        if self._http_client is None:
            self._http_client = SyncHTTPClient(self.config)
        return self._http_client

    @property
    def prompts(self) -> PromptManager:
        """
        Access prompt management operations.

        Returns a PromptManager for fetching and managing prompts.
        All methods are synchronous.

        Returns:
            PromptManager instance

        Example:
            >>> prompt = client.prompts.get("greeting", label="production")
            >>> compiled = prompt.compile({"name": "Alice"})
        """
        if self._prompts_manager is None:
            self._prompts_manager = PromptManager(
                http_client=self._http,
                config=self.config,
                prompt_config=self._prompt_config,
            )
        return self._prompts_manager

    @property
    def datasets(self) -> DatasetsManager:
        """
        Access dataset management operations.

        Returns a DatasetsManager for creating and managing datasets.
        All methods are synchronous.

        Returns:
            DatasetsManager instance

        Example:
            >>> dataset = client.datasets.create(name="qa-pairs")
            >>> dataset.insert([{"input": {"q": "2+2?"}, "expected": {"a": "4"}}])
            >>> for item in dataset:
            ...     print(item.input, item.expected)
        """
        if self._datasets_manager is None:
            self._datasets_manager = DatasetsManager(
                http_client=self._http,
                config=self.config,
            )
        return self._datasets_manager

    @property
    def scores(self) -> ScoresManager:
        """
        Access score submission operations.

        Returns a ScoresManager for submitting quality scores.
        All methods are synchronous.

        Returns:
            ScoresManager instance

        Example:
            >>> client.scores.submit(
            ...     trace_id="abc123",
            ...     name="quality",
            ...     value=0.9,
            ... )
        """
        if self._scores_manager is None:
            self._scores_manager = ScoresManager(
                http_client=self._http,
                config=self.config,
            )
        return self._scores_manager

    @property
    def experiments(self) -> ExperimentsManager:
        """
        Access experiment operations.

        Returns an ExperimentsManager for running evaluation experiments.
        All methods are synchronous.

        Returns:
            ExperimentsManager instance

        Example:
            >>> from brokle.scorers import ExactMatch
            >>> results = client.experiments.run(
            ...     name="gpt4-test",
            ...     dataset=dataset,
            ...     task=my_task,
            ...     scorers=[ExactMatch()],
            ... )
            >>> print(results.summary)
        """
        if self._experiments_manager is None:
            self._experiments_manager = ExperimentsManager(
                http_client=self._http,
                config=self.config,
            )
        return self._experiments_manager

    @property
    def query(self) -> QueryManager:
        """
        Access query operations for production spans (THE WEDGE).

        Returns a QueryManager for querying production spans and
        running retrospective evaluations without re-instrumenting applications.
        All methods are synchronous.

        Returns:
            QueryManager instance

        Example:
            >>> from datetime import datetime, timedelta
            >>>
            >>> # Query production spans
            >>> result = client.query.query(
            ...     filter="service.name=chatbot AND gen_ai.provider.name=openai",
            ...     start_time=datetime.now() - timedelta(days=7),
            ... )
            >>>
            >>> # Evaluate queried spans
            >>> from brokle.scorers import ExactMatch
            >>> eval_result = client.experiments.run(
            ...     name="retrospective-analysis",
            ...     spans=result.spans,
            ...     scorers=[ExactMatch()],
            ...     extract_input=lambda s: {"prompt": s.input},
            ...     extract_output=lambda s: s.output,
            ... )
        """
        if self._query_manager is None:
            self._query_manager = QueryManager(
                http_client=self._http,
                config=self.config,
            )
        return self._query_manager

    @property
    def annotations(self) -> AnnotationQueuesManager:
        """
        Access annotation queue management operations.

        Returns an AnnotationQueuesManager for adding items to annotation
        queues for human-in-the-loop (HITL) evaluation workflows.
        All methods are synchronous.

        Returns:
            AnnotationQueuesManager instance

        Example:
            >>> # Add traces to an annotation queue
            >>> result = client.annotations.add_traces(
            ...     queue_id="queue123",
            ...     trace_ids=["trace1", "trace2", "trace3"],
            ...     priority=5,
            ... )
            >>> print(f"Added {result['created']} items")
            >>>
            >>> # Add items with mixed types
            >>> client.annotations.add_items(
            ...     queue_id="queue123",
            ...     items=[
            ...         {"object_id": "trace1", "object_type": "trace"},
            ...         {"object_id": "span1", "object_type": "span", "priority": 10},
            ...     ]
            ... )
        """
        if self._annotations_manager is None:
            self._annotations_manager = AnnotationQueuesManager(
                http_client=self._http,
                config=self.config,
            )
        return self._annotations_manager

    def auth_check(self) -> bool:
        """
        Verify connection to Brokle server.

        Makes a synchronous request to validate API key.
        Use for development/testing only - adds latency.

        Returns:
            True if authenticated, False otherwise

        Example:
            >>> if client.auth_check():
            ...     print("Connected!")
        """
        try:
            # _http.post raises typed exceptions on 4xx/5xx (bad key →
            # AuthenticationError, network failure → ConnectionError,
            # etc.). Reaching here means the backend validated the key.
            self._http.post("/v1/auth/validate-key", json={})
            return True
        except Exception:
            return False

    def shutdown(self, timeout_seconds: int = 30) -> bool:
        """Shutdown with manager cleanup."""
        success = super().shutdown(timeout_seconds)

        if self._http_client:
            self._http_client.close()
        if self._prompts_manager:
            self._prompts_manager._shutdown()

        return success

    def close(self):
        """Close the client (alias for shutdown)."""
        self.shutdown()
        # Clear global registration if this is the registered client
        if _client_context.get() is self:
            _client_context.set(None)

    def __enter__(self) -> "Brokle":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class AsyncBrokle(BaseBrokleClient):
    """
    Asynchronous Brokle client for OpenTelemetry-native LLM observability.

    Core responsibility: Telemetry (traces, spans, metrics, logs)
    Feature APIs: Prompts, datasets, and scores

    This client provides async methods for all operations.
    Uses AsyncHTTPClient (httpx.AsyncClient) internally.

    Example:
        >>> from brokle import AsyncBrokle
        >>>
        >>> # Context manager (recommended)
        >>> async with AsyncBrokle(api_key="bk_...") as client:
        ...     # Core: Telemetry - traces and spans
        ...     with client.start_as_current_span("process") as span:
        ...         result = await do_work()
        ...         span.set_attribute("result", result)
        ...
        ...     # Feature: Prompt management (optional)
        ...     prompt = await client.prompts.get("greeting")
        ...     messages = prompt.to_openai_messages({"name": "Alice"})
    """

    def __init__(self, *args, **kwargs):
        """Initialize async Brokle client with AsyncHTTPClient."""
        super().__init__(*args, **kwargs)
        self._http_client: Optional[AsyncHTTPClient] = None
        # Auto-register as global client (first-write-wins)
        if _async_client_context.get() is None:
            _async_client_context.set(self)

    @property
    def _http(self) -> AsyncHTTPClient:
        """Lazy-init async HTTP client."""
        if self._http_client is None:
            self._http_client = AsyncHTTPClient(self.config)
        return self._http_client

    @property
    def prompts(self) -> AsyncPromptManager:
        """
        Access prompt management operations.

        Returns an AsyncPromptManager for fetching and managing prompts.
        All methods are async and must be awaited.

        Returns:
            AsyncPromptManager instance

        Example:
            >>> prompt = await client.prompts.get("greeting", label="production")
            >>> compiled = prompt.compile({"name": "Alice"})
        """
        if self._prompts_manager is None:
            self._prompts_manager = AsyncPromptManager(
                http_client=self._http,
                config=self.config,
                prompt_config=self._prompt_config,
            )
        return self._prompts_manager

    @property
    def datasets(self) -> AsyncDatasetsManager:
        """
        Access dataset management operations.

        Returns an AsyncDatasetsManager for creating and managing datasets.
        All methods are async and must be awaited.

        Returns:
            AsyncDatasetsManager instance

        Example:
            >>> dataset = await client.datasets.create(name="qa-pairs")
            >>> await dataset.insert([{"input": {"q": "2+2?"}, "expected": {"a": "4"}}])
            >>> async for item in dataset:
            ...     print(item.input, item.expected)
        """
        if self._datasets_manager is None:
            self._datasets_manager = AsyncDatasetsManager(
                http_client=self._http,
                config=self.config,
            )
        return self._datasets_manager

    @property
    def scores(self) -> AsyncScoresManager:
        """
        Access score submission operations.

        Returns an AsyncScoresManager for submitting quality scores.
        All methods are async and must be awaited.

        Returns:
            AsyncScoresManager instance

        Example:
            >>> await client.scores.submit(
            ...     trace_id="abc123",
            ...     name="quality",
            ...     value=0.9,
            ... )
        """
        if self._scores_manager is None:
            self._scores_manager = AsyncScoresManager(
                http_client=self._http,
                config=self.config,
            )
        return self._scores_manager

    @property
    def experiments(self) -> AsyncExperimentsManager:
        """
        Access experiment operations.

        Returns an AsyncExperimentsManager for running evaluation experiments.
        All methods are async and must be awaited.

        Returns:
            AsyncExperimentsManager instance

        Example:
            >>> from brokle.scorers import ExactMatch
            >>> results = await client.experiments.run(
            ...     name="gpt4-test",
            ...     dataset=dataset,
            ...     task=my_task,
            ...     scorers=[ExactMatch()],
            ... )
            >>> print(results.summary)
        """
        if self._experiments_manager is None:
            self._experiments_manager = AsyncExperimentsManager(
                http_client=self._http,
                config=self.config,
            )
        return self._experiments_manager

    @property
    def query(self) -> AsyncQueryManager:
        """
        Access query operations for production spans (THE WEDGE).

        Returns an AsyncQueryManager for querying production spans and
        running retrospective evaluations without re-instrumenting applications.
        All methods are async and must be awaited.

        Returns:
            AsyncQueryManager instance

        Example:
            >>> from datetime import datetime, timedelta
            >>>
            >>> # Query production spans
            >>> result = await client.query.query(
            ...     filter="service.name=chatbot AND gen_ai.provider.name=openai",
            ...     start_time=datetime.now() - timedelta(days=7),
            ... )
            >>>
            >>> # Evaluate queried spans
            >>> from brokle.scorers import ExactMatch
            >>> eval_result = await client.experiments.run(
            ...     name="retrospective-analysis",
            ...     spans=result.spans,
            ...     scorers=[ExactMatch()],
            ...     extract_input=lambda s: {"prompt": s.input},
            ...     extract_output=lambda s: s.output,
            ... )
        """
        if self._query_manager is None:
            self._query_manager = AsyncQueryManager(
                http_client=self._http,
                config=self.config,
            )
        return self._query_manager

    @property
    def annotations(self) -> AsyncAnnotationQueuesManager:
        """
        Access annotation queue management operations.

        Returns an AsyncAnnotationQueuesManager for adding items to annotation
        queues for human-in-the-loop (HITL) evaluation workflows.
        All methods are async and must be awaited.

        Returns:
            AsyncAnnotationQueuesManager instance

        Example:
            >>> # Add traces to an annotation queue
            >>> result = await client.annotations.add_traces(
            ...     queue_id="queue123",
            ...     trace_ids=["trace1", "trace2", "trace3"],
            ...     priority=5,
            ... )
            >>> print(f"Added {result['created']} items")
        """
        if self._annotations_manager is None:
            self._annotations_manager = AsyncAnnotationQueuesManager(
                http_client=self._http,
                config=self.config,
            )
        return self._annotations_manager

    async def auth_check(self) -> bool:
        """
        Verify connection to Brokle server.

        Makes an async request to validate API key.
        Use for development/testing only - adds latency.

        Returns:
            True if authenticated, False otherwise

        Example:
            >>> if await client.auth_check():
            ...     print("Connected!")
        """
        try:
            # _http.post raises typed exceptions on 4xx/5xx (bad key →
            # AuthenticationError, network failure → ConnectionError,
            # etc.). Reaching here means the backend validated the key —
            # matches the sync auth_check contract above. The raw body
            # has no `success` flag under the Stripe/OpenAI-style wire
            # contract; HTTP status is the signal.
            await self._http.post("/v1/auth/validate-key", json={})
            return True
        except Exception:
            return False

    async def shutdown(self, timeout_seconds: int = 30) -> bool:
        """Shutdown with manager cleanup."""
        success = super().shutdown(timeout_seconds)

        if self._http_client:
            await self._http_client.close()
        if self._prompts_manager:
            await self._prompts_manager._shutdown()

        return success

    async def close(self):
        """Close the client (alias for shutdown)."""
        await self.shutdown()
        # Clear global registration if this is the registered client
        if _async_client_context.get() is self:
            _async_client_context.set(None)

    async def __aenter__(self) -> "AsyncBrokle":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# ContextVar for async-safe multi-project support
_client_context: ContextVar[Optional[Brokle]] = ContextVar(
    "brokle_client", default=None
)
_async_client_context: ContextVar[Optional[AsyncBrokle]] = ContextVar(
    "brokle_async_client", default=None
)


def get_client(**overrides) -> Brokle:
    """
    Get or create Brokle client from context.

    Uses ContextVar for async-safe multi-project support. Configuration is read
    from environment variables on first call. Subsequent calls in the same
    context return the same instance.

    Args:
        **overrides: Override specific configuration values

    Returns:
        Brokle instance from current context

    Raises:
        ValueError: If BROKLE_API_KEY environment variable is missing

    Example:
        >>> from brokle import get_client
        >>> client = get_client()
        >>> prompt = client.prompts.get("greeting")
    """
    client = _client_context.get()

    if client is None:
        config = BrokleConfig.from_env(**overrides)
        client = Brokle(config=config)
        _client_context.set(client)

    return client


def set_client(client: Brokle) -> None:
    """
    Set client in current context.

    Useful for multi-project scenarios where different requests need
    different Brokle clients (e.g., multi-tenant applications).

    Args:
        client: Brokle client instance to set

    Example:
        >>> import contextvars
        >>> from brokle import Brokle, set_client
        >>>
        >>> project_a_client = Brokle(api_key="bk_project_a_key")
        >>> project_b_client = Brokle(api_key="bk_project_b_key")
        >>>
        >>> # Use different clients in different contexts
        >>> ctx = contextvars.copy_context()
        >>> ctx.run(set_client, project_a_client)
    """
    _client_context.set(client)


def reset_client() -> None:
    """
    Reset client in current context.

    Closes the current client if it exists and removes it from context.
    Useful for testing and cleanup.
    """
    client = _client_context.get()
    if client:
        client.close()
    _client_context.set(None)


async def get_async_client(**overrides) -> AsyncBrokle:
    """
    Get or create AsyncBrokle client from context.

    Uses ContextVar for async-safe multi-project support. Configuration is read
    from environment variables on first call. Subsequent calls in the same
    context return the same instance.

    Args:
        **overrides: Override specific configuration values

    Returns:
        AsyncBrokle instance from current context

    Raises:
        ValueError: If BROKLE_API_KEY environment variable is missing

    Example:
        >>> from brokle import get_async_client
        >>> client = await get_async_client()
        >>> prompt = await client.prompts.get("greeting")
    """
    client = _async_client_context.get()

    if client is None:
        config = BrokleConfig.from_env(**overrides)
        client = AsyncBrokle(config=config)
        _async_client_context.set(client)

    return client


def set_async_client(client: AsyncBrokle) -> None:
    """
    Set async client in current context.

    Useful for multi-project scenarios where different requests need
    different AsyncBrokle clients (e.g., multi-tenant applications).

    Args:
        client: AsyncBrokle client instance to set
    """
    _async_client_context.set(client)


async def reset_async_client() -> None:
    """
    Reset async client in current context.

    Closes the current client if it exists and removes it from context.
    Useful for testing and cleanup.
    """
    client = _async_client_context.get()
    if client:
        await client.close()
    _async_client_context.set(None)


@contextmanager
def brokle_context(client: Brokle):
    """
    Temporarily override the global Brokle client for a block.

    Useful for multi-tenant or per-request client overrides.
    Restores the previous client when the block exits (even on exception).

    Args:
        client: Brokle client to use within the block

    Yields:
        The provided client

    Example:
        >>> tenant_client = Brokle(api_key="bk_tenant_key")
        >>> with brokle_context(tenant_client) as c:
        ...     # All get_client() calls within this block return tenant_client
        ...     wrapped = wrap_openai(openai.OpenAI())
        ...     response = wrapped.chat.completions.create(...)
    """
    token = _client_context.set(client)
    try:
        yield client
    finally:
        _client_context.reset(token)


@asynccontextmanager
async def async_brokle_context(client: AsyncBrokle):
    """
    Temporarily override the global AsyncBrokle client for a block.

    Args:
        client: AsyncBrokle client to use within the block

    Yields:
        The provided client
    """
    token = _async_client_context.set(client)
    try:
        yield client
    finally:
        _async_client_context.reset(token)
