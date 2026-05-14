"""
Annotation Queues Manager

Provides both synchronous and asynchronous management for Brokle annotation queues.

Annotation queues enable human-in-the-loop (HITL) evaluation workflows where
human annotators review and score AI outputs.

Sync Usage:
    >>> from brokle import Brokle
    >>>
    >>> client = Brokle(api_key="bk_...")
    >>>
    >>> # Add items to a queue
    >>> result = client.annotations.add_items(
    ...     queue_id="queue123",
    ...     items=[
    ...         {"object_id": "trace1", "object_type": "trace"},
    ...         {"object_id": "span1", "object_type": "span", "priority": 10},
    ...     ]
    ... )
    >>> print(f"Added {result['created']} items")

Async Usage:
    >>> async with AsyncBrokle(api_key="bk_...") as client:
    ...     result = await client.annotations.add_items(
    ...         queue_id="queue123",
    ...         items=[{"object_id": "trace1", "object_type": "trace"}]
    ...     )
"""

from typing import Any, Dict, List, Optional

from .._http import AsyncHTTPClient, SyncHTTPClient
from ..config import BrokleConfig
from .exceptions import (
    AnnotationError,
    ItemLockedError,
    ItemNotFoundError,
    NoItemsAvailableError,
    QueueNotFoundError,
)
from .types import (
    AddItemRequest,
    AnnotationQueue,
    ItemStatus,
    ObjectType,
    QueueItem,
    ScoreSubmission,
)


class _BaseAnnotationManagerMixin:
    """
    Shared functionality for both sync and async annotation managers.
    """

    _config: BrokleConfig

    def _log(self, message: str, *args: Any) -> None:
        """Log debug messages."""
        if self._config.debug:
            print(f"[Brokle Annotations] {message}", *args)

    def _handle_error(self, e: Exception, operation: str) -> None:
        """Handle and transform errors from API responses."""
        error_str = str(e).lower()

        if "not found" in error_str:
            if "queue" in error_str:
                raise QueueNotFoundError(f"Queue not found: {e}")
            elif "item" in error_str:
                raise ItemNotFoundError(f"Item not found: {e}")
        elif "locked" in error_str or "forbidden" in error_str:
            raise ItemLockedError(f"Item is locked by another user: {e}")
        elif "no items available" in error_str or "no pending items" in error_str:
            raise NoItemsAvailableError(f"No items available for annotation: {e}")

        raise AnnotationError(f"Failed to {operation}: {e}")


class AnnotationQueuesManager(_BaseAnnotationManagerMixin):
    """
    Sync annotation queues manager for Brokle.

    All methods are synchronous. Uses SyncHTTPClient (httpx.Client) internally.

    Example:
        >>> from brokle import Brokle
        >>>
        >>> client = Brokle(api_key="bk_...")
        >>>
        >>> # Add items to queue
        >>> result = client.annotations.add_items(
        ...     queue_id="queue123",
        ...     items=[{"object_id": "trace1", "object_type": "trace"}]
        ... )
    """

    def __init__(
        self,
        http_client: SyncHTTPClient,
        config: BrokleConfig,
    ):
        """
        Initialize sync annotation queues manager.

        Args:
            http_client: Sync HTTP client
            config: Brokle configuration
        """
        self._http = http_client
        self._config = config

    def add_items(
        self,
        queue_id: str,
        items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Add items to an annotation queue.

        Supports adding traces or spans for human annotation.

        Args:
            queue_id: ID of the annotation queue
            items: List of items to add, each containing:
                - object_id: ID of the trace or span
                - object_type: "trace" or "span"
                - priority: Optional priority (higher = processed first)
                - metadata: Optional metadata dict

        Returns:
            Dict with 'created' count of items added

        Raises:
            QueueNotFoundError: If the queue doesn't exist
            AnnotationError: If the request fails

        Example:
            >>> client.annotations.add_items(
            ...     queue_id="queue123",
            ...     items=[
            ...         {"object_id": "trace1", "object_type": "trace"},
            ...         {"object_id": "span1", "object_type": "span", "priority": 10},
            ...     ]
            ... )
            {'created': 2}
        """
        self._log(f"Adding {len(items)} items to queue {queue_id}")

        # Normalize items
        normalized_items = []
        for item in items:
            object_type = item.get("object_type", "trace")
            normalized = {
                "object_id": item["object_id"],
                "object_type": object_type,
            }
            if "priority" in item:
                normalized["priority"] = item["priority"]
            if "metadata" in item:
                normalized["metadata"] = item["metadata"]
            normalized_items.append(normalized)

        payload = {"items": normalized_items}

        try:
            raw_response = self._http.post(
                f"/v1/annotation-queues/{queue_id}/items",
                json=payload,
            )
            return raw_response
        except Exception as e:
            self._handle_error(e, "add items to queue")

    def list_items(
        self,
        queue_id: str,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        List items in an annotation queue.

        Args:
            queue_id: ID of the annotation queue
            status: Optional filter by status ("pending", "completed", "skipped")
            limit: Maximum number of items to return (default: 50)
            offset: Number of items to skip for pagination (default: 0)

        Returns:
            Dict with 'items' list and 'total' count

        Raises:
            QueueNotFoundError: If the queue doesn't exist
            AnnotationError: If the request fails

        Example:
            >>> result = client.annotations.list_items(
            ...     queue_id="queue123",
            ...     status="pending",
            ...     limit=20,
            ... )
            >>> for item in result['items']:
            ...     print(f"{item['object_id']}: {item['status']}")
        """
        self._log(f"Listing items for queue {queue_id}")

        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        try:
            raw_response = self._http.get(
                f"/v1/annotation-queues/{queue_id}/items",
                params=params,
            )
            return raw_response
        except Exception as e:
            self._handle_error(e, "list items in queue")

    def add_traces(
        self,
        queue_id: str,
        trace_ids: List[str],
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Convenience method to add traces to an annotation queue.

        Args:
            queue_id: ID of the annotation queue
            trace_ids: List of trace IDs to add
            priority: Priority for all items (default: 0)
            metadata: Optional metadata for all items

        Returns:
            Dict with 'created' count of items added

        Example:
            >>> client.annotations.add_traces(
            ...     queue_id="queue123",
            ...     trace_ids=["trace1", "trace2", "trace3"],
            ...     priority=5,
            ... )
            {'created': 3}
        """
        items = [
            {
                "object_id": trace_id,
                "object_type": "trace",
                "priority": priority,
                **({"metadata": metadata} if metadata else {}),
            }
            for trace_id in trace_ids
        ]
        return self.add_items(queue_id=queue_id, items=items)

    def add_spans(
        self,
        queue_id: str,
        span_ids: List[str],
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Convenience method to add spans to an annotation queue.

        Args:
            queue_id: ID of the annotation queue
            span_ids: List of span IDs to add
            priority: Priority for all items (default: 0)
            metadata: Optional metadata for all items

        Returns:
            Dict with 'created' count of items added

        Example:
            >>> client.annotations.add_spans(
            ...     queue_id="queue123",
            ...     span_ids=["span1", "span2"],
            ... )
            {'created': 2}
        """
        items = [
            {
                "object_id": span_id,
                "object_type": "span",
                "priority": priority,
                **({"metadata": metadata} if metadata else {}),
            }
            for span_id in span_ids
        ]
        return self.add_items(queue_id=queue_id, items=items)


class AsyncAnnotationQueuesManager(_BaseAnnotationManagerMixin):
    """
    Async annotation queues manager for AsyncBrokle.

    All methods are async and return coroutines that must be awaited.
    Uses AsyncHTTPClient (httpx.AsyncClient) internally.

    Example:
        >>> async with AsyncBrokle(api_key="bk_...") as client:
        ...     result = await client.annotations.add_items(
        ...         queue_id="queue123",
        ...         items=[{"object_id": "trace1", "object_type": "trace"}]
        ...     )
    """

    def __init__(
        self,
        http_client: AsyncHTTPClient,
        config: BrokleConfig,
    ):
        """
        Initialize async annotation queues manager.

        Args:
            http_client: Async HTTP client
            config: Brokle configuration
        """
        self._http = http_client
        self._config = config

    async def add_items(
        self,
        queue_id: str,
        items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Add items to an annotation queue (async).

        Supports adding traces or spans for human annotation.

        Args:
            queue_id: ID of the annotation queue
            items: List of items to add, each containing:
                - object_id: ID of the trace or span
                - object_type: "trace" or "span"
                - priority: Optional priority (higher = processed first)
                - metadata: Optional metadata dict

        Returns:
            Dict with 'created' count of items added

        Raises:
            QueueNotFoundError: If the queue doesn't exist
            AnnotationError: If the request fails
        """
        self._log(f"Adding {len(items)} items to queue {queue_id}")

        # Normalize items
        normalized_items = []
        for item in items:
            object_type = item.get("object_type", "trace")
            normalized = {
                "object_id": item["object_id"],
                "object_type": object_type,
            }
            if "priority" in item:
                normalized["priority"] = item["priority"]
            if "metadata" in item:
                normalized["metadata"] = item["metadata"]
            normalized_items.append(normalized)

        payload = {"items": normalized_items}

        try:
            raw_response = await self._http.post(
                f"/v1/annotation-queues/{queue_id}/items",
                json=payload,
            )
            return raw_response
        except Exception as e:
            self._handle_error(e, "add items to queue")

    async def list_items(
        self,
        queue_id: str,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        List items in an annotation queue (async).

        Args:
            queue_id: ID of the annotation queue
            status: Optional filter by status ("pending", "completed", "skipped")
            limit: Maximum number of items to return (default: 50)
            offset: Number of items to skip for pagination (default: 0)

        Returns:
            Dict with 'items' list and 'total' count

        Raises:
            QueueNotFoundError: If the queue doesn't exist
            AnnotationError: If the request fails
        """
        self._log(f"Listing items for queue {queue_id}")

        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        try:
            raw_response = await self._http.get(
                f"/v1/annotation-queues/{queue_id}/items",
                params=params,
            )
            return raw_response
        except Exception as e:
            self._handle_error(e, "list items in queue")

    async def add_traces(
        self,
        queue_id: str,
        trace_ids: List[str],
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Convenience method to add traces to an annotation queue (async).

        Args:
            queue_id: ID of the annotation queue
            trace_ids: List of trace IDs to add
            priority: Priority for all items (default: 0)
            metadata: Optional metadata for all items

        Returns:
            Dict with 'created' count of items added
        """
        items = [
            {
                "object_id": trace_id,
                "object_type": "trace",
                "priority": priority,
                **({"metadata": metadata} if metadata else {}),
            }
            for trace_id in trace_ids
        ]
        return await self.add_items(queue_id=queue_id, items=items)

    async def add_spans(
        self,
        queue_id: str,
        span_ids: List[str],
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Convenience method to add spans to an annotation queue (async).

        Args:
            queue_id: ID of the annotation queue
            span_ids: List of span IDs to add
            priority: Priority for all items (default: 0)
            metadata: Optional metadata for all items

        Returns:
            Dict with 'created' count of items added
        """
        items = [
            {
                "object_id": span_id,
                "object_type": "span",
                "priority": priority,
                **({"metadata": metadata} if metadata else {}),
            }
            for span_id in span_ids
        ]
        return await self.add_items(queue_id=queue_id, items=items)
