"""
Datasets Manager

Provides both synchronous and asynchronous dataset management for Brokle.

Sync Usage:
    >>> from brokle import Brokle
    >>>
    >>> client = Brokle(api_key="bk_...")
    >>>
    >>> # Create dataset
    >>> dataset = client.datasets.create(
    ...     name="qa-pairs",
    ...     description="Question-answer test cases"
    ... )
    >>>
    >>> # Get existing
    >>> dataset = client.datasets.get("01HXYZ...")
    >>>
    >>> # List all
    >>> datasets = client.datasets.list()

Async Usage:
    >>> async with AsyncBrokle(api_key="bk_...") as client:
    ...     dataset = await client.datasets.create(name="test")
    ...     datasets = await client.datasets.list()
"""

from typing import Any, Dict, List, Optional

from .._http import AsyncHTTPClient, SyncHTTPClient
from ..config import BrokleConfig
from .dataset import AsyncDataset, Dataset
from .exceptions import DatasetError


class _BaseDatasetsManagerMixin:
    """
    Shared functionality for both sync and async datasets managers.

    Contains utility methods that don't depend on HTTP client type.
    """

    _config: BrokleConfig

    def _log(self, message: str, *args: Any) -> None:
        """Log debug messages."""
        if self._config.debug:
            print(f"[Brokle Datasets] {message}", *args)


class DatasetsManager(_BaseDatasetsManagerMixin):
    """
    Sync datasets manager for Brokle.

    All methods are synchronous. Uses SyncHTTPClient (httpx.Client) internally -
    no event loop involvement.

    Example:
        >>> from brokle import Brokle
        >>>
        >>> client = Brokle(api_key="bk_...")
        >>>
        >>> # Create dataset
        >>> dataset = client.datasets.create(name="qa-pairs")
        >>>
        >>> # Insert items
        >>> dataset.insert([
        ...     {"input": {"q": "2+2?"}, "expected": {"a": "4"}},
        ... ])
        >>>
        >>> # Iterate
        >>> for item in dataset:
        ...     print(item.input, item.expected)
    """

    def __init__(
        self,
        http_client: SyncHTTPClient,
        config: BrokleConfig,
    ):
        """
        Initialize sync datasets manager.

        Args:
            http_client: Sync HTTP client
            config: Brokle configuration
        """
        self._http = http_client
        self._config = config

    def create(
        self,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dataset:
        """
        Create a new dataset.

        Args:
            name: Dataset name
            description: Optional description
            metadata: Optional additional metadata

        Returns:
            Dataset object for managing items

        Raises:
            DatasetError: If the API request fails

        Example:
            >>> dataset = client.datasets.create(
            ...     name="qa-pairs",
            ...     description="Question-answer test cases"
            ... )
            >>> dataset.insert([
            ...     {"input": {"q": "2+2?"}, "expected": {"a": "4"}},
            ... ])
        """
        self._log(f"Creating dataset: {name}")

        payload: Dict[str, Any] = {"name": name}
        if description:
            payload["description"] = description
        if metadata:
            payload["metadata"] = metadata

        try:
            raw_response = self._http.post("/v1/datasets", json=payload)
            data = raw_response
            return Dataset(
                id=data["id"],
                name=data["name"],
                description=data.get("description"),
                metadata=data.get("metadata"),
                created_at=data["created_at"],
                updated_at=data["updated_at"],
                _http_client=self._http,
                _debug=self._config.debug,
            )
        except ValueError as e:
            raise DatasetError(f"Failed to create dataset: {e}")
        except Exception as e:
            raise DatasetError(f"Failed to create dataset: {e}")

    def get(self, dataset_id: str) -> Dataset:
        """
        Get an existing dataset by ID.

        Args:
            dataset_id: Dataset ID

        Returns:
            Dataset object for managing items

        Raises:
            DatasetError: If the API request fails or dataset not found

        Example:
            >>> dataset = client.datasets.get("01HXYZ...")
            >>> for item in dataset:
            ...     print(item.input, item.expected)
        """
        self._log(f"Getting dataset: {dataset_id}")

        try:
            raw_response = self._http.get(f"/v1/datasets/{dataset_id}")
            data = raw_response
            return Dataset(
                id=data["id"],
                name=data["name"],
                description=data.get("description"),
                metadata=data.get("metadata"),
                created_at=data["created_at"],
                updated_at=data["updated_at"],
                _http_client=self._http,
                _debug=self._config.debug,
            )
        except ValueError as e:
            raise DatasetError(f"Failed to get dataset: {e}")
        except Exception as e:
            raise DatasetError(f"Failed to get dataset: {e}")

    def list(
        self,
        limit: int = 50,
        page: int = 1,
    ) -> List[Dataset]:
        """
        List all datasets.

        Args:
            limit: Maximum number of datasets to return (default: 50, valid: 10, 25, 50, 100)
            page: Page number to fetch (default: 1, 1-indexed)

        Returns:
            List of Dataset objects

        Raises:
            DatasetError: If the API request fails

        Example:
            >>> datasets = client.datasets.list()
            >>> for ds in datasets:
            ...     print(ds.name, ds.id)
        """
        self._log(f"Listing datasets: limit={limit}, page={page}")

        try:
            raw_response = self._http.get(
                "/v1/datasets",
                params={"limit": limit, "page": page},
            )
            data = raw_response["data"]
            return [
                Dataset(
                    id=ds["id"],
                    name=ds["name"],
                    description=ds.get("description"),
                    metadata=ds.get("metadata"),
                    created_at=ds["created_at"],
                    updated_at=ds["updated_at"],
                    _http_client=self._http,
                    _debug=self._config.debug,
                )
                for ds in data
            ]
        except ValueError as e:
            raise DatasetError(f"Failed to list datasets: {e}")
        except Exception as e:
            raise DatasetError(f"Failed to list datasets: {e}")

    def update(
        self,
        dataset_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dataset:
        """
        Update an existing dataset.

        Args:
            dataset_id: Dataset ID to update
            name: New name (optional)
            description: New description (optional)
            metadata: New metadata (optional)

        Returns:
            Updated Dataset object

        Raises:
            DatasetError: If the API request fails or dataset not found

        Example:
            >>> dataset = client.datasets.update(
            ...     "01HXYZ...",
            ...     name="updated-name",
            ...     description="New description"
            ... )
        """
        self._log(f"Updating dataset: {dataset_id}")

        payload: Dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if metadata is not None:
            payload["metadata"] = metadata

        if not payload:
            raise DatasetError("At least one field must be provided to update")

        try:
            raw_response = self._http.patch(f"/v1/datasets/{dataset_id}", json=payload)
            data = raw_response
            return Dataset(
                id=data["id"],
                name=data["name"],
                description=data.get("description"),
                metadata=data.get("metadata"),
                created_at=data["created_at"],
                updated_at=data["updated_at"],
                _http_client=self._http,
                _debug=self._config.debug,
            )
        except ValueError as e:
            raise DatasetError(f"Failed to update dataset: {e}")
        except Exception as e:
            raise DatasetError(f"Failed to update dataset: {e}")

    def delete(self, dataset_id: str) -> None:
        """
        Delete a dataset by ID.

        Args:
            dataset_id: Dataset ID to delete

        Raises:
            DatasetError: If the API request fails or dataset not found

        Example:
            >>> client.datasets.delete("01HXYZ...")
        """
        self._log(f"Deleting dataset: {dataset_id}")

        try:
            # _http.delete raises typed exceptions on 4xx/5xx. A
            # successful return (None for 204, body for 200) means
            # the dataset was deleted.
            self._http.delete(f"/v1/datasets/{dataset_id}")
        except DatasetError:
            raise
        except Exception as e:
            raise DatasetError(f"Failed to delete dataset: {e}")


class AsyncDatasetsManager(_BaseDatasetsManagerMixin):
    """
    Async datasets manager for AsyncBrokle.

    All methods are async and return coroutines that must be awaited.
    Uses AsyncHTTPClient (httpx.AsyncClient) internally.

    Example:
        >>> async with AsyncBrokle(api_key="bk_...") as client:
        ...     dataset = await client.datasets.create(name="test")
        ...     await dataset.insert([{"input": {"q": "test"}}])
        ...     async for item in dataset:
        ...         print(item.input)
    """

    def __init__(
        self,
        http_client: AsyncHTTPClient,
        config: BrokleConfig,
    ):
        """
        Initialize async datasets manager.

        Args:
            http_client: Async HTTP client
            config: Brokle configuration
        """
        self._http = http_client
        self._config = config

    async def create(
        self,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AsyncDataset:
        """
        Create a new dataset (async).

        Args:
            name: Dataset name
            description: Optional description
            metadata: Optional additional metadata

        Returns:
            AsyncDataset object for managing items

        Raises:
            DatasetError: If the API request fails

        Example:
            >>> dataset = await client.datasets.create(
            ...     name="qa-pairs",
            ...     description="Question-answer test cases"
            ... )
            >>> await dataset.insert([
            ...     {"input": {"q": "2+2?"}, "expected": {"a": "4"}},
            ... ])
        """
        self._log(f"Creating dataset: {name}")

        payload: Dict[str, Any] = {"name": name}
        if description:
            payload["description"] = description
        if metadata:
            payload["metadata"] = metadata

        try:
            raw_response = await self._http.post("/v1/datasets", json=payload)
            data = raw_response
            return AsyncDataset(
                id=data["id"],
                name=data["name"],
                description=data.get("description"),
                metadata=data.get("metadata"),
                created_at=data["created_at"],
                updated_at=data["updated_at"],
                _http_client=self._http,
                _debug=self._config.debug,
            )
        except ValueError as e:
            raise DatasetError(f"Failed to create dataset: {e}")
        except Exception as e:
            raise DatasetError(f"Failed to create dataset: {e}")

    async def get(self, dataset_id: str) -> AsyncDataset:
        """
        Get an existing dataset by ID (async).

        Args:
            dataset_id: Dataset ID

        Returns:
            AsyncDataset object for managing items

        Raises:
            DatasetError: If the API request fails or dataset not found

        Example:
            >>> dataset = await client.datasets.get("01HXYZ...")
            >>> async for item in dataset:
            ...     print(item.input, item.expected)
        """
        self._log(f"Getting dataset: {dataset_id}")

        try:
            raw_response = await self._http.get(f"/v1/datasets/{dataset_id}")
            data = raw_response
            return AsyncDataset(
                id=data["id"],
                name=data["name"],
                description=data.get("description"),
                metadata=data.get("metadata"),
                created_at=data["created_at"],
                updated_at=data["updated_at"],
                _http_client=self._http,
                _debug=self._config.debug,
            )
        except ValueError as e:
            raise DatasetError(f"Failed to get dataset: {e}")
        except Exception as e:
            raise DatasetError(f"Failed to get dataset: {e}")

    async def list(
        self,
        limit: int = 50,
        page: int = 1,
    ) -> List[AsyncDataset]:
        """
        List all datasets (async).

        Args:
            limit: Maximum number of datasets to return (default: 50, valid: 10, 25, 50, 100)
            page: Page number to fetch (default: 1, 1-indexed)

        Returns:
            List of AsyncDataset objects

        Raises:
            DatasetError: If the API request fails

        Example:
            >>> datasets = await client.datasets.list()
            >>> for ds in datasets:
            ...     print(ds.name, ds.id)
        """
        self._log(f"Listing datasets: limit={limit}, page={page}")

        try:
            raw_response = await self._http.get(
                "/v1/datasets",
                params={"limit": limit, "page": page},
            )
            data = raw_response["data"]
            return [
                AsyncDataset(
                    id=ds["id"],
                    name=ds["name"],
                    description=ds.get("description"),
                    metadata=ds.get("metadata"),
                    created_at=ds["created_at"],
                    updated_at=ds["updated_at"],
                    _http_client=self._http,
                    _debug=self._config.debug,
                )
                for ds in data
            ]
        except ValueError as e:
            raise DatasetError(f"Failed to list datasets: {e}")
        except Exception as e:
            raise DatasetError(f"Failed to list datasets: {e}")

    async def update(
        self,
        dataset_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AsyncDataset:
        """
        Update an existing dataset (async).

        Args:
            dataset_id: Dataset ID to update
            name: New name (optional)
            description: New description (optional)
            metadata: New metadata (optional)

        Returns:
            Updated AsyncDataset object

        Raises:
            DatasetError: If the API request fails or dataset not found

        Example:
            >>> dataset = await client.datasets.update(
            ...     "01HXYZ...",
            ...     name="updated-name",
            ...     description="New description"
            ... )
        """
        self._log(f"Updating dataset: {dataset_id}")

        payload: Dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if metadata is not None:
            payload["metadata"] = metadata

        if not payload:
            raise DatasetError("At least one field must be provided to update")

        try:
            raw_response = await self._http.patch(
                f"/v1/datasets/{dataset_id}", json=payload
            )
            data = raw_response
            return AsyncDataset(
                id=data["id"],
                name=data["name"],
                description=data.get("description"),
                metadata=data.get("metadata"),
                created_at=data["created_at"],
                updated_at=data["updated_at"],
                _http_client=self._http,
                _debug=self._config.debug,
            )
        except ValueError as e:
            raise DatasetError(f"Failed to update dataset: {e}")
        except Exception as e:
            raise DatasetError(f"Failed to update dataset: {e}")

    async def delete(self, dataset_id: str) -> None:
        """
        Delete a dataset by ID (async).

        Args:
            dataset_id: Dataset ID to delete

        Raises:
            DatasetError: If the API request fails or dataset not found

        Example:
            >>> await client.datasets.delete("01HXYZ...")
        """
        self._log(f"Deleting dataset: {dataset_id}")

        try:
            # _http.delete raises typed exceptions on 4xx/5xx. A
            # successful return (None for 204, body for 200) means
            # the dataset was deleted.
            await self._http.delete(f"/v1/datasets/{dataset_id}")
        except DatasetError:
            raise
        except Exception as e:
            raise DatasetError(f"Failed to delete dataset: {e}")
