"""
Dataset Management

Provides Dataset and AsyncDataset classes for managing evaluation datasets.
Datasets are collections of input/expected pairs used for systematic evaluation.

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
    >>> # Insert items
    >>> dataset.insert([
    ...     {"input": {"question": "What is 2+2?"}, "expected": {"answer": "4"}},
    ... ])
    >>>
    >>> # Iterate with auto-pagination
    >>> for item in dataset:
    ...     print(item.input, item.expected)

Async Usage:
    >>> async with AsyncBrokle(api_key="bk_...") as client:
    ...     dataset = await client.datasets.create(name="test")
    ...     await dataset.insert([{"input": {"q": "test"}}])
    ...     async for item in dataset:
    ...         print(item.input)
"""

from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from .._http import AsyncHTTPClient, SyncHTTPClient
from .exceptions import DatasetError


@dataclass
class DatasetItem:
    """
    A single item in a dataset.

    Attributes:
        id: Unique identifier for the item
        dataset_id: ID of the parent dataset
        input: Input data for evaluation (arbitrary dict)
        expected: Expected output for comparison (optional)
        metadata: Additional metadata (optional)
        source: Item source (manual, trace, span, csv, json, sdk)
        source_trace_id: Source trace ID if created from trace
        source_span_id: Source span ID if created from span
        created_at: ISO timestamp when created
    """

    id: str
    dataset_id: str
    input: Dict[str, Any]
    expected: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    source: str = "manual"
    source_trace_id: Optional[str] = None
    source_span_id: Optional[str] = None
    created_at: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetItem":
        """Create DatasetItem from API response dict."""
        return cls(
            id=data["id"],
            dataset_id=data["dataset_id"],
            input=data.get("input", {}),
            expected=data.get("expected"),
            metadata=data.get("metadata"),
            source=data.get("source", "manual"),
            source_trace_id=data.get("source_trace_id"),
            source_span_id=data.get("source_span_id"),
            created_at=data.get("created_at"),
        )


@dataclass
class KeysMapping:
    """
    Field mapping for bulk import operations.

    Attributes:
        input_keys: Keys to extract for input field
        expected_keys: Keys to extract for expected field
        metadata_keys: Keys to extract for metadata field
    """

    input_keys: Optional[List[str]] = None
    expected_keys: Optional[List[str]] = None
    metadata_keys: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API request format."""
        result: Dict[str, Any] = {}
        if self.input_keys:
            result["input_keys"] = self.input_keys
        if self.expected_keys:
            result["expected_keys"] = self.expected_keys
        if self.metadata_keys:
            result["metadata_keys"] = self.metadata_keys
        return result


@dataclass
class CSVColumnMapping:
    """
    Column mapping for CSV import operations.

    Attributes:
        input_column: Column name to use for input field (required)
        expected_column: Column name to use for expected field (optional)
        metadata_columns: Column names to include as metadata (optional)
    """

    input_column: str
    expected_column: Optional[str] = None
    metadata_columns: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API request format."""
        result: Dict[str, Any] = {"input_column": self.input_column}
        if self.expected_column:
            result["expected_column"] = self.expected_column
        if self.metadata_columns:
            result["metadata_columns"] = self.metadata_columns
        return result


@dataclass
class BulkImportResult:
    """
    Result of a bulk import operation.

    Attributes:
        created: Number of items created
        skipped: Number of items skipped (duplicates)
        errors: List of error messages
    """

    created: int
    skipped: int
    errors: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BulkImportResult":
        """Create from API response."""
        return cls(
            created=data.get("created", 0),
            skipped=data.get("skipped", 0),
            errors=data.get("errors"),
        )


DatasetItemInput = Union[Dict[str, Any], DatasetItem]


@dataclass
class DatasetVersion:
    """
    A dataset version snapshot.

    Attributes:
        id: Unique identifier for the version
        dataset_id: ID of the parent dataset
        version: Version number (auto-incremented)
        item_count: Number of items in this version snapshot
        description: Optional description of the version
        metadata: Additional metadata (optional)
        created_by: User ID who created the version (optional)
        created_at: ISO timestamp when created
    """

    id: str
    dataset_id: str
    version: int
    item_count: int
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_by: Optional[str] = None
    created_at: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetVersion":
        """Create DatasetVersion from API response dict."""
        return cls(
            id=data["id"],
            dataset_id=data["dataset_id"],
            version=data["version"],
            item_count=data["item_count"],
            description=data.get("description"),
            metadata=data.get("metadata"),
            created_by=data.get("created_by"),
            created_at=data.get("created_at"),
        )


@dataclass
class DatasetWithVersionInfo:
    """
    A dataset with its version information.

    Attributes:
        id: Dataset ID
        project_id: Project ID
        name: Dataset name
        description: Dataset description (optional)
        metadata: Dataset metadata (optional)
        current_version_id: Currently pinned version ID (optional)
        current_version: Currently pinned version number (optional)
        latest_version: Latest available version number (optional)
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """

    id: str
    project_id: str
    name: str
    created_at: str
    updated_at: str
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    current_version_id: Optional[str] = None
    current_version: Optional[int] = None
    latest_version: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetWithVersionInfo":
        """Create DatasetWithVersionInfo from API response dict."""
        return cls(
            id=data["id"],
            project_id=data["project_id"],
            name=data["name"],
            description=data.get("description"),
            metadata=data.get("metadata"),
            current_version_id=data.get("current_version_id"),
            current_version=data.get("current_version"),
            latest_version=data.get("latest_version"),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )


class Dataset:
    """
    A dataset for evaluation (sync).

    Supports batch insert and auto-pagination for iteration.
    Uses SyncHTTPClient internally - no event loop involvement.

    Example:
        >>> dataset = client.datasets.create(name="my-dataset")
        >>> dataset.insert([
        ...     {"input": {"text": "hello"}, "expected": {"label": "greeting"}},
        ... ])
        >>> for item in dataset:
        ...     print(item.input, item.expected)
    """

    def __init__(
        self,
        id: str,
        name: str,
        description: Optional[str],
        metadata: Optional[Dict[str, Any]],
        created_at: str,
        updated_at: str,
        _http_client: SyncHTTPClient,
        _debug: bool = False,
    ):
        """
        Initialize Dataset.

        Args:
            id: Dataset ID
            name: Dataset name
            description: Dataset description
            metadata: Additional metadata
            created_at: Creation timestamp
            updated_at: Last update timestamp
            _http_client: Internal HTTP client (injected by manager)
            _debug: Enable debug logging
        """
        self._id = id
        self._name = name
        self._description = description
        self._metadata = metadata
        self._created_at = created_at
        self._updated_at = updated_at
        self._http = _http_client
        self._debug = _debug

    @property
    def id(self) -> str:
        """Dataset ID."""
        return self._id

    @property
    def name(self) -> str:
        """Dataset name."""
        return self._name

    @property
    def description(self) -> Optional[str]:
        """Dataset description."""
        return self._description

    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        """Dataset metadata."""
        return self._metadata

    @property
    def created_at(self) -> str:
        """Creation timestamp."""
        return self._created_at

    @property
    def updated_at(self) -> str:
        """Last update timestamp."""
        return self._updated_at

    def _log(self, message: str, *args: Any) -> None:
        """Log debug messages."""
        if self._debug:
            print(f"[Brokle Dataset] {message}", *args)

    def _normalize_item(self, item: DatasetItemInput) -> Dict[str, Any]:
        """Normalize item input to API format."""
        if isinstance(item, DatasetItem):
            result: Dict[str, Any] = {"input": item.input}
            if item.expected is not None:
                result["expected"] = item.expected
            if item.metadata is not None:
                result["metadata"] = item.metadata
            return result
        elif isinstance(item, dict):
            if "input" not in item:
                raise ValueError("Item dict must have 'input' key")
            return item
        else:
            raise TypeError(f"Item must be dict or DatasetItem, got {type(item)}")

    def insert(self, items: List[DatasetItemInput], deduplicate: bool = False) -> int:
        """
        Insert items into the dataset.

        Args:
            items: List of items to insert. Each item can be:
                - A dict with 'input' (required), 'expected' (optional), 'metadata' (optional)
                - A DatasetItem instance
            deduplicate: If True, skip items with duplicate content (based on input+expected hash).
                         Checks against existing items in the dataset and within the batch.

        Returns:
            Number of items created

        Raises:
            DatasetError: If the API request fails
            ValueError: If item format is invalid

        Example:
            >>> dataset.insert([
            ...     {"input": {"q": "2+2?"}, "expected": {"a": "4"}},
            ... ])
            1

            >>> # With deduplication
            >>> dataset.insert([item1, item1], deduplicate=True)
            1  # Second item skipped as duplicate
        """
        if not items:
            return 0

        normalized = [self._normalize_item(item) for item in items]
        self._log(f"Inserting {len(normalized)} items into dataset {self._id}")

        try:
            raw_response = self._http.post(
                f"/v1/datasets/{self._id}/items",
                json={"items": normalized, "deduplicate": deduplicate},
            )
            data = raw_response
            return int(data.get("created", len(normalized)))
        except ValueError as e:
            raise DatasetError(f"Failed to insert items: {e}")
        except Exception as e:
            raise DatasetError(f"Failed to insert items: {e}")

    def get_items(
        self,
        limit: int = 50,
        page: int = 1,
    ) -> List[DatasetItem]:
        """
        Fetch items with pagination.

        Args:
            limit: Maximum number of items to return (default: 50, valid: 10, 25, 50, 100)
            page: Page number to fetch (default: 1, 1-indexed)

        Returns:
            List of DatasetItem objects

        Raises:
            DatasetError: If the API request fails

        Example:
            >>> items = dataset.get_items(limit=10, page=1)
            >>> for item in items:
            ...     print(item.input)
        """
        self._log(
            f"Fetching items from dataset {self._id}: limit={limit}, page={page}"
        )

        try:
            raw_response = self._http.get(
                f"/v1/datasets/{self._id}/items",
                params={"limit": limit, "page": page},
            )
            data = raw_response["data"]
            return [DatasetItem.from_dict(item) for item in data]
        except ValueError as e:
            raise DatasetError(f"Failed to fetch items: {e}")
        except Exception as e:
            raise DatasetError(f"Failed to fetch items: {e}")

    def __iter__(self) -> Iterator[DatasetItem]:
        """
        Auto-paginating iterator over all items.

        Transparently fetches pages as needed.

        Example:
            >>> for item in dataset:
            ...     print(item.input, item.expected)
        """
        page = 1
        limit = 50
        while True:
            items = self.get_items(limit=limit, page=page)
            if not items:
                break
            yield from items
            if len(items) < limit:
                break
            page += 1

    def __len__(self) -> int:
        """
        Return total item count.

        Note: This requires an API call to fetch the count.

        Example:
            >>> len(dataset)
            42
        """
        try:
            raw_response = self._http.get(
                f"/v1/datasets/{self._id}/items",
                params={"limit": 1, "page": 1},
            )
            return int(raw_response.get("pagination", {}).get("total", 0))
        except Exception:
            return 0

    def __repr__(self) -> str:
        """String representation."""
        return f"Dataset(id='{self._id}', name='{self._name}')"

    # =========================================================================
    # Import Methods
    # =========================================================================

    def insert_from_json(
        self,
        file_path: str,
        keys_mapping: Optional[KeysMapping] = None,
        deduplicate: bool = True,
    ) -> BulkImportResult:
        """
        Import dataset items from a JSON or JSONL file.

        Args:
            file_path: Path to JSON file (array) or JSONL file (one object per line)
            keys_mapping: Optional field mapping for extraction
            deduplicate: Skip items with duplicate content (default: True)

        Returns:
            BulkImportResult with created/skipped counts

        Raises:
            DatasetError: If the file cannot be read or API request fails
            FileNotFoundError: If file doesn't exist

        Example:
            >>> result = dataset.insert_from_json("data.json")
            >>> print(f"Created: {result.created}, Skipped: {result.skipped}")
        """
        import json

        self._log(f"Importing items from {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            # Try parsing as JSON array first
            try:
                items = json.loads(content)
                if not isinstance(items, list):
                    items = [items]
            except json.JSONDecodeError:
                # Try parsing as JSONL (one JSON object per line)
                items = []
                for line in content.split("\n"):
                    line = line.strip()
                    if line:
                        items.append(json.loads(line))

            return self._import_items(items, keys_mapping, deduplicate, source="json")
        except FileNotFoundError:
            raise
        except json.JSONDecodeError as e:
            raise DatasetError(f"Invalid JSON in {file_path}: {e}")
        except Exception as e:
            raise DatasetError(f"Failed to import from JSON: {e}")

    def insert_from_pandas(
        self,
        df: "Any",  # pandas.DataFrame - lazy import
        keys_mapping: Optional[KeysMapping] = None,
        deduplicate: bool = True,
    ) -> BulkImportResult:
        """
        Import dataset items from a pandas DataFrame.

        Args:
            df: pandas DataFrame with columns to import
            keys_mapping: Optional field mapping for extraction
            deduplicate: Skip items with duplicate content (default: True)

        Returns:
            BulkImportResult with created/skipped counts

        Raises:
            DatasetError: If the API request fails
            ImportError: If pandas is not installed

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({"question": ["Q1"], "answer": ["A1"]})
            >>> result = dataset.insert_from_pandas(
            ...     df,
            ...     keys_mapping=KeysMapping(
            ...         input_keys=["question"],
            ...         expected_keys=["answer"]
            ...     )
            ... )
        """
        self._log(f"Importing {len(df)} items from DataFrame")

        try:
            items = df.to_dict(orient="records")
            return self._import_items(items, keys_mapping, deduplicate, source="sdk")
        except Exception as e:
            raise DatasetError(f"Failed to import from DataFrame: {e}")

    def from_traces(
        self,
        trace_ids: List[str],
        keys_mapping: Optional[KeysMapping] = None,
        deduplicate: bool = True,
    ) -> BulkImportResult:
        """
        Create dataset items from production traces (OTEL-native).

        This is Brokle's differentiating feature - no competitor exposes this in SDK.
        Extracts input/output from trace spans to create evaluation dataset items.

        Args:
            trace_ids: List of trace IDs to import from
            keys_mapping: Optional field mapping for extraction
            deduplicate: Skip items with duplicate content (default: True)

        Returns:
            BulkImportResult with created/skipped counts

        Raises:
            DatasetError: If the API request fails

        Example:
            >>> result = dataset.from_traces(
            ...     trace_ids=["01HXYZ...", "01HABC..."],
            ...     keys_mapping=KeysMapping(input_keys=["user_input"])
            ... )
        """
        if not trace_ids:
            return BulkImportResult(created=0, skipped=0)

        self._log(f"Creating items from {len(trace_ids)} traces")

        try:
            payload: Dict[str, Any] = {
                "trace_ids": trace_ids,
                "deduplicate": deduplicate,
            }
            if keys_mapping:
                payload["keys_mapping"] = keys_mapping.to_dict()

            raw_response = self._http.post(
                f"/v1/datasets/{self._id}/items/from-traces",
                json=payload,
            )
            data = raw_response
            return BulkImportResult.from_dict(data)
        except Exception as e:
            raise DatasetError(f"Failed to create items from traces: {e}")

    def from_spans(
        self,
        span_ids: List[str],
        keys_mapping: Optional[KeysMapping] = None,
        deduplicate: bool = True,
    ) -> BulkImportResult:
        """
        Create dataset items from production spans.

        Args:
            span_ids: List of span IDs to import from
            keys_mapping: Optional field mapping for extraction
            deduplicate: Skip items with duplicate content (default: True)

        Returns:
            BulkImportResult with created/skipped counts

        Raises:
            DatasetError: If the API request fails

        Example:
            >>> result = dataset.from_spans(span_ids=["span1", "span2"])
        """
        if not span_ids:
            return BulkImportResult(created=0, skipped=0)

        self._log(f"Creating items from {len(span_ids)} spans")

        try:
            payload: Dict[str, Any] = {
                "span_ids": span_ids,
                "deduplicate": deduplicate,
            }
            if keys_mapping:
                payload["keys_mapping"] = keys_mapping.to_dict()

            raw_response = self._http.post(
                f"/v1/datasets/{self._id}/items/from-spans",
                json=payload,
            )
            data = raw_response
            return BulkImportResult.from_dict(data)
        except Exception as e:
            raise DatasetError(f"Failed to create items from spans: {e}")

    def _import_items(
        self,
        items: List[Dict[str, Any]],
        keys_mapping: Optional[KeysMapping],
        deduplicate: bool,
        source: str,
    ) -> BulkImportResult:
        """Internal method to import items via API."""
        if not items:
            return BulkImportResult(created=0, skipped=0)

        payload: Dict[str, Any] = {
            "items": items,
            "deduplicate": deduplicate,
            "source": source,
        }
        if keys_mapping:
            payload["keys_mapping"] = keys_mapping.to_dict()

        try:
            raw_response = self._http.post(
                f"/v1/datasets/{self._id}/items/import-json",
                json=payload,
            )
            data = raw_response
            return BulkImportResult.from_dict(data)
        except Exception as e:
            raise DatasetError(f"Failed to import items: {e}")

    def insert_from_csv(
        self,
        file_path: str,
        column_mapping: CSVColumnMapping,
        has_header: bool = True,
        deduplicate: bool = True,
    ) -> BulkImportResult:
        """
        Import dataset items from a CSV file.

        Reads a CSV file from disk and imports items using the specified column mapping.
        The CSV content is sent to the backend API for processing.

        Args:
            file_path: Path to the CSV file to import
            column_mapping: CSVColumnMapping specifying which columns to use:
                - input_column: Column name for input data (required)
                - expected_column: Column name for expected output (optional)
                - metadata_columns: List of column names for metadata (optional)
            has_header: Whether the CSV has a header row (default: True)
            deduplicate: Skip items with duplicate content (default: True)

        Returns:
            BulkImportResult with created/skipped counts and any errors

        Raises:
            DatasetError: If the file cannot be read or API request fails
            FileNotFoundError: If the file doesn't exist

        Example:
            >>> from brokle.datasets import CSVColumnMapping
            >>> result = dataset.insert_from_csv(
            ...     "qa_pairs.csv",
            ...     column_mapping=CSVColumnMapping(
            ...         input_column="question",
            ...         expected_column="answer",
            ...         metadata_columns=["category", "difficulty"]
            ...     ),
            ...     has_header=True,
            ...     deduplicate=True
            ... )
            >>> print(f"Created: {result.created}, Skipped: {result.skipped}")
        """
        self._log(f"Importing items from CSV file: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            payload: Dict[str, Any] = {
                "content": content,
                "column_mapping": column_mapping.to_dict(),
                "has_header": has_header,
                "deduplicate": deduplicate,
            }

            raw_response = self._http.post(
                f"/v1/datasets/{self._id}/items/import-csv",
                json=payload,
            )
            data = raw_response
            return BulkImportResult.from_dict(data)
        except FileNotFoundError:
            raise
        except Exception as e:
            raise DatasetError(f"Failed to import from CSV: {e}")

    # =========================================================================
    # Export Methods
    # =========================================================================

    def to_json(self, file_path: str) -> None:
        """
        Export dataset items to a JSON file.

        Args:
            file_path: Path to write the JSON file

        Raises:
            DatasetError: If the export fails

        Example:
            >>> dataset.to_json("exported_data.json")
        """
        import json

        self._log(f"Exporting items to {file_path}")

        try:
            items = self._export_items()
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(items, f, indent=2, default=str)
        except Exception as e:
            raise DatasetError(f"Failed to export to JSON: {e}")

    def to_pandas(self) -> "Any":  # Returns pandas.DataFrame
        """
        Export dataset items as a pandas DataFrame.

        Returns:
            pandas.DataFrame with dataset items

        Raises:
            DatasetError: If the export fails
            ImportError: If pandas is not installed

        Example:
            >>> df = dataset.to_pandas()
            >>> print(df.head())
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_pandas(). Install with: pip install pandas"
            )

        self._log("Exporting items to DataFrame")

        try:
            items = self._export_items()
            return pd.DataFrame(items)
        except ImportError:
            raise
        except Exception as e:
            raise DatasetError(f"Failed to export to DataFrame: {e}")

    def _export_items(self) -> List[Dict[str, Any]]:
        """Internal method to fetch all items for export."""
        try:
            raw_response = self._http.get(
                f"/v1/datasets/{self._id}/items/export",
            )
            data = raw_response
            return data
        except Exception as e:
            raise DatasetError(f"Failed to export items: {e}")

    # =========================================================================
    # Versioning Methods
    # =========================================================================

    def create_version(
        self,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DatasetVersion:
        """
        Create a new version snapshot of the current dataset items.

        Versions allow you to pin a dataset to a specific point in time for
        reproducible evaluations.

        Args:
            description: Optional description for this version
            metadata: Optional metadata for this version

        Returns:
            DatasetVersion object representing the new version

        Raises:
            DatasetError: If the API request fails

        Example:
            >>> version = dataset.create_version(
            ...     description="Baseline evaluation dataset v1"
            ... )
            >>> print(f"Created version {version.version} with {version.item_count} items")
        """
        self._log(f"Creating version for dataset {self._id}")

        try:
            payload: Dict[str, Any] = {}
            if description is not None:
                payload["description"] = description
            if metadata is not None:
                payload["metadata"] = metadata

            raw_response = self._http.post(
                f"/v1/datasets/{self._id}/versions",
                json=payload,
            )
            data = raw_response
            return DatasetVersion.from_dict(data)
        except Exception as e:
            raise DatasetError(f"Failed to create version: {e}")

    def list_versions(self) -> List[DatasetVersion]:
        """
        List all versions for this dataset.

        Returns:
            List of DatasetVersion objects

        Raises:
            DatasetError: If the API request fails

        Example:
            >>> versions = dataset.list_versions()
            >>> for v in versions:
            ...     print(f"v{v.version}: {v.item_count} items")
        """
        self._log(f"Listing versions for dataset {self._id}")

        try:
            raw_response = self._http.get(
                f"/v1/datasets/{self._id}/versions",
            )
            data = raw_response.get("data", [])
            if isinstance(data, list):
                return [DatasetVersion.from_dict(v) for v in data]
            return []
        except Exception as e:
            raise DatasetError(f"Failed to list versions: {e}")

    def get_version(self, version_id: str) -> DatasetVersion:
        """
        Get a specific version by ID.

        Args:
            version_id: The version ID to retrieve

        Returns:
            DatasetVersion object

        Raises:
            DatasetError: If the API request fails or version not found

        Example:
            >>> version = dataset.get_version("01HXYZ...")
            >>> print(f"Version {version.version} has {version.item_count} items")
        """
        self._log(f"Getting version {version_id} for dataset {self._id}")

        try:
            raw_response = self._http.get(
                f"/v1/datasets/{self._id}/versions/{version_id}",
            )
            data = raw_response
            return DatasetVersion.from_dict(data)
        except Exception as e:
            raise DatasetError(f"Failed to get version: {e}")

    def get_version_items(
        self,
        version_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[DatasetItem]:
        """
        Get items for a specific version with pagination.

        Args:
            version_id: The version ID to get items from
            limit: Maximum number of items to return (default: 50)
            offset: Number of items to skip (default: 0)

        Returns:
            List of DatasetItem objects for this version

        Raises:
            DatasetError: If the API request fails

        Example:
            >>> items = dataset.get_version_items("01HXYZ...", limit=10)
            >>> for item in items:
            ...     print(item.input)
        """
        self._log(
            f"Getting items for version {version_id}: limit={limit}, offset={offset}"
        )

        try:
            raw_response = self._http.get(
                f"/v1/datasets/{self._id}/versions/{version_id}/items",
                params={"limit": limit, "offset": offset},
            )
            data = raw_response
            items_data = data.get("items", [])
            return [DatasetItem.from_dict(item) for item in items_data]
        except Exception as e:
            raise DatasetError(f"Failed to get version items: {e}")

    def pin_version(self, version_id: Optional[str] = None) -> "Dataset":
        """
        Pin this dataset to a specific version for reproducible evaluations.

        When pinned, iterations and exports will use the pinned version's items
        instead of the current items.

        Args:
            version_id: Version ID to pin to, or None to unpin (use latest)

        Returns:
            Updated Dataset object

        Raises:
            DatasetError: If the API request fails

        Example:
            >>> # Pin to a specific version
            >>> dataset.pin_version("01HXYZ...")
            >>> # Unpin to use latest items
            >>> dataset.pin_version(None)
        """
        action = f"version {version_id}" if version_id else "latest"
        self._log(f"Pinning dataset {self._id} to {action}")

        try:
            payload: Dict[str, Any] = {"version_id": version_id}
            raw_response = self._http.post(
                f"/v1/datasets/{self._id}/pin",
                json=payload,
            )
            data = raw_response
            # Return updated dataset
            return Dataset(
                id=data["id"],
                name=data["name"],
                description=data.get("description"),
                metadata=data.get("metadata"),
                created_at=data["created_at"],
                updated_at=data["updated_at"],
                _http_client=self._http,
                _debug=self._debug,
            )
        except Exception as e:
            raise DatasetError(f"Failed to pin version: {e}")

    def get_info(self) -> DatasetWithVersionInfo:
        """
        Get dataset with version information.

        Returns information about the current pinned version and latest version.

        Returns:
            DatasetWithVersionInfo object

        Raises:
            DatasetError: If the API request fails

        Example:
            >>> info = dataset.get_info()
            >>> print(f"Current version: {info.current_version}")
            >>> print(f"Latest version: {info.latest_version}")
        """
        self._log(f"Getting info for dataset {self._id}")

        try:
            raw_response = self._http.get(
                f"/v1/datasets/{self._id}/info",
            )
            data = raw_response
            return DatasetWithVersionInfo.from_dict(data)
        except Exception as e:
            raise DatasetError(f"Failed to get dataset info: {e}")


class AsyncDataset:
    """
    A dataset for evaluation (async).

    Supports batch insert and auto-pagination for async iteration.
    Uses AsyncHTTPClient internally.

    Example:
        >>> dataset = await client.datasets.create(name="my-dataset")
        >>> await dataset.insert([
        ...     {"input": {"text": "hello"}, "expected": {"label": "greeting"}},
        ... ])
        >>> async for item in dataset:
        ...     print(item.input, item.expected)
    """

    def __init__(
        self,
        id: str,
        name: str,
        description: Optional[str],
        metadata: Optional[Dict[str, Any]],
        created_at: str,
        updated_at: str,
        _http_client: AsyncHTTPClient,
        _debug: bool = False,
    ):
        """
        Initialize AsyncDataset.

        Args:
            id: Dataset ID
            name: Dataset name
            description: Dataset description
            metadata: Additional metadata
            created_at: Creation timestamp
            updated_at: Last update timestamp
            _http_client: Internal async HTTP client (injected by manager)
            _debug: Enable debug logging
        """
        self._id = id
        self._name = name
        self._description = description
        self._metadata = metadata
        self._created_at = created_at
        self._updated_at = updated_at
        self._http = _http_client
        self._debug = _debug

    @property
    def id(self) -> str:
        """Dataset ID."""
        return self._id

    @property
    def name(self) -> str:
        """Dataset name."""
        return self._name

    @property
    def description(self) -> Optional[str]:
        """Dataset description."""
        return self._description

    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        """Dataset metadata."""
        return self._metadata

    @property
    def created_at(self) -> str:
        """Creation timestamp."""
        return self._created_at

    @property
    def updated_at(self) -> str:
        """Last update timestamp."""
        return self._updated_at

    def _log(self, message: str, *args: Any) -> None:
        """Log debug messages."""
        if self._debug:
            print(f"[Brokle AsyncDataset] {message}", *args)

    def _normalize_item(self, item: DatasetItemInput) -> Dict[str, Any]:
        """Normalize item input to API format."""
        if isinstance(item, DatasetItem):
            result: Dict[str, Any] = {"input": item.input}
            if item.expected is not None:
                result["expected"] = item.expected
            if item.metadata is not None:
                result["metadata"] = item.metadata
            return result
        elif isinstance(item, dict):
            if "input" not in item:
                raise ValueError("Item dict must have 'input' key")
            return item
        else:
            raise TypeError(f"Item must be dict or DatasetItem, got {type(item)}")

    async def insert(self, items: List[DatasetItemInput], deduplicate: bool = False) -> int:
        """
        Insert items into the dataset (async).

        Args:
            items: List of items to insert. Each item can be:
                - A dict with 'input' (required), 'expected' (optional), 'metadata' (optional)
                - A DatasetItem instance
            deduplicate: If True, skip items with duplicate content (based on input+expected hash).
                         Checks against existing items in the dataset and within the batch.

        Returns:
            Number of items created

        Raises:
            DatasetError: If the API request fails
            ValueError: If item format is invalid

        Example:
            >>> await dataset.insert([
            ...     {"input": {"q": "2+2?"}, "expected": {"a": "4"}},
            ... ])
            1

            >>> # With deduplication
            >>> await dataset.insert([item1, item1], deduplicate=True)
            1  # Second item skipped as duplicate
        """
        if not items:
            return 0

        normalized = [self._normalize_item(item) for item in items]
        self._log(f"Inserting {len(normalized)} items into dataset {self._id}")

        try:
            raw_response = await self._http.post(
                f"/v1/datasets/{self._id}/items",
                json={"items": normalized, "deduplicate": deduplicate},
            )
            data = raw_response
            return int(data.get("created", len(normalized)))
        except ValueError as e:
            raise DatasetError(f"Failed to insert items: {e}")
        except Exception as e:
            raise DatasetError(f"Failed to insert items: {e}")

    async def get_items(
        self,
        limit: int = 50,
        page: int = 1,
    ) -> List[DatasetItem]:
        """
        Fetch items with pagination (async).

        Args:
            limit: Maximum number of items to return (default: 50, valid: 10, 25, 50, 100)
            page: Page number to fetch (default: 1, 1-indexed)

        Returns:
            List of DatasetItem objects

        Raises:
            DatasetError: If the API request fails

        Example:
            >>> items = await dataset.get_items(limit=10, page=1)
            >>> for item in items:
            ...     print(item.input)
        """
        self._log(
            f"Fetching items from dataset {self._id}: limit={limit}, page={page}"
        )

        try:
            raw_response = await self._http.get(
                f"/v1/datasets/{self._id}/items",
                params={"limit": limit, "page": page},
            )
            data = raw_response["data"]
            return [DatasetItem.from_dict(item) for item in data]
        except ValueError as e:
            raise DatasetError(f"Failed to fetch items: {e}")
        except Exception as e:
            raise DatasetError(f"Failed to fetch items: {e}")

    async def __aiter__(self) -> AsyncIterator[DatasetItem]:
        """
        Auto-paginating async iterator over all items.

        Transparently fetches pages as needed.

        Example:
            >>> async for item in dataset:
            ...     print(item.input, item.expected)
        """
        page = 1
        limit = 50
        while True:
            items = await self.get_items(limit=limit, page=page)
            if not items:
                break
            for item in items:
                yield item
            if len(items) < limit:
                break
            page += 1

    async def count(self) -> int:
        """
        Return total item count (async).

        Example:
            >>> total = await dataset.count()
            >>> print(f"Dataset has {total} items")
        """
        try:
            raw_response = await self._http.get(
                f"/v1/datasets/{self._id}/items",
                params={"limit": 1, "page": 1},
            )
            return int(raw_response.get("pagination", {}).get("total", 0))
        except Exception:
            return 0

    def __repr__(self) -> str:
        """String representation."""
        return f"AsyncDataset(id='{self._id}', name='{self._name}')"

    # =========================================================================
    # Import Methods (Async)
    # =========================================================================

    async def insert_from_json(
        self,
        file_path: str,
        keys_mapping: Optional[KeysMapping] = None,
        deduplicate: bool = True,
    ) -> BulkImportResult:
        """
        Import dataset items from a JSON or JSONL file (async).

        Args:
            file_path: Path to JSON file (array) or JSONL file (one object per line)
            keys_mapping: Optional field mapping for extraction
            deduplicate: Skip items with duplicate content (default: True)

        Returns:
            BulkImportResult with created/skipped counts

        Raises:
            DatasetError: If the file cannot be read or API request fails
            FileNotFoundError: If file doesn't exist

        Example:
            >>> result = await dataset.insert_from_json("data.json")
            >>> print(f"Created: {result.created}, Skipped: {result.skipped}")
        """
        import json

        self._log(f"Importing items from {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            # Try parsing as JSON array first
            try:
                items = json.loads(content)
                if not isinstance(items, list):
                    items = [items]
            except json.JSONDecodeError:
                # Try parsing as JSONL (one JSON object per line)
                items = []
                for line in content.split("\n"):
                    line = line.strip()
                    if line:
                        items.append(json.loads(line))

            return await self._import_items(
                items, keys_mapping, deduplicate, source="json"
            )
        except FileNotFoundError:
            raise
        except json.JSONDecodeError as e:
            raise DatasetError(f"Invalid JSON in {file_path}: {e}")
        except Exception as e:
            raise DatasetError(f"Failed to import from JSON: {e}")

    async def insert_from_pandas(
        self,
        df: "Any",  # pandas.DataFrame - lazy import
        keys_mapping: Optional[KeysMapping] = None,
        deduplicate: bool = True,
    ) -> BulkImportResult:
        """
        Import dataset items from a pandas DataFrame (async).

        Args:
            df: pandas DataFrame with columns to import
            keys_mapping: Optional field mapping for extraction
            deduplicate: Skip items with duplicate content (default: True)

        Returns:
            BulkImportResult with created/skipped counts

        Raises:
            DatasetError: If the API request fails
            ImportError: If pandas is not installed

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({"question": ["Q1"], "answer": ["A1"]})
            >>> result = await dataset.insert_from_pandas(
            ...     df,
            ...     keys_mapping=KeysMapping(
            ...         input_keys=["question"],
            ...         expected_keys=["answer"]
            ...     )
            ... )
        """
        self._log(f"Importing {len(df)} items from DataFrame")

        try:
            items = df.to_dict(orient="records")
            return await self._import_items(
                items, keys_mapping, deduplicate, source="sdk"
            )
        except Exception as e:
            raise DatasetError(f"Failed to import from DataFrame: {e}")

    async def from_traces(
        self,
        trace_ids: List[str],
        keys_mapping: Optional[KeysMapping] = None,
        deduplicate: bool = True,
    ) -> BulkImportResult:
        """
        Create dataset items from production traces (OTEL-native, async).

        This is Brokle's differentiating feature - no competitor exposes this in SDK.
        Extracts input/output from trace spans to create evaluation dataset items.

        Args:
            trace_ids: List of trace IDs to import from
            keys_mapping: Optional field mapping for extraction
            deduplicate: Skip items with duplicate content (default: True)

        Returns:
            BulkImportResult with created/skipped counts

        Raises:
            DatasetError: If the API request fails

        Example:
            >>> result = await dataset.from_traces(
            ...     trace_ids=["01HXYZ...", "01HABC..."],
            ...     keys_mapping=KeysMapping(input_keys=["user_input"])
            ... )
        """
        if not trace_ids:
            return BulkImportResult(created=0, skipped=0)

        self._log(f"Creating items from {len(trace_ids)} traces")

        try:
            payload: Dict[str, Any] = {
                "trace_ids": trace_ids,
                "deduplicate": deduplicate,
            }
            if keys_mapping:
                payload["keys_mapping"] = keys_mapping.to_dict()

            raw_response = await self._http.post(
                f"/v1/datasets/{self._id}/items/from-traces",
                json=payload,
            )
            data = raw_response
            return BulkImportResult.from_dict(data)
        except Exception as e:
            raise DatasetError(f"Failed to create items from traces: {e}")

    async def from_spans(
        self,
        span_ids: List[str],
        keys_mapping: Optional[KeysMapping] = None,
        deduplicate: bool = True,
    ) -> BulkImportResult:
        """
        Create dataset items from production spans (async).

        Args:
            span_ids: List of span IDs to import from
            keys_mapping: Optional field mapping for extraction
            deduplicate: Skip items with duplicate content (default: True)

        Returns:
            BulkImportResult with created/skipped counts

        Raises:
            DatasetError: If the API request fails

        Example:
            >>> result = await dataset.from_spans(span_ids=["span1", "span2"])
        """
        if not span_ids:
            return BulkImportResult(created=0, skipped=0)

        self._log(f"Creating items from {len(span_ids)} spans")

        try:
            payload: Dict[str, Any] = {
                "span_ids": span_ids,
                "deduplicate": deduplicate,
            }
            if keys_mapping:
                payload["keys_mapping"] = keys_mapping.to_dict()

            raw_response = await self._http.post(
                f"/v1/datasets/{self._id}/items/from-spans",
                json=payload,
            )
            data = raw_response
            return BulkImportResult.from_dict(data)
        except Exception as e:
            raise DatasetError(f"Failed to create items from spans: {e}")

    async def _import_items(
        self,
        items: List[Dict[str, Any]],
        keys_mapping: Optional[KeysMapping],
        deduplicate: bool,
        source: str,
    ) -> BulkImportResult:
        """Internal method to import items via API (async)."""
        if not items:
            return BulkImportResult(created=0, skipped=0)

        payload: Dict[str, Any] = {
            "items": items,
            "deduplicate": deduplicate,
            "source": source,
        }
        if keys_mapping:
            payload["keys_mapping"] = keys_mapping.to_dict()

        try:
            raw_response = await self._http.post(
                f"/v1/datasets/{self._id}/items/import-json",
                json=payload,
            )
            data = raw_response
            return BulkImportResult.from_dict(data)
        except Exception as e:
            raise DatasetError(f"Failed to import items: {e}")

    async def insert_from_csv(
        self,
        file_path: str,
        column_mapping: CSVColumnMapping,
        has_header: bool = True,
        deduplicate: bool = True,
    ) -> BulkImportResult:
        """
        Import dataset items from a CSV file (async).

        Reads a CSV file from disk and imports items using the specified column mapping.
        The CSV content is sent to the backend API for processing.

        Args:
            file_path: Path to the CSV file to import
            column_mapping: CSVColumnMapping specifying which columns to use:
                - input_column: Column name for input data (required)
                - expected_column: Column name for expected output (optional)
                - metadata_columns: List of column names for metadata (optional)
            has_header: Whether the CSV has a header row (default: True)
            deduplicate: Skip items with duplicate content (default: True)

        Returns:
            BulkImportResult with created/skipped counts and any errors

        Raises:
            DatasetError: If the file cannot be read or API request fails
            FileNotFoundError: If the file doesn't exist

        Example:
            >>> from brokle.datasets import CSVColumnMapping
            >>> result = await dataset.insert_from_csv(
            ...     "qa_pairs.csv",
            ...     column_mapping=CSVColumnMapping(
            ...         input_column="question",
            ...         expected_column="answer",
            ...         metadata_columns=["category", "difficulty"]
            ...     ),
            ...     has_header=True,
            ...     deduplicate=True
            ... )
            >>> print(f"Created: {result.created}, Skipped: {result.skipped}")
        """
        self._log(f"Importing items from CSV file: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            payload: Dict[str, Any] = {
                "content": content,
                "column_mapping": column_mapping.to_dict(),
                "has_header": has_header,
                "deduplicate": deduplicate,
            }

            raw_response = await self._http.post(
                f"/v1/datasets/{self._id}/items/import-csv",
                json=payload,
            )
            data = raw_response
            return BulkImportResult.from_dict(data)
        except FileNotFoundError:
            raise
        except Exception as e:
            raise DatasetError(f"Failed to import from CSV: {e}")

    # =========================================================================
    # Export Methods (Async)
    # =========================================================================

    async def to_json(self, file_path: str) -> None:
        """
        Export dataset items to a JSON file (async).

        Args:
            file_path: Path to write the JSON file

        Raises:
            DatasetError: If the export fails

        Example:
            >>> await dataset.to_json("exported_data.json")
        """
        import json

        self._log(f"Exporting items to {file_path}")

        try:
            items = await self._export_items()
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(items, f, indent=2, default=str)
        except Exception as e:
            raise DatasetError(f"Failed to export to JSON: {e}")

    async def to_pandas(self) -> "Any":  # Returns pandas.DataFrame
        """
        Export dataset items as a pandas DataFrame (async).

        Returns:
            pandas.DataFrame with dataset items

        Raises:
            DatasetError: If the export fails
            ImportError: If pandas is not installed

        Example:
            >>> df = await dataset.to_pandas()
            >>> print(df.head())
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_pandas(). Install with: pip install pandas"
            )

        self._log("Exporting items to DataFrame")

        try:
            items = await self._export_items()
            return pd.DataFrame(items)
        except ImportError:
            raise
        except Exception as e:
            raise DatasetError(f"Failed to export to DataFrame: {e}")

    async def _export_items(self) -> List[Dict[str, Any]]:
        """Internal method to fetch all items for export (async)."""
        try:
            raw_response = await self._http.get(
                f"/v1/datasets/{self._id}/items/export",
            )
            data = raw_response
            return data
        except Exception as e:
            raise DatasetError(f"Failed to export items: {e}")

    # =========================================================================
    # Versioning Methods (Async)
    # =========================================================================

    async def create_version(
        self,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DatasetVersion:
        """
        Create a new version snapshot of the current dataset items (async).

        Versions allow you to pin a dataset to a specific point in time for
        reproducible evaluations.

        Args:
            description: Optional description for this version
            metadata: Optional metadata for this version

        Returns:
            DatasetVersion object representing the new version

        Raises:
            DatasetError: If the API request fails

        Example:
            >>> version = await dataset.create_version(
            ...     description="Baseline evaluation dataset v1"
            ... )
            >>> print(f"Created version {version.version} with {version.item_count} items")
        """
        self._log(f"Creating version for dataset {self._id}")

        try:
            payload: Dict[str, Any] = {}
            if description is not None:
                payload["description"] = description
            if metadata is not None:
                payload["metadata"] = metadata

            raw_response = await self._http.post(
                f"/v1/datasets/{self._id}/versions",
                json=payload,
            )
            data = raw_response
            return DatasetVersion.from_dict(data)
        except Exception as e:
            raise DatasetError(f"Failed to create version: {e}")

    async def list_versions(self) -> List[DatasetVersion]:
        """
        List all versions for this dataset (async).

        Returns:
            List of DatasetVersion objects

        Raises:
            DatasetError: If the API request fails

        Example:
            >>> versions = await dataset.list_versions()
            >>> for v in versions:
            ...     print(f"v{v.version}: {v.item_count} items")
        """
        self._log(f"Listing versions for dataset {self._id}")

        try:
            raw_response = await self._http.get(
                f"/v1/datasets/{self._id}/versions",
            )
            data = raw_response.get("data", [])
            if isinstance(data, list):
                return [DatasetVersion.from_dict(v) for v in data]
            return []
        except Exception as e:
            raise DatasetError(f"Failed to list versions: {e}")

    async def get_version(self, version_id: str) -> DatasetVersion:
        """
        Get a specific version by ID (async).

        Args:
            version_id: The version ID to retrieve

        Returns:
            DatasetVersion object

        Raises:
            DatasetError: If the API request fails or version not found

        Example:
            >>> version = await dataset.get_version("01HXYZ...")
            >>> print(f"Version {version.version} has {version.item_count} items")
        """
        self._log(f"Getting version {version_id} for dataset {self._id}")

        try:
            raw_response = await self._http.get(
                f"/v1/datasets/{self._id}/versions/{version_id}",
            )
            data = raw_response
            return DatasetVersion.from_dict(data)
        except Exception as e:
            raise DatasetError(f"Failed to get version: {e}")

    async def get_version_items(
        self,
        version_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[DatasetItem]:
        """
        Get items for a specific version with pagination (async).

        Args:
            version_id: The version ID to get items from
            limit: Maximum number of items to return (default: 50)
            offset: Number of items to skip (default: 0)

        Returns:
            List of DatasetItem objects for this version

        Raises:
            DatasetError: If the API request fails

        Example:
            >>> items = await dataset.get_version_items("01HXYZ...", limit=10)
            >>> for item in items:
            ...     print(item.input)
        """
        self._log(
            f"Getting items for version {version_id}: limit={limit}, offset={offset}"
        )

        try:
            raw_response = await self._http.get(
                f"/v1/datasets/{self._id}/versions/{version_id}/items",
                params={"limit": limit, "offset": offset},
            )
            data = raw_response
            items_data = data.get("items", [])
            return [DatasetItem.from_dict(item) for item in items_data]
        except Exception as e:
            raise DatasetError(f"Failed to get version items: {e}")

    async def pin_version(self, version_id: Optional[str] = None) -> "AsyncDataset":
        """
        Pin this dataset to a specific version for reproducible evaluations (async).

        When pinned, iterations and exports will use the pinned version's items
        instead of the current items.

        Args:
            version_id: Version ID to pin to, or None to unpin (use latest)

        Returns:
            Updated AsyncDataset object

        Raises:
            DatasetError: If the API request fails

        Example:
            >>> # Pin to a specific version
            >>> await dataset.pin_version("01HXYZ...")
            >>> # Unpin to use latest items
            >>> await dataset.pin_version(None)
        """
        action = f"version {version_id}" if version_id else "latest"
        self._log(f"Pinning dataset {self._id} to {action}")

        try:
            payload: Dict[str, Any] = {"version_id": version_id}
            raw_response = await self._http.post(
                f"/v1/datasets/{self._id}/pin",
                json=payload,
            )
            data = raw_response
            # Return updated dataset
            return AsyncDataset(
                id=data["id"],
                name=data["name"],
                description=data.get("description"),
                metadata=data.get("metadata"),
                created_at=data["created_at"],
                updated_at=data["updated_at"],
                _http_client=self._http,
                _debug=self._debug,
            )
        except Exception as e:
            raise DatasetError(f"Failed to pin version: {e}")

    async def get_info(self) -> DatasetWithVersionInfo:
        """
        Get dataset with version information (async).

        Returns information about the current pinned version and latest version.

        Returns:
            DatasetWithVersionInfo object

        Raises:
            DatasetError: If the API request fails

        Example:
            >>> info = await dataset.get_info()
            >>> print(f"Current version: {info.current_version}")
            >>> print(f"Latest version: {info.latest_version}")
        """
        self._log(f"Getting info for dataset {self._id}")

        try:
            raw_response = await self._http.get(
                f"/v1/datasets/{self._id}/info",
            )
            data = raw_response
            return DatasetWithVersionInfo.from_dict(data)
        except Exception as e:
            raise DatasetError(f"Failed to get dataset info: {e}")
