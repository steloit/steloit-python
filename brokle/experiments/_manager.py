"""
Experiments Manager

Provides both synchronous and asynchronous experiment management for Brokle.

Sync Usage:
    >>> from brokle import Brokle
    >>>
    >>> client = Brokle(api_key="bk_...")
    >>>
    >>> results = client.experiments.run(
    ...     name="gpt4-test",
    ...     dataset=dataset,
    ...     task=my_task,
    ...     scorers=[exact, relevance],
    ... )
    >>>
    >>> for name, stats in results.summary.items():
    ...     print(f"{name}: mean={stats['mean']:.3f}")

Async Usage:
    >>> async with AsyncBrokle(api_key="bk_...") as client:
    ...     results = await client.experiments.run(
    ...         name="test",
    ...         dataset=dataset,
    ...         task=my_task,
    ...         scorers=[exact],
    ...     )
"""

import asyncio
import inspect
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Union

from .._http import AsyncHTTPClient, SyncHTTPClient
from ..config import BrokleConfig
from ..datasets.dataset import AsyncDataset, Dataset, DatasetItem
from ..query.types import QueriedSpan
from ..scores.types import ScoreResult, ScorerProtocol, ScoreType, ScoreValue
from .exceptions import EvaluationError
from .types import (
    AsyncTaskFunction,
    ComparisonResult,
    EvaluationItem,
    EvaluationResults,
    Experiment,
    ProgressCallback,
    SpanExtractExpected,
    SpanExtractInput,
    SpanExtractOutput,
    SummaryStats,
    TaskFunction,
)


def _normalize_score_result(
    value: ScoreValue,
    scorer_name: str,
) -> List[ScoreResult]:
    """
    Normalize various scorer return types to List[ScoreResult].

    Args:
        value: The scorer return value
        scorer_name: Name to use if value is a primitive

    Returns:
        List of ScoreResult objects
    """
    if value is None:
        return []

    if isinstance(value, ScoreResult):
        return [value]

    if isinstance(value, list):
        results = []
        for v in value:
            if isinstance(v, ScoreResult):
                results.append(v)
        return results

    if isinstance(value, bool):
        return [
            ScoreResult(
                name=scorer_name,
                value=1.0 if value else 0.0,
                type=ScoreType.BOOLEAN,
            )
        ]

    if isinstance(value, (int, float)):
        return [
            ScoreResult(
                name=scorer_name,
                value=float(value),
                type=ScoreType.NUMERIC,
            )
        ]

    return []


def _get_scorer_name(scorer: ScorerProtocol) -> str:
    """Get the name of a scorer."""
    if hasattr(scorer, "name"):
        return str(scorer.name)
    if hasattr(scorer, "__name__"):
        return str(scorer.__name__)
    return scorer.__class__.__name__


def _run_scorer_safe(
    scorer: ScorerProtocol,
    output: Any,
    expected: Any,
    input_data: Dict[str, Any],
) -> List[ScoreResult]:
    """
    Run a scorer safely, catching any exceptions.

    Returns list of ScoreResults. On failure, returns a single ScoreResult
    with scoring_failed=True.
    """
    scorer_name = _get_scorer_name(scorer)
    try:
        result = scorer(output=output, expected=expected, input=input_data)
        return _normalize_score_result(result, scorer_name)
    except Exception as e:
        return [
            ScoreResult(
                name=scorer_name,
                value=0.0,
                type=ScoreType.NUMERIC,
                scoring_failed=True,
                reason=f"Scorer failed: {str(e)}",
            )
        ]


async def _run_scorer_safe_async(
    scorer: ScorerProtocol,
    output: Any,
    expected: Any,
    input_data: Dict[str, Any],
) -> List[ScoreResult]:
    """
    Run a scorer safely (async-aware), catching any exceptions.

    Handles both sync and async scorers.
    """
    scorer_name = _get_scorer_name(scorer)
    try:
        result = scorer(output=output, expected=expected, input=input_data)
        if inspect.isawaitable(result):
            result = await result
        return _normalize_score_result(result, scorer_name)
    except Exception as e:
        return [
            ScoreResult(
                name=scorer_name,
                value=0.0,
                type=ScoreType.NUMERIC,
                scoring_failed=True,
                reason=f"Scorer failed: {str(e)}",
            )
        ]


def _compute_summary(items: List[EvaluationItem]) -> Dict[str, SummaryStats]:
    """
    Compute per-scorer summary statistics.

    Only includes non-failed scores in mean/std_dev/min/max.
    pass_rate = successful_scores / total_scores.
    """
    summary: Dict[str, SummaryStats] = {}

    # Collect scores by name
    scores_by_name: Dict[str, List[ScoreResult]] = defaultdict(list)
    for item in items:
        for score in item.scores:
            scores_by_name[score.name].append(score)

    for name, scores in scores_by_name.items():
        # Filter successful scores for statistics
        successful = [s.value for s in scores if not s.scoring_failed]
        total = len(scores)
        passed = len(successful)

        if successful:
            summary[name] = SummaryStats(
                mean=statistics.mean(successful),
                std_dev=statistics.stdev(successful) if len(successful) > 1 else 0.0,
                min=min(successful),
                max=max(successful),
                count=total,
                pass_rate=passed / total if total > 0 else 0.0,
            )
        else:
            # All scores failed
            summary[name] = SummaryStats(
                mean=0.0,
                std_dev=0.0,
                min=0.0,
                max=0.0,
                count=total,
                pass_rate=0.0,
            )

    return summary


class _BaseExperimentsManagerMixin:
    """
    Shared functionality for both sync and async experiments managers.
    """

    _config: BrokleConfig

    def _log(self, message: str, *args: Any) -> None:
        """Log debug messages."""
        if self._config.debug:
            print(f"[Brokle Experiments] {message}", *args)


class ExperimentsManager(_BaseExperimentsManagerMixin):
    """
    Sync experiments manager for Brokle.

    All methods are synchronous. Uses SyncHTTPClient (httpx.Client) internally.

    Example:
        >>> from brokle import Brokle
        >>> from brokle.scorers import ExactMatch
        >>>
        >>> client = Brokle(api_key="bk_...")
        >>> dataset = client.datasets.get("dataset_id")
        >>>
        >>> def my_task(input):
        ...     return f"Response to: {input['question']}"
        >>>
        >>> results = client.experiments.run(
        ...     name="test-experiment",
        ...     dataset=dataset,
        ...     task=my_task,
        ...     scorers=[ExactMatch()],
        ... )
        >>> print(results.summary)
    """

    def __init__(
        self,
        http_client: SyncHTTPClient,
        config: BrokleConfig,
    ):
        """
        Initialize sync experiments manager.

        Args:
            http_client: Sync HTTP client
            config: Brokle configuration
        """
        self._http = http_client
        self._config = config

    def run(
        self,
        name: str,
        scorers: List[ScorerProtocol],
        *,
        # Dataset-based evaluation (existing)
        dataset: Optional[Union[Dataset, str]] = None,
        task: Optional[TaskFunction] = None,
        # Span-based evaluation (THE WEDGE - new)
        spans: Optional[List[QueriedSpan]] = None,
        extract_input: Optional[SpanExtractInput] = None,
        extract_output: Optional[SpanExtractOutput] = None,
        extract_expected: Optional[SpanExtractExpected] = None,
        # Common options
        max_concurrency: int = 10,
        trial_count: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> EvaluationResults:
        """
        Run an evaluation experiment.

        Two modes:
        - Dataset-based: Provide dataset + task
        - Span-based (THE WEDGE): Provide spans + extract_input + extract_output

        Args:
            name: Experiment name
            scorers: List of scorer callables

            # Dataset-based parameters
            dataset: Dataset object or dataset ID (for dataset-based)
            task: Function that takes input dict and returns output (for dataset-based)

            # Span-based parameters (THE WEDGE)
            spans: List of QueriedSpan objects (for span-based)
            extract_input: Function to extract input from span (required for span-based)
            extract_output: Function to extract output from span (required for span-based)
            extract_expected: Function to extract expected from span (optional)

            # Common options
            max_concurrency: Maximum parallel executions (default: 10)
            trial_count: Number of times to run each item (default: 1, only for dataset-based)
            metadata: Optional experiment metadata
            on_progress: Optional callback for progress updates

        Returns:
            EvaluationResults with summary statistics and all items

        Raises:
            EvaluationError: If experiment creation or submission fails

        Example (dataset-based):
            >>> results = client.experiments.run(
            ...     name="gpt4-accuracy",
            ...     dataset=dataset,
            ...     task=lambda x: call_gpt4(x["prompt"]),
            ...     scorers=[ExactMatch(), Contains()],
            ... )

        Example (span-based - THE WEDGE):
            >>> spans = client.query.query(filter="gen_ai.provider.name=openai").spans
            >>> results = client.experiments.run(
            ...     name="retrospective-analysis",
            ...     spans=spans,
            ...     scorers=[Relevance()],
            ...     extract_input=lambda s: {"prompt": s.input},
            ...     extract_output=lambda s: s.output,
            ... )
        """
        # Validate mode
        if dataset is not None and spans is not None:
            raise EvaluationError(
                "Cannot specify both 'dataset' and 'spans'. Choose one mode."
            )

        if spans is not None:
            # Span-based evaluation (THE WEDGE)
            if extract_input is None:
                raise EvaluationError("'extract_input' is required when using 'spans'")
            if extract_output is None:
                raise EvaluationError("'extract_output' is required when using 'spans'")

            return self._run_span_based(
                name=name,
                spans=spans,
                extract_input=extract_input,
                extract_output=extract_output,
                extract_expected=extract_expected,
                scorers=scorers,
                max_concurrency=max_concurrency,
                metadata=metadata,
                on_progress=on_progress,
            )

        # Dataset-based evaluation (existing logic)
        if dataset is None:
            raise EvaluationError("Either 'dataset' or 'spans' must be provided")
        if task is None:
            raise EvaluationError("'task' is required when using 'dataset'")

        self._log(f"Starting experiment: {name}")

        # 1. Resolve dataset
        if isinstance(dataset, str):
            self._log(f"Fetching dataset: {dataset}")
            resolved_dataset = self._fetch_dataset(dataset)
        else:
            resolved_dataset = dataset

        # 2. Collect dataset items FIRST (before creating experiment)
        items_list: List[DatasetItem] = list(resolved_dataset)
        if not items_list:
            self._log("Dataset is empty, returning early")
            return EvaluationResults(
                experiment_id="",
                experiment_name=name,
                dataset_id=resolved_dataset.id,
                summary={},
                items=[],
                url=None,
            )

        # 3. Create experiment via API (only if we have items)
        experiment = self._create_experiment(
            name=name,
            dataset_id=resolved_dataset.id,
            metadata=metadata,
        )
        self._log(f"Created experiment: {experiment.id}")

        # 4. Flatten items with trials
        work_items: List[tuple[DatasetItem, int]] = []
        for item in items_list:
            for trial in range(1, trial_count + 1):
                work_items.append((item, trial))

        total = len(work_items)
        completed = 0
        results: List[EvaluationItem] = []

        # 5. Process items with ThreadPoolExecutor
        def process_item(
            dataset_item: DatasetItem, trial_number: int
        ) -> EvaluationItem:
            input_data = dataset_item.input
            expected = dataset_item.expected

            # Run task
            try:
                output = task(input_data)
            except Exception as e:
                return EvaluationItem(
                    input=input_data,
                    output=None,
                    expected=expected,
                    scores=[],
                    trial_number=trial_number,
                    error=f"Task failed: {str(e)}",
                    dataset_item_id=dataset_item.id,
                )

            # Run scorers
            all_scores: List[ScoreResult] = []
            for scorer in scorers:
                scorer_results = _run_scorer_safe(scorer, output, expected, input_data)
                all_scores.extend(scorer_results)

            return EvaluationItem(
                input=input_data,
                output=output,
                expected=expected,
                scores=all_scores,
                trial_number=trial_number,
                dataset_item_id=dataset_item.id,
            )

        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = {
                executor.submit(process_item, item, trial): (item, trial)
                for item, trial in work_items
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1
                if on_progress:
                    on_progress(completed, total)

        # 6. Submit items to API
        self._submit_items(experiment.id, results)

        # 7. Update experiment status
        self._update_experiment_status(experiment.id, "completed")

        # 8. Compute summary
        summary = _compute_summary(results)

        # 9. Return results
        return EvaluationResults(
            experiment_id=experiment.id,
            experiment_name=name,
            dataset_id=resolved_dataset.id,
            summary=summary,
            items=results,
            url=self._get_experiment_url(experiment.id),
        )

    def get(self, experiment_id: str) -> Experiment:
        """
        Get an existing experiment by ID.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment object

        Raises:
            EvaluationError: If the API request fails or experiment not found

        Example:
            >>> experiment = client.experiments.get("exp_123")
            >>> print(experiment.status)
        """
        self._log(f"Getting experiment: {experiment_id}")

        try:
            raw_response = self._http.get(f"/v1/experiments/{experiment_id}")
            data = raw_response
            return Experiment.from_dict(data)
        except ValueError as e:
            raise EvaluationError(f"Failed to get experiment: {e}")
        except Exception as e:
            raise EvaluationError(f"Failed to get experiment: {e}")

    def list(
        self,
        limit: int = 50,
        page: int = 1,
    ) -> List[Experiment]:
        """
        List all experiments.

        Args:
            limit: Maximum number of experiments to return (default: 50, valid: 10, 25, 50, 100)
            page: Page number to fetch (default: 1, 1-indexed)

        Returns:
            List of Experiment objects

        Raises:
            EvaluationError: If the API request fails

        Example:
            >>> experiments = client.experiments.list(limit=10)
            >>> for exp in experiments:
            ...     print(exp.name, exp.status)
        """
        self._log(f"Listing experiments: limit={limit}, page={page}")

        try:
            raw_response = self._http.get(
                "/v1/experiments",
                params={"limit": limit, "page": page},
            )
            data = raw_response["data"]
            return [Experiment.from_dict(exp) for exp in data]
        except ValueError as e:
            raise EvaluationError(f"Failed to list experiments: {e}")
        except Exception as e:
            raise EvaluationError(f"Failed to list experiments: {e}")

    def compare(
        self,
        experiment_ids: List[str],
        baseline_id: Optional[str] = None,
    ) -> ComparisonResult:
        """
        Compare multiple experiments.

        Compares score metrics across experiments. Optionally specify a baseline
        for calculating score differences.

        Args:
            experiment_ids: List of experiment IDs to compare (2-10 experiments)
            baseline_id: Optional baseline experiment ID for diff calculations

        Returns:
            ComparisonResult with score aggregations and diffs

        Raises:
            EvaluationError: If the API request fails or experiments not found

        Example:
            >>> result = client.experiments.compare(
            ...     experiment_ids=["exp_1", "exp_2", "exp_3"],
            ...     baseline_id="exp_1",
            ... )
            >>> for scorer, exp_scores in result.scores.items():
            ...     print(f"{scorer}:")
            ...     for exp_id, stats in exp_scores.items():
            ...         print(f"  {exp_id}: mean={stats['mean']:.3f}")
        """
        self._log(f"Comparing experiments: {experiment_ids}")

        if len(experiment_ids) < 2:
            raise EvaluationError("At least 2 experiments are required for comparison")
        if len(experiment_ids) > 10:
            raise EvaluationError("Maximum 10 experiments can be compared at once")

        payload: Dict[str, Any] = {"experiment_ids": experiment_ids}
        if baseline_id:
            payload["baseline_id"] = baseline_id

        try:
            raw_response = self._http.post("/v1/experiments/compare", json=payload)
            data = raw_response
            return ComparisonResult.from_dict(data)
        except ValueError as e:
            raise EvaluationError(f"Failed to compare experiments: {e}")
        except Exception as e:
            raise EvaluationError(f"Failed to compare experiments: {e}")

    def rerun(
        self,
        experiment_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Experiment:
        """
        Re-run an experiment.

        Creates a new experiment based on an existing one, using the same dataset.
        The new experiment starts in pending status, ready for the SDK to run
        with a new task function.

        Args:
            experiment_id: Source experiment ID to re-run
            name: Optional new name (defaults to "{original_name}-rerun-{timestamp}")
            description: Optional new description
            metadata: Optional new metadata

        Returns:
            New Experiment object in pending status

        Raises:
            EvaluationError: If the API request fails or source experiment not found

        Example:
            >>> # Re-run with same configuration
            >>> new_exp = client.experiments.rerun("exp_123")
            >>> print(new_exp.id, new_exp.status)  # new ID, "pending"
            >>>
            >>> # Re-run with custom name
            >>> new_exp = client.experiments.rerun(
            ...     "exp_123",
            ...     name="gpt4-retest-v2",
            ... )
        """
        self._log(f"Re-running experiment: {experiment_id}")

        payload: Dict[str, Any] = {}
        if name:
            payload["name"] = name
        if description:
            payload["description"] = description
        if metadata:
            payload["metadata"] = metadata

        try:
            raw_response = self._http.post(
                f"/v1/experiments/{experiment_id}/rerun",
                json=payload,
            )
            data = raw_response
            return Experiment.from_dict(data)
        except ValueError as e:
            raise EvaluationError(f"Failed to re-run experiment: {e}")
        except Exception as e:
            raise EvaluationError(f"Failed to re-run experiment: {e}")

    def _fetch_dataset(self, dataset_id: str) -> Dataset:
        """Fetch dataset by ID."""
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
        except Exception as e:
            raise EvaluationError(f"Failed to fetch dataset: {e}")

    def _create_experiment(
        self,
        name: str,
        dataset_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Experiment:
        """Create a new experiment via API."""
        payload: Dict[str, Any] = {
            "name": name,
            "dataset_id": dataset_id,
            "status": "running",
        }
        if metadata:
            payload["metadata"] = metadata

        try:
            raw_response = self._http.post("/v1/experiments", json=payload)
            data = raw_response
            return Experiment.from_dict(data)
        except Exception as e:
            raise EvaluationError(f"Failed to create experiment: {e}")

    def _submit_items(
        self,
        experiment_id: str,
        items: List[EvaluationItem],
    ) -> None:
        """Submit evaluation items to API."""
        if not items:
            return

        payload = {"items": [item.to_dict() for item in items]}

        try:
            self._http.post(f"/v1/experiments/{experiment_id}/items", json=payload)
        except Exception as e:
            raise EvaluationError(f"Failed to submit items: {e}")

    def _update_experiment_status(
        self,
        experiment_id: str,
        status: str,
    ) -> None:
        """Update experiment status via API."""
        try:
            self._http.patch(
                f"/v1/experiments/{experiment_id}",
                json={"status": status},
            )
        except Exception as e:
            self._log(f"Failed to update experiment status: {e}")

    def _get_experiment_url(self, experiment_id: str) -> Optional[str]:
        """Generate dashboard URL for experiment."""
        base_url = self._config.base_url or ""
        if base_url.endswith("/api") or "/api" in base_url:
            dashboard_url = base_url.replace("/api", "")
        else:
            dashboard_url = base_url.replace(":8080", ":3000")
        return f"{dashboard_url}/experiments/{experiment_id}"

    def _create_experiment_for_spans(
        self,
        name: str,
        span_count: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Experiment:
        """Create a new experiment for span-based evaluation via API."""
        payload: Dict[str, Any] = {
            "name": name,
            "status": "running",
            "source": "spans",
        }
        if metadata:
            payload["metadata"] = {**(metadata or {}), "span_count": span_count}
        else:
            payload["metadata"] = {"span_count": span_count}

        try:
            raw_response = self._http.post("/v1/experiments", json=payload)
            data = raw_response
            return Experiment.from_dict(data)
        except Exception as e:
            raise EvaluationError(f"Failed to create experiment: {e}")

    def _run_span_based(
        self,
        name: str,
        spans: List[QueriedSpan],
        extract_input: SpanExtractInput,
        extract_output: SpanExtractOutput,
        extract_expected: Optional[SpanExtractExpected],
        scorers: List[ScorerProtocol],
        max_concurrency: int,
        metadata: Optional[Dict[str, Any]],
        on_progress: Optional[ProgressCallback],
    ) -> EvaluationResults:
        """
        Run span-based evaluation (THE WEDGE).

        Evaluates existing production spans without re-instrumenting applications.
        """
        self._log(f"Starting span-based experiment: {name} ({len(spans)} spans)")

        if not spans:
            self._log("No spans provided, returning early")
            return EvaluationResults(
                experiment_id="",
                experiment_name=name,
                summary={},
                items=[],
                url=None,
                source="spans",
            )

        # 1. Create experiment (without dataset_id)
        experiment = self._create_experiment_for_spans(name, len(spans), metadata)
        self._log(f"Created experiment: {experiment.id}")

        # 2. Process spans
        total = len(spans)
        completed = 0
        results: List[EvaluationItem] = []

        def process_span(span: QueriedSpan) -> EvaluationItem:
            # Extract data from span
            try:
                input_data = extract_input(span)
                output = extract_output(span)
                expected = extract_expected(span) if extract_expected else None
            except Exception as e:
                return EvaluationItem(
                    input={},
                    output=None,
                    error=f"Extraction failed: {str(e)}",
                    span_id=span.span_id,
                )

            # Run scorers
            all_scores: List[ScoreResult] = []
            for scorer in scorers:
                scorer_results = _run_scorer_safe(scorer, output, expected, input_data)
                all_scores.extend(scorer_results)

            return EvaluationItem(
                input=input_data,
                output=output,
                expected=expected,
                scores=all_scores,
                trial_number=1,
                span_id=span.span_id,
            )

        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = {executor.submit(process_span, span): span for span in spans}

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1
                if on_progress:
                    on_progress(completed, total)

        # 3. Submit items to API
        self._submit_items(experiment.id, results)

        # 4. Update experiment status
        self._update_experiment_status(experiment.id, "completed")

        # 5. Compute summary
        summary = _compute_summary(results)

        # 6. Return results
        return EvaluationResults(
            experiment_id=experiment.id,
            experiment_name=name,
            summary=summary,
            items=results,
            url=self._get_experiment_url(experiment.id),
            source="spans",
        )


class AsyncExperimentsManager(_BaseExperimentsManagerMixin):
    """
    Async experiments manager for AsyncBrokle.

    All methods are async and return coroutines that must be awaited.
    Uses AsyncHTTPClient (httpx.AsyncClient) internally.

    Example:
        >>> async with AsyncBrokle(api_key="bk_...") as client:
        ...     results = await client.experiments.run(
        ...         name="test",
        ...         dataset=dataset,
        ...         task=my_task,
        ...         scorers=[ExactMatch()],
        ...     )
        ...     print(results.summary)
    """

    def __init__(
        self,
        http_client: AsyncHTTPClient,
        config: BrokleConfig,
    ):
        """
        Initialize async experiments manager.

        Args:
            http_client: Async HTTP client
            config: Brokle configuration
        """
        self._http = http_client
        self._config = config

    async def run(
        self,
        name: str,
        scorers: List[ScorerProtocol],
        *,
        # Dataset-based evaluation (existing)
        dataset: Optional[Union[AsyncDataset, str]] = None,
        task: Optional[Union[TaskFunction, AsyncTaskFunction]] = None,
        # Span-based evaluation (THE WEDGE - new)
        spans: Optional[List[QueriedSpan]] = None,
        extract_input: Optional[SpanExtractInput] = None,
        extract_output: Optional[SpanExtractOutput] = None,
        extract_expected: Optional[SpanExtractExpected] = None,
        # Common options
        max_concurrency: int = 10,
        trial_count: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> EvaluationResults:
        """
        Run an evaluation experiment (async).

        Two modes:
        - Dataset-based: Provide dataset + task
        - Span-based (THE WEDGE): Provide spans + extract_input + extract_output

        Args:
            name: Experiment name
            scorers: List of scorer callables

            # Dataset-based parameters
            dataset: AsyncDataset object or dataset ID (for dataset-based)
            task: Function (sync or async) that takes input dict and returns output (for dataset-based)

            # Span-based parameters (THE WEDGE)
            spans: List of QueriedSpan objects (for span-based)
            extract_input: Function to extract input from span (required for span-based)
            extract_output: Function to extract output from span (required for span-based)
            extract_expected: Function to extract expected from span (optional)

            # Common options
            max_concurrency: Maximum parallel executions (default: 10)
            trial_count: Number of times to run each item (default: 1, only for dataset-based)
            metadata: Optional experiment metadata
            on_progress: Optional callback for progress updates

        Returns:
            EvaluationResults with summary statistics and all items

        Raises:
            EvaluationError: If experiment creation or submission fails

        Example (dataset-based):
            >>> results = await client.experiments.run(
            ...     name="gpt4-accuracy",
            ...     dataset=dataset,
            ...     task=lambda x: call_gpt4(x["prompt"]),
            ...     scorers=[ExactMatch(), Contains()],
            ... )

        Example (span-based - THE WEDGE):
            >>> result = await client.query.query(filter="gen_ai.provider.name=openai")
            >>> results = await client.experiments.run(
            ...     name="retrospective-analysis",
            ...     spans=result.spans,
            ...     scorers=[Relevance()],
            ...     extract_input=lambda s: {"prompt": s.input},
            ...     extract_output=lambda s: s.output,
            ... )
        """
        # Validate mode
        if dataset is not None and spans is not None:
            raise EvaluationError(
                "Cannot specify both 'dataset' and 'spans'. Choose one mode."
            )

        if spans is not None:
            # Span-based evaluation (THE WEDGE)
            if extract_input is None:
                raise EvaluationError("'extract_input' is required when using 'spans'")
            if extract_output is None:
                raise EvaluationError("'extract_output' is required when using 'spans'")

            return await self._run_span_based(
                name=name,
                spans=spans,
                extract_input=extract_input,
                extract_output=extract_output,
                extract_expected=extract_expected,
                scorers=scorers,
                max_concurrency=max_concurrency,
                metadata=metadata,
                on_progress=on_progress,
            )

        # Dataset-based evaluation (existing logic)
        if dataset is None:
            raise EvaluationError("Either 'dataset' or 'spans' must be provided")
        if task is None:
            raise EvaluationError("'task' is required when using 'dataset'")

        self._log(f"Starting experiment: {name}")

        # 1. Resolve dataset
        if isinstance(dataset, str):
            self._log(f"Fetching dataset: {dataset}")
            resolved_dataset = await self._fetch_dataset(dataset)
        else:
            resolved_dataset = dataset

        # 2. Collect dataset items FIRST (before creating experiment)
        items_list: List[DatasetItem] = []
        async for item in resolved_dataset:
            items_list.append(item)

        if not items_list:
            self._log("Dataset is empty, returning early")
            return EvaluationResults(
                experiment_id="",
                experiment_name=name,
                dataset_id=resolved_dataset.id,
                summary={},
                items=[],
                url=None,
            )

        # 3. Create experiment via API (only if we have items)
        experiment = await self._create_experiment(
            name=name,
            dataset_id=resolved_dataset.id,
            metadata=metadata,
        )
        self._log(f"Created experiment: {experiment.id}")

        # 4. Flatten items with trials
        work_items: List[tuple[DatasetItem, int]] = []
        for item in items_list:
            for trial in range(1, trial_count + 1):
                work_items.append((item, trial))

        total = len(work_items)
        completed = 0
        results: List[EvaluationItem] = []
        lock = asyncio.Lock()

        # 5. Process items with asyncio.Semaphore
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_item(
            dataset_item: DatasetItem, trial_number: int
        ) -> EvaluationItem:
            nonlocal completed
            async with semaphore:
                input_data = dataset_item.input
                expected = dataset_item.expected

                # Run task
                try:
                    output = task(input_data)
                    if inspect.isawaitable(output):
                        output = await output
                except Exception as e:
                    async with lock:
                        completed += 1
                        if on_progress:
                            on_progress(completed, total)
                    return EvaluationItem(
                        input=input_data,
                        output=None,
                        expected=expected,
                        scores=[],
                        trial_number=trial_number,
                        error=f"Task failed: {str(e)}",
                        dataset_item_id=dataset_item.id,
                    )

                # Run scorers
                all_scores: List[ScoreResult] = []
                for scorer in scorers:
                    scorer_results = await _run_scorer_safe_async(
                        scorer, output, expected, input_data
                    )
                    all_scores.extend(scorer_results)

                async with lock:
                    completed += 1
                    if on_progress:
                        on_progress(completed, total)

                return EvaluationItem(
                    input=input_data,
                    output=output,
                    expected=expected,
                    scores=all_scores,
                    trial_number=trial_number,
                    dataset_item_id=dataset_item.id,
                )

        tasks = [process_item(item, trial) for item, trial in work_items]
        results = await asyncio.gather(*tasks)

        # 6. Submit items to API
        await self._submit_items(experiment.id, list(results))

        # 7. Update experiment status
        await self._update_experiment_status(experiment.id, "completed")

        # 8. Compute summary
        summary = _compute_summary(list(results))

        # 9. Return results
        return EvaluationResults(
            experiment_id=experiment.id,
            experiment_name=name,
            dataset_id=resolved_dataset.id,
            summary=summary,
            items=list(results),
            url=self._get_experiment_url(experiment.id),
        )

    async def get(self, experiment_id: str) -> Experiment:
        """
        Get an existing experiment by ID (async).

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment object

        Raises:
            EvaluationError: If the API request fails or experiment not found
        """
        self._log(f"Getting experiment: {experiment_id}")

        try:
            raw_response = await self._http.get(f"/v1/experiments/{experiment_id}")
            data = raw_response
            return Experiment.from_dict(data)
        except ValueError as e:
            raise EvaluationError(f"Failed to get experiment: {e}")
        except Exception as e:
            raise EvaluationError(f"Failed to get experiment: {e}")

    async def list(
        self,
        limit: int = 50,
        page: int = 1,
    ) -> List[Experiment]:
        """
        List all experiments (async).

        Args:
            limit: Maximum number of experiments to return (default: 50, valid: 10, 25, 50, 100)
            page: Page number to fetch (default: 1, 1-indexed)

        Returns:
            List of Experiment objects

        Raises:
            EvaluationError: If the API request fails
        """
        self._log(f"Listing experiments: limit={limit}, page={page}")

        try:
            raw_response = await self._http.get(
                "/v1/experiments",
                params={"limit": limit, "page": page},
            )
            data = raw_response["data"]
            return [Experiment.from_dict(exp) for exp in data]
        except ValueError as e:
            raise EvaluationError(f"Failed to list experiments: {e}")
        except Exception as e:
            raise EvaluationError(f"Failed to list experiments: {e}")

    async def compare(
        self,
        experiment_ids: List[str],
        baseline_id: Optional[str] = None,
    ) -> ComparisonResult:
        """
        Compare multiple experiments (async).

        Compares score metrics across experiments. Optionally specify a baseline
        for calculating score differences.

        Args:
            experiment_ids: List of experiment IDs to compare (2-10 experiments)
            baseline_id: Optional baseline experiment ID for diff calculations

        Returns:
            ComparisonResult with score aggregations and diffs

        Raises:
            EvaluationError: If the API request fails or experiments not found

        Example:
            >>> result = await client.experiments.compare(
            ...     experiment_ids=["exp_1", "exp_2", "exp_3"],
            ...     baseline_id="exp_1",
            ... )
            >>> for scorer, exp_scores in result.scores.items():
            ...     print(f"{scorer}:")
            ...     for exp_id, stats in exp_scores.items():
            ...         print(f"  {exp_id}: mean={stats['mean']:.3f}")
        """
        self._log(f"Comparing experiments: {experiment_ids}")

        if len(experiment_ids) < 2:
            raise EvaluationError("At least 2 experiments are required for comparison")
        if len(experiment_ids) > 10:
            raise EvaluationError("Maximum 10 experiments can be compared at once")

        payload: Dict[str, Any] = {"experiment_ids": experiment_ids}
        if baseline_id:
            payload["baseline_id"] = baseline_id

        try:
            raw_response = await self._http.post(
                "/v1/experiments/compare", json=payload
            )
            data = raw_response
            return ComparisonResult.from_dict(data)
        except ValueError as e:
            raise EvaluationError(f"Failed to compare experiments: {e}")
        except Exception as e:
            raise EvaluationError(f"Failed to compare experiments: {e}")

    async def rerun(
        self,
        experiment_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Experiment:
        """
        Re-run an experiment (async).

        Creates a new experiment based on an existing one, using the same dataset.
        The new experiment starts in pending status, ready for the SDK to run
        with a new task function.

        Args:
            experiment_id: Source experiment ID to re-run
            name: Optional new name (defaults to "{original_name}-rerun-{timestamp}")
            description: Optional new description
            metadata: Optional new metadata

        Returns:
            New Experiment object in pending status

        Raises:
            EvaluationError: If the API request fails or source experiment not found

        Example:
            >>> # Re-run with same configuration
            >>> new_exp = await client.experiments.rerun("exp_123")
            >>> print(new_exp.id, new_exp.status)  # new ID, "pending"
            >>>
            >>> # Re-run with custom name
            >>> new_exp = await client.experiments.rerun(
            ...     "exp_123",
            ...     name="gpt4-retest-v2",
            ... )
        """
        self._log(f"Re-running experiment: {experiment_id}")

        payload: Dict[str, Any] = {}
        if name:
            payload["name"] = name
        if description:
            payload["description"] = description
        if metadata:
            payload["metadata"] = metadata

        try:
            raw_response = await self._http.post(
                f"/v1/experiments/{experiment_id}/rerun",
                json=payload,
            )
            data = raw_response
            return Experiment.from_dict(data)
        except ValueError as e:
            raise EvaluationError(f"Failed to re-run experiment: {e}")
        except Exception as e:
            raise EvaluationError(f"Failed to re-run experiment: {e}")

    async def _fetch_dataset(self, dataset_id: str) -> AsyncDataset:
        """Fetch dataset by ID (async)."""
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
        except Exception as e:
            raise EvaluationError(f"Failed to fetch dataset: {e}")

    async def _create_experiment(
        self,
        name: str,
        dataset_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Experiment:
        """Create a new experiment via API (async)."""
        payload: Dict[str, Any] = {
            "name": name,
            "dataset_id": dataset_id,
            "status": "running",
        }
        if metadata:
            payload["metadata"] = metadata

        try:
            raw_response = await self._http.post("/v1/experiments", json=payload)
            data = raw_response
            return Experiment.from_dict(data)
        except Exception as e:
            raise EvaluationError(f"Failed to create experiment: {e}")

    async def _submit_items(
        self,
        experiment_id: str,
        items: List[EvaluationItem],
    ) -> None:
        """Submit evaluation items to API (async)."""
        if not items:
            return

        payload = {"items": [item.to_dict() for item in items]}

        try:
            await self._http.post(
                f"/v1/experiments/{experiment_id}/items", json=payload
            )
        except Exception as e:
            raise EvaluationError(f"Failed to submit items: {e}")

    async def _update_experiment_status(
        self,
        experiment_id: str,
        status: str,
    ) -> None:
        """Update experiment status via API (async)."""
        try:
            await self._http.patch(
                f"/v1/experiments/{experiment_id}",
                json={"status": status},
            )
        except Exception as e:
            self._log(f"Failed to update experiment status: {e}")

    def _get_experiment_url(self, experiment_id: str) -> Optional[str]:
        """Generate dashboard URL for experiment."""
        base_url = self._config.base_url or ""
        if base_url.endswith("/api") or "/api" in base_url:
            dashboard_url = base_url.replace("/api", "")
        else:
            dashboard_url = base_url.replace(":8080", ":3000")
        return f"{dashboard_url}/experiments/{experiment_id}"

    async def _create_experiment_for_spans(
        self,
        name: str,
        span_count: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Experiment:
        """Create a new experiment for span-based evaluation via API (async)."""
        payload: Dict[str, Any] = {
            "name": name,
            "status": "running",
            "source": "spans",
        }
        if metadata:
            payload["metadata"] = {**(metadata or {}), "span_count": span_count}
        else:
            payload["metadata"] = {"span_count": span_count}

        try:
            raw_response = await self._http.post("/v1/experiments", json=payload)
            data = raw_response
            return Experiment.from_dict(data)
        except Exception as e:
            raise EvaluationError(f"Failed to create experiment: {e}")

    async def _run_span_based(
        self,
        name: str,
        spans: List[QueriedSpan],
        extract_input: SpanExtractInput,
        extract_output: SpanExtractOutput,
        extract_expected: Optional[SpanExtractExpected],
        scorers: List[ScorerProtocol],
        max_concurrency: int,
        metadata: Optional[Dict[str, Any]],
        on_progress: Optional[ProgressCallback],
    ) -> EvaluationResults:
        """
        Run span-based evaluation (THE WEDGE) - async version.

        Evaluates existing production spans without re-instrumenting applications.
        """
        self._log(f"Starting span-based experiment: {name} ({len(spans)} spans)")

        if not spans:
            self._log("No spans provided, returning early")
            return EvaluationResults(
                experiment_id="",
                experiment_name=name,
                summary={},
                items=[],
                url=None,
                source="spans",
            )

        # 1. Create experiment (without dataset_id)
        experiment = await self._create_experiment_for_spans(name, len(spans), metadata)
        self._log(f"Created experiment: {experiment.id}")

        # 2. Process spans with async semaphore
        total = len(spans)
        completed = 0
        results: List[EvaluationItem] = []
        lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_span(span: QueriedSpan) -> EvaluationItem:
            nonlocal completed
            async with semaphore:
                # Extract data from span
                try:
                    input_data = extract_input(span)
                    output = extract_output(span)
                    expected = extract_expected(span) if extract_expected else None
                except Exception as e:
                    async with lock:
                        completed += 1
                        if on_progress:
                            on_progress(completed, total)
                    return EvaluationItem(
                        input={},
                        output=None,
                        error=f"Extraction failed: {str(e)}",
                        span_id=span.span_id,
                    )

                # Run scorers
                all_scores: List[ScoreResult] = []
                for scorer in scorers:
                    scorer_results = await _run_scorer_safe_async(
                        scorer, output, expected, input_data
                    )
                    all_scores.extend(scorer_results)

                async with lock:
                    completed += 1
                    if on_progress:
                        on_progress(completed, total)

                return EvaluationItem(
                    input=input_data,
                    output=output,
                    expected=expected,
                    scores=all_scores,
                    trial_number=1,
                    span_id=span.span_id,
                )

        tasks = [process_span(span) for span in spans]
        results = await asyncio.gather(*tasks)

        # 3. Submit items to API
        await self._submit_items(experiment.id, list(results))

        # 4. Update experiment status
        await self._update_experiment_status(experiment.id, "completed")

        # 5. Compute summary
        summary = _compute_summary(list(results))

        # 6. Return results
        return EvaluationResults(
            experiment_id=experiment.id,
            experiment_name=name,
            summary=summary,
            items=list(results),
            url=self._get_experiment_url(experiment.id),
            source="spans",
        )
