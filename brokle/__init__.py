"""
Brokle SDK - OpenTelemetry-native observability for AI applications.

Basic Usage:
    >>> from brokle import Brokle
    >>> client = Brokle(api_key="bk_your_secret")
    >>> with client.start_as_current_span("my-operation") as span:
    ...     span.set_attribute("output", "Hello, world!")
    >>> client.flush()

Singleton Pattern:
    >>> from brokle import get_client
    >>> client = get_client()  # Reads from BROKLE_* env vars

LLM Generation Tracking:
    >>> with client.start_as_current_generation(
    ...     name="chat", model="gpt-4", provider="openai"
    ... ) as gen:
    ...     response = openai_client.chat.completions.create(...)
    ...     gen.set_attribute("gen_ai.output.messages", [...])
"""

import importlib
from typing import TYPE_CHECKING

# Version info is always imported (small, always needed)
from .version import __version__, __version_info__

# Lazy loading configuration
# Format: "exported_name": ("module_path", "attribute_name")
_LAZY_MODULES: dict[str, tuple[str, str]] = {
    # Client
    "AsyncBrokle": ("._client", "AsyncBrokle"),
    "Brokle": ("._client", "Brokle"),
    "get_async_client": ("._client", "get_async_client"),
    "get_client": ("._client", "get_client"),
    "reset_async_client": ("._client", "reset_async_client"),
    "reset_client": ("._client", "reset_client"),
    "set_async_client": ("._client", "set_async_client"),
    "set_client": ("._client", "set_client"),
    "brokle_context": ("._client", "brokle_context"),
    "async_brokle_context": ("._client", "async_brokle_context"),
    # Sync utilities
    "run_sync": ("._utils.sync", "run_sync"),
    "run_sync_safely": ("._utils.sync", "run_sync_safely"),
    # Config
    "BrokleConfig": (".config", "BrokleConfig"),
    # Datasets
    "AsyncDataset": (".datasets", "AsyncDataset"),
    "AsyncDatasetsManager": (".datasets", "AsyncDatasetsManager"),
    "Dataset": (".datasets", "Dataset"),
    "DatasetData": (".datasets", "DatasetData"),
    "DatasetError": (".datasets", "DatasetError"),
    "DatasetItem": (".datasets", "DatasetItem"),
    "DatasetItemInput": (".datasets", "DatasetItemInput"),
    "DatasetsManager": (".datasets", "DatasetsManager"),
    # Decorators
    "observe": (".decorators", "observe"),
    # Experiments
    "AsyncExperimentsManager": (".experiments", "AsyncExperimentsManager"),
    "EvaluationError": (".experiments", "EvaluationError"),
    "EvaluationItem": (".experiments", "EvaluationItem"),
    "EvaluationResults": (".experiments", "EvaluationResults"),
    "Experiment": (".experiments", "Experiment"),
    "ExperimentsManager": (".experiments", "ExperimentsManager"),
    "ScorerExecutionError": (".experiments", "ScorerExecutionError"),
    "SummaryStats": (".experiments", "SummaryStats"),
    "TaskError": (".experiments", "TaskError"),
    # Evaluate
    "async_evaluate": (".evaluate", "async_evaluate"),
    "evaluate": (".evaluate", "evaluate"),
    # Experiments types
    "SpanExtractExpected": (".experiments.types", "SpanExtractExpected"),
    "SpanExtractInput": (".experiments.types", "SpanExtractInput"),
    "SpanExtractOutput": (".experiments.types", "SpanExtractOutput"),
    # Metrics
    "DURATION_BOUNDARIES": (".metrics", "DURATION_BOUNDARIES"),
    "GenAIMetrics": (".metrics", "GenAIMetrics"),
    "MetricNames": (".metrics", "MetricNames"),
    "TOKEN_BOUNDARIES": (".metrics", "TOKEN_BOUNDARIES"),
    "TTFT_BOUNDARIES": (".metrics", "TTFT_BOUNDARIES"),
    "create_genai_metrics": (".metrics", "create_genai_metrics"),
    # Observations
    "BrokleAgent": (".observations", "BrokleAgent"),
    "BrokleEvent": (".observations", "BrokleEvent"),
    "BrokleGeneration": (".observations", "BrokleGeneration"),
    "BrokleObservation": (".observations", "BrokleObservation"),
    "BrokleRetrieval": (".observations", "BrokleRetrieval"),
    "BrokleTool": (".observations", "BrokleTool"),
    "ObservationType": (".observations", "ObservationType"),
    # Prompts
    "AnthropicMessage": (".prompts", "AnthropicMessage"),
    "AnthropicRequest": (".prompts", "AnthropicRequest"),
    "AsyncPromptManager": (".prompts", "AsyncPromptManager"),
    "CacheEntry": (".prompts", "CacheEntry"),
    "CacheOptions": (".prompts", "CacheOptions"),
    "ChatFallback": (".prompts", "ChatFallback"),
    "ChatMessage": (".prompts", "ChatMessage"),
    "ChatTemplate": (".prompts", "ChatTemplate"),
    "Fallback": (".prompts", "Fallback"),
    "GetPromptOptions": (".prompts", "GetPromptOptions"),
    "ListPromptsOptions": (".prompts", "ListPromptsOptions"),
    "MessageRole": (".prompts", "MessageRole"),
    "ModelConfig": (".prompts", "ModelConfig"),
    "OpenAIMessage": (".prompts", "OpenAIMessage"),
    "PaginatedResponse": (".prompts", "PaginatedResponse"),
    "Pagination": (".prompts", "Pagination"),
    "Prompt": (".prompts", "Prompt"),
    "PromptCache": (".prompts", "PromptCache"),
    "PromptCompileError": (".prompts", "PromptCompileError"),
    "PromptConfig": (".prompts", "PromptConfig"),
    "PromptData": (".prompts", "PromptData"),
    "PromptError": (".prompts", "PromptError"),
    "PromptFetchError": (".prompts", "PromptFetchError"),
    "PromptManager": (".prompts", "PromptManager"),
    "PromptNotFoundError": (".prompts", "PromptNotFoundError"),
    "PromptType": (".prompts", "PromptType"),
    "PromptVersion": (".prompts", "PromptVersion"),
    "Template": (".prompts", "Template"),
    "TextFallback": (".prompts", "TextFallback"),
    "TextTemplate": (".prompts", "TextTemplate"),
    "UpsertPromptRequest": (".prompts", "UpsertPromptRequest"),
    "Variables": (".prompts", "Variables"),
    "compile_chat_template": (".prompts", "compile_chat_template"),
    "compile_template": (".prompts", "compile_template"),
    "compile_text_template": (".prompts", "compile_text_template"),
    "extract_variables": (".prompts", "extract_variables"),
    "get_compiled_content": (".prompts", "get_compiled_content"),
    "get_compiled_messages": (".prompts", "get_compiled_messages"),
    "is_chat_template": (".prompts", "is_chat_template"),
    "is_text_template": (".prompts", "is_text_template"),
    "validate_variables": (".prompts", "validate_variables"),
    # Query
    "AsyncQueryManager": (".query", "AsyncQueryManager"),
    "InvalidFilterError": (".query", "InvalidFilterError"),
    "QueriedSpan": (".query", "QueriedSpan"),
    "QueryManager": (".query", "QueryManager"),
    "QueryResult": (".query", "QueryResult"),
    "SpanEvent": (".query", "SpanEvent"),
    "TokenUsage": (".query", "TokenUsage"),
    "ValidationResult": (".query", "ValidationResult"),
    # Scorers
    "Contains": (".scorers", "Contains"),
    "ExactMatch": (".scorers", "ExactMatch"),
    "JSONValid": (".scorers", "JSONValid"),
    "LLMScorer": (".scorers", "LLMScorer"),
    "LengthCheck": (".scorers", "LengthCheck"),
    "RegexMatch": (".scorers", "RegexMatch"),
    "multi_scorer": (".scorers", "multi_scorer"),
    "scorer": (".scorers", "scorer"),
    # Scores
    "AsyncScoresManager": (".scores", "AsyncScoresManager"),
    "ScoreError": (".scores", "ScoreError"),
    "ScoreResult": (".scores", "ScoreResult"),
    "ScoreSource": (".scores", "ScoreSource"),
    "ScoreValue": (".scores", "ScoreValue"),
    "Scorer": (".scores", "Scorer"),
    "ScorerArgs": (".scores", "ScorerArgs"),
    "ScorerError": (".scores", "ScorerError"),
    "ScorerProtocol": (".scores", "ScorerProtocol"),
    "ScoresManager": (".scores", "ScoresManager"),
    # Streaming
    "StreamingAccumulator": (".streaming", "StreamingAccumulator"),
    "StreamingMetrics": (".streaming", "StreamingMetrics"),
    "StreamingResult": (".streaming", "StreamingResult"),
    # Transport
    "TransportType": (".transport", "TransportType"),
    "create_metric_exporter": (".transport", "create_metric_exporter"),
    "create_trace_exporter": (".transport", "create_trace_exporter"),
    # Types
    "Attrs": (".types", "Attrs"),
    "BrokleOtelSpanAttributes": (".types", "BrokleOtelSpanAttributes"),
    "LLMProvider": (".types", "LLMProvider"),
    "OperationType": (".types", "OperationType"),
    "SchemaURLs": (".types", "SchemaURLs"),
    "ScoreDataType": (".types", "ScoreDataType"),
    "ScoreType": (".scores", "ScoreType"),
    "SpanLevel": (".types", "SpanLevel"),
    "SpanType": (".types", "SpanType"),
    # Utilities
    "MaskingHelper": (".utils.masking", "MaskingHelper"),
    # Errors (enhanced with actionable guidance) - Langfuse pattern (no prefix)
    "BrokleError": ("._http.errors", "BrokleError"),
    "AuthenticationError": ("._http.errors", "AuthenticationError"),
    "ConnectionError": ("._http.errors", "ConnectionError"),
    "ValidationError": ("._http.errors", "ValidationError"),
    "RateLimitError": ("._http.errors", "RateLimitError"),
    "NotFoundError": ("._http.errors", "NotFoundError"),
    "ServerError": ("._http.errors", "ServerError"),
}

# Cache for loaded modules
_module_cache: dict[str, object] = {}


def __getattr__(name: str) -> object:
    """Lazy load attributes on first access."""
    if name in _LAZY_MODULES:
        if name in _module_cache:
            return _module_cache[name]

        module_path, attr_name = _LAZY_MODULES[name]
        module = importlib.import_module(module_path, __name__)
        attr = getattr(module, attr_name)
        _module_cache[name] = attr
        return attr

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return all public names including lazy-loaded ones."""
    return list(_LAZY_MODULES.keys()) + ["__version__", "__version_info__"]


# Wrappers are imported separately to avoid requiring provider SDKs
# Usage: from brokle.wrappers import wrap_openai, wrap_anthropic, wrap_google, etc.
# Available wrappers:
#   - wrap_openai, wrap_openai_async (OpenAI)
#   - wrap_anthropic, wrap_anthropic_async (Anthropic)
#   - wrap_azure_openai, wrap_azure_openai_async (Azure OpenAI)
#   - wrap_google (Google GenAI)
#   - wrap_mistral (Mistral AI)
#   - wrap_cohere (Cohere)
#   - wrap_bedrock (AWS Bedrock)

__all__ = [
    # Version
    "__version__",
    "__version_info__",
    # Client
    "Brokle",
    "AsyncBrokle",
    "BrokleConfig",
    "get_client",
    "set_client",
    "reset_client",
    "get_async_client",
    "set_async_client",
    "reset_async_client",
    "brokle_context",
    "async_brokle_context",
    # Sync Utilities
    "run_sync",
    "run_sync_safely",
    # Decorators
    "observe",
    # Types
    "BrokleOtelSpanAttributes",
    "Attrs",
    "SpanType",
    "SpanLevel",
    "LLMProvider",
    "OperationType",
    "ScoreType",
    "ScoreDataType",
    "SchemaURLs",
    # Metrics
    "GenAIMetrics",
    "create_genai_metrics",
    "MetricNames",
    "TOKEN_BOUNDARIES",
    "DURATION_BOUNDARIES",
    "TTFT_BOUNDARIES",
    # Transport
    "TransportType",
    "create_trace_exporter",
    "create_metric_exporter",
    # Streaming
    "StreamingAccumulator",
    "StreamingResult",
    "StreamingMetrics",
    # Observations
    "ObservationType",
    "BrokleObservation",
    "BrokleGeneration",
    "BrokleEvent",
    "BrokleAgent",
    "BrokleTool",
    "BrokleRetrieval",
    # Utilities
    "MaskingHelper",
    # Prompts
    "PromptManager",
    "AsyncPromptManager",
    "Prompt",
    "PromptCache",
    "CacheOptions",
    "PromptError",
    "PromptNotFoundError",
    "PromptCompileError",
    "PromptFetchError",
    "extract_variables",
    "compile_template",
    "compile_text_template",
    "compile_chat_template",
    "validate_variables",
    "is_text_template",
    "is_chat_template",
    "get_compiled_content",
    "get_compiled_messages",
    "PromptType",
    "MessageRole",
    "ChatMessage",
    "TextTemplate",
    "ChatTemplate",
    "Template",
    "ModelConfig",
    "PromptConfig",
    "PromptVersion",
    "PromptData",
    "GetPromptOptions",
    "ListPromptsOptions",
    "Pagination",
    "PaginatedResponse",
    "UpsertPromptRequest",
    "CacheEntry",
    "OpenAIMessage",
    "AnthropicMessage",
    "AnthropicRequest",
    "Variables",
    "Fallback",
    "TextFallback",
    "ChatFallback",
    # Datasets
    "DatasetsManager",
    "AsyncDatasetsManager",
    "Dataset",
    "AsyncDataset",
    "DatasetItem",
    "DatasetItemInput",
    "DatasetData",
    "DatasetError",
    # Scores
    "ScoresManager",
    "AsyncScoresManager",
    "ScoreType",
    "ScoreSource",
    "ScoreResult",
    "ScoreValue",
    "ScorerProtocol",
    "Scorer",
    "ScorerArgs",
    "ScoreError",
    "ScorerError",
    # Scorers
    "ExactMatch",
    "Contains",
    "RegexMatch",
    "JSONValid",
    "LengthCheck",
    "LLMScorer",
    "scorer",
    "multi_scorer",
    # Experiments
    "ExperimentsManager",
    "AsyncExperimentsManager",
    "EvaluationResults",
    "EvaluationItem",
    "SummaryStats",
    "Experiment",
    "EvaluationError",
    "TaskError",
    "ScorerExecutionError",
    # Query (THE WEDGE)
    "QueryManager",
    "AsyncQueryManager",
    "QueriedSpan",
    "QueryResult",
    "ValidationResult",
    "TokenUsage",
    "SpanEvent",
    "InvalidFilterError",
    # Span Extract Types (for span-based evaluation)
    "SpanExtractInput",
    "SpanExtractOutput",
    "SpanExtractExpected",
    # Top-level evaluate functions (competitor pattern)
    "evaluate",
    "async_evaluate",
    # Errors (enhanced with actionable guidance) - Langfuse pattern (no prefix)
    "BrokleError",
    "AuthenticationError",
    "ConnectionError",
    "ValidationError",
    "RateLimitError",
    "NotFoundError",
    "ServerError",
]

# TYPE_CHECKING block for IDE support
if TYPE_CHECKING:
    from ._client import (
        AsyncBrokle,
        Brokle,
        async_brokle_context,
        brokle_context,
        get_async_client,
        get_client,
        reset_async_client,
        reset_client,
        set_async_client,
        set_client,
    )
    from ._utils.sync import run_sync, run_sync_safely
    from .config import BrokleConfig
    from .datasets import (
        AsyncDataset,
        AsyncDatasetsManager,
        Dataset,
        DatasetData,
        DatasetError,
        DatasetItem,
        DatasetItemInput,
        DatasetsManager,
    )
    from .decorators import observe
    from .evaluate import async_evaluate, evaluate
    from .experiments import (
        AsyncExperimentsManager,
        EvaluationError,
        EvaluationItem,
        EvaluationResults,
        Experiment,
        ExperimentsManager,
        ScorerExecutionError,
        SummaryStats,
        TaskError,
    )
    from .experiments.types import (
        SpanExtractExpected,
        SpanExtractInput,
        SpanExtractOutput,
    )
    from .metrics import (
        DURATION_BOUNDARIES,
        TOKEN_BOUNDARIES,
        TTFT_BOUNDARIES,
        GenAIMetrics,
        MetricNames,
        create_genai_metrics,
    )
    from .observations import (
        BrokleAgent,
        BrokleEvent,
        BrokleGeneration,
        BrokleObservation,
        BrokleRetrieval,
        BrokleTool,
        ObservationType,
    )
    from .prompts import (
        AnthropicMessage,
        AnthropicRequest,
        AsyncPromptManager,
        CacheEntry,
        CacheOptions,
        ChatFallback,
        ChatMessage,
        ChatTemplate,
        Fallback,
        GetPromptOptions,
        ListPromptsOptions,
        MessageRole,
        ModelConfig,
        OpenAIMessage,
        PaginatedResponse,
        Pagination,
        Prompt,
        PromptCache,
        PromptCompileError,
        PromptConfig,
        PromptData,
        PromptError,
        PromptFetchError,
        PromptManager,
        PromptNotFoundError,
        PromptType,
        PromptVersion,
        Template,
        TextFallback,
        TextTemplate,
        UpsertPromptRequest,
        Variables,
        compile_chat_template,
        compile_template,
        compile_text_template,
        extract_variables,
        get_compiled_content,
        get_compiled_messages,
        is_chat_template,
        is_text_template,
        validate_variables,
    )
    from .query import (
        AsyncQueryManager,
        InvalidFilterError,
        QueriedSpan,
        QueryManager,
        QueryResult,
        SpanEvent,
        TokenUsage,
        ValidationResult,
    )
    from .scorers import (
        Contains,
        ExactMatch,
        JSONValid,
        LengthCheck,
        LLMScorer,
        RegexMatch,
        multi_scorer,
        scorer,
    )
    from .scores import (
        AsyncScoresManager,
        ScoreError,
        ScoreResult,
        ScoreSource,
        ScoreType,
        ScoreValue,
        Scorer,
        ScorerArgs,
        ScorerError,
        ScorerProtocol,
        ScoresManager,
    )
    from .streaming import (
        StreamingAccumulator,
        StreamingMetrics,
        StreamingResult,
    )
    from .transport import (
        TransportType,
        create_metric_exporter,
        create_trace_exporter,
    )
    from .types import (
        Attrs,
        BrokleOtelSpanAttributes,
        LLMProvider,
        OperationType,
        SchemaURLs,
        ScoreDataType,
        SpanLevel,
        SpanType,
    )
    from .utils.masking import MaskingHelper
    from ._http.errors import (
        AuthenticationError,
        BrokleError,
        ConnectionError,
        NotFoundError,
        RateLimitError,
        ServerError,
        ValidationError,
    )
