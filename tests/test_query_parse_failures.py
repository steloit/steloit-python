"""
Parse-wrapper tests for query module.

Locks the invariant from CLAUDE.md gotcha #30b: a malformed 2xx response
body (wrong shape) NEVER leaks an untyped exception past the manager
boundary. The shared `_ensure_dict` guard is the authoritative check —
every non-dict JSON type must surface as `BrokleError` with the raw
payload preserved in `.details["response"]`.
"""

import pytest

from brokle._http.errors import BrokleError
from brokle._http.errors import ValidationError as HTTPValidationError
from brokle.query._managers import (
    _ensure_dict,
    _is_invalid_filter_error,
    _parse_query_result,
    _parse_validation_result,
)


# ---- _ensure_dict ----------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        [],
        [{"spans": []}],
        "ok",
        42,
        3.14,
        True,
        False,
        None,
    ],
    ids=["empty-list", "list-of-dicts", "string", "int", "float", "bool-true", "bool-false", "null"],
)
def test_ensure_dict_rejects_non_dict(raw):
    with pytest.raises(BrokleError) as exc_info:
        _ensure_dict(raw, resource="test resource")

    err = exc_info.value
    assert "expected JSON object" in str(err)
    assert "test resource" in str(err)
    assert err.details["response"] == raw


def test_ensure_dict_accepts_dict():
    payload = {"any": "shape"}
    assert _ensure_dict(payload, resource="x") is payload


# ---- _parse_query_result ---------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [[], [{"spans": []}], "ok", 42, 3.14, True, None],
    ids=["empty-list", "list-of-dicts", "string", "int", "float", "bool", "null"],
)
def test_parse_query_result_rejects_non_dict(raw):
    with pytest.raises(BrokleError) as exc_info:
        _parse_query_result(raw)

    err = exc_info.value
    assert "expected JSON object" in str(err)
    assert err.details["response"] == raw


def test_parse_query_result_wraps_from_dict_exceptions(monkeypatch):
    # Exercise the broadened except clause: if from_dict raises anything
    # in (AttributeError, KeyError, TypeError, ValueError), the wrapper
    # must convert it to BrokleError with the original error chained.
    from brokle.query import types as types_mod

    def _broken_from_dict(cls, data):
        raise KeyError("missing 'spans'")

    monkeypatch.setattr(types_mod.QueryResult, "from_dict", classmethod(_broken_from_dict))

    with pytest.raises(BrokleError) as exc_info:
        _parse_query_result({"anything": "goes"})

    err = exc_info.value
    assert "Failed to parse query response" in str(err)
    assert isinstance(err.original_error, KeyError)
    assert err.details["response"] == {"anything": "goes"}


def test_parse_query_result_accepts_valid_payload():
    result = _parse_query_result(
        {
            "spans": [],
            "total_count": 0,
            "has_more": False,
        }
    )
    assert result.total == 0
    assert result.has_more is False
    assert result.spans == []


# ---- _parse_validation_result ---------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [[], "ok", 42, None, True],
    ids=["list", "string", "int", "null", "bool"],
)
def test_parse_validation_result_rejects_non_dict(raw):
    with pytest.raises(BrokleError) as exc_info:
        _parse_validation_result(raw)

    assert "expected JSON object" in str(exc_info.value)


def test_parse_validation_result_accepts_valid_payload():
    result = _parse_validation_result({"valid": True, "message": "ok"})
    assert result.valid is True
    assert result.message == "ok"


def test_parse_validation_result_accepts_invalid_payload():
    # Backend's "filter parsed OK, result is valid=False" shape.
    result = _parse_validation_result({"valid": False, "error": "bad syntax"})
    assert result.valid is False
    assert result.error == "bad syntax"


# ---- _is_invalid_filter_error ---------------------------------------------
#
# Discriminator for 422 sub-kinds on /v1/spans/query. Filter-parser
# rejection carries `error.code = "invalid_filter_expression"`; generic
# input validation (invalid limit/page/timestamp) carries
# `error.code = "validation_error"` (the default).


def _make_validation_error(body):
    """Build an HTTPValidationError matching what the shared client raises."""
    return HTTPValidationError("test", details={"response": body})


def test_is_invalid_filter_error_true_when_code_matches():
    err = _make_validation_error(
        {
            "error": {
                "type": "validation_error",
                "code": "invalid_filter_expression",
                "message": "invalid filter expression",
                "details": "unexpected token at position 5",
            }
        }
    )
    assert _is_invalid_filter_error(err) is True


def test_is_invalid_filter_error_false_on_generic_validation_code():
    err = _make_validation_error(
        {
            "error": {
                "type": "validation_error",
                "code": "validation_error",
                "message": "validation failed",
                "errors": [
                    {"location": "body.limit", "message": "expected number <= 1000"}
                ],
            }
        }
    )
    assert _is_invalid_filter_error(err) is False


def test_is_invalid_filter_error_false_on_missing_code():
    err = _make_validation_error(
        {"error": {"type": "validation_error", "message": "no code field"}}
    )
    assert _is_invalid_filter_error(err) is False


@pytest.mark.parametrize(
    "body",
    [
        None,
        "not a dict",
        [],
        {"error": "not a dict"},
        {"error": ["list"]},
        {"no_error_key": {}},
        {},
    ],
)
def test_is_invalid_filter_error_false_on_malformed_body(body):
    err = _make_validation_error(body)
    assert _is_invalid_filter_error(err) is False


def test_is_invalid_filter_error_handles_missing_details():
    # Shared client always populates details, but the classifier
    # defends against an empty mapping anyway.
    err = HTTPValidationError("test", details={})
    assert _is_invalid_filter_error(err) is False


# ---- query() integration: promotion gated on code -------------------------
#
# End-to-end: the two 422 sub-kinds round-trip to the documented
# exception types. Uses a stubbed HTTP client so we don't need the live
# backend to exercise the promotion logic.


class _StubSyncHTTPClient:
    def __init__(self, raises):
        self._raises = raises

    def post(self, path, json=None):
        raise self._raises


def _make_sync_query_manager(raises):
    from brokle.config import BrokleConfig
    from brokle.query._managers import QueryManager

    http = _StubSyncHTTPClient(raises)
    cfg = BrokleConfig(api_key="bk_" + "a" * 40)
    return QueryManager(http_client=http, config=cfg)


def test_query_filter_code_promotes_to_invalid_filter_error():
    from brokle.query.exceptions import InvalidFilterError

    err = HTTPValidationError(
        "invalid filter expression",
        details={
            "response": {
                "error": {
                    "type": "validation_error",
                    "code": "invalid_filter_expression",
                    "message": "invalid filter expression",
                    "details": "unexpected token",
                }
            }
        },
    )
    mgr = _make_sync_query_manager(err)

    with pytest.raises(InvalidFilterError) as exc_info:
        mgr.query(filter="service.name=")

    assert exc_info.value.filter == "service.name="


def test_query_generic_validation_propagates_as_validation_error():
    from brokle.query.exceptions import InvalidFilterError

    err = HTTPValidationError(
        "validation failed",
        details={
            "response": {
                "error": {
                    "type": "validation_error",
                    "code": "validation_error",
                    "message": "validation failed",
                    "errors": [
                        {
                            "location": "body.limit",
                            "message": "expected number <= 1000",
                            "value": 9999,
                        }
                    ],
                }
            }
        },
    )
    mgr = _make_sync_query_manager(err)

    with pytest.raises(HTTPValidationError) as exc_info:
        mgr.query(filter="service.name=x", limit=9999)

    # Not promoted — the caller sees the shared ValidationError with
    # per-field diagnostics preserved.
    assert not isinstance(exc_info.value, InvalidFilterError)
    errors_array = exc_info.value.details["response"]["error"]["errors"]
    assert errors_array[0]["location"] == "body.limit"


def test_query_missing_code_propagates_as_validation_error():
    from brokle.query.exceptions import InvalidFilterError

    err = HTTPValidationError(
        "validation failed",
        details={
            "response": {
                "error": {
                    "type": "validation_error",
                    # No `code` field — conservative default is "do not
                    # promote"; the caller gets the shared ValidationError.
                    "message": "some validation problem",
                }
            }
        },
    )
    mgr = _make_sync_query_manager(err)

    with pytest.raises(HTTPValidationError) as exc_info:
        mgr.query(filter="anything")

    assert not isinstance(exc_info.value, InvalidFilterError)
