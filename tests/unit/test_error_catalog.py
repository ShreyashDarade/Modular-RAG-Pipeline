"""The error catalog: one wire code, one exception class, in both transports."""

from __future__ import annotations

import pytest
from src.core import errors
from src.core.errors import NotFoundError, RagError, error_catalog, error_from_code


def _defined_codes() -> list[str]:
    return [cls.code for cls in errors._walk(RagError) if "code" in vars(cls)]


def test_every_built_in_class_defines_its_own_unique_code():
    codes = _defined_codes()
    assert len(codes) == len(set(codes)), "two built-in errors share a wire code"
    classes = [cls for cls in errors._walk(RagError) if cls.__module__ == errors.__name__]
    assert all("code" in vars(cls) for cls in classes), (
        "a built-in error inherits its parent's code, so the wire could not tell them apart: "
        + ", ".join(cls.__name__ for cls in classes if "code" not in vars(cls))
    )


def test_a_plug_in_subclass_that_inherits_a_code_does_not_take_it_over():
    class MyNotFound(NotFoundError):
        """A plug-in's own subtype: same wire code as its parent."""

    try:
        catalog = error_catalog()  # must not raise, and must not hand the code to the subclass
        assert catalog["not_found"] is NotFoundError
        assert type(error_from_code("not_found", "gone")) is NotFoundError
        assert MyNotFound not in catalog.values()
    finally:
        del MyNotFound


def test_a_plug_in_error_with_its_own_code_is_reachable_by_that_code():
    class QuotaExceeded(RagError):
        code = "plugin_quota_exceeded"
        status_code = 429

    assert error_catalog()["plugin_quota_exceeded"] is QuotaExceeded
    rebuilt = error_from_code("plugin_quota_exceeded", "slow down", status=429)
    assert type(rebuilt) is QuotaExceeded and rebuilt.status_code == 429


@pytest.mark.parametrize("code", sorted(_defined_codes()))
def test_every_code_round_trips_to_the_class_that_defines_it(code):
    cls = error_catalog()[code]
    assert cls.code == code
    rebuilt = error_from_code(code, "public message", request_id="r-1", details={"k": "v"})
    assert type(rebuilt) is cls and str(rebuilt) == "public message"
    assert rebuilt.request_id == "r-1" and dict(rebuilt.details) == {"k": "v"}
