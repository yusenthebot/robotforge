"""Operation-scoped preparation feedback; background operations never share callbacks."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_REPORT: ContextVar[Callable[[dict], None] | None] = ContextVar("viewer_preparation_progress", default=None)


@contextmanager
def reporting(callback: Callable[[dict], None]) -> Iterator[None]:
    token = _REPORT.set(callback)
    try:
        yield
    finally:
        _REPORT.reset(token)


def report(stage: str, fraction: float | None = None, message: str | None = None) -> None:
    if callback := _REPORT.get():
        callback({"stage": stage, **({"fraction": fraction} if fraction is not None else {})})


def download_callback(stage: str) -> Callable[[float, str], None]:
    """Bind before a downloader starts its threads, which have no ContextVar context."""
    callback = _REPORT.get()

    def emit(fraction: float, _message: str) -> None:
        if callback is not None:
            callback({"stage": stage, "fraction": fraction})

    return emit
