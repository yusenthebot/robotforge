"""Keep native pointer gestures outside Chrome's temporary screenshot scaling."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from test_viewer_creation_browser import click


class NativeCapture:
    """One gate per tab; route every captured click and screenshot through it.

    Other RPCs remain concurrent. An interrupted operation leaves Chrome's state
    unknown: subsequent operations fail without dispatch, and the owner closes
    the tab. Cancelling a queued waiter does not invalidate the active owner.
    """

    def __init__(self, tab: Any) -> None:
        self._tab = tab
        self._lock = asyncio.Lock()
        self._failed = False

    @asynccontextmanager
    async def _operation(self) -> AsyncIterator[None]:
        async with self._lock:
            if self._failed:
                raise RuntimeError("Native capture interrupted; close this tab before further capture or input")
            try:
                yield
            except BaseException:
                self._failed = True
                raise

    async def click(self, selector: str | None = None, point: Sequence[float] | None = None) -> None:
        """Delegate the complete existing mouse/touch gesture, using CSS pixels."""
        async with self._operation():
            await click(self._tab, selector=selector, point=point)

    async def screenshot(self, **params: Any) -> dict[str, Any]:
        """Keep the gate until the native screenshot RPC has completed."""
        async with self._operation():
            return await self._tab.page.call("Page.captureScreenshot", **params)
