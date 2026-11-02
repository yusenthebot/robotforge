"""Capture exclusion must survive cancellation and preserve unrelated concurrency."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import viewer_capture
from viewer_capture import NativeCapture


@pytest.mark.parametrize("first", ["click", "screenshot"])
def test_complete_operations_exclude_each_other_and_forward_unchanged(monkeypatch, first):
    async def scenario():
        entered, release, queued = asyncio.Event(), asyncio.Event(), asyncio.Event()
        calls = []
        response = {"data": "original-native-image"}

        async def perform(kind, value):
            calls.append((kind, value))
            if kind == first:
                entered.set()
                await release.wait()
            return response

        async def click(tab, selector, point):
            assert tab is target
            await perform("click", (selector, point))

        async def call(method, **params):
            if method != "Page.captureScreenshot":
                return "unrelated response"
            return await perform("screenshot", params)

        monkeypatch.setattr(viewer_capture, "click", click)
        target = SimpleNamespace(page=SimpleNamespace(call=call))
        gate = NativeCapture(target)
        operations = {
            "click": lambda: gate.click("#button", point=[14.25, 27.5]),
            "screenshot": lambda: gate.screenshot(format="jpeg", quality=90, fromSurface=True),
        }
        owner = asyncio.create_task(operations[first]())
        await entered.wait()

        async def competitor():
            queued.set()
            return await operations["click" if first == "screenshot" else "screenshot"]()

        waiter = asyncio.create_task(competitor())
        await queued.wait()
        assert len(calls) == 1 and not waiter.done()
        assert await target.page.call("Runtime.evaluate", expression="1") == "unrelated response"
        release.set()
        results = await asyncio.gather(owner, waiter)
        assert results[0 if first == "screenshot" else 1] is response
        assert dict(calls) == {
            "click": ("#button", [14.25, 27.5]),
            "screenshot": {"format": "jpeg", "quality": 90, "fromSurface": True},
        }

    asyncio.run(asyncio.wait_for(scenario(), 3))


def test_separate_tabs_and_cancelled_waiter_do_not_block_valid_work(monkeypatch):
    async def scenario():
        entered, release, queued = asyncio.Event(), asyncio.Event(), asyncio.Event()
        calls = []

        async def call(method, **params):
            calls.append((method, params))
            if params.get("hold"):
                entered.set()
                await release.wait()
            return params

        target = SimpleNamespace(page=SimpleNamespace(call=call))
        gate, other = NativeCapture(target), NativeCapture(SimpleNamespace(page=target.page))
        owner = asyncio.create_task(gate.screenshot(hold=True))
        await entered.wait()
        assert await other.screenshot(format="png") == {"format": "png"}

        async def waiting():
            queued.set()
            await gate.screenshot(format="jpeg")

        waiter = asyncio.create_task(waiting())
        await queued.wait()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        release.set()
        await owner
        assert await gate.screenshot(quality=90) == {"quality": 90}
        assert len(calls) == 3

    asyncio.run(asyncio.wait_for(scenario(), 3))


@pytest.mark.parametrize("operation", ["click", "screenshot"])
@pytest.mark.parametrize("failure", ["cancel", "error"])
def test_interrupted_operation_rejects_queued_and_future_dispatch(monkeypatch, operation, failure):
    async def scenario():
        entered, release, queued = asyncio.Event(), asyncio.Event(), asyncio.Event()
        calls = []

        async def perform(kind):
            calls.append(kind)
            entered.set()
            await release.wait()
            raise ValueError("incomplete operation")

        async def click(tab, selector, point):
            await perform("click")

        async def call(method, **params):
            return await perform("screenshot")

        monkeypatch.setattr(viewer_capture, "click", click)
        gate = NativeCapture(SimpleNamespace(page=SimpleNamespace(call=call)))
        owner = asyncio.create_task(gate.click("#button") if operation == "click" else gate.screenshot())
        await entered.wait()

        async def waiting():
            queued.set()
            await gate.screenshot()

        waiter = asyncio.create_task(waiting())
        await queued.wait()
        if failure == "cancel":
            owner.cancel()
            with pytest.raises(asyncio.CancelledError):
                await owner
        else:
            release.set()
            with pytest.raises(ValueError, match="incomplete operation"):
                await owner
        with pytest.raises(RuntimeError, match="close this tab"):
            await waiter
        with pytest.raises(RuntimeError, match="close this tab"):
            await gate.click("#other")
        assert calls == [operation]

    asyncio.run(asyncio.wait_for(scenario(), 3))
