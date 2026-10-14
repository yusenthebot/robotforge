"""The learning path: the catalog, its Chinese pages, and how MCP clients reach it (tool and prompts)."""

import json
import re
from pathlib import Path

import pytest
from mcp import Client

from robotforge.cli import find_free_port
from robotforge.lessons import LESSONS
from robotforge.mcp_server import build_server
from robotforge.playground import Playground

ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / "docs" / "lessons"
TOOL_CALL = re.compile(r"`([a-z_]+)\(")


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _text(result) -> str:
    return "".join(c.text for c in result.content if getattr(c, "type", "") == "text")


def test_catalog_is_numbered_in_order_with_unique_names():
    assert [lesson.number for lesson in LESSONS] == list(range(1, len(LESSONS) + 1))
    assert 6 <= len(LESSONS) <= 9
    assert len({lesson.slug for lesson in LESSONS}) == len(LESSONS)
    for lesson in LESSONS:
        assert lesson.ideas and lesson.steps and lesson.check and lesson.challenge


@pytest.mark.parametrize("lesson", LESSONS, ids=lambda lesson: lesson.slug)
def test_each_lesson_page_matches_the_catalog(lesson):
    page = ROOT / lesson.page
    assert page.exists(), f"missing lesson page {lesson.page}"
    text = page.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == f"# 第 {lesson.number} 课 {lesson.title_zh}"
    # The page quotes, word for word, what the learner can say; the agent's plan starts from the same request.
    say = lines.index("## 你可以这样说")
    assert f"> {lesson.ask}" in lines[say : say + 4]
    assert "—" not in text, "no em dashes in docs"


def test_the_index_links_every_lesson():
    index = (PAGES / "README.md").read_text(encoding="utf-8")
    for lesson in LESSONS:
        assert f"({Path(lesson.page).name})" in index


@pytest.mark.anyio
async def test_lesson_plans_only_call_tools_the_server_has():
    server = build_server(Playground.with_demo(), find_free_port(8790))
    async with Client(server) as client:
        tools = {t.name for t in (await client.list_tools()).tools}
    for lesson in LESSONS:
        called = {name for step in lesson.steps for name in TOOL_CALL.findall(step)}
        assert called, f"lesson {lesson.number} names no tools"
        missing = called - tools - set(lesson.optional_tools)
        assert not missing, f"lesson {lesson.number} uses unknown tools {missing}"


@pytest.mark.anyio
async def test_lessons_tool_lists_and_explains_lessons():
    server = build_server(Playground.with_demo(), find_free_port(8790))
    async with Client(server) as client:
        listed = json.loads(_text(await client.call_tool("lessons", {})))
        assert [entry["number"] for entry in listed["lessons"]] == [lesson.number for lesson in LESSONS]
        assert listed["lessons"][2]["ask"] == LESSONS[2].ask

        plan = json.loads(_text(await client.call_tool("lessons", {"number": 3})))
        assert plan["lesson"]["title_zh"] == LESSONS[2].title_zh
        assert plan["lesson"]["steps"] and plan["lesson"]["check"]
        assert plan["lesson"]["page"].endswith(LESSONS[2].page)
        assert "user's language" in plan["how_to_teach"]

        bad = await client.call_tool("lessons", {"number": 42})
        assert bad.is_error and f"from 1 to {len(LESSONS)}" in _text(bad)


@pytest.mark.anyio
async def test_every_lesson_is_an_mcp_prompt():
    server = build_server(Playground.with_demo(), find_free_port(8790))
    async with Client(server) as client:
        prompts = {p.name: p for p in (await client.list_prompts()).prompts}
        assert {"start", *(lesson.prompt_name for lesson in LESSONS)} <= set(prompts)
        assert prompts["lesson_4"].title.startswith("Lesson 4:")

        got = await client.get_prompt("lesson_4", {})
        body = got.messages[0].content.text
        assert got.messages[0].role == "user"
        assert LESSONS[3].ask in body and "pick_and_place" in body and "Check before you say it worked" in body

        start = (await client.get_prompt("start", {})).messages[0].content.text
        assert "lessons()" in start
