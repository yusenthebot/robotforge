"""Shared fixtures. Rendering tests need an OpenGL context, which headless machines do not have."""

from __future__ import annotations

import os

import pytest

from robotforge.sim.render import render_unavailable_reason

# CI jobs that are meant to cover rendering set this, so a broken GL setup fails instead of skipping.
REQUIRE_GL_ENV = "ROBOTFORGE_REQUIRE_GL"


@pytest.fixture(scope="session")
def gl_reason() -> str | None:
    """Why offscreen rendering is unavailable on this machine, or None when it works."""
    return render_unavailable_reason()


@pytest.fixture
def needs_gl(gl_reason: str | None) -> None:
    if gl_reason is None:
        return
    if os.environ.get(REQUIRE_GL_ENV):
        pytest.fail(f"{REQUIRE_GL_ENV} is set but offscreen OpenGL is unavailable: {gl_reason}")
    pytest.skip(f"no offscreen OpenGL on this machine: {gl_reason}")
