"""Shared fixtures for extractor unit tests."""

import shutil
from pathlib import Path

import pytest


@pytest.fixture
def sandbox(request: pytest.FixtureRequest) -> Path:
    """
    Isolated temporary directory scoped to one test.

    Each test gets its own folder (named after the module + test function) so
    tests that write real files to disk never interfere with each other.  The
    directory is removed both before (in case a previous run crashed) and after
    the test to keep the workspace clean.
    """
    module = request.node.module.__name__.split(".")[-1]
    name = request.node.name
    root = Path(f"_sandbox_{module}_{name}")
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    yield root
    shutil.rmtree(root, ignore_errors=True)
