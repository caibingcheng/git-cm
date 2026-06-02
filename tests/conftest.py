"""pytest fixtures for git-cm tests."""

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def mock_config_dir(monkeypatch, temp_dir):
    """Mock the config directory to use a temp directory."""
    config_dir = temp_dir / ".config" / "git-cm"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # We need to patch the Config class to use this directory
    # This will be done in individual tests as needed
    return config_dir
