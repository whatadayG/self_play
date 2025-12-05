"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

import pytest

# Add project root to path so we can import modules
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add scripts directory to path
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture
def base_seed():
    """Base random seed for deterministic tests."""
    return 42


@pytest.fixture
def group_size():
    """Default GRPO group size for tests."""
    return 8


@pytest.fixture
def temp_test_dir(tmp_path):
    """Create a temporary directory for test outputs."""
    test_dir = tmp_path / "test_outputs"
    test_dir.mkdir(exist_ok=True)
    return test_dir
