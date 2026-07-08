"""Version resolution for git-cm."""

import importlib.metadata
from pathlib import Path

from git import Repo

from git_cm.git_utils import get_git_version


def get_version() -> str:
    """Return the version string for git-cm.

    Tries to derive the version from version tags in the git-cm repository.
    Falls back to the installed package version, and finally to a static
    fallback if neither is available.
    """
    package_dir = Path(__file__).parent.resolve()
    try:
        repo = Repo(package_dir, search_parent_directories=True)
        return get_git_version(repo)
    except Exception:
        pass

    try:
        return importlib.metadata.version("git-cm")
    except Exception:
        return "0.0.0"
