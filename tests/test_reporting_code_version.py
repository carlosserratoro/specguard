import shutil
import subprocess

import pytest

from specguard.adapters.reporting import code_version

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    (tmp_path / "code.py").write_text("x = 1\n")
    _git(tmp_path, "add", "code.py")
    _git(tmp_path, "commit", "-qm", "init")
    return tmp_path


def test_clean_repo(tmp_path):
    repo = _repo(tmp_path)
    version = code_version.__wrapped__(repo)
    assert version["commit"] == _git(repo, "rev-parse", "HEAD")
    assert version["dirty"] is False


def test_modified_tracked_file_is_dirty(tmp_path):
    repo = _repo(tmp_path)
    (repo / "code.py").write_text("x = 2\n")
    assert code_version.__wrapped__(repo)["dirty"] is True


def test_untracked_results_dont_make_it_dirty(tmp_path):
    repo = _repo(tmp_path)
    (repo / "result.json").write_text("{}")
    assert code_version.__wrapped__(repo)["dirty"] is False


def test_outside_a_repo_returns_none(tmp_path):
    assert code_version.__wrapped__(tmp_path) is None
