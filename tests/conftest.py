"""Test setup: use a throwaway working database so tests never touch data/working.sqlite."""
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_tmp = Path(tempfile.mkdtemp(prefix="influencer_tests_"))
os.environ["INFLUENCER_DB"] = str(_tmp / "test.sqlite")
os.environ.pop("ANTHROPIC_API_KEY", None)

import pytest  # noqa: E402


@pytest.fixture()
def fresh_db():
    """A fresh copy of the baseline database for tests that write; restored again afterwards so tests stay independent."""
    from core import db
    path = db.reset_db()
    yield path
    db.reset_db()


@pytest.fixture(scope="session")
def tmp_db_path():
    return _tmp / "test.sqlite"
