import os
import shutil
import tempfile
from pathlib import Path

TEST_ROOT = Path(tempfile.mkdtemp(prefix="researchflow-tests-"))
os.environ["DATABASE_URL"] = f"sqlite:///{(TEST_ROOT / 'test.db').as_posix()}"
os.environ["STORAGE_ROOT"] = str(TEST_ROOT / "storage")
os.environ["APP_ENV"] = "test"


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(TEST_ROOT, ignore_errors=True)
