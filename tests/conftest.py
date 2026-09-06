"""Test environment, established before anything imports audbre.

`audbre.config` reads the environment exactly once, at import time. Any
`import audbre.*` at module scope in a test file therefore freezes the config
before fixtures get a chance to run — so these have to be set here, in
conftest, which pytest imports first.

They are assigned rather than defaulted on purpose: a developer's local `.env`
must not be able to change what the suite does. Before this file existed the
tests picked up a real `AUDBRE_MODAL_URL` from `.env` and quietly ran against
the live GPU engine locally, while failing on CI where no `.env` exists.
"""
import os
import shutil
import tempfile

_STORAGE = tempfile.mkdtemp(prefix="audbre-tests-")

os.environ["AUDBRE_ENGINE"] = "stub"
os.environ["AUDBRE_STORAGE"] = _STORAGE
# Nothing in the suite may reach a real worker, whatever the developer has set.
os.environ.pop("AUDBRE_MODAL_URL", None)
os.environ.pop("AUDBRE_WORKER_TOKEN", None)


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_STORAGE, ignore_errors=True)
