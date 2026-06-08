"""Regression test for the Windows ('spawn') parallel-worker crash.

Running a parallel FDDC optimization under the 'spawn' start method used to fail
with "'NoneType' object is not callable" because worker processes re-import the
module fresh and the pool's worker globals stayed None. The fix sends picklable
worker callables and populates the globals via Pool(initializer=...).

The actual spawn run lives in tests/_spawn_worker_check.py and is executed in a
separate process so that pytest's import/collection does not interfere with
spawn's re-import of the main module.
"""

import subprocess
import sys
from pathlib import Path

CHECK_SCRIPT = Path(__file__).parent / "_spawn_worker_check.py"


def test_parallel_optimization_runs_under_spawn():
    proc = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT)],
        capture_output=True, text=True, timeout=180,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, (
        f"spawn parallel run failed (exit {proc.returncode}):\n{combined}")
    assert "SUCCESS" in proc.stdout, combined
    # The exact crash signature this fix removes.
    assert "is not callable" not in combined, combined
    assert "FAIL" not in combined, combined
