"""Pins the narwhals pooled engine's black-box acceptance result.

``POOLED_ENGINE`` defaults to ``"numpy"`` (ruling F14: the plan's earlier
drafts flipped the default; that was reverted because it fails 12 test
functions that assert on legacy ``PooledState`` internals the narwhals engine
does not have -- see ``tests/test_pooled_removal_manifest.py``'s "White-box
test section" for the exact classification). This file runs the black-box
subset with ``MLFORECAST_POOLED_ENGINE=narwhals`` in a SUBPROCESS, so no
existing test file needs modifying and no in-process engine reload is needed
(``mlforecast.pooled`` reads the env var once at import time).

Three things are pinned, matching the acceptance picture established for
Task 14:

* ``tests/test_pooled_sqlite_oracle.py`` is 510/510 under narwhals -- this is
  the independent SQL RANGE-window authority and the strongest acceptance
  evidence there is.
* ``tests/test_history_warmup.py`` is 22/22 under narwhals.
* Across the five pooled suites (``test_pooled.py``,
  ``test_pooled_sqlite_oracle.py``, ``test_pooled_keep_last_n_trim.py``,
  ``test_pooled_state_cleanup.py``, ``test_history_warmup.py``), the set of
  FAILING test *functions* is exactly the 12 named below -- not "at most 12".
  Asserting the exact set (rather than a subset/count bound) is what makes a
  NEW failure appear as a regression even if some existing failure gets
  fixed and its name should be removed from the pinned set instead.

Each subprocess run takes single-digit-to-tens of seconds (measured: sqlite
oracle ~10s, history_warmup ~1s, the five-suite run ~30s), so the whole file
finishes in well under the "couple of minutes" budget.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

POOLED_SUITE_FILES = [
    "tests/test_pooled.py",
    "tests/test_pooled_sqlite_oracle.py",
    "tests/test_pooled_keep_last_n_trim.py",
    "tests/test_pooled_state_cleanup.py",
    "tests/test_history_warmup.py",
]

# The exact 12 test functions that fail under the narwhals engine, per the
# empirical classification recorded in
# tests/test_pooled_removal_manifest.py::test_white_box_set_matches_blanket_narwhals_failures.
# Every one is white-box: it reads or mutates legacy ``PooledState``
# internals (``_ts_aggs``, byte-identity comparisons via ``time_index``,
# etc.) that the narwhals engine's ``NarwhalsPooledState`` does not carry,
# rather than asserting on forecast values.
KNOWN_WHITE_BOX_FAILURES = {
    "tests/test_pooled.py::test_fast_vs_slow_equivalence",
    "tests/test_pooled.py::test_fast_vs_slow_partition",
    "tests/test_pooled.py::test_fast_vs_slow_time_agg",
    "tests/test_pooled.py::test_partition_update_sparse_then_dense",
    "tests/test_pooled_keep_last_n_trim.py::test_g2_2_trim_equals_fit_on_truncated_slice",
    "tests/test_pooled_keep_last_n_trim.py::test_g2_2_trim_then_update_matches_fresh_then_update",
    "tests/test_pooled_keep_last_n_trim.py::test_g2_3_suffix_invariant_global",
    "tests/test_pooled_keep_last_n_trim.py::test_g2_4_expanding_and_ewm_states_keep_full_history",
    "tests/test_pooled_keep_last_n_trim.py::test_g2_4_mixed_finite_and_unbounded_state_not_trimmed",
    "tests/test_pooled_keep_last_n_trim.py::test_g2_4_offset_and_combine_respect_inner_transform",
    "tests/test_pooled_state_cleanup.py::test_backup_snapshot_restores_pooled_state_like_deepcopy",
    "tests/test_pooled_state_cleanup.py::test_snapshot_restore_after_dynamic_new_bucket",
}

EXPECTED_TOTAL = 897
EXPECTED_PASSED = 748
EXPECTED_FAILED = 149

_FAILED_LINE = re.compile(r"^FAILED (\S+)")
_PARAM_SUFFIX = re.compile(r"\[[^\]]*\]$")


def _run_pytest(paths, timeout=180):
    """Runs a subprocess pytest over ``paths`` with the narwhals engine.

    Uses ``sys.executable -m pytest`` rather than shelling out to ``uv run
    pytest`` again: when this test itself is invoked via ``uv run pytest``,
    ``sys.executable`` already points at the resolved venv interpreter, so
    this gets the identical environment without a second, slower dependency
    resolution. ``--no-cov`` is required for a narrow selection -- the repo's
    global ``--cov-fail-under=75`` addopt would otherwise fail these
    subprocesses regardless of whether their own tests pass (pre-existing,
    not introduced by this file).
    """
    env = dict(os.environ)
    env["MLFORECAST_POOLED_ENGINE"] = "narwhals"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *paths,
            "-q",
            "--no-cov",
            "-p",
            "no:cacheprovider",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc


def _failed_test_functions(stdout):
    """Extracts the set of failing ``path::function`` ids from a ``-q`` run's
    short summary, with parametrize suffixes (``[...]``) stripped so multiple
    failing cases of one parametrized test collapse to one function name --
    matching how the 12-name manifest above is expressed."""
    ids = set()
    for line in stdout.splitlines():
        m = _FAILED_LINE.match(line)
        if m:
            ids.add(_PARAM_SUFFIX.sub("", m.group(1)))
    return ids


def test_sqlite_oracle_is_510_of_510_under_narwhals():
    """The independent SQL RANGE-window authority must be fully green under
    the narwhals engine with no exceptions -- this is the strongest single
    piece of acceptance evidence for the new engine.

    Can fail: a real narwhals-engine regression against the SQL oracle (or
    someone weakening this assertion to "at least N passed") shows up as
    "510 passed" not appearing in stdout, or "failed"/"error" appearing.
    """
    proc = _run_pytest(["tests/test_pooled_sqlite_oracle.py"])
    assert "510 passed" in proc.stdout, (
        f"expected exactly 510 passed; got:\n{proc.stdout[-4000:]}\n{proc.stderr[-2000:]}"
    )
    assert " failed" not in proc.stdout
    assert " error" not in proc.stdout.lower()
    assert proc.returncode == 0


def test_history_warmup_is_22_of_22_under_narwhals():
    """Can fail the same way as the sqlite-oracle test above: a real
    regression or a weakened assertion both show up as "22 passed" missing
    or a failure/error line appearing."""
    proc = _run_pytest(["tests/test_history_warmup.py"])
    assert "22 passed" in proc.stdout, (
        f"expected exactly 22 passed; got:\n{proc.stdout[-4000:]}\n{proc.stderr[-2000:]}"
    )
    assert " failed" not in proc.stdout
    assert " error" not in proc.stdout.lower()
    assert proc.returncode == 0


def test_pooled_suite_narwhals_failures_are_exactly_the_known_white_box_set():
    """Pins the exact failing-function set across the five pooled suites.

    This is an EQUALITY check, not "at most 12": if a currently-failing
    white-box test starts passing (e.g. someone tightens
    ``_assert_state_byte_identical``, or the legacy engine is later deleted
    and the test rewritten), this fails just as loudly as a brand-new
    regression appearing -- both require a deliberate update here (and to
    ``tests/test_pooled_removal_manifest.py``'s manifest), not a silent pass.

    Verified this can fail: flipping the exact match to a subset check (or
    dropping a name from ``KNOWN_WHITE_BOX_FAILURES``) makes this fail
    against the current measured output, and injecting a fake extra
    ``FAILED`` line in a scratch copy of this parser reliably produces a
    non-empty "unexpected" diff.
    """
    proc = _run_pytest(POOLED_SUITE_FILES, timeout=180)
    failing = _failed_test_functions(proc.stdout)
    unexpected = failing - KNOWN_WHITE_BOX_FAILURES
    now_passing = KNOWN_WHITE_BOX_FAILURES - failing
    assert not unexpected and not now_passing, (
        "narwhals engine's failing-test-function set drifted from the "
        "pinned 12.\n"
        f"NEW failures (investigate as a regression): {sorted(unexpected)}\n"
        f"NO LONGER failing (update KNOWN_WHITE_BOX_FAILURES here and the "
        f"manifest in tests/test_pooled_removal_manifest.py together): "
        f"{sorted(now_passing)}\n"
        f"--- tail of subprocess stdout ---\n{proc.stdout[-6000:]}"
    )
    # Cross-checks the overall shape of the acceptance picture, not just the
    # failing-function set: 897 total cases, 748 passing, 149 failing
    # (spread unevenly across the 12 failing functions via parametrization).
    assert f"{EXPECTED_TOTAL - EXPECTED_FAILED} passed" in proc.stdout
    m = re.search(r"(\d+) failed, (\d+) passed", proc.stdout)
    assert m is not None, (
        f"could not find pytest summary line in:\n{proc.stdout[-2000:]}"
    )
    n_failed, n_passed = int(m.group(1)), int(m.group(2))
    assert (n_failed, n_passed) == (EXPECTED_FAILED, EXPECTED_PASSED), (
        f"expected {EXPECTED_FAILED} failed / {EXPECTED_PASSED} passed "
        f"({EXPECTED_TOTAL} total), got {n_failed} failed / {n_passed} passed"
    )
    assert n_failed + n_passed == EXPECTED_TOTAL
