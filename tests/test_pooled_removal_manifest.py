"""Machine-checked inventory of the code iteration two removes.

The spec's removal manifest is prose; this is the executable form. In iteration
one every entry must still EXIST (proving the manifest is accurate). Flip
ITERATION to 2 after the deletion commit and the same test proves none remains.

Beyond the original brief, this file also records (per a standing ruling, see
progress.md Task 4 / task-14-brief.md) that 48 of the pooled suite's tests are
"white-box": they read or mutate legacy-engine internals rather than assert on
feature values, so they get deleted or rewritten alongside the legacy engine
in iteration two instead of being held up as an acceptance bar for narwhals.

That 48 figure turned out to be an order-of-magnitude estimate from an earlier
exploratory pass (raw substring occurrence counts across whole files, noted in
progress.md), not a rigorous per-test-function classification. Redoing it
properly here -- AST-classifying each test function's own source (plus, one
level deep, any same-file helper it calls) for references to the seven named
legacy ``PooledState`` fields -- measures 31 white-box / 116 black-box out of
147. That 31 was cross-checked empirically: running the pooled suite under
``MLFORECAST_POOLED_ENGINE=narwhals`` fails exactly 12 distinct test functions
(AttributeError on the ``_ts_aggs``/fast-slow split, or AssertionError
comparing legacy-only structural fields via the ``_assert_state_equal`` /
``_assert_state_byte_identical`` helpers), and all 12 are inside this 31.
The other 19 don't currently crash under narwhals -- some legacy fields
(``bucket_df``, ``time_index``, ...) already have narwhals-side analogues --
but they still assert on the legacy representation's shape rather than on
forecast values, so the ruling still routes them to iteration two.

We record and assert the number we actually measure (31), not the prose
estimate, per the same policy this file already applies to the hook and
wrapper counts below: the spec is prose, this file is ground truth, and a
mismatch means the spec drifted, not this test.
"""

import ast
from pathlib import Path

import pytest

ITERATION = 1

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "mlforecast"
TESTS = ROOT / "tests"

LEGACY_FILE = SRC / "_pooled_legacy.py"

HOOKS = {
    "_bucket_feature_from_aggs_impl": 10,
    "_bucket_feature_rows_impl": 6,
    "_latest_from_aggs_impl": 10,
    "_ts_level_from_aggs_impl": 10,
}
WRAPPERS = {
    "_compute_bucket_feature": 4,  # brief said 5; verified 4 at this commit, see docstring below
    "_compute_latest_from_aggs": 4,
    "_compute_ts_level_from_aggs": 3,
    "_compute_bucket_feature_collapsed": 1,
    "_maybe_reagg": 1,
    "_pooled_time_agg": 2,
}
FREE_HELPERS = [
    "_build_sparse_table",
    "_query_sparse_table",
    "_rolling_mean_from_agg",
    "_rolling_std_from_agg",
    "_rolling_min_from_agg",
    "_rolling_max_from_agg",
    "_expanding_mean_from_agg",
    "_expanding_std_from_agg",
    "_expanding_min_from_agg",
    "_expanding_max_from_agg",
    "_ewm_from_agg",
]


def _def_counts(path):
    tree = ast.parse(path.read_text())
    counts = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            counts[node.name] = counts.get(node.name, 0) + 1
    return counts


@pytest.mark.parametrize("name,expected", sorted({**HOOKS, **WRAPPERS}.items()))
def test_hook_definition_counts(name, expected):
    """NOTE: ``_compute_bucket_feature`` is 4 at this commit, not the 5 the
    original brief recorded. Verified by grepping `def _compute_bucket_feature`
    in mlforecast/lag_transforms.py: it is defined on `_BaseLagTransform`,
    `LookupLag`, `Offset`, and `Combine` (4 classes) -- lag_transforms.py has
    drifted since the brief was written, so this test uses the real count."""
    counts = _def_counts(SRC / "lag_transforms.py")
    got = counts.get(name, 0)
    if ITERATION == 1:
        assert got == expected, (
            f"{name}: manifest says {expected} definitions, found {got}. "
            "Update the spec's removal manifest and this test together."
        )
    else:
        assert got == 0, f"{name} survived iteration two ({got} definitions)"


@pytest.mark.parametrize("name", FREE_HELPERS)
def test_free_helpers(name):
    counts = _def_counts(SRC / "lag_transforms.py")
    if ITERATION == 1:
        assert counts.get(name, 0) >= 1, f"{name} missing; manifest is stale"
    else:
        assert counts.get(name, 0) == 0, f"{name} survived iteration two"


def test_legacy_engine_file():
    if ITERATION == 1:
        assert LEGACY_FILE.exists()
        assert len(LEGACY_FILE.read_text().splitlines()) > 1000
    else:
        assert not LEGACY_FILE.exists(), "_pooled_legacy.py survived iteration two"


def test_migration_module_is_retained_not_removed():
    """``_pooled_migrate.py`` is explicitly OUT of scope for this removal.

    It is a migration utility for models pickled before the narwhals engine
    existed (pre-narwhals ``PooledState`` snapshots), not part of the legacy
    *compute* engine this manifest tracks. It has its own end of life -- once
    the deprecation window for loading pre-narwhals saved models closes -- on
    a schedule independent of the iteration-two engine deletion. Both
    iterations must therefore find it present.
    """
    assert (SRC / "_pooled_migrate.py").exists()


def test_engine_switch_is_scaffolding():
    import mlforecast.pooled as mp

    if ITERATION == 1:
        assert hasattr(mp, "POOLED_ENGINE")
    else:
        assert not hasattr(mp, "POOLED_ENGINE"), "engine switch survived iteration two"


# === White-box test section ===============================================
#
# "White-box" here means: the test's own source, or a same-file helper it
# calls (one level of indirection -- e.g. ``_assert_state_equal`` /
# ``_assert_state_byte_identical``, which compare PooledState field-by-field
# via ``getattr(state, "bucket_id")`` and friends), references one of the
# seven fields below that are specific to the legacy ``PooledState``
# representation (``_pooled_legacy.py``) -- as attribute access
# (``state._ts_aggs``), a bare name (imported directly off ``PooledState``),
# or a constructor/dict keyword (``PooledState(bucket_id=...)``). The
# dominant pattern is ``state._ts_aggs = {}``, which forces the legacy engine
# off its O(1) incremental fast path and onto its from-scratch slow path -- a
# bifurcation that does not exist in the narwhals engine, so a test built to
# force it has nothing left to assert once the legacy engine is gone.
#
# The one-level indirection through helpers matters: without it, tests that
# call a shared byte-identity assertion helper (rather than naming the fields
# inline) are invisible to a purely-direct scan, but they still fail under a
# blanket narwhals run for exactly the reason every directly-flagged test
# does -- see the module docstring for the empirical cross-check.
LEGACY_STATE_FIELDS = {
    "_ts_aggs",
    "bucket_id",
    "time_index",
    "next_time_index_by_bucket",
    "bucket_df",
    "_idsorted_to_bucket_pos",
    "_parent_time_grids",
}

# The five files that make up "the pooled suite" referenced by the standing
# ruling. Chosen because they are the pre-narwhals-acceptance test files most
# tightly coupled to PooledState's legacy shape; test_pooled_differential.py,
# test_pooled_narwhals.py, and test_pooled_migration.py are new tests written
# *for* this design and are not part of what gets deleted.
POOLED_SUITE_FILES = [
    "test_pooled.py",
    "test_pooled_sqlite_oracle.py",
    "test_pooled_keep_last_n_trim.py",
    "test_pooled_state_cleanup.py",
    "test_history_warmup.py",
]


def _directly_references_legacy_state(func: ast.FunctionDef) -> bool:
    """True if `func`'s own body touches a legacy-only PooledState field, not
    counting helpers it calls. Scoped to `func`'s own AST subtree -- a
    reference buried in a module-level fixture is not attributed to every
    test that uses the fixture. This keeps the base case auditable: anyone
    can grep the field name inside the test body and see why it was flagged.
    """
    for node in ast.walk(func):
        if isinstance(node, ast.Attribute) and node.attr in LEGACY_STATE_FIELDS:
            return True
        if isinstance(node, ast.Name) and node.id in LEGACY_STATE_FIELDS:
            return True
        if isinstance(node, ast.keyword) and node.arg in LEGACY_STATE_FIELDS:
            return True
    return False


def _called_names(func: ast.FunctionDef) -> set:
    names = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, ast.Attribute):
                names.add(target.attr)
    return names


def _classify_pooled_suite():
    """Returns (total, per_file) computed fresh from disk.

    per_file maps each suite filename to the sorted list of its white-box test
    function names, so a mismatch against the recorded manifest below points
    directly at which file changed.

    A test function is white-box if it (or a same-file helper function it
    calls, one level of indirection resolved transitively through further
    same-file helpers) directly references a ``LEGACY_STATE_FIELDS`` name.
    """
    per_file = {}
    total = 0
    for fname in POOLED_SUITE_FILES:
        tree = ast.parse((TESTS / fname).read_text())
        all_funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        funcs_by_name: dict = {}
        for n in all_funcs:
            funcs_by_name.setdefault(n.name, []).append(n)
        direct = {n: _directly_references_legacy_state(n) for n in all_funcs}
        calls = {n: _called_names(n) for n in all_funcs}
        memo: dict = {}

        def is_white(node, stack=frozenset()):
            if node in memo:
                return memo[node]
            if node in stack:
                return False
            if direct[node]:
                memo[node] = True
                return True
            result = False
            for callee_name in calls[node]:
                for callee in funcs_by_name.get(callee_name, []):
                    if callee is node:
                        continue
                    if is_white(callee, stack | {node}):
                        result = True
                        break
                if result:
                    break
            memo[node] = result
            return result

        white = []
        for node in all_funcs:
            if node.name.startswith("test"):
                total += 1
                if is_white(node):
                    white.append(node.name)
        per_file[fname] = sorted(white)
    return total, per_file


# Recorded manifest: the exact white-box test names measured at this commit by
# running the classifier above. This is *not* hand-typed -- it is the printed
# output of `_classify_pooled_suite()`, pasted in so iteration two has a fixed
# list of names to check for absence once the source they reference is gone
# (at which point the classifier itself can no longer find them, since the
# fields it looks for will have been deleted along with the tests).
WHITE_BOX_TESTS = {
    "test_pooled.py": [
        "test_append_predictions_preserves_time_dtype",
        "test_categorical_groupby_update_with_new_group",
        "test_compute_pooled_features_raises_for_unsupported",
        "test_fast_vs_slow_equivalence",
        "test_fast_vs_slow_local_partition_with_nan",
        "test_fast_vs_slow_partition",
        "test_fast_vs_slow_time_agg",
        "test_global_partition_new_bucket_inherits_parent_calendar",
        "test_global_partition_update_advances_sibling_calendar",
        "test_global_sequential_updates",
        "test_global_update_preserves_bucket_df",
        "test_group_update_preserves_bucket_df",
        "test_groupby_partition_update_advances_sibling_calendar",
        "test_local_partition_update_advances_sibling_calendar",
        "test_new_partition_bucket_uses_existing_parent_calendar",
        "test_new_series_new_group_update_then_predict",
        "test_partition_backup_restore_with_dynamic_buckets",
        "test_partition_by_update",
        "test_partition_datetime_update_new_bucket",
        "test_partition_ordinals_have_parent_gaps",
        "test_partition_update_batch_multiple_ids_new_buckets",
        "test_partition_update_sparse_then_dense",
    ],
    "test_pooled_sqlite_oracle.py": [],
    "test_pooled_keep_last_n_trim.py": [
        "test_g2_2_trim_equals_fit_on_truncated_slice",
        "test_g2_2_trim_then_update_matches_fresh_then_update",
        "test_g2_3_suffix_invariant_global",
        "test_g2_4_expanding_and_ewm_states_keep_full_history",
        "test_g2_4_mixed_finite_and_unbounded_state_not_trimmed",
        "test_g2_4_offset_and_combine_respect_inner_transform",
    ],
    "test_pooled_state_cleanup.py": [
        "test_backup_snapshot_restores_pooled_state_like_deepcopy",
        "test_snapshot_restore_after_dynamic_new_bucket",
    ],
    "test_history_warmup.py": ["test_history_warmup_trims_like_fit"],
}

TOTAL_POOLED_TESTS = 147
WHITE_BOX_COUNT = 31
BLACK_BOX_COUNT = TOTAL_POOLED_TESTS - WHITE_BOX_COUNT  # 116


def test_pooled_suite_size_is_147():
    """Sanity-checks the population this section classifies. If this drifts,
    someone added/removed a test in the suite and the white-box counts below
    need to be regenerated, not just re-asserted."""
    total, _ = _classify_pooled_suite()
    assert total == TOTAL_POOLED_TESTS, (
        f"pooled suite now has {total} test functions across "
        f"{POOLED_SUITE_FILES}, not {TOTAL_POOLED_TESTS}. Regenerate "
        "WHITE_BOX_TESTS with _classify_pooled_suite() and update this file."
    )


@pytest.mark.parametrize("fname", POOLED_SUITE_FILES)
def test_white_box_manifest_matches_current_source(fname):
    """The recorded per-file white-box list must equal what the live
    classifier finds right now. This is the drift guard: if a test's
    reference to `_ts_aggs`/`bucket_id`/etc. is added, removed, or the test
    itself is renamed or deleted without updating WHITE_BOX_TESTS, this fails.
    """
    if ITERATION == 1:
        _, per_file = _classify_pooled_suite()
        assert per_file[fname] == WHITE_BOX_TESTS[fname], (
            f"live white-box classification of {fname} no longer matches the "
            "recorded manifest -- update WHITE_BOX_TESTS to match reality."
        )
    else:
        # The file may have been deleted outright, or rewritten in place with
        # the white-box tests removed; either way none of the recorded names
        # may still be a test function in it.
        path = TESTS / fname
        if not path.exists():
            return
        tree = ast.parse(path.read_text())
        current_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test")
        }
        survivors = sorted(current_names & set(WHITE_BOX_TESTS[fname]))
        assert not survivors, (
            f"{fname}: white-box tests survived iteration two: {survivors}"
        )


def test_white_box_count_is_31_not_the_estimated_48():
    """Checks the recorded manifest against its own totals -- exists so a
    change to WHITE_BOX_TESTS (e.g. someone adding an entry by hand without
    re-running the classifier) can't silently drift from the counts everyone
    will cite. See the module docstring for why this is 31, not the 48 from
    the original exploratory pass."""
    flat = sorted(name for names in WHITE_BOX_TESTS.values() for name in names)
    assert len(flat) == len(set(flat)), "duplicate test name across suite files"
    assert len(flat) == WHITE_BOX_COUNT
    assert TOTAL_POOLED_TESTS - WHITE_BOX_COUNT == BLACK_BOX_COUNT


def test_white_box_set_matches_blanket_narwhals_failures():
    """Empirical cross-check, not a re-derivation: every test function that
    actually fails when the pooled suite runs under
    ``MLFORECAST_POOLED_ENGINE=narwhals`` (AttributeError on a legacy-only
    field, or AssertionError inside a byte-identity helper comparing
    legacy-only structure) must be inside the recorded white-box set. This
    doesn't run the suite itself (that's `tests/test_pooled_narwhals.py`'s
    job, and doing it here would make this file slow and engine-order
    sensitive) -- it pins the 12 function names measured that way at the time
    this manifest was written, so a future edit to WHITE_BOX_TESTS can't
    accidentally drop one of the entries that's empirically load-bearing.
    """
    measured_failing_under_narwhals = {
        "test_fast_vs_slow_equivalence",
        "test_fast_vs_slow_partition",
        "test_fast_vs_slow_time_agg",
        "test_partition_update_sparse_then_dense",
        "test_g2_2_trim_equals_fit_on_truncated_slice",
        "test_g2_2_trim_then_update_matches_fresh_then_update",
        "test_g2_3_suffix_invariant_global",
        "test_g2_4_expanding_and_ewm_states_keep_full_history",
        "test_g2_4_mixed_finite_and_unbounded_state_not_trimmed",
        "test_g2_4_offset_and_combine_respect_inner_transform",
        "test_backup_snapshot_restores_pooled_state_like_deepcopy",
        "test_snapshot_restore_after_dynamic_new_bucket",
    }
    all_white = {name for names in WHITE_BOX_TESTS.values() for name in names}
    missing = measured_failing_under_narwhals - all_white
    assert not missing, (
        f"{missing} fail under the narwhals engine but are not in the "
        "recorded white-box manifest -- the manifest under-covers reality."
    )
