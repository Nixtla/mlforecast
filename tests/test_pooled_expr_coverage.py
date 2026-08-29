"""Proves every pooled-capable transform has a narwhals expression.

Task 14 deliverable 3: with the legacy-fallback scaffolding removed from
``compute_pooled_features``/``core.py`` (see ``tests/test_pooled_acceptance_
narwhals.py`` and the removal in ``mlforecast/pooled.py``/``mlforecast/
core.py``), no transform in ``lag_transforms.__all__`` may still depend on
that fallback. Quantile transforms are the one deliberate exception: they
have no sufficient statistic, so they never define ``_pooled_expr`` at all
(``_pooled_quantile = True`` marks them for the separate flat values+offsets
store instead -- see ``NarwhalsPooledState._quantile_columns``).

This does NOT use a try/except-continue construction ladder (the brief's
original sketch did: ``cls(window_size=7, **kwargs)`` falling back to
``cls(**kwargs)`` falling back to ``continue``). That pattern is exactly the
"test that cannot fail" shape flagged by this plan's own retro: a class this
repo's constructors dict doesn't yet know how to build would silently vanish
from consideration instead of failing the test. Instead, every class in
``__all__`` has an explicit, correctly-shaped constructor call below, and
``test_constructors_cover_every_name_in_lag_transforms_all`` fails loudly if
a class is ever added to ``__all__`` without one.
"""

import operator

import narwhals as nw

from mlforecast import lag_transforms as lt
from mlforecast.pooled import _resolve_ctx

# One correctly-shaped constructor per class in `lag_transforms.__all__`, so
# every class actually gets built and probed rather than silently skipped.
# `groupby=["store"]`/`partition_by=["store"]` puts each transform in a
# pooled mode (required for `_pooled_expr` to be meaningful at all -- the
# per-series/local path never calls it).
CONSTRUCTORS = {
    "RollingMean": lambda: lt.RollingMean(window_size=7, groupby=["store"]),
    "RollingStd": lambda: lt.RollingStd(window_size=7, groupby=["store"]),
    "RollingMin": lambda: lt.RollingMin(window_size=7, groupby=["store"]),
    "RollingMax": lambda: lt.RollingMax(window_size=7, groupby=["store"]),
    "RollingQuantile": lambda: lt.RollingQuantile(
        p=0.5, window_size=7, groupby=["store"]
    ),
    "SeasonalRollingMean": lambda: lt.SeasonalRollingMean(
        season_length=7, window_size=3, groupby=["store"]
    ),
    "SeasonalRollingStd": lambda: lt.SeasonalRollingStd(
        season_length=7, window_size=3, groupby=["store"]
    ),
    "SeasonalRollingMin": lambda: lt.SeasonalRollingMin(
        season_length=7, window_size=3, groupby=["store"]
    ),
    "SeasonalRollingMax": lambda: lt.SeasonalRollingMax(
        season_length=7, window_size=3, groupby=["store"]
    ),
    "SeasonalRollingQuantile": lambda: lt.SeasonalRollingQuantile(
        p=0.5, season_length=7, window_size=3, groupby=["store"]
    ),
    "ExpandingMean": lambda: lt.ExpandingMean(groupby=["store"]),
    "ExpandingStd": lambda: lt.ExpandingStd(groupby=["store"]),
    "ExpandingMin": lambda: lt.ExpandingMin(groupby=["store"]),
    "ExpandingMax": lambda: lt.ExpandingMax(groupby=["store"]),
    "ExpandingQuantile": lambda: lt.ExpandingQuantile(p=0.5, groupby=["store"]),
    "ExponentiallyWeightedMean": lambda: lt.ExponentiallyWeightedMean(
        alpha=0.3, groupby=["store"]
    ),
    "LookupLag": lambda: lt.LookupLag(partition_by=["store"]),
    "Offset": lambda: lt.Offset(
        tfm=lt.RollingMean(window_size=7, groupby=["store"]), n=1
    ),
    "Combine": lambda: lt.Combine(
        tfm1=lt.RollingMean(window_size=7, groupby=["store"]),
        tfm2=lt.RollingMean(window_size=14, groupby=["store"]),
        operator=operator.truediv,
    ),
}

# Fixed at this commit: 19 classes in `lag_transforms.__all__`, 3 of which
# (RollingQuantile/SeasonalRollingQuantile/ExpandingQuantile) are
# `_pooled_quantile`-exempt, leaving 16 that must return a real `nw.Expr`.
# Verified by running the enumeration below: it prints exactly these counts.
# Checked here directly (rather than only via `len(missing) == 0`, which a
# fully-skipped enumeration also satisfies) so the test cannot degenerate
# into vacuous success.
EXPECTED_TOTAL = 19
EXPECTED_QUANTILE_EXEMPT = 3
EXPECTED_CHECKED = EXPECTED_TOTAL - EXPECTED_QUANTILE_EXEMPT


def test_constructors_cover_every_name_in_lag_transforms_all():
    """Guards the enumeration itself: a class added to ``__all__`` without a
    matching entry here must fail loudly, not vanish from the walk below."""
    assert set(CONSTRUCTORS) == set(lt.__all__)
    assert len(lt.__all__) == EXPECTED_TOTAL, (
        f"lag_transforms.__all__ has {len(lt.__all__)} entries, expected "
        f"{EXPECTED_TOTAL} -- update EXPECTED_TOTAL/EXPECTED_CHECKED and "
        "CONSTRUCTORS together with whatever class was added or removed."
    )


def test_every_pooled_capable_transform_has_a_narwhals_expression():
    """No pooled-capable transform may still rely on the legacy fallback."""
    missing = []
    checked = []
    for name in lt.__all__:
        tfm = CONSTRUCTORS[name]()
        # The real construction path (also used by `feature_frame`/
        # `_build`), not a fake object standing in for `_core_tfm` -- for
        # `Offset`/`Combine` this recursively sets the inner transforms'
        # `_core_tfm` too, which their own `_pooled_expr` delegation depends
        # on (see `Offset._pooled_expr`/`Combine._pooled_expr`).
        tfm._set_core_tfm(lag=1)
        if getattr(tfm, "_pooled_quantile", False):
            continue
        checked.append(name)
        ctx = _resolve_ctx(tfm, ["_bucket_id"])
        expr = tfm._pooled_expr(ctx)
        if expr is None:
            missing.append(name)
            continue
        assert isinstance(expr, nw.Expr), (
            f"{name}._pooled_expr returned {type(expr)!r}, not an nw.Expr"
        )
    assert not missing, f"no _pooled_expr and not a quantile: {missing}"
    # The floor that keeps this from degenerating: if every class silently
    # failed construction and got skipped, `checked` would be empty and
    # `not missing` would pass vacuously.
    assert len(checked) == EXPECTED_CHECKED, (
        f"checked {len(checked)} transforms ({sorted(checked)}), expected "
        f"{EXPECTED_CHECKED} -- the enumeration likely degenerated (or the "
        "set of quantile-exempt transforms changed; update "
        "EXPECTED_QUANTILE_EXEMPT accordingly)"
    )
