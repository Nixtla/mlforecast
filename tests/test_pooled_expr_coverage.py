"""Proves every pooled-capable transform has a narwhals expression.

Task 14 deliverable 3: with the legacy-fallback scaffolding removed from
``compute_pooled_features``/``core.py`` (see ``tests/test_pooled_acceptance_
narwhals.py`` and the removal in ``mlforecast/pooled.py``/``mlforecast/
core.py``), no transform in ``lag_transforms.__all__`` may still depend on
that fallback. Quantile transforms have no sufficient statistic, so their
value cannot be an expression over the aggregate columns
(``_pooled_quantile = True`` marks them for the separate flat values+offsets
store -- see ``NarwhalsPooledState._quantile_columns``). They still define
``_pooled_expr``: ``feature_frame`` materializes each quantile LEAF into a
column and the transform's expression references it, so ``Offset``/
``Combine`` have exactly one hook to delegate through. The enumeration below
skips them because their expression is a column reference rather than the
computed statistic; the composite section at the bottom of this file covers
that path instead.

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


# ---------------------------------------------------------------------------
# Composites over quantiles.
#
# The enumeration above only builds `Offset(RollingMean)` and
# `Combine(RollingMean, RollingMean)` -- two composites whose inner transforms
# both HAVE a `_pooled_expr`, so it could never notice that composites failed
# to forward the quantile half of the contract. `Offset(RollingQuantile(...))`
# and `Combine(RollingQuantile(...), ...)` both raised
# `AttributeError: 'NoneType' object has no attribute 'alias'` under the
# narwhals engine while computing fine under numpy: the composite forwarded
# `_pooled_expr` but not the `_pooled_quantile` marker, so `feature_frame`
# routed it to the expression branch where the quantile leaf's base
# `_pooled_expr` returned `None`.
#
# The fix makes the contract uniform instead of special-casing the crash:
# `feature_frame` materializes each quantile LEAF as a column and the quantile
# transform's own `_pooled_expr` references it, so there is exactly one hook
# for composites to delegate through -- as there was with the legacy engine's
# uniform `_compute_bucket_feature`.
# ---------------------------------------------------------------------------

QUANTILE_LEAVES = {
    "RollingQuantile": lambda: lt.RollingQuantile(
        p=0.5, window_size=7, groupby=["store"]
    ),
    "SeasonalRollingQuantile": lambda: lt.SeasonalRollingQuantile(
        p=0.5, season_length=7, window_size=3, groupby=["store"]
    ),
    "ExpandingQuantile": lambda: lt.ExpandingQuantile(p=0.5, groupby=["store"]),
}

# Every composite shape that can wrap a quantile: Offset(q), Combine(q, q),
# Combine(q, non-quantile), Combine(non-quantile, q), and a nested
# Offset(Combine(...)) to check the recursion in `_iter_leaf_tfms`.
QUANTILE_COMPOSITES = {}
for _name, _make in QUANTILE_LEAVES.items():
    QUANTILE_COMPOSITES[f"Offset({_name})"] = lambda _make=_make: lt.Offset(
        tfm=_make(), n=2
    )
    QUANTILE_COMPOSITES[f"Combine({_name},{_name})"] = lambda _make=_make: lt.Combine(
        tfm1=_make(), tfm2=_make(), operator=operator.truediv
    )
    QUANTILE_COMPOSITES[f"Combine({_name},RollingMean)"] = (
        lambda _make=_make: lt.Combine(
            tfm1=_make(),
            tfm2=lt.RollingMean(window_size=7, groupby=["store"]),
            operator=operator.truediv,
        )
    )
    QUANTILE_COMPOSITES[f"Combine(RollingMean,{_name})"] = (
        lambda _make=_make: lt.Combine(
            tfm1=lt.RollingMean(window_size=7, groupby=["store"]),
            tfm2=_make(),
            operator=operator.truediv,
        )
    )
    # Nesting: `Combine(Offset(q), ...)`. (`Offset(Combine(...))` is NOT
    # buildable in this repo -- `Offset._set_core_tfm` reads
    # `self.tfm._core_tfm`, which `Combine` does not define. Pre-existing and
    # unrelated to the quantile contract, so it is not exercised here.)
    QUANTILE_COMPOSITES[f"Combine(Offset({_name}),RollingMean)"] = (
        lambda _make=_make: lt.Combine(
            tfm1=lt.Offset(tfm=_make(), n=2),
            tfm2=lt.RollingMean(window_size=7, groupby=["store"]),
            operator=operator.truediv,
        )
    )

EXPECTED_QUANTILE_COMPOSITES = 15


def test_quantile_composites_return_a_real_expression():
    """`Offset`/`Combine` over a quantile must produce an `nw.Expr`, not None.

    Reverting the fix (removing `_pooled_expr` from the three quantile
    classes, so the base class's `None` is reached again) makes every one of
    these 15 cases report `None`.
    """
    checked, missing = [], []
    for name, make in QUANTILE_COMPOSITES.items():
        tfm = make()
        tfm._set_core_tfm(lag=1)
        ctx = _resolve_ctx(tfm, ["_bucket_id"])
        expr = tfm._pooled_expr(ctx)
        if expr is None:
            missing.append(name)
            continue
        assert isinstance(expr, nw.Expr), (
            f"{name}._pooled_expr returned {type(expr)!r}, not an nw.Expr"
        )
        checked.append(name)
    assert not missing, (
        "these composites over a quantile have no pooled expression -- "
        f"`feature_frame` would crash on `None.alias(...)`: {missing}"
    )
    assert len(checked) == EXPECTED_QUANTILE_COMPOSITES, (
        f"checked {len(checked)}, expected {EXPECTED_QUANTILE_COMPOSITES} -- "
        "the enumeration degenerated"
    )


def test_quantile_leaves_are_discoverable_through_every_composite():
    """`feature_frame` materializes the quantile columns by walking
    `_iter_leaf_tfms`; if a composite hid its quantile leaf, the column the
    expression references would never be built and the evaluation would fail
    on a missing column instead."""
    from mlforecast.pooled import _iter_leaf_tfms

    for name, make in QUANTILE_COMPOSITES.items():
        tfm = make()
        tfm._set_core_tfm(lag=1)
        leaves = list(_iter_leaf_tfms(tfm))
        quantile_leaves = [
            leaf for leaf in leaves if getattr(leaf, "_pooled_quantile", False)
        ]
        assert quantile_leaves, f"{name}: no quantile leaf found in {leaves}"
        for leaf in quantile_leaves:
            ctx = _resolve_ctx(leaf, ["_bucket_id"])
            col = leaf._pooled_quantile_name(ctx)
            # leading underscore keeps the scratch column out of the
            # user-visible feature namespace, and out of the way of every
            # aggregate column name on `self.agg`
            assert col.startswith("_pq__"), col
            assert isinstance(leaf._pooled_expr(ctx), nw.Expr)


def test_quantile_column_names_dedupe_by_configuration_only():
    """The materialized column name must depend on the transform's resolved
    configuration and nothing else: two identical leaves reached through
    different composites share one column (computed once), while any
    configuration difference gets its own."""
    keys = ["_bucket_id"]

    def name_of(tfm):
        tfm._set_core_tfm(lag=1)
        return tfm._pooled_quantile_name(_resolve_ctx(tfm, keys))

    a = name_of(lt.RollingQuantile(p=0.5, window_size=7, groupby=["store"]))
    b = name_of(lt.RollingQuantile(p=0.5, window_size=7, groupby=["store"]))
    assert a == b, "identical leaves must share one materialized column"

    different = {
        "p": lt.RollingQuantile(p=0.9, window_size=7, groupby=["store"]),
        "window": lt.RollingQuantile(p=0.5, window_size=14, groupby=["store"]),
        "time_agg": lt.RollingQuantile(
            p=0.5, window_size=7, groupby=["store"], time_agg="mean"
        ),
        "family": lt.ExpandingQuantile(p=0.5, groupby=["store"]),
        "seasonal": lt.SeasonalRollingQuantile(
            p=0.5, season_length=7, window_size=3, groupby=["store"]
        ),
    }
    for label, tfm in different.items():
        assert name_of(tfm) != a, (
            f"{label} differs from the reference transform but got the same "
            f"materialized column name {a!r} -- one would silently overwrite "
            "the other"
        )
