"""One-way migration for models saved before the narwhals pooled engine.

cloudpickle stores library-defined classes BY REFERENCE (module + qualname),
so a saved ``ts.pkl`` needs ``PooledState``/``_TimestampAggregates`` importable
under whatever module they were defined in at save time -- either the 1.1.0-era
``mlforecast.pooled`` or the current ``mlforecast._pooled_legacy`` (verified
with ``pickletools.genops`` on a real saved model: both classes are recorded
as ``(module, qualname)`` references, never inlined by value). The shipped
engine keeps neither name importable forever: all legacy-pickle knowledge
lives here, in throwaway stub classes plus a ``pickle.Unpickler.find_class``
remap, so the rest of the library never has to.

This module is a migration utility, not legacy engine code. Unlike
``mlforecast._pooled_legacy`` (the numpy pooled engine itself, scheduled for
removal once the narwhals engine has fully replaced it -- see
``mlforecast.pooled``'s module docstring), this file must keep working AFTER
that removal, so a model saved years ago can still be converted. It is
deliberately excluded from that removal manifest and has its own, separate
end of life: once the deprecation window for pre-narwhals saved models closes.
Consequently it never imports ``mlforecast._pooled_legacy`` -- the numpy-engine
fallback aggregation below (``_rebuild_agg_from_legacy``'s empty-``_ts_aggs``
branch) reimplements the small piece of ``_build_ts_aggs`` it needs instead of
importing it.
"""

import contextlib
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import cloudpickle
import narwhals as nw
import numpy as np
import utilsforecast.processing as ufp

from ._pooled_engine import _add_ordinals, _add_prefix_sums, _derive_time_agg_family


class LegacyPickleError(RuntimeError):
    """Raised when a saved model predates the narwhals pooled engine."""


class _LegacyStub:
    """Stand-in for a removed legacy class. Migration-only.

    Generic on purpose: it never needs to know which of ``PooledState`` /
    ``_TimestampAggregates`` it is replacing, only that a dataclass's pickled
    state is a plain ``__dict__`` that can be restored verbatim.
    """

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)


# Both the true 1.1.0-era module path (`PooledState`/`_TimestampAggregates`
# were defined directly in `mlforecast.pooled` back then) and the current one
# (moved to `mlforecast._pooled_legacy` -- see that module) need covering: a
# class's pickle reference is fixed by its `__module__`/`__qualname__` at
# definition time, not by whatever module happens to re-export it later.
_LEGACY_MODULES = ("mlforecast.pooled", "mlforecast._pooled_legacy")
_LEGACY_NAMES = ("PooledState", "_TimestampAggregates")
_LEGACY = {(m, n): _LegacyStub for m in _LEGACY_MODULES for n in _LEGACY_NAMES}


class _MigrationUnpickler(pickle.Unpickler):
    """Unpickles a legacy ``TimeSeries`` without the legacy classes importable.

    Every other class reference (models, transforms, ``GroupedArray``, the
    narwhals pooled state, ...) resolves normally.
    """

    def find_class(self, module: str, name: str):
        stub = _LEGACY.get((module, name))
        if stub is not None:
            return stub
        return super().find_class(module, name)


@contextlib.contextmanager
def _simulate_missing_legacy() -> Iterator[None]:
    """Test-only: temporarily delete the legacy classes from ``_pooled_legacy``.

    Simulates the post-removal library (once ``mlforecast._pooled_legacy`` is
    dropped) against TODAY's codebase, so ``TimeSeries.load``'s error handling
    can be exercised without waiting for that removal.
    """
    import mlforecast._pooled_legacy as legacy_mod

    removed: Dict[str, Any] = {}
    for name in _LEGACY_NAMES:
        if hasattr(legacy_mod, name):
            removed[name] = getattr(legacy_mod, name)
            delattr(legacy_mod, name)
    try:
        yield
    finally:
        for name, cls in removed.items():
            setattr(legacy_mod, name, cls)


def _group_slices(
    bucket_ids: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, Tuple[int, int]]]:
    """Stable per-bucket contiguous slices via a single ``argsort``.

    The antidote to a ``bucket_id == bid`` boolean-mask scan repeated once per
    bucket (O(buckets * rows), the exact antipattern this project removes
    everywhere else): one sort groups every bucket's rows into one contiguous
    range of the sorted arrays, so later per-bucket work only ever touches
    that bucket's own rows.

    Returns the sort order (to reindex sibling arrays the same way) and a
    ``{bucket_id: (start, end)}`` map into the sorted arrays.
    """
    order = np.argsort(bucket_ids, kind="stable")
    sorted_ids = bucket_ids[order]
    if len(sorted_ids) == 0:
        return order, {}
    uniq, starts = np.unique(sorted_ids, return_index=True)
    ends = np.append(starts[1:], len(sorted_ids))
    return order, {int(b): (int(s), int(e)) for b, s, e in zip(uniq, starts, ends)}


def _rebuild_agg_from_legacy(legacy_state: Any, time_col: str):
    """Legacy ``_ts_aggs`` -> the new aggregate table. Verified bit-exact.

    The legacy per-bucket ``_ts_aggs`` dict-of-arrays (``sums``, ``counts``,
    ``mins``, ``maxs``) IS the new table's ``s``/``c``/``mn``/``mx`` columns,
    transposed -- reusing those arrays verbatim (rather than re-aggregating
    raw rows through a different summation order) is what makes those columns
    bit-exact rather than merely close.

    The variance moments are the exception: legacy stores ``sum_sq`` (centred
    on zero) while this engine stores ``sK``/``qK`` centred on the bucket's
    own reference ``K`` (see ``_pooled_engine.compute_kref`` -- the whole
    point being that ``sum_sq`` loses the variance to cancellation at large
    magnitude). ``sum_sq`` cannot be re-centred after the fact, so these two
    columns -- AND the reference itself, which the migrated state must then
    carry for the rest of its life exactly like a freshly-fit one -- are
    recomputed from the row-aligned raw ``y`` this state also carries. Same
    data, same bincount grouping as the fallback path below.

    Falls back to rebuilding from the row-aligned ``bucket_id``/
    ``time_index``/``y`` arrays -- the same bincount algorithm as legacy's
    ``_build_ts_aggs``, reimplemented here rather than imported (see the
    module docstring) -- when ``_ts_aggs`` is empty, e.g. after the slow-path
    ``_transform`` clears the cache.

    Every mode's real timestamp is recovered from ``_ts_aggs``'s bare integer
    ordinals via ``_group_slices`` (a single stable sort) plus one
    ``np.unique(..., return_index=True)`` per bucket over that bucket's own
    contiguous slice -- never a ``bucket_id == bid`` scan of the full row
    arrays repeated per bucket.

    The per-bucket dense rank (``0..m-1`` in ascending-time order) is used for
    ``ord``, not the legacy ordinal value itself: for `global`/`groupby`
    states these coincide (legacy renumbers those ordinals to `0..n-1`
    already), but a `partition_by` bucket's legacy ordinal is parent-calendar-
    relative and can be gapped, whereas the *sparse* fit-time ``agg`` table
    (before ``ensure_densified`` ever runs) always uses the dense per-bucket
    rank -- exactly what ``_add_ordinals`` below recomputes.
    """
    d = legacy_state.__dict__
    bucket_ids = np.asarray(d["bucket_id"], dtype=np.int64)
    ordinals = np.asarray(d["time_index"], dtype=np.int64)
    times = np.asarray(d["time"])
    y = np.asarray(d["y"], dtype=np.float64)
    ts_aggs: Dict[int, Any] = d.get("_ts_aggs") or {}
    has_keys = d.get("group_cols") is not None or d.get("key_cols") is not None
    keys: List[str] = ["_bucket_id"] if has_keys else []

    order, slices = _group_slices(bucket_ids)
    ord_sorted = ordinals[order]
    time_sorted = times[order]
    y_sorted = y[order]

    bucket_iter = sorted(ts_aggs) if ts_aggs else sorted(slices)

    bids_out: List[np.ndarray] = []
    times_out: List[np.ndarray] = []
    s_out: List[np.ndarray] = []
    c_out: List[np.ndarray] = []
    sk_out: List[np.ndarray] = []
    qk_out: List[np.ndarray] = []
    mn_out: List[np.ndarray] = []
    mx_out: List[np.ndarray] = []
    kref_bids: List[int] = []
    kref_k: List[float] = []
    kref_avgc: List[float] = []

    for bid in bucket_iter:
        s_idx, e_idx = slices[bid]
        ord_slice = ord_sorted[s_idx:e_idx]
        time_slice = time_sorted[s_idx:e_idx]
        uniq_ord, first_idx = np.unique(ord_slice, return_index=True)
        time_for_ord = time_slice[first_idx]
        m = len(uniq_ord)
        if m == 0:
            continue
        # The shifted moments always come from the raw rows -- `_ts_aggs`
        # only ever cached the zero-centred `sum_sq`, which cannot be
        # re-centred. `K` is the bucket mean, matching `compute_kref`.
        y_raw = y_sorted[s_idx:e_idx]
        _, inv_raw = np.unique(ord_slice, return_inverse=True)
        valid_raw = ~np.isnan(y_raw)
        n_valid = int(valid_raw.sum())
        k_bucket = float(y_raw[valid_raw].mean()) if n_valid else 0.0
        dy = np.where(valid_raw, y_raw - k_bucket, 0.0)
        sk = np.bincount(inv_raw, weights=dy, minlength=m).astype(np.float64)
        qk = np.bincount(inv_raw, weights=dy**2, minlength=m).astype(np.float64)
        kref_bids.append(int(bid))
        kref_k.append(k_bucket)
        kref_avgc.append(n_valid / m)
        if ts_aggs:
            agg = ts_aggs[bid].__dict__
            sums = np.asarray(agg["sums"], dtype=np.float64)
            counts = np.asarray(agg["counts"], dtype=np.float64)
            mins = np.asarray(agg["mins"], dtype=np.float64)
            maxs = np.asarray(agg["maxs"], dtype=np.float64)
        else:
            y_slice = y_sorted[s_idx:e_idx]
            _, inv = np.unique(ord_slice, return_inverse=True)
            valid = ~np.isnan(y_slice)
            y_valid = np.where(valid, y_slice, 0.0)
            sums = np.bincount(inv, weights=y_valid, minlength=m).astype(np.float64)
            counts = np.bincount(inv, weights=valid.astype(float), minlength=m).astype(
                np.float64
            )
            mins = np.full(m, np.inf)
            maxs = np.full(m, -np.inf)
            if valid.any():
                np.minimum.at(mins, inv[valid], y_slice[valid])
                np.maximum.at(maxs, inv[valid], y_slice[valid])
            no_valid = mins == np.inf
            mins[no_valid] = np.nan
            maxs[no_valid] = np.nan
        bids_out.append(np.full(m, bid, dtype=np.int64))
        times_out.append(time_for_ord)
        s_out.append(sums)
        c_out.append(counts)
        sk_out.append(sk)
        qk_out.append(qk)
        mn_out.append(mins)
        mx_out.append(maxs)

    def _cat(parts: List[np.ndarray], dtype=None) -> np.ndarray:
        if not parts:
            return np.array([], dtype=dtype)
        return np.concatenate(parts)

    data: Dict[str, Any] = {}
    if keys:
        data["_bucket_id"] = _cat(bids_out, np.int64)
    data[time_col] = _cat(times_out)
    data["s"] = _cat(s_out, np.float64)
    data["c"] = _cat(c_out, np.float64)
    data["sK"] = _cat(sk_out, np.float64)
    data["qK"] = _cat(qk_out, np.float64)
    data["mn"] = _cat(mn_out, np.float64)
    data["mx"] = _cat(mx_out, np.float64)

    backend = nw.get_native_namespace(
        nw.from_native(legacy_state.bucket_df, eager_only=True)
    )
    tbl = nw.from_dict(data, backend=backend)
    sort_keys = keys + [time_col] if keys else [time_col]
    tbl = tbl.sort(sort_keys)
    # Reuse the real engine's own ordinal/derived-family/prefix-sum
    # machinery rather than duplicating it, so the migrated table's schema
    # can never drift out of sync with what a fresh fit produces.
    tbl = _add_ordinals(tbl, keys, time_col)
    tbl = _derive_time_agg_family(tbl, {None})
    kref_data: Dict[str, Any] = {}
    if keys:
        kref_data["_bucket_id"] = np.asarray(kref_bids, dtype=np.int64)
    k_arr = np.asarray(kref_k, dtype=np.float64)
    avgc = np.asarray(kref_avgc, dtype=np.float64)
    if not keys:
        # a single implicit bucket: `_group_slices` still yields exactly one
        # entry for it, so the arrays are length 1 (or 0 for an empty state).
        k_arr = k_arr[:1] if len(k_arr) else np.zeros(1)
        avgc = avgc[:1] if len(avgc) else np.zeros(1)
    kref_data["K"] = k_arr
    kref_data["K__sum"] = k_arr * avgc
    kref_data["K__mean"] = k_arr
    kref_data["K__count"] = avgc
    kref_data["K__min"] = k_arr
    kref_data["K__max"] = k_arr
    kref = nw.from_dict(kref_data, backend=backend).to_native()
    return _add_prefix_sums(tbl.to_native(), keys, {None}), kref


def _migrate_one_state(
    legacy_state: Any, target_col: str, leaf_tfms: Optional[list] = None
):
    """One legacy ``PooledState`` -> one fully-functional ``NarwhalsPooledState``.

    Sets every field a fresh ``NarwhalsPooledState._build`` would: the
    densification/seed/quantile/pending machinery the class has grown since
    the original migration proof-of-concept (``tmp/pooled-audit/migrate_poc3.py``,
    which only reconstructed the aggregate table) all start at the same
    defaults a brand-new fit leaves them at -- there is nothing to recover for
    them from the legacy pickle, since the legacy engine never had them
    (density estimation, quantile offset stores and predict-time seeding are
    narwhals-engine-only concepts). They are computed lazily on first use
    (``_quantile_columns``, ``_ensure_predict_init``) exactly as for a
    freshly-fit state.

    ``ensure_densified`` is the one exception -- it is NOT left lazy here.
    Normally it is only ever invoked from ``feature_frame`` (the FIT-time
    evaluator); ``latest_features`` (the PREDICT-time evaluator) assumes
    densification already happened, since a naturally-fit-then-predicted
    model always runs `feature_frame` at least once before any predict call.
    A migrated model skips `feature_frame` entirely, so without this call a
    `partition_by` state's `_densified` flag would incorrectly stay `False`
    forever, and `latest_features` would evaluate every non-`LookupLag`
    family's row-position shift against the SPARSE (gapped) table instead of
    the dense parent-calendar grid -- silently wrong predictions, not a
    crash (measured: ~0.1 absolute divergence on this project's own
    RollingMean/ExpandingMean partition_by fixtures). Densifying here, once,
    up front, makes a migrated state behave exactly like one that HAD been
    through `feature_frame` already. A no-op for `global`/`groupby` states
    (`ensure_densified` returns immediately for `_KNOWN_DENSE_MODES`).
    """
    from .pooled import NarwhalsPooledState

    d = legacy_state.__dict__
    time_col = d["join_cols"][1]
    group_cols = d.get("group_cols")
    key_cols = d.get("key_cols")
    has_keys = group_cols is not None or key_cols is not None
    keys: List[str] = ["_bucket_id"] if has_keys else []
    if group_cols is None and key_cols is None:
        mode = "global"
    elif key_cols is None:
        mode = "groupby"
    else:
        # "local" or "nonlocal" -- legacy's `PooledState.from_partition` and
        # the narwhals engine's `NarwhalsPooledState.from_partition` use the
        # identical two literal strings for this parameter.
        mode = d["mode"]

    state = NarwhalsPooledState(
        agg=None,
        groups=d["groups"],
        group_cols=list(group_cols) if group_cols is not None else None,
        series_bucket_id=np.asarray(d["series_bucket_id"], dtype=np.int64),
        join_cols=list(d["join_cols"]),
        keys=keys,
        time_col=time_col,
        mode=mode,
    )
    if key_cols is not None:
        state.key_cols = list(key_cols)
        state.partition_cols = list(d.get("partition_cols") or [])
        parent_scope_cols = d.get("parent_scope_cols")
        state._parent_scope_cols = (
            list(parent_scope_cols) if parent_scope_cols is not None else None
        )

    # `bucket_df` already carries every column a fresh `_build` input would
    # (id_col, group/key cols, `_bucket_id`, time_col, target_col) -- reused
    # as-is except for `target_col`, overwritten with the row-aligned `y`
    # field: `y` already went through the model's working-dtype narrowing at
    # fit time (`.astype(ga_data_dtype).astype(float)`), while `bucket_df`'s
    # own target column never did, so it can hold different (wider) floats
    # than what a fresh fit's rounding would have produced.
    y_arr = np.asarray(d["y"], dtype=np.float64)
    bucket_df_nw = nw.from_native(legacy_state.bucket_df, eager_only=True)
    df_nw = bucket_df_nw.with_columns(
        nw.new_series(
            target_col,
            y_arr,
            dtype=nw.Float64,
            backend=nw.get_native_namespace(bucket_df_nw),
        )
    )
    state._df = ufp.drop_index_if_pandas(df_nw.to_native())
    state._target_col = target_col
    state._time_aggs = {None}
    state._qvalues = None
    # The frozen centring reference is part of the state's shape contract
    # from here on: `ensure_time_aggs` below rebuilds the whole table from
    # `state._df` and MUST re-centre it on this same reference, and every
    # later `append_observations`/`trim_to_last` depends on it never moving.
    state.agg, state._kref = _rebuild_agg_from_legacy(legacy_state, time_col)

    if key_cols is not None:
        # Mirrors `NarwhalsPooledState.from_partition`'s own call: primes the
        # `should_densify` size guard `ensure_densified` needs.
        state._compute_density_estimate()
        state.ensure_densified(leaf_tfms or [])
    # UNCONDITIONAL, unlike the densification above: `global_`/`groupby`
    # states need their accumulate columns just as much as `partition_by`
    # ones do. This settles the same three-step invariant `feature_frame`
    # (fit) and `core.py:_initialize_lag_transform_states` (`history_warmup` /
    # `predict(new_df=...)`) settle -- `ensure_time_aggs`, then
    # `ensure_densified`, then `ensure_accumulates`. A migrated state is the
    # THIRD entry point that reaches the predict path without ever going
    # through `feature_frame`: `_make_seeds` reads the seed row's running
    # accumulate value out of the `A<col>` columns before `feature_frame`
    # would have had a chance to build them. Without this, migrating an
    # `ExpandingMin`, `ExpandingMax` or `ExponentiallyWeightedMean` model and
    # predicting raised `RuntimeError: _make_seeds: ... missing accumulate
    # column(s) [...]` -- and before that seed guard existed, silently
    # produced a wrong number instead.
    #
    # `ensure_time_aggs` is needed for the same reason and must come FIRST:
    # `_rebuild_agg_from_legacy` reconstructs only the bare ``{None}`` family
    # (``state._time_aggs = {None}`` above), so a ``time_agg``-carrying
    # transform's suffixed columns -- which its accumulate is derived FROM --
    # do not exist on a freshly migrated table at all. That is not an exotic
    # case: ``ExponentiallyWeightedMean``'s ``time_agg`` defaults to
    # ``"mean"``, so EVERY pooled EWM needs ``ewm__mean``. With
    # ``ensure_accumulates`` alone, those models fail with
    # ``ColumnNotFoundError: ewm__mean`` instead.
    state.ensure_time_aggs({getattr(t, "time_agg", None) for t in (leaf_tfms or [])})
    state.ensure_accumulates(leaf_tfms or [])

    return state


def migrate_saved_model(src: Union[str, Path], dst: Union[str, Path]) -> None:
    """Rewrite a saved model directory for the narwhals pooled engine.

    ``src`` must be a directory produced by ``MLForecast.save`` (or
    ``TimeSeries.save``) under the numpy pooled engine
    (``MLFORECAST_POOLED_ENGINE=numpy``, or an even older pre-engine-split
    release). ``dst`` is created (if needed) and populated with an equivalent
    ``ts.pkl`` whose pooled states are ``NarwhalsPooledState`` instances, plus
    unchanged copies of ``models.pkl`` and (if present) ``intervals.pkl``.
    The result loads under ``MLFORECAST_POOLED_ENGINE=narwhals`` (the
    default) via ``MLForecast.load``/``TimeSeries.load`` and predicts
    identically to the original model.
    """
    src = Path(src)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    with open(src / "ts.pkl", "rb") as f:
        ts = _MigrationUnpickler(f).load()

    from .pooled import _iter_leaf_tfms

    target_col = ts.target_col
    # `TimeSeries._get_pooled_tfms` groups the (plain, non-legacy) transform
    # objects by the same key as `_pooled_states` -- needed so
    # `_migrate_one_state` can call `ensure_densified` with the actual leaf
    # transforms for each `partition_by` state (see its docstring).
    pooled_tfms = ts._get_pooled_tfms()
    ts._pooled_states = {
        key: _migrate_one_state(
            state,
            target_col,
            leaf_tfms=[
                leaf
                for tfm in pooled_tfms.get(key, {}).values()
                for leaf in _iter_leaf_tfms(tfm)
            ],
        )
        for key, state in ts._pooled_states.items()
    }

    with open(dst / "ts.pkl", "wb") as f:
        cloudpickle.dump(ts, f)

    for name in ("models.pkl", "intervals.pkl"):
        src_file = src / name
        if src_file.exists():
            shutil.copyfile(src_file, dst / name)
