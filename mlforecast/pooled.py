"""Pooled (cross-series) lag transform engine.

A *pooled* lag transform computes its statistic over a **bucket** of series
aggregated by timestamp rather than over a single series, so every series in a
bucket receives the same value at each timestamp.

The engine rests on one observation: **a bucket is a series**.  Collapse the
panel to one cell per ``(bucket, timestamp)`` and a bucket becomes an ordinary
time series, so the *existing* ``coreforecast`` transforms can be run on it
unchanged -- no new rolling/expanding kernels.  Two consequences fall out:

* ``RANGE`` semantics for free.  One cell per calendar step means a
  row-position window over the collapsed series *is* a timestamp-distance
  window over the panel.  Series that start late simply contribute nothing to
  early cells, so no phantom zeros enter the window.
* Cheap recursion.  ``coreforecast`` transforms are stateful and expose
  ``update()`` (last value per group), which is what the local path already
  relies on, so a horizon step costs ``O(n_series + n_buckets)``.

Running a transform on a single collapsed channel would only give
*statistic-of-per-timestamp-aggregates* (that is what ``time_agg`` means).  The
default is the *row-weighted* pooled statistic, where every observation in the
window counts once.  Both come from the same machinery by keeping a few
**channels** per cell and combining ordinary results over them, e.g.
``pooled_mean = rolling_mean(sum) / rolling_mean(count)`` (the window length
cancels).  See ``_PooledKernel`` subclasses for the full table.
"""

__all__ = ["PooledState"]

from typing import Any, Collection, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import coreforecast.lag_transforms as core_tfms
from coreforecast.grouped_array import GroupedArray as CoreGroupedArray

# Value every missing/null key is encoded to, so missing matches missing (and
# only missing) across backends and key dtypes. NUL-prefixed so a collision
# with a real key value is negligible.
_NULL_KEY = "\x00__MLF_NULL__"


# %% key encoding
def _encode_column(values: np.ndarray) -> np.ndarray:
    """Encode one key column as strings, canonically across dtypes.

    Two things have to hold. Every missing value (null, NaN, ``None``) maps to a
    single sentinel, so missing matches missing and nothing else -- SQL
    ``PARTITION BY`` semantics. And a key must encode the same whether it
    arrives as an int or as a float: a column can be float at fit (one NaN is
    enough to widen it) and int in ``X_df`` at predict, and those must land in
    the same bucket. So integral values encode as an int string and only
    fractional ones keep a decimal point, which cannot collide with an integer.
    """
    values = np.asarray(values)
    if values.dtype.kind in "fc":
        missing = np.isnan(values)
    elif values.dtype.kind == "O":
        missing = np.array([v is None or v != v for v in values], dtype=bool)
    elif values.dtype.kind == "M":
        missing = np.isnat(values)
    else:
        missing = np.zeros(values.shape, dtype=bool)
    if values.dtype.kind == "f":
        safe = np.where(missing, 0.0, values)
        integral = safe == np.floor(safe)
        out = np.empty(values.shape, dtype=object)
        if integral.any():
            out[integral] = safe[integral].astype(np.int64).astype(str)
        if (~integral).any():
            out[~integral] = safe[~integral].astype(str)
    elif values.dtype.kind == "O":
        # object columns may mix ints and floats; normalise the numeric ones
        out = np.array(
            [
                str(int(v))
                if isinstance(v, float) and not (v != v) and v == int(v)
                else str(v)
                for v in values
            ],
            dtype=object,
        )
    else:
        out = values.astype(str).astype(object)
    if missing.any():
        out[missing] = _NULL_KEY
    return out


def _join_encoded(parts: Sequence[np.ndarray]) -> np.ndarray:
    """Join already-encoded key columns into one key per row."""
    if len(parts) == 1:
        return parts[0]
    return np.array(["\x1f".join(t) for t in zip(*parts)], dtype=object)


def _join_keys(arrays: Sequence[np.ndarray]) -> np.ndarray:
    return _join_encoded([_encode_column(a) for a in arrays])


def encode_keys(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Encoded bucket key per row, in the same space as ``bucket_uniques``."""
    return _join_keys(arrays)


def factorize(arrays: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Map a set of key columns to dense bucket ids. Returns (ids, uniques).

    Builds one key string per distinct *combination* rather than one per row,
    which is the difference between ``O(n_rows)`` and ``O(n_buckets)`` Python
    string work on a partitioned panel. The columns are hash-factorized to
    integer codes, combined as a mixed radix, and only the surviving
    combinations are encoded and joined.

    The vocabulary is still sorted in *joined-string* space, which is not the
    same as tuple order: a key value may contain a byte below the ``\\x1f``
    separator (``"a"`` sorts before ``"a\\nb"`` as a tuple, after it once
    joined), so the survivors are joined first and sorted second.
    """
    codes: List[np.ndarray] = []
    encoded: List[np.ndarray] = []
    for arr in arrays:
        col_codes, col_uniques = pd.factorize(np.asarray(arr))
        enc = _encode_column(np.asarray(col_uniques))
        if (col_codes < 0).any():
            # pandas parks every missing value at -1; give it a slot of its own
            # so it encodes to the sentinel like any other key
            enc = np.append(enc, _NULL_KEY)
            col_codes = np.where(col_codes < 0, len(col_uniques), col_codes)
        codes.append(col_codes.astype(np.int64, copy=False))
        encoded.append(enc)
    combined = codes[0]
    for col_codes, enc in zip(codes[1:], encoded[1:]):
        # re-compressed every step so the radix product cannot overflow int64
        combined = np.unique(combined * len(enc) + col_codes, return_inverse=True)[1]
        combined = combined.ravel().astype(np.int64, copy=False)
    combo_uniques, combo_ids = np.unique(combined, return_inverse=True)
    combo_ids = combo_ids.ravel()
    # one representative row per surviving combination, taken in reverse so the
    # earliest row wins -- any row of the combination encodes the same
    first = np.zeros(len(combo_uniques), dtype=np.int64)
    first[combo_ids[::-1]] = np.arange(len(combo_ids))[::-1]
    keys = _join_encoded([enc[c[first]] for c, enc in zip(codes, encoded)])
    # sorts, and dedupes where two raw values encode alike (None beside NaN)
    uniques, inv = np.unique(keys, return_inverse=True)
    return inv.ravel()[combo_ids].astype(np.int64, copy=False), uniques


def lookup(arrays: Sequence[np.ndarray], uniques: np.ndarray) -> np.ndarray:
    """Map key columns onto an existing vocabulary; unseen keys get ``-1``."""
    keys = _join_keys(arrays)
    if len(uniques) == 0:
        return np.full(len(keys), -1, dtype=np.int64)
    pos = np.clip(np.searchsorted(uniques, keys), 0, len(uniques) - 1)
    return np.where(uniques[pos] == keys, pos, -1).astype(np.int64, copy=False)


# %% cell aggregates
#: what an unoccupied cell holds per channel, so coreforecast never sees a null;
#: cells with no data are masked out later using the observation count
_FILL = {"min": np.inf, "max": -np.inf}


class _CellStore:
    """Occupied ``(bucket, ordinal)`` cells of the fit panel, bucket-major.

    The dense ``(n_buckets, width)`` block has a cell at every calendar position
    of every bucket whether rows land there or not; a partitioned panel occupies
    ``1 / cardinality`` of it. Holding only the occupied cells keeps the fit-time
    store ``O(n_rows)`` however wide the calendar, and any column range of the
    dense block is scattered out of it on demand.

    The per-cell aggregates come from one ``bincount`` over the cell index,
    which accumulates in row order exactly as a bincount over the flat
    ``bucket * width + ordinal`` index does, so a scattered block is
    bit-identical to one built dense.
    """

    def __init__(
        self,
        bucket_id: np.ndarray,
        ordinal: np.ndarray,
        y: np.ndarray,
        n_buckets: int,
        width: int,
        names: Sequence[str],
    ):
        self.n_buckets = n_buckets
        self.width = width
        flat = bucket_id.astype(np.int64) * width + ordinal
        keys, cell = np.unique(flat, return_inverse=True)
        self.keys = keys
        self.cell_of_row = cell.ravel().astype(np.int64, copy=False)
        self.bucket = keys // width
        self.ordinal = keys - self.bucket * width
        n = keys.size
        valid = ~np.isnan(y)
        cell_v = self.cell_of_row[valid]
        y_v = y[valid]
        self.chan: Dict[str, np.ndarray] = {}
        if "count" in names:
            self.chan["count"] = np.bincount(cell_v, minlength=n).astype(np.float64)
        if "sum" in names:
            self.chan["sum"] = np.bincount(cell_v, weights=y_v, minlength=n)
        if "sumsq" in names:
            self.chan["sumsq"] = np.bincount(cell_v, weights=y_v * y_v, minlength=n)
        if "min" in names:
            buf = np.full(n, np.inf)
            np.minimum.at(buf, cell_v, y_v)
            self.chan["min"] = buf
        if "max" in names:
            buf = np.full(n, -np.inf)
            np.maximum.at(buf, cell_v, y_v)
            self.chan["max"] = buf
        self._by_ordinal: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._starts: Optional[np.ndarray] = None
        self._views: Dict[str, Dict[str, np.ndarray]] = {}
        self._seasonal: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    @property
    def size(self) -> int:
        return self.bucket.size

    def channels_for(
        self, time_agg: Optional[str], names: Collection[str]
    ) -> Dict[str, np.ndarray]:
        """Per-cell channels, collapsed by ``time_agg`` if given (cached)."""
        if time_agg is None:
            return {name: self.chan[name] for name in names}
        want = set(names) | {"count"}
        view = self._views.get(time_agg)
        if view is None:
            view = self._views[time_agg] = _collapse(self.chan, time_agg, want)
        elif not want <= view.keys():
            view.update(_collapse(self.chan, time_agg, want - view.keys()))
        return {name: view[name] for name in names}

    @property
    def starts(self) -> np.ndarray:
        """CSR offsets: bucket ``b`` owns cells ``starts[b]:starts[b + 1]``."""
        if self._starts is None:
            self._starts = np.searchsorted(self.bucket, np.arange(self.n_buckets + 1))
        return self._starts

    def ragged(self, values: np.ndarray, keep: np.ndarray) -> CoreGroupedArray:
        """The cells selected by ``keep``, as one coreforecast group per bucket.

        Cells are ordered by ordinal within a bucket, so a coreforecast
        transform walking a group sees that bucket's history in time order,
        with the empty calendar positions simply absent.
        """
        starts = np.searchsorted(self.bucket[keep], np.arange(self.n_buckets + 1))
        return CoreGroupedArray(values[keep], starts.astype(np.int32))

    # -- windows ---------------------------------------------------------
    # A window is a run of consecutive cells in some layout of the store: the
    # store itself for calendar windows, or a phase-major reordering for
    # seasonal ones. Bounds come from two searchsorted calls over every cell at
    # once, and a window of `w` calendar positions holds at most `w` cells.

    def source_bound(self, lag: int) -> np.ndarray:
        """Index after the last cell at or before ``ordinal - lag``, per cell.

        The clamp keeps the key inside the cell's own bucket, so a source
        before the calendar starts yields the bucket's first index -- an
        empty window, never a cell of the bucket before.
        """
        src = np.maximum(self.ordinal - lag, -1)
        return np.searchsorted(self.keys, self.bucket * self.width + src, side="right")

    def rolling_bounds(self, lag: int, window: int) -> Tuple[np.ndarray, np.ndarray]:
        """Half-open cell bounds of ``[t - lag - window + 1, t - lag]`` per cell.

        A window that reaches before the calendar starts is simply shorter.
        """
        first = np.maximum(self.ordinal - lag - window + 1, 0)
        lo = np.searchsorted(self.keys, self.bucket * self.width + first, side="left")
        return lo, self.source_bound(lag)

    def seasonal_layout(self, season_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Cells reordered by ``(bucket, ordinal % season_length, ordinal)``.

        In that order the cells of one seasonal window -- same bucket, same
        phase, ``window_size`` seasons back -- are consecutive.
        """
        layout = self._seasonal.get(season_length)
        if layout is None:
            phase = self.ordinal % season_length
            keys = (self.bucket * season_length + phase) * self.width + self.ordinal
            perm = np.argsort(keys, kind="stable")
            layout = self._seasonal[season_length] = (keys[perm], perm)
        return layout

    def seasonal_bounds(
        self, lag: int, window: int, season_length: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bounds in the seasonal layout, plus the layout's cell permutation."""
        keys, perm = self.seasonal_layout(season_length)
        src = self.ordinal - lag
        has = src >= 0
        phase = np.where(has, src % season_length, 0)
        base = (self.bucket * season_length + phase) * self.width
        # the earliest ordinal on this phase is the phase itself
        first = np.maximum(src - (window - 1) * season_length, phase)
        lo = np.searchsorted(keys, base + first, side="left")
        hi = np.searchsorted(keys, base + np.where(has, src, 0), side="right")
        return lo, np.where(has, hi, lo), perm

    def reduce_windows(
        self,
        lo: np.ndarray,
        hi: np.ndarray,
        window: int,
        arrays: Dict[str, np.ndarray],
        perm: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Reduce each window's cells: sums, or extremes for ``min``/``max``.

        Gathered in row blocks capped at ``_MAX_GATHER`` cells, so the transient
        is bounded whatever the window or the panel. Cells past ``hi`` are
        masked with the channel's fill, which every reduction ignores.
        """
        n = lo.size
        out = {name: np.full(n, _FILL.get(name, 0.0)) for name in arrays}
        offsets = np.arange(window)
        step = max(1, _MAX_GATHER // window)
        last = max(self.size - 1, 0)
        for a in range(0, n, step):
            z = slice(a, min(a + step, n))
            idx = lo[z][:, None] + offsets
            ok = idx < hi[z][:, None]
            np.minimum(idx, last, out=idx)
            if perm is not None:
                idx = perm[idx]
            for name, arr in arrays.items():
                vals = np.where(ok, arr[idx], _FILL.get(name, 0.0))
                if name == "min":
                    out[name][z] = vals.min(axis=1)
                elif name == "max":
                    out[name][z] = vals.max(axis=1)
                else:
                    out[name][z] = vals.sum(axis=1)
        return out

    @property
    def nbytes(self) -> int:
        arrays = [self.cell_of_row, self.bucket, self.ordinal, *self.chan.values()]
        return sum(a.nbytes for a in arrays)

    def cells_in(self, lo: int, hi: int) -> np.ndarray:
        """Indices of the cells whose ordinal lies in ``[lo, hi)``.

        The store is bucket-major, so a column range is not a slice of it; one
        stable argsort by ordinal, cached, turns every range into one.
        """
        if self._by_ordinal is None:
            perm = np.argsort(self.ordinal, kind="stable")
            offsets = np.searchsorted(self.ordinal[perm], np.arange(self.width + 1))
            self._by_ordinal = (perm, offsets)
        perm, offsets = self._by_ordinal
        return perm[offsets[lo] : offsets[hi]]

    def dense(
        self, lo: int, hi: int, names: Optional[Collection[str]] = None
    ) -> Dict[str, np.ndarray]:
        """Dense ``(n_buckets, hi - lo)`` block per channel over columns ``[lo, hi)``."""
        cells = self.cells_in(lo, hi)
        b = self.bucket[cells]
        o = self.ordinal[cells] - lo
        out: Dict[str, np.ndarray] = {}
        for name in names or self.chan:
            block = np.full((self.n_buckets, hi - lo), _FILL.get(name, 0.0))
            block[b, o] = self.chan[name][cells]
            out[name] = block
        return out

    def gather(
        self, block: np.ndarray, lo: int, cells: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Per-cell values read off a block whose first column is ``lo``."""
        if cells is None:
            return block[self.bucket, self.ordinal - lo]
        return block[self.bucket[cells], self.ordinal[cells] - lo]

    def row_values(self, cell_values: np.ndarray) -> np.ndarray:
        return cell_values[self.cell_of_row]


def _build_cells(
    bucket_id: np.ndarray,
    ordinal: np.ndarray,
    y: np.ndarray,
    n_buckets: int,
    width: int,
    names: Sequence[str],
) -> Dict[str, np.ndarray]:
    """Aggregate rows into a dense ``(n_buckets, width)`` array per requested channel."""
    return _CellStore(bucket_id, ordinal, y, n_buckets, width, names).dense(0, width)


_ALL_CHANNELS = ("count", "sum", "sumsq", "min", "max")

#: Cells per temporary gather block, for the window reductions at fit and the
#: row kernels' gathers: a few same-shaped temporaries live at once, so this
#: bounds those transients at a small multiple of ``8 * _MAX_GATHER`` bytes
#: whatever the window or the panel.
_MAX_GATHER = 1 << 20


def _block_ga(block: np.ndarray) -> CoreGroupedArray:
    """Wrap a ``(n_buckets, n_cols)`` block as one coreforecast group per bucket."""
    n_buckets, n_cols = block.shape
    data = np.ascontiguousarray(block).ravel()
    indptr = np.arange(0, (n_buckets + 1) * n_cols, n_cols, dtype=np.int32)
    return CoreGroupedArray(data, indptr)


def _collapse(
    cells: Dict[str, np.ndarray],
    time_agg: str,
    names: Optional[Collection[str]] = None,
) -> Dict[str, np.ndarray]:
    """Re-express cell aggregates as channels over the collapsed value ``v_t``.

    With ``time_agg`` each timestamp contributes a single value, so the channels
    become "one observation per observed timestamp".  Because the shape matches
    the row-weighted channels exactly, every kernel combine below works
    unchanged for both modes.

    ``names`` limits the output to the channels a kernel actually reads; the
    result is cached, so building all five would also keep them all alive.
    """
    counts = cells["count"]
    obs = counts > 0
    if time_agg == "count":
        v = counts
    elif time_agg == "sum":
        v = cells["sum"]
    elif time_agg == "min":
        v = cells["min"]
    elif time_agg == "max":
        v = cells["max"]
    else:  # mean
        v = np.divide(cells["sum"], counts, out=np.zeros_like(cells["sum"]), where=obs)
    zero = np.zeros_like(v)
    builders = {
        "count": lambda: obs.astype(np.float64),
        "sum": lambda: np.where(obs, v, zero),
        "sumsq": lambda: np.where(obs, v * v, zero),
        "min": lambda: np.where(obs, v, np.inf),
        "max": lambda: np.where(obs, v, -np.inf),
    }
    return {name: builders[name]() for name in (names or _ALL_CHANNELS)}


# %% kernels
def _grow_rows(arr: np.ndarray, remap: np.ndarray, n_new: int, fill) -> np.ndarray:
    """Permute per-bucket rows into a grown vocabulary; new buckets get ``fill``."""
    out = np.empty((n_new,) + arr.shape[1:], dtype=arr.dtype)
    out[...] = fill
    out[remap] = arr
    return out


def _expanding_fill(tfm, stats: np.ndarray):
    """Inner state of a bucket that has only ever seen empty cells."""
    if isinstance(tfm, core_tfms.ExpandingMean):
        # ``[cells consumed, cumsum]``. The cell count is read off an existing
        # bucket rather than derived from ``n_ordinals``: a multi-timestamp
        # ``update`` appends several columns but advances the accumulator once,
        # so the two disagree.
        return np.array([stats[0, 0] if stats.shape[0] else 0.0, 0.0])
    if isinstance(tfm, core_tfms.ExpandingMin):
        return np.inf
    if isinstance(tfm, core_tfms.ExpandingMax):
        return -np.inf
    raise NotImplementedError(
        f"pooled bucket growth for inner transform {type(tfm).__name__!r}"
    )


class _PooledKernel:
    """Maps a user-facing transform onto channels + inner transforms + a combine.

    ``channels`` names the cell aggregates needed.  ``make_inner`` builds one
    stateful ``coreforecast`` transform per channel (they are advanced in
    lockstep by ``update``).  ``window_cells`` gives the number of *source
    cells* the window spans at each ordinal, which converts a channel mean back
    into a channel sum so observation counts can be recovered.
    """

    channels: Tuple[str, ...] = ("sum", "count")
    #: kernels that carry their own recurrence instead of delegating to a
    #: stateful coreforecast transform (see :class:`EwmK`).
    custom: bool = False
    #: whether the fit pass leaves behind state a later ``update`` reads back.
    #: False for the kernels whose inners recompute from the block on every
    #: update (Rolling*/SeasonalRolling*/Lag) and for the row kernels, which
    #: re-gather from the raw observations; only the accumulator-carrying
    #: kernels (Expanding*, EWM) need priming at all.
    primes_state: bool = True

    def __init__(self, tfm):
        self.tfm = tfm
        self.lag = tfm._core_tfm.lag

    def make_inner(self) -> Dict[str, Any]:
        raise NotImplementedError

    def window_cells(self, ordinals: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_from_store(
        self, _store: "_CellStore", _inner: Optional[Dict[str, Any]] = None
    ) -> Optional[np.ndarray]:
        """Fit-time value per occupied cell, computed on the cell store.

        Every channel kernel defines it, two ways. A bounded window is a run of
        at most ``window_size`` consecutive cells in some layout of the store,
        gathered and reduced outright; the sums are combined with ``k = 1``,
        so the count needs no ``k * mean`` round trip and is exact. An
        accumulator kernel runs its inner transforms over each bucket's
        occupied cells instead, and when ``inner`` is given primes it off the
        same pass. ``None`` (the default) sends the kernel down the dense
        ``transform`` path -- the row kernels.
        """
        return None

    def min_samples(self) -> float:
        ms = getattr(self.tfm, "min_samples", None)
        if ms is not None:
            return float(ms)
        # a local partition bucket holds a single series, so requiring a full
        # window of observations would blank out most of the partition
        if self.tfm._pooled_mode == "local":
            return 1.0
        return float(getattr(self.tfm, "window_size", 1))

    def combine(self, res: Dict[str, np.ndarray], k: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # implemented by kernels with ``custom = True``, which carry their own
    # recurrence or row gather instead of a coreforecast transform per channel
    def run_transform(self, state: "PooledState", st: Dict) -> np.ndarray:
        raise NotImplementedError

    def run_update(self, state: "PooledState", st: Dict) -> np.ndarray:
        raise NotImplementedError

    def remap_buckets(
        self, inner: Dict[str, Any], remap: np.ndarray, n_new: int
    ) -> None:
        """Permute/extend per-bucket inner state after ``grow_buckets``.

        Stateless inners (``Rolling*``/``SeasonalRolling*``/``Lag``) recompute
        from the stored block on every ``update``, so only the accumulator-carrying
        ones need this. A new bucket's channels were padded over the whole existing
        calendar, so its inner state is that of a bucket which consumed exactly
        those empty cells.
        """
        for tfm in inner.values():
            stats = getattr(tfm, "stats_", None)
            if stats is None:
                continue
            tfm.stats_ = _grow_rows(stats, remap, n_new, _expanding_fill(tfm, stats))

    @staticmethod
    def _n_obs(res: Dict[str, np.ndarray], k: np.ndarray) -> np.ndarray:
        """Total observations inside the window: ``k * mean(count)``.

        Rounded because the count only ever comes back as a *mean* over ``k``
        cells, and ``k * (S / k)`` is not exactly ``S`` in float64 -- ``49 *
        (1 / 49)`` is ``0.9999999999999999``, which would fail a
        ``min_samples=1`` gate and blank out a real value. ``S`` is an integer
        and the round trip is accurate to a few ULP, so rounding recovers it.
        """
        n = k * res["count"]
        # in place: on a wide state this is a full block, and the rounded copy
        # would be a second one
        return np.rint(n, out=n)


class _MeanKernel(_PooledKernel):
    channels = ("sum", "count")

    def combine(self, res, k):
        # inline rather than named so the observation count, a full block, is
        # freed at the comparison instead of living to the end of the call
        ok = (self._n_obs(res, k) >= self.min_samples()) & (res["count"] > 0)
        out = np.full(res["sum"].shape, np.nan)
        # `out` starts as nan and `where=ok` writes nowhere else, so it already
        # carries the nans a trailing ``np.where`` would add -- at one block per
        # call, which is a tenth of the fit-time transient on a wide state
        np.divide(res["sum"], res["count"], out=out, where=ok)
        return out


class _StdKernel(_PooledKernel):
    channels = ("sum", "sumsq", "count")

    def combine(self, res, k):
        n = self._n_obs(res, k)
        ok = (n >= max(self.min_samples(), 2.0)) & (n > 1)
        # 2.0 keeps ``safe_n - 1`` non-zero where ~ok; those cells are nan'd below
        safe_n = np.where(ok, n, 2.0)
        s1 = k * res["sum"]
        var = k * res["sumsq"]
        # accumulated in place: each temporary here is a full (bucket, cell) block
        s1 *= s1
        s1 /= safe_n
        var -= s1
        safe_n -= 1.0
        var /= safe_n
        # tiny negatives are float cancellation, not real variance
        np.clip(var, 0.0, None, out=var)
        np.sqrt(var, out=var)
        var[~ok] = np.nan
        return var


class _MinKernel(_PooledKernel):
    channels = ("min", "count")

    def combine(self, res, k):
        n = self._n_obs(res, k)
        ok = (n >= self.min_samples()) & np.isfinite(res["min"])
        return np.where(ok, res["min"], np.nan)


class _MaxKernel(_PooledKernel):
    channels = ("max", "count")

    def combine(self, res, k):
        n = self._n_obs(res, k)
        ok = (n >= self.min_samples()) & np.isfinite(res["max"])
        return np.where(ok, res["max"], np.nan)


class _RollingMixin:
    primes_state = False

    def window_cells(self, ordinals):
        return np.clip(ordinals - self.lag + 1, 0, self.tfm.window_size).astype(float)

    def fit_from_store(self, store, _inner=None):
        window = self.tfm.window_size
        lo, hi = store.rolling_bounds(self.lag, window)
        cells = store.channels_for(self.tfm.time_agg, self.channels)
        return self.combine(store.reduce_windows(lo, hi, window, cells), 1.0)

    def _inner(self, cls, **kw):
        return cls(lag=self.lag, window_size=self.tfm.window_size, min_samples=1, **kw)


class _SeasonalMixin:
    primes_state = False

    def window_cells(self, ordinals):
        sl = self.tfm.season_length
        avail = np.floor_divide(np.maximum(ordinals - self.lag, -1), sl) + 1
        return np.clip(avail, 0, self.tfm.window_size).astype(float)

    def fit_from_store(self, store, _inner=None):
        window = self.tfm.window_size
        lo, hi, perm = store.seasonal_bounds(self.lag, window, self.tfm.season_length)
        cells = store.channels_for(self.tfm.time_agg, self.channels)
        return self.combine(store.reduce_windows(lo, hi, window, cells, perm), 1.0)

    def _inner(self, cls, **kw):
        return cls(
            lag=self.lag,
            season_length=self.tfm.season_length,
            window_size=self.tfm.window_size,
            min_samples=1,
            **kw,
        )


class _ExpandingMixin:
    def window_cells(self, ordinals):
        return np.clip(ordinals - self.lag + 1, 0, None).astype(float)

    def _inner(self, cls, lag=None, **kw):
        return cls(lag=self.lag if lag is None else lag, **kw)

    def min_samples(self):
        return 1.0

    def fit_from_store(self, store, inner=None):
        """Running statistics over each bucket's occupied cells, read at ``t - lag``.

        An empty calendar position adds nothing to an expanding window, so the
        inner transforms can walk the occupied cells alone, with ``lag=0``
        since the lag is applied when the running value is looked up. Only the
        cells at or before ``width - 1 - lag`` are ever a source at fit, so the
        pass stops there -- which leaves the lag-0 inners holding the
        accumulator the dense pass would have left in the state's own, to
        within an ulp (the same values, minus the zeros). Priming copies it
        over, with the cell count set to the calendar cells the dense pass
        consumed, which is what ``window_cells`` derives the factor from.
        """
        cells = store.channels_for(self.tfm.time_agg, self.channels)
        consumed = max(store.width - self.lag, 0)
        keep = store.ordinal < consumed
        n_kept = np.bincount(store.bucket[keep], minlength=store.n_buckets)
        primed = self.make_inner(lag=0)
        # coreforecast reads each group's last output for its state, which an
        # empty group has to borrow from the group before -- so those buckets
        # are refilled below, and a pass with nothing kept is skipped outright
        running = {
            name: tfm.transform(store.ragged(cells[name], keep))
            if keep.any()
            else np.empty(0)
            for name, tfm in primed.items()
        }
        starts = store.starts
        first = starts[store.bucket]
        hi = store.source_bound(self.lag)
        has = hi > first
        # `hi` indexes the full store; the pass skipped the cells past the
        # cutoff of every earlier bucket, so shift by that many
        kept_before = np.concatenate([[0], np.cumsum(keep)])
        at = (hi - 1 - (first - kept_before[first]))[has]
        res = {}
        for name, vals in running.items():
            out = np.full(store.size, _FILL.get(name, 0.0))
            out[has] = vals[at]
            res[name] = out
        k = np.where(has, hi - first, 0).astype(np.float64)
        values = self.combine(res, k)
        if inner is not None:
            empty = n_kept == 0
            for name, tfm in inner.items():
                if keep.any():
                    stats = np.array(primed[name].stats_, copy=True)
                elif isinstance(tfm, core_tfms.ExpandingMean):
                    stats = np.zeros((store.n_buckets, 2))
                else:
                    stats = np.zeros(store.n_buckets)
                if stats.ndim == 2:
                    # ``[cells consumed, cumsum]``: the dense pass consumes the
                    # calendar, which is what ``window_cells`` counts against
                    stats[:, 0] = consumed
                    stats[empty, 1] = 0.0
                else:
                    stats[empty] = _expanding_fill(tfm, stats)
                tfm.stats_ = stats
        return values


class RollingMeanK(_RollingMixin, _MeanKernel):
    def make_inner(self):
        return {c: self._inner(core_tfms.RollingMean) for c in self.channels}


class RollingStdK(_RollingMixin, _StdKernel):
    def make_inner(self):
        return {c: self._inner(core_tfms.RollingMean) for c in self.channels}


class RollingMinK(_RollingMixin, _MinKernel):
    def make_inner(self):
        return {
            "min": self._inner(core_tfms.RollingMin),
            "count": self._inner(core_tfms.RollingMean),
        }


class RollingMaxK(_RollingMixin, _MaxKernel):
    def make_inner(self):
        return {
            "max": self._inner(core_tfms.RollingMax),
            "count": self._inner(core_tfms.RollingMean),
        }


class SeasonalRollingMeanK(_SeasonalMixin, _MeanKernel):
    def make_inner(self):
        return {c: self._inner(core_tfms.SeasonalRollingMean) for c in self.channels}


class SeasonalRollingStdK(_SeasonalMixin, _StdKernel):
    def make_inner(self):
        return {c: self._inner(core_tfms.SeasonalRollingMean) for c in self.channels}


class SeasonalRollingMinK(_SeasonalMixin, _MinKernel):
    def make_inner(self):
        return {
            "min": self._inner(core_tfms.SeasonalRollingMin),
            "count": self._inner(core_tfms.SeasonalRollingMean),
        }


class SeasonalRollingMaxK(_SeasonalMixin, _MaxKernel):
    def make_inner(self):
        return {
            "max": self._inner(core_tfms.SeasonalRollingMax),
            "count": self._inner(core_tfms.SeasonalRollingMean),
        }


class ExpandingMeanK(_ExpandingMixin, _MeanKernel):
    def make_inner(self, lag=None):
        return {c: self._inner(core_tfms.ExpandingMean, lag) for c in self.channels}


class ExpandingStdK(_ExpandingMixin, _StdKernel):
    def make_inner(self, lag=None):
        return {c: self._inner(core_tfms.ExpandingMean, lag) for c in self.channels}


class ExpandingMinK(_ExpandingMixin, _MinKernel):
    def make_inner(self, lag=None):
        return {
            "min": self._inner(core_tfms.ExpandingMin, lag),
            "count": self._inner(core_tfms.ExpandingMean, lag),
        }


class ExpandingMaxK(_ExpandingMixin, _MaxKernel):
    def make_inner(self, lag=None):
        return {
            "max": self._inner(core_tfms.ExpandingMax, lag),
            "count": self._inner(core_tfms.ExpandingMean, lag),
        }


class LagK(_PooledKernel):
    """Pooled ``Lag``: the collapsed bucket value at ``ordinal - lag``."""

    channels = ("sum", "count")
    primes_state = False

    def make_inner(self):
        return {c: core_tfms.Lag(lag=self.tfm.lag) for c in self.channels}

    def window_cells(self, ordinals):
        return np.where(ordinals - self.tfm.lag >= 0, 1.0, 0.0)

    def fit_from_store(self, store, _inner=None):
        # a window of one calendar position: the cell at ``ordinal - lag``, if any
        lo, hi = store.rolling_bounds(self.tfm.lag, 1)
        cells = store.channels_for(self.tfm.time_agg, self.channels)
        return self.combine(store.reduce_windows(lo, hi, 1, cells), 1.0)

    def min_samples(self):
        return 1.0

    def combine(self, res, k):
        ok = (self._n_obs(res, k) >= 1.0) & (res["count"] > 0)
        out = np.full(res["sum"].shape, np.nan)
        np.divide(res["sum"], res["count"], out=out, where=ok)
        return out


class EwmK(_PooledKernel):
    """Pooled exponentially weighted mean over the per-timestamp value.

    This is the one kernel whose predict step cannot delegate to
    ``coreforecast``.  A cell with no observations must leave the decay
    untouched, but the grouped ``update`` advances every bucket in lockstep and
    cannot skip one, and a ratio of two exponentially weighted channels does
    *not* reproduce the skip either -- it decays the accumulated mass and so
    over-weights the next observation.  The recurrence is three lines, so it is
    carried here and applied only where the bucket is actually observed. At
    fit the observed cells of a bucket are exactly the ragged series that
    recurrence walks, so there ``coreforecast`` runs it after all.
    """

    channels = ("sum", "count")
    custom = True

    def make_inner(self):
        return {"_state": {"s": None, "started": None, "next_src": 0}}

    def fit_from_store(self, store, inner=None):
        cells = store.channels_for(self.tfm.time_agg, self.channels)
        counts = cells["count"]
        consumed = max(store.width - self.lag, 0)
        # the cells the fit folds: observed, and a source for some ordinal
        obs = (counts > 0) & (store.ordinal < consumed)
        v = np.divide(cells["sum"], counts, out=np.zeros(counts.shape), where=obs)
        ewm = core_tfms.ExponentiallyWeightedMean(lag=0, alpha=self.tfm.alpha)
        s = ewm.transform(store.ragged(v, obs)) if obs.any() else np.empty(0)
        # the value at t is the state after the last observed source cell,
        # found by counting observed cells up to the source bound
        seen = np.concatenate([[0], np.cumsum(obs)])
        first = store.starts[store.bucket]
        pos = seen[store.source_bound(self.lag)]
        started = pos > seen[first]
        values = np.full(store.size, np.nan)
        values[started] = s[pos[started] - 1]
        if inner is not None:
            st = inner["_state"]
            last = seen[store.starts[1:]]
            st["started"] = last > seen[store.starts[:-1]]
            st["s"] = np.zeros(store.n_buckets)
            st["s"][st["started"]] = s[last[st["started"]] - 1]
            # negative when the lag outruns the calendar; `run_update` skips
            # sources that predate it, so the cursor can start there
            st["next_src"] = store.width - self.lag
        return values

    def remap_buckets(self, inner, remap, n_new):
        # next_src is a calendar cursor shared by every bucket, so it doesn't move
        st = inner["_state"]
        if st["s"] is None:
            return
        st["s"] = _grow_rows(st["s"], remap, n_new, 0.0)
        st["started"] = _grow_rows(st["started"], remap, n_new, False)

    def _fold(self, s, started, vals, obs):
        a = self.tfm.alpha
        stepped = np.where(started, (1.0 - a) * s + a * vals, vals)
        return np.where(obs, stepped, s), started | obs

    @staticmethod
    def _cell(cells, col):
        counts = cells["count"][:, col]
        obs = counts > 0
        vals = np.divide(
            cells["sum"][:, col], counts, out=np.zeros(counts.shape), where=obs
        )
        return vals, obs

    def run_transform(self, state, st):
        cells = state.channels(self.tfm.time_agg, self.channels)
        n_buckets, width = cells["count"].shape
        out = np.full((n_buckets, width), np.nan)
        s = np.zeros(n_buckets)
        started = np.zeros(n_buckets, dtype=bool)
        for t in range(width):
            src = t - self.lag
            if src >= 0:
                s, started = self._fold(s, started, *self._cell(cells, src))
            out[:, t] = np.where(started, s, np.nan)
        st["s"], st["started"] = s, started
        st["next_src"] = width - self.lag
        return out

    def run_update(self, state, st):
        cells = state.channels(self.tfm.time_agg, self.channels)
        n_ordinals, width = state.n_ordinals, state.width
        s, started = st["s"], st["started"]
        target = n_ordinals - self.lag
        offset = n_ordinals - width
        for src in range(st["next_src"], target + 1):
            if src < 0:
                continue  # source ordinal predates the calendar; nothing to fold
            col = src - offset
            if col < 0:
                # the cell existed and was trimmed away; folding on regardless
                # would silently under-decay, so fail instead of skipping
                raise RuntimeError(
                    f"pooled EWM needs calendar column {src} but the state starts "
                    f"at {offset} (lag={self.lag}); it was trimmed below its "
                    "retention."
                )
            s, started = self._fold(s, started, *self._cell(cells, col))
        st["s"], st["started"] = s, started
        st["next_src"] = max(st["next_src"], target + 1)
        return np.where(started, s, np.nan)


class _RowKernel(_PooledKernel):
    """Kernel that needs the raw observations, not the moment channels.

    Quantiles cannot be recovered from sums and counts, so these gather the rows
    inside each window instead.  Two things keep that affordable: values are only
    produced at the cells that actually carry rows (the only ones `scatter` ever
    reads back), and windows of equal length are gathered into one rectangular
    block so the statistic runs once per distinct length rather than once per
    position.
    """

    #: not read by the row gather, but the state needs one channel for its
    #: width/append/trim bookkeeping
    channels = ("count",)
    custom = True
    needs_rows = True
    #: the gather reads `rows_ord`/`rows_y`, never a primed inner
    primes_state = False
    #: cap on a temporary gather block, so a long expanding window is processed
    #: in chunks instead of being materialised all at once
    _max_gather = _MAX_GATHER

    def make_inner(self):
        return {"_state": {}}

    def stat(self, mat: np.ndarray) -> np.ndarray:
        """Reduce each row of a ``(n_windows, window_len)`` block."""
        raise NotImplementedError

    def window_bounds(self, ords: np.ndarray, targets: np.ndarray):
        """Half-open row bounds of each target's window, vectorised."""
        raise NotImplementedError

    def values_at(self, ords, ys, targets):
        lo, hi = self.window_bounds(ords, targets)
        out = np.full(targets.shape, np.nan)
        lengths = hi - lo
        usable = np.flatnonzero((lengths > 0) & (lengths >= self.min_samples()))
        if usable.size == 0:
            return out
        order = usable[np.argsort(lengths[usable], kind="stable")]
        sorted_len = lengths[order]
        starts = np.flatnonzero(np.r_[True, sorted_len[1:] != sorted_len[:-1]])
        for a, b in zip(starts, np.r_[starts[1:], order.size]):
            grp = order[a:b]
            width = int(sorted_len[a])
            step = max(1, self._max_gather // width)
            offsets = np.arange(width)
            for c in range(0, grp.size, step):
                sel = grp[c : c + step]
                out[sel] = self.stat(ys[lo[sel][:, None] + offsets])
        return out

    @staticmethod
    def _occupied(ords: np.ndarray) -> np.ndarray:
        """Distinct ordinals present, which are the only cells read back."""
        if ords.size == 0:
            return ords
        return ords[np.r_[True, ords[1:] != ords[:-1]]]

    def run_transform(self, state, _st):
        out = np.full((state.n_buckets, state.width), np.nan)
        for b in range(state.n_buckets):
            ords, ys = state.rows(b, self.tfm.time_agg)
            if ords.size == 0:
                continue
            targets = self._occupied(ords)
            targets = targets[targets < state.width]
            if targets.size:
                out[b, targets] = self.values_at(ords, ys, targets)
        return out

    def run_update(self, state, _st):
        out = np.full(state.n_buckets, np.nan)
        target = np.array([state.n_ordinals])
        for b in range(state.n_buckets):
            ords, ys = state.rows(b, self.tfm.time_agg)
            if ords.size:
                out[b] = self.values_at(ords, ys, target)[0]
        return out


class _RollingRowMixin:
    def window_bounds(self, ords, targets):
        hi_ord = targets - self.lag
        lo_ord = hi_ord - self.tfm.window_size + 1
        return (
            np.searchsorted(ords, lo_ord, side="left"),
            np.searchsorted(ords, hi_ord, side="right"),
        )


class _ExpandingRowMixin:
    def window_bounds(self, ords, targets):
        return (
            np.zeros(np.shape(targets), dtype=np.int64),
            np.searchsorted(ords, targets - self.lag, side="right"),
        )

    def min_samples(self):
        return 1.0


class RollingQuantileK(_RollingRowMixin, _RowKernel):
    def stat(self, mat):
        return np.quantile(mat, self.tfm.p, axis=1)


class ExpandingQuantileK(_ExpandingRowMixin, _RowKernel):
    def stat(self, mat):
        return np.quantile(mat, self.tfm.p, axis=1)


class SeasonalRollingQuantileK(_RowKernel):
    """Seasonal windows are strided, so the rows are gathered per season offset."""

    def values_at(self, ords, ys, targets):
        sl, w = self.tfm.season_length, self.tfm.window_size
        ms = self.min_samples()
        out = np.full(targets.shape, np.nan)
        # one searchsorted pair per (target, season offset) instead of a full
        # membership scan per target
        wanted = targets[:, None] - self.lag - np.arange(w) * sl
        lo = np.searchsorted(ords, wanted, side="left")
        hi = np.searchsorted(ords, np.where(wanted >= 0, wanted, -1), side="right")
        hi = np.where(wanted >= 0, hi, lo)
        counts = (hi - lo).sum(axis=1)
        for i in np.flatnonzero(counts >= max(ms, 1)):
            vals = np.concatenate(
                [ys[lo[i, j] : hi[i, j]] for j in range(w) if hi[i, j] > lo[i, j]]
            )
            out[i] = float(np.quantile(vals, self.tfm.p))
        return out


class LookupLagK(_RowKernel):
    """Target from the previous matching occurrence within the bucket.

    Position-based rather than calendar-based: the value is the observation
    ``lag`` *occurrences* back inside the (id, partition) bucket, however far
    away in time that is.
    """

    def values_at(self, ords, ys, targets):
        j = np.searchsorted(ords, targets, side="left") - self.tfm._core_tfm.lag
        out = np.full(np.shape(targets), np.nan)
        ok = (j >= 0) & (j < ys.size)
        if ok.any():
            out[ok] = ys[j[ok]]
        return out


_KERNELS = {
    "Lag": LagK,
    "RollingQuantile": RollingQuantileK,
    "ExpandingQuantile": ExpandingQuantileK,
    "SeasonalRollingQuantile": SeasonalRollingQuantileK,
    "LookupLag": LookupLagK,
    "RollingMean": RollingMeanK,
    "RollingStd": RollingStdK,
    "RollingMin": RollingMinK,
    "RollingMax": RollingMaxK,
    "SeasonalRollingMean": SeasonalRollingMeanK,
    "SeasonalRollingStd": SeasonalRollingStdK,
    "SeasonalRollingMin": SeasonalRollingMinK,
    "SeasonalRollingMax": SeasonalRollingMaxK,
    "ExpandingMean": ExpandingMeanK,
    "ExpandingStd": ExpandingStdK,
    "ExpandingMin": ExpandingMinK,
    "ExpandingMax": ExpandingMaxK,
    "ExponentiallyWeightedMean": EwmK,
}


#: base aggregate a ``time_agg`` collapses from
_TIME_AGG_SOURCE = {
    "count": "count",
    "sum": "sum",
    "mean": "sum",
    "min": "min",
    "max": "max",
}


def base_channels(kernel_channels: Sequence[str], time_agg: Optional[str]) -> set:
    """Base aggregates needed to serve a kernel's channels under ``time_agg``."""
    if time_agg:
        return {"count", _TIME_AGG_SOURCE[time_agg]}
    return set(kernel_channels) | {"count"}


def get_kernel(tfm) -> _PooledKernel:
    name = type(tfm).__name__
    kernel_cls = _KERNELS.get(name)
    if kernel_cls is None:
        raise NotImplementedError(
            f"{name!r} does not support pooled (global_/groupby/partition_by) "
            "computation. Supported: " + ", ".join(sorted(_KERNELS))
        )
    return kernel_cls(tfm)


# %% state
class PooledState:
    """Per-bucket aggregate store shared by every leaf with the same bucket key.

    The key is ``(mode, group_cols, partition_cols)`` -- deliberately *not*
    including ``time_agg``. ``time_agg`` doesn't change which rows land in a
    bucket, only how they are summarised, so it is a *view* over the same store
    rather than a separate one. That keeps the expensive part (one ``bincount``
    pass over the panel) shared, and makes each collapsed view an O(buckets x
    width) derivation that is cached.

    Layout is a dense ``(n_buckets, width)`` block per aggregate over the shared
    calendar, which is what makes the window a true ``RANGE`` window. Buckets
    that start late carry ``count == 0`` cells and so contribute nothing.

    At fit the block is not built up front: `build` keeps a `_CellStore` of the
    occupied cells, the bounded-window kernels compute on it directly, and the
    dense block is only derived when something reads it whole -- or, after
    `trim_to_last`, at the width the predict loop keeps.
    """

    def __init__(
        self,
        mode: str,
        group_cols: List[str],
        partition_cols: List[str],
        n_buckets: int,
        n_ordinals: int,
        base: Optional[Dict[str, np.ndarray]],
        series_bucket_id: np.ndarray,
        bucket_uniques: Optional[np.ndarray] = None,
        store: Optional[_CellStore] = None,
    ):
        self.mode = mode
        self.group_cols = group_cols
        self.partition_cols = partition_cols
        self.n_buckets = n_buckets
        self.n_ordinals = n_ordinals
        #: fit-time cell store; dropped by `finish_fit` once the features exist
        self._store = store
        self._base: Optional[Dict[str, np.ndarray]] = None
        if base is not None:
            self.base = base
        elif store is not None:
            self.channel_names = tuple(store.chan)
            self._width = store.width
        else:
            raise ValueError("PooledState needs either `base` or `store`")
        self.series_bucket_id = series_bucket_id
        self.bucket_uniques = bucket_uniques
        self._views: Dict[str, Dict[str, np.ndarray]] = {}
        # raw observations, kept only when a row-level kernel needs them
        self.rows_ord: Optional[List[np.ndarray]] = None
        self.rows_y: Optional[List[np.ndarray]] = None
        self._pend_ord: Optional[List[List[int]]] = None
        self._pend_y: Optional[List[List[float]]] = None
        #: rows appended since the last flush; a counter rather than a scan, so
        #: `rows()` stays O(1) when nothing is pending
        self._n_pending = 0
        self._rows_views: Dict[Optional[str], Any] = {}

    # -- construction ----------------------------------------------------
    @classmethod
    def build(
        cls,
        *,
        mode: str,
        group_cols: List[str],
        partition_cols: List[str],
        bucket_id_by_row: np.ndarray,
        ordinal_by_row: np.ndarray,
        y: np.ndarray,
        n_buckets: int,
        n_ordinals: int,
        series_bucket_id: np.ndarray,
        needed: Sequence[str],
        bucket_uniques: Optional[np.ndarray] = None,
        needs_rows: bool = False,
    ) -> "PooledState":
        store = _CellStore(
            bucket_id_by_row, ordinal_by_row, y, n_buckets, n_ordinals, sorted(needed)
        )
        obj = cls(
            mode=mode,
            group_cols=group_cols,
            partition_cols=partition_cols,
            n_buckets=n_buckets,
            n_ordinals=n_ordinals,
            base=None,
            series_bucket_id=series_bucket_id,
            bucket_uniques=bucket_uniques,
            store=store,
        )
        if needs_rows:
            obj._set_rows(bucket_id_by_row, ordinal_by_row, y, n_buckets)
        return obj

    def _set_rows(self, bid, ordinal, y, n_buckets) -> None:
        order = np.lexsort((ordinal, bid))
        rb, ro, ry = bid[order], ordinal[order], y[order]
        idx = np.arange(n_buckets)
        starts = np.searchsorted(rb, idx, side="left")
        ends = np.searchsorted(rb, idx, side="right")
        self.rows_ord = [ro[a:b].astype(np.int64) for a, b in zip(starts, ends)]
        self.rows_y = [ry[a:b].astype(np.float64) for a, b in zip(starts, ends)]
        self._pend_ord = [[] for _ in range(n_buckets)]
        self._pend_y = [[] for _ in range(n_buckets)]
        self._n_pending = 0
        self._rows_views = {}

    @property
    def base(self) -> Dict[str, np.ndarray]:
        """Dense ``(n_buckets, width)`` block per channel.

        Derived from the cell store on first use, so a fit that only ever reads
        column ranges never pays for the full width.
        """
        if self._base is None:
            assert self._store is not None
            self._base = self._store.dense(0, self._width)
        return self._base

    @base.setter
    def base(self, value: Dict[str, np.ndarray]) -> None:
        self._base = value
        self.channel_names = tuple(value)
        self._width = next(iter(value.values())).shape[1]

    @property
    def width(self) -> int:
        return self._width

    def finish_fit(self) -> None:
        """Materialise the predict-time block and drop the fit-time store."""
        if self._base is None and self._store is not None:
            self._base = self._store.dense(0, self._width)
        self._store = None

    @property
    def ordinal_offset(self) -> int:
        """Calendar cells dropped by ``trim_to_last``; 0 while untrimmed.

        ``append`` advances ``n_ordinals`` and ``width`` together, so this is
        fixed by the trim and then constant.
        """
        return self.n_ordinals - self.width

    # -- views -----------------------------------------------------------
    def channels(
        self, time_agg: Optional[str], names: Optional[Collection[str]] = None
    ) -> Dict[str, np.ndarray]:
        """Aggregate block per channel, collapsed by ``time_agg`` if given.

        ``names`` is what the calling kernel reads; a cached view holding fewer
        channels than that is widened in place, which can only happen while
        priming (before any append) and at most once per channel.
        """
        if time_agg is None:
            return self.base
        want = set(names or _ALL_CHANNELS) | {"count"}
        view = self._views.get(time_agg)
        if view is None:
            self._views[time_agg] = _collapse(self.base, time_agg, want)
        elif not want <= view.keys():
            view.update(_collapse(self.base, time_agg, want - view.keys()))
        return self._views[time_agg]

    def rows(self, b: int, time_agg: Optional[str] = None):
        """Materialised ``(ordinals, values)`` for one bucket, newest last."""
        self._flush_pending()
        if time_agg is None:
            assert self.rows_ord is not None and self.rows_y is not None
            return self.rows_ord[b], self.rows_y[b]
        view = self._rows_views.get(time_agg)
        if view is None:
            view = self._collapse_rows(time_agg)
            self._rows_views[time_agg] = view
        return view[0][b], view[1][b]

    def _flush_pending(self) -> None:
        if self._pend_ord is None or self._pend_y is None:
            return
        if not self._n_pending:
            return
        assert self.rows_ord is not None and self.rows_y is not None
        for b, pend in enumerate(self._pend_ord):
            if not pend:
                continue
            self.rows_ord[b] = np.concatenate(
                [self.rows_ord[b], np.asarray(pend, dtype=np.int64)]
            )
            self.rows_y[b] = np.concatenate(
                [self.rows_y[b], np.asarray(self._pend_y[b], dtype=np.float64)]
            )
            pend.clear()
            self._pend_y[b].clear()
        self._n_pending = 0
        self._rows_views = {}

    def _collapse_rows(self, time_agg: str):
        """One row per observed (bucket, timestamp), collapsed by ``time_agg``."""
        assert self.rows_ord is not None and self.rows_y is not None
        ords_out, vals_out = [], []
        for o, v in zip(self.rows_ord, self.rows_y):
            if o.size == 0:
                ords_out.append(o)
                vals_out.append(v)
                continue
            starts = np.flatnonzero(np.r_[True, o[1:] != o[:-1]])
            uniq = o[starts]
            if time_agg == "count":
                vals = np.diff(np.r_[starts, o.size]).astype(np.float64)
            elif time_agg == "sum":
                vals = np.add.reduceat(v, starts)
            elif time_agg == "mean":
                vals = np.add.reduceat(v, starts) / np.diff(np.r_[starts, o.size])
            elif time_agg == "min":
                vals = np.minimum.reduceat(v, starts)
            else:
                vals = np.maximum.reduceat(v, starts)
            ords_out.append(uniq)
            vals_out.append(vals)
        return ords_out, vals_out

    # -- feature computation --------------------------------------------
    def _channel_ga(self, cells: Dict[str, np.ndarray], name: str) -> CoreGroupedArray:
        return _block_ga(cells[name])

    def _require_untrimmed(self) -> None:
        if self.ordinal_offset:
            raise RuntimeError(
                "PooledState.transform() can't run on a trimmed state: it reads "
                "the window factor off relative column positions and would re-prime "
                "the inner transforms from a truncated prefix. Priming must precede "
                "the keep_last_n trim."
            )

    def _fit_store(self) -> _CellStore:
        self._require_untrimmed()
        if self._store is None:
            raise RuntimeError(
                "PooledState needs the fit-time cell store for this, which "
                "finish_fit() drops once the features exist."
            )
        return self._store

    def fit_values(self, kernel, inner: Dict[str, Any]) -> np.ndarray:
        """Per-row feature values at fit, priming ``inner`` where it needs it.

        The channel kernels compute on the cell store, so nothing full-width is
        built for them. The row kernels take the dense block -- which
        materialises ``(n_buckets, width)`` per channel -- and read the occupied
        cells out of it.
        """
        store = self._fit_store()
        values = kernel.fit_from_store(store, inner)
        if values is None:
            values = store.gather(self.transform(kernel, inner), 0)
        return store.row_values(values)

    def prime(self, kernel, inner: Dict[str, Any]) -> None:
        """Leave ``inner`` as a fit would, without keeping the features."""
        if kernel.fit_from_store(self._fit_store(), inner) is None:
            self.transform(kernel, inner)

    def transform(self, kernel, inner: Dict[str, Any]) -> np.ndarray:
        """Full ``(n_buckets, width)`` feature block, priming the inner state."""
        self._require_untrimmed()
        if kernel.custom:
            return kernel.run_transform(self, inner["_state"])
        cells = self.channels(kernel.tfm.time_agg, kernel.channels)
        res = {
            name: tfm.transform(self._channel_ga(cells, name)).reshape(
                self.n_buckets, self.width
            )
            for name, tfm in inner.items()
        }
        k = kernel.window_cells(np.arange(self.width))[None, :]
        return kernel.combine(res, k)

    def update(self, kernel, inner: Dict[str, Any]) -> np.ndarray:
        """One value per bucket for the next timestamp, advancing inner state."""
        if kernel.custom:
            return kernel.run_update(self, inner["_state"])
        cells = self.channels(kernel.tfm.time_agg, kernel.channels)
        res = {
            name: tfm.update(self._channel_ga(cells, name)).reshape(self.n_buckets, 1)
            for name, tfm in inner.items()
        }
        k = kernel.window_cells(np.array([self.n_ordinals]))[None, :]
        return kernel.combine(res, k)[:, 0]

    def broadcast(self, values: np.ndarray) -> np.ndarray:
        """Map per-bucket values onto series using the current assignment."""
        bid = self.series_bucket_id
        out = np.full(bid.shape, np.nan)
        known = bid >= 0
        out[known] = values[bid[known]]
        return out

    # -- mutation --------------------------------------------------------
    def _cells_for(
        self, bucket_ids: np.ndarray, y: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """One new column per base aggregate from a single timestamp of values."""
        b = bucket_ids.astype(np.int64)
        v = np.asarray(y, dtype=np.float64)
        col: Dict[str, np.ndarray] = {}
        names = self.channel_names
        if "count" in names:
            col["count"] = np.bincount(b, minlength=self.n_buckets).astype(np.float64)
        if "sum" in names:
            col["sum"] = np.bincount(b, weights=v, minlength=self.n_buckets)
        if "sumsq" in names:
            col["sumsq"] = np.bincount(b, weights=v * v, minlength=self.n_buckets)
        if "min" in names:
            buf = np.full(self.n_buckets, np.inf)
            np.minimum.at(buf, b, v)
            col["min"] = buf
        if "max" in names:
            buf = np.full(self.n_buckets, -np.inf)
            np.maximum.at(buf, b, v)
            col["max"] = buf
        return col

    def append(
        self, y_hat: np.ndarray, bucket_ids: Optional[np.ndarray] = None
    ) -> None:
        """Fold one new timestamp into every aggregate."""
        bid = self.series_bucket_id if bucket_ids is None else bucket_ids
        known = bid >= 0
        b = bid[known].astype(np.int64)
        v = np.asarray(y_hat, dtype=np.float64)[known]
        col = self._cells_for(b, v)
        for name in self.base:
            self.base[name] = np.concatenate(
                [self.base[name], col[name][:, None]], axis=1
            )
        self._width += 1
        self._extend_views(col)
        if self._pend_ord is not None and self._pend_y is not None:
            # row kernels re-gather from history, so predictions land here too;
            # never trimmed, which is why they cost O(n_rows) memory
            for bi, yi in zip(b.tolist(), v.tolist()):
                self._pend_ord[bi].append(self.n_ordinals)
                self._pend_y[bi].append(yi)
            self._n_pending += len(b)
        self.n_ordinals += 1

    def _extend_views(self, col: Dict[str, np.ndarray]) -> None:
        """Extend each cached collapsed view with the newly appended column.

        ``_collapse`` is elementwise per cell, so collapsing the single new
        column yields exactly the column a full recollapse would produce.
        """
        if not self._views:
            return
        cells = {name: values[:, None] for name, values in col.items()}
        for time_agg, view in self._views.items():
            new = _collapse(cells, time_agg, list(view))
            for name in list(view):
                view[name] = np.concatenate([view[name], new[name]], axis=1)

    def grow_buckets(self, new_uniques: np.ndarray) -> Optional[np.ndarray]:
        """Merge new bucket keys in, returning the old -> new id mapping.

        ``bucket_uniques`` must stay sorted for `lookup`, so absorbing new keys
        can renumber existing buckets; every per-bucket structure is permuted to
        match. Returns ``None`` when the vocabulary didn't move, so the caller
        knows there is no per-kernel state to remap either.
        """
        if self.bucket_uniques is None:
            return None
        merged = np.union1d(self.bucket_uniques, new_uniques)
        if merged.size == self.bucket_uniques.size:
            return None
        remap = np.searchsorted(merged, self.bucket_uniques).astype(np.int64)
        n_new = merged.size
        for name, arr in self.base.items():
            grown = np.zeros((n_new, arr.shape[1]), dtype=arr.dtype)
            if name == "min":
                grown[:] = np.inf
            elif name == "max":
                grown[:] = -np.inf
            grown[remap] = arr
            self.base[name] = grown
        if self.rows_ord is not None and self.rows_y is not None:
            ro: List[np.ndarray] = [np.empty(0, np.int64) for _ in range(n_new)]
            ry: List[np.ndarray] = [np.empty(0, np.float64) for _ in range(n_new)]
            po: List[List[int]] = [[] for _ in range(n_new)]
            py: List[List[float]] = [[] for _ in range(n_new)]
            for old, new in enumerate(remap):
                ro[new], ry[new] = self.rows_ord[old], self.rows_y[old]
                if self._pend_ord is not None and self._pend_y is not None:
                    po[new], py[new] = self._pend_ord[old], self._pend_y[old]
            self.rows_ord, self.rows_y = ro, ry
            if self._pend_ord is not None:
                self._pend_ord, self._pend_y = po, py
                self._n_pending = sum(len(x) for x in po)
        known = self.series_bucket_id >= 0
        self.series_bucket_id = np.where(
            known, remap[np.clip(self.series_bucket_id, 0, None)], -1
        )
        self.bucket_uniques = merged
        self.n_buckets = n_new
        self._views = {}
        self._rows_views = {}
        return remap

    def trim_to_last(self, n: int) -> None:
        """Keep only the last `n` calendar cells. ``n_ordinals`` keeps counting.

        The raw row store is trimmed on the same criterion, by absolute ordinal:
        a row-level kernel reaching further back than `n` would be reading cells
        that no longer exist either.
        """
        if n <= 0 or n >= self.width:
            return
        if self._base is None:
            # straight from the cell store, so only the kept tail is ever built
            assert self._store is not None
            self._base = self._store.dense(self.width - n, self.width)
        else:
            for name in self._base:
                self._base[name] = np.ascontiguousarray(self._base[name][:, -n:])
        self._width = n
        self._store = None
        self._views = {}
        if self.rows_ord is not None and self.rows_y is not None:
            self._flush_pending()
            cutoff = self.n_ordinals - n
            for b, ords in enumerate(self.rows_ord):
                if ords.size and ords[0] < cutoff:
                    keep = int(np.searchsorted(ords, cutoff, side="left"))
                    self.rows_ord[b] = ords[keep:]
                    self.rows_y[b] = self.rows_y[b][keep:]
            self._rows_views = {}

    def set_series_bucket_id(self, bucket_id: np.ndarray) -> None:
        self.series_bucket_id = bucket_id

    # -- cheap rollback for TimeSeries._backup ---------------------------
    def snapshot(self):
        # aggregates are only appended to during predict, so copying the array
        # references is enough -- no deep copy of the whole state per model
        self._flush_pending()
        return (
            dict(self.base),
            self.n_ordinals,
            self.series_bucket_id,
            self.n_buckets,
            self.bucket_uniques,
            list(self.rows_ord) if self.rows_ord is not None else None,
            list(self.rows_y) if self.rows_y is not None else None,
        )

    def restore(self, snap) -> None:
        (
            base,
            self.n_ordinals,
            self.series_bucket_id,
            self.n_buckets,
            self.bucket_uniques,
            rows_ord,
            rows_y,
        ) = snap
        self.base = dict(base)
        self.rows_ord = list(rows_ord) if rows_ord is not None else None
        self.rows_y = list(rows_y) if rows_y is not None else None
        if self.rows_ord is not None:
            self._pend_ord = [[] for _ in range(self.n_buckets)]
            self._pend_y = [[] for _ in range(self.n_buckets)]
            self._n_pending = 0
        self._views = {}
        self._rows_views = {}

    def __repr__(self) -> str:
        return (
            f"PooledState(mode={self.mode}, n_buckets={self.n_buckets}, "
            f"width={self.width}, n_ordinals={self.n_ordinals})"
        )
