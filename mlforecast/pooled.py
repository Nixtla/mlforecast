__all__ = ["PooledState", "compute_pooled_features"]

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import narwhals as nw
import numpy as np
import utilsforecast.processing as ufp

from .grouped_array import GroupedArray
from .lag_transforms import _BaseLagTransform


def _dedupe_preserve_order(items):
    return list(dict.fromkeys(items))


def add_bucket_id(data, cols):
    cols = list(cols)
    data_nw = nw.from_native(data)
    groups_nw = data_nw.select(cols).unique(maintain_order=True)
    groups_nw = groups_nw.with_row_index(name="_bucket_id").with_columns(
        nw.col("_bucket_id").cast(nw.Int64)
    )
    merged_nw = data_nw.join(groups_nw, on=cols, how="left")
    return (
        ufp.drop_index_if_pandas(nw.to_native(merged_nw)),
        ufp.drop_index_if_pandas(nw.to_native(groups_nw)),
    )


def lookup_bucket_ids(data, groups, cols):
    cols = list(cols)
    data_nw = nw.from_native(data)
    groups_nw = nw.from_native(groups)
    data_slice = data_nw.select(cols)
    for col in cols:
        s1, s2 = ufp.match_if_categorical(
            nw.to_native(data_slice.get_column(col)),
            nw.to_native(groups_nw.get_column(col)),
        )
        data_slice = data_slice.with_columns(nw.from_native(s1, series_only=True))
        groups_nw = groups_nw.with_columns(nw.from_native(s2, series_only=True))
    joined = data_slice.join(groups_nw, on=cols, how="left")
    return joined.get_column("_bucket_id").to_numpy()


@dataclass
class _TimestampAggregates:
    """Per-timestamp aggregates for a single bucket."""

    unique_times: np.ndarray
    sums: np.ndarray
    counts: np.ndarray
    n_rows: np.ndarray
    is_balanced: bool
    sum_sq: np.ndarray
    mins: np.ndarray
    maxs: np.ndarray


def _build_ts_aggs(
    bid_arr: np.ndarray,
    ord_arr: np.ndarray,
    y_arr: np.ndarray,
) -> Dict[int, _TimestampAggregates]:
    aggs: Dict[int, _TimestampAggregates] = {}
    for bid in np.unique(bid_arr):
        mask = bid_arr == bid
        ord_b = ord_arr[mask]
        y_b = y_arr[mask]
        unique_ord, inv = np.unique(ord_b, return_inverse=True)
        m = len(unique_ord)
        valid = ~np.isnan(y_b)
        y_valid = np.where(valid, y_b, 0.0)
        sums = np.bincount(inv, weights=y_valid, minlength=m)
        counts = np.bincount(inv, weights=valid.astype(float), minlength=m)
        n_rows = np.bincount(inv, minlength=m).astype(float)
        is_balanced = bool(n_rows.size > 0 and np.all(n_rows == n_rows[0]))
        sum_sq = np.bincount(inv, weights=np.where(valid, y_b**2, 0.0), minlength=m)
        mins = np.full(m, np.inf)
        maxs = np.full(m, -np.inf)
        valid_inv = inv[valid]
        valid_y = y_b[valid]
        if len(valid_y) > 0:
            np.minimum.at(mins, valid_inv, valid_y)
            np.maximum.at(maxs, valid_inv, valid_y)
        no_valid = mins == np.inf
        mins[no_valid] = np.nan
        maxs[no_valid] = np.nan
        aggs[int(bid)] = _TimestampAggregates(
            unique_times=unique_ord,
            sums=sums,
            counts=counts,
            n_rows=n_rows,
            is_balanced=is_balanced,
            sum_sq=sum_sq,
            mins=mins,
            maxs=maxs,
        )
    return aggs


def _compute_time_index(bid_arr, ts_arr):
    """Assign integer period coordinates over the validated regular time grid.

    For each bucket, maps raw timestamps to consecutive integers based on
    the bucket's sorted unique timestamps. Because the input grid is validated
    as regular (no gaps within a series), this is equivalent to SQL RANGE
    interval semantics. Series that start later simply have no rows at earlier
    periods — no synthetic zeros are injected.
    """
    idx_arr = np.empty(len(ts_arr), dtype=np.int64)
    next_by_bucket = {}
    for bid in np.unique(bid_arr):
        mask = bid_arr == bid
        ts_b = ts_arr[mask]
        unique_ts = np.unique(ts_b)
        idx_arr[mask] = np.searchsorted(unique_ts, ts_b)
        next_by_bucket[int(bid)] = len(unique_ts)
    return idx_arr, next_by_bucket


def _compute_time_index_from_parent(bid_arr, ts_arr, parent_grids):
    """Assign ordinals using the parent calendar's time grid.

    For partition_by states, ordinals must reflect the parent calendar so
    that missing partition values leave holes.  E.g. if the parent has
    timestamps [1,2,3,4,5] but a partition bucket only has [1,3,5], the
    ordinals are [0,2,4] (not [0,1,2]).

    Parameters
    ----------
    bid_arr : np.ndarray
        Bucket ID for each observation.
    ts_arr : np.ndarray
        Timestamp for each observation.
    parent_grids : Dict[int, np.ndarray]
        Maps each bucket_id to the sorted unique timestamps of its parent
        calendar.

    Returns
    -------
    idx_arr : np.ndarray
        Ordinal coordinates derived from parent calendar positions.
    next_by_bucket : Dict[int, int]
        Next ordinal for each bucket (= len(parent_grid)).
    """
    idx_arr = np.empty(len(ts_arr), dtype=np.int64)
    next_by_bucket = {}
    for bid in np.unique(bid_arr):
        mask = bid_arr == bid
        ts_b = ts_arr[mask]
        parent_ts = parent_grids[int(bid)]
        idx_arr[mask] = np.searchsorted(parent_ts, ts_b)
        next_by_bucket[int(bid)] = len(parent_ts)
    return idx_arr, next_by_bucket


def _compute_idsorted_to_bucket_pos(bucket_df, id_col, time_col):
    _pos_col = "_idsorted_tmp_pos"
    df_nw = nw.from_native(bucket_df)
    df_nw = (
        df_nw.select([id_col, time_col])
        .with_row_index(name=_pos_col)
        .with_columns(nw.col(_pos_col).cast(nw.Int64))
    )
    idsorted = ufp.sort(nw.to_native(df_nw), by=[id_col, time_col])
    return idsorted[_pos_col].to_numpy()


@dataclass
class PooledState:
    """Holds all arrays and metadata for a pooled bucket.

    Covers global, groupby, and partition_by modes (and combinations thereof).

    ``time_index`` is an integer period coordinate over a validated regular
    time grid.  For global/group states (no partition), the input is validated
    to have no missing timestamps within any series.  Different series in the
    same bucket may have different start dates; only actual observations are
    included (no synthetic zeros).  Under this invariant, ``time_index``
    is equivalent to SQL ``RANGE BETWEEN … PRECEDING`` interval semantics.

    For partition_by buckets, ``time_index`` is derived from the parent
    series/global calendar — not the bucket's observed timestamps — so that
    missing partition values leave holes and window bounds remain
    interval-correct.

    **Ordering contract**: ``bucket_id``, ``time``, ``time_index``, and ``y``
    arrays are positionally aligned — row *i* in each describes the same
    observation.  ``ga`` is **not** positionally aligned with these arrays
    after mutations (``append_predictions`` / ``append_observations``):
    ``ga.append_several`` interleaves new values within each group, while the
    flat arrays append at the tail.  ``ga`` is only used for
    ``_initialize_lag_transform_states`` (called once before any mutations);
    all feature computation uses the flat arrays via ``compute_pooled_features``.
    """

    ga: GroupedArray
    bucket_df: Any
    groups: Any
    group_cols: Optional[List[str]]
    series_bucket_id: np.ndarray
    bucket_id: np.ndarray
    time: np.ndarray
    time_index: np.ndarray
    y: np.ndarray
    next_time_index_by_bucket: Dict[int, int]
    join_cols: List[str]
    mode: str = "nonlocal"
    partition_cols: Optional[List[str]] = None
    key_cols: Optional[List[str]] = None
    parent_scope_cols: Optional[List[str]] = None
    _parent_time_grids: Optional[Dict[int, np.ndarray]] = None
    _bucket_to_parent_id: Optional[Dict[int, int]] = None
    _parent_to_buckets: Optional[Dict[int, List[int]]] = None
    _scope_key_to_parent_id: Optional[Dict[tuple, int]] = None
    _ts_aggs: Dict[int, _TimestampAggregates] = field(default_factory=dict)
    _idsorted_to_bucket_pos: Optional[np.ndarray] = None

    @property
    def group_uids(self):
        if self.groups is None:
            ns = nw.get_native_namespace(self.bucket_df)
            return nw.to_native(nw.new_series("_bucket_id", [0], dtype=nw.Int64, backend=ns))
        return nw.to_native(nw.from_native(self.groups).get_column("_bucket_id").sort())

    @classmethod
    def from_global(
        cls,
        sorted_df,
        id_col: str,
        time_col: str,
        target_col: str,
        ga_data_dtype,
        n_series: int,
    ):
        keep_cols = _dedupe_preserve_order([id_col, time_col, target_col])
        global_df = sorted_df[keep_cols]
        global_df = ufp.sort(global_df, by=[time_col, id_col])
        global_df = ufp.drop_index_if_pandas(global_df)
        ts_raw = global_df[time_col].to_numpy()
        y_raw = global_df[target_col].to_numpy().astype(ga_data_dtype)
        global_df_nw = nw.from_native(global_df)
        global_df_nw = global_df_nw.with_row_index(name="_bucket_pos").with_columns(
            nw.col("_bucket_pos").cast(nw.Int64)
        )
        process_df_nw = global_df_nw.select([
            nw.lit(0).cast(nw.Int64).alias("_bucket_id"),
            "_bucket_pos",
            target_col,
        ])
        global_df = nw.to_native(global_df_nw)
        process_df = nw.to_native(process_df_nw)
        processed = ufp.process_df(
            process_df,
            id_col="_bucket_id",
            time_col="_bucket_pos",
            target_col=target_col,
        )
        ga = GroupedArray(processed.data[:, 0], processed.indptr)
        unique_ts = np.unique(ts_raw)
        ord_raw = np.searchsorted(unique_ts, ts_raw).astype(np.int64)
        bid_arr = np.zeros(len(global_df), dtype=np.int64)
        y_float = y_raw.astype(float)
        return cls(
            ga=ga,
            bucket_df=global_df,
            groups=None,
            group_cols=None,
            series_bucket_id=np.zeros(n_series, dtype=np.int64),
            bucket_id=bid_arr,
            time=ts_raw,
            time_index=ord_raw,
            y=y_float,
            next_time_index_by_bucket={0: len(unique_ts)},
            join_cols=[id_col, time_col],
            _ts_aggs=_build_ts_aggs(bid_arr, ord_raw, y_float),
        )

    @classmethod
    def from_groupby(
        cls,
        df_for_group,
        group_cols_list: List[str],
        id_col: str,
        time_col: str,
        target_col: str,
        ga_data_dtype,
        static_features,
    ):
        keep_cols = _dedupe_preserve_order(
            [id_col] + group_cols_list + [time_col, target_col]
        )
        bucket_df = df_for_group[keep_cols]
        bucket_df = ufp.sort(bucket_df, by=group_cols_list + [time_col, id_col])
        bucket_df = ufp.drop_index_if_pandas(bucket_df)
        bucket_df, groups = add_bucket_id(bucket_df, group_cols_list)
        ts_raw = bucket_df[time_col].to_numpy()
        y_raw = bucket_df[target_col].to_numpy().astype(ga_data_dtype)
        bid_raw = bucket_df["_bucket_id"].to_numpy()
        bucket_df_nw = nw.from_native(bucket_df)
        bucket_df_nw = bucket_df_nw.with_columns(
            (nw.col("_bucket_id").cum_count() - 1).cast(nw.Int64).alias("_global_idx")
        )
        group_starts = bucket_df_nw.group_by("_bucket_id").agg(
            nw.col("_global_idx").min().alias("_group_start")
        )
        bucket_df_nw = (
            bucket_df_nw
            .join(group_starts, on="_bucket_id", how="left")
            .with_columns(
                (nw.col("_global_idx") - nw.col("_group_start")).cast(nw.Int64).alias("_bucket_pos")
            )
            .drop(["_global_idx", "_group_start"])
        )
        process_df_nw = bucket_df_nw.select(["_bucket_id", "_bucket_pos", target_col])
        bucket_df = nw.to_native(bucket_df_nw)
        process_df = nw.to_native(process_df_nw)
        processed = ufp.process_df(
            process_df,
            id_col="_bucket_id",
            time_col="_bucket_pos",
            target_col=target_col,
        )
        if processed.sort_idxs is not None:
            bucket_df = ufp.take_rows(bucket_df, processed.sort_idxs)
            ts_raw = ts_raw[processed.sort_idxs]
            y_raw = y_raw[processed.sort_idxs]
            bid_raw = bid_raw[processed.sort_idxs]
        bucket_df = ufp.drop_index_if_pandas(bucket_df)
        ga = GroupedArray(processed.data[:, 0], processed.indptr)
        bid_arr = bid_raw.astype(np.int64)
        ord_arr, next_by_bucket = _compute_time_index(bid_arr, ts_raw)
        series_bucket_id = lookup_bucket_ids(
            static_features, groups, group_cols_list
        ).astype(np.int64, copy=False)
        y_float = y_raw.astype(float)
        return cls(
            ga=ga,
            bucket_df=bucket_df,
            groups=groups,
            group_cols=group_cols_list,
            series_bucket_id=series_bucket_id,
            bucket_id=bid_arr,
            time=ts_raw,
            time_index=ord_arr,
            y=y_float,
            next_time_index_by_bucket=next_by_bucket,
            join_cols=[id_col, time_col],
            _ts_aggs=_build_ts_aggs(bid_arr, ord_arr, y_float),
        )

    @classmethod
    def from_partition(
        cls,
        sorted_df,
        mode: str,
        group_cols_list: Optional[List[str]],
        partition_cols_list: List[str],
        id_col: str,
        time_col: str,
        target_col: str,
        ga_data_dtype,
        static_features,
        n_series: int,
    ):
        """Build a PooledState for partition_by transforms.

        Parameters
        ----------
        mode : str
            "local" — each (id, partition_vals) is its own bucket.
            "nonlocal" — bucket key is (*group_cols, *partition_cols) or
            (*partition_cols) for global+partition_by.
        group_cols_list : list of str or None
            Group columns (from ``groupby``). None for global+partition or
            local+partition.
        partition_cols_list : list of str
            Partition columns (from ``partition_by``).
        """
        if mode == "local":
            key_cols = _dedupe_preserve_order([id_col] + partition_cols_list)
        elif group_cols_list:
            key_cols = _dedupe_preserve_order(
                list(group_cols_list) + partition_cols_list
            )
        else:
            key_cols = _dedupe_preserve_order(partition_cols_list)

        if mode == "local":
            join_cols = key_cols + [time_col]
            join_cols = _dedupe_preserve_order(join_cols)
        else:
            join_cols = [id_col, time_col]

        keep_cols = _dedupe_preserve_order(
            [id_col] + key_cols + [time_col, target_col]
        )
        bucket_df = sorted_df[keep_cols]
        bucket_df = ufp.sort(bucket_df, by=key_cols + [time_col, id_col])
        bucket_df = ufp.drop_index_if_pandas(bucket_df)
        bucket_df, groups = add_bucket_id(bucket_df, key_cols)
        ts_raw = bucket_df[time_col].to_numpy()
        y_raw = bucket_df[target_col].to_numpy().astype(ga_data_dtype)
        bid_raw = bucket_df["_bucket_id"].to_numpy()

        bucket_df_nw = nw.from_native(bucket_df)
        bucket_df_nw = bucket_df_nw.with_columns(
            (nw.col("_bucket_id").cum_count() - 1).cast(nw.Int64).alias("_global_idx")
        )
        group_starts = bucket_df_nw.group_by("_bucket_id").agg(
            nw.col("_global_idx").min().alias("_group_start")
        )
        bucket_df_nw = (
            bucket_df_nw
            .join(group_starts, on="_bucket_id", how="left")
            .with_columns(
                (nw.col("_global_idx") - nw.col("_group_start")).cast(nw.Int64).alias("_bucket_pos")
            )
            .drop(["_global_idx", "_group_start"])
        )
        process_df_nw = bucket_df_nw.select(["_bucket_id", "_bucket_pos", target_col])
        bucket_df = nw.to_native(bucket_df_nw)
        process_df = nw.to_native(process_df_nw)

        processed = ufp.process_df(
            process_df,
            id_col="_bucket_id",
            time_col="_bucket_pos",
            target_col=target_col,
        )
        if processed.sort_idxs is not None:
            bucket_df = ufp.take_rows(bucket_df, processed.sort_idxs)
            ts_raw = ts_raw[processed.sort_idxs]
            y_raw = y_raw[processed.sort_idxs]
            bid_raw = bid_raw[processed.sort_idxs]
        bucket_df = ufp.drop_index_if_pandas(bucket_df)
        ga = GroupedArray(processed.data[:, 0], processed.indptr)
        bid_arr = bid_raw.astype(np.int64)

        if mode == "local":
            parent_scope_cols = [id_col]
        elif group_cols_list:
            parent_scope_cols = list(group_cols_list)
        else:
            parent_scope_cols = None

        parent_id_map: Dict[tuple, int] = {}
        bucket_to_parent: Dict[int, int] = {}
        parent_grids: Dict[int, np.ndarray] = {}
        next_pid = 0

        if parent_scope_cols is not None:
            groups_nw = nw.from_native(groups)
            sorted_nw = nw.from_native(sorted_df)
            for bid in np.unique(bid_arr):
                row_nw = groups_nw.filter(nw.col("_bucket_id") == int(bid))
                scope_key = row_nw.select(parent_scope_cols).row(0)

                if scope_key not in parent_id_map:
                    pid = next_pid
                    next_pid += 1
                    parent_id_map[scope_key] = pid
                    exprs = [nw.col(c) == v for c, v in zip(parent_scope_cols, scope_key)]
                    combined = exprs[0]
                    for e in exprs[1:]:
                        combined = combined & e
                    filtered = sorted_nw.filter(combined)
                    parent_ts = np.sort(
                        filtered.get_column(time_col).unique().to_numpy()
                    )
                    parent_grids[pid] = parent_ts

                bucket_to_parent[int(bid)] = parent_id_map[scope_key]
        else:
            all_ts = sorted_df[time_col].to_numpy()
            global_ts = np.sort(np.unique(all_ts))
            parent_grids[0] = global_ts
            for bid in np.unique(bid_arr):
                bucket_to_parent[int(bid)] = 0

        parent_to_buckets: Dict[int, List[int]] = {}
        for bid, pid in bucket_to_parent.items():
            parent_to_buckets.setdefault(pid, []).append(bid)

        if parent_scope_cols is not None:
            scope_key_to_parent = dict(parent_id_map)
        else:
            scope_key_to_parent = {(): 0}

        per_bucket_grids = {
            bid: parent_grids[bucket_to_parent[bid]]
            for bid in bucket_to_parent
        }
        ord_arr, next_by_bucket = _compute_time_index_from_parent(
            bid_arr, ts_raw, per_bucket_grids
        )

        sf_cols = set(static_features.columns)
        can_lookup = all(c in sf_cols for c in key_cols)
        if can_lookup:
            series_bucket_id = lookup_bucket_ids(
                static_features, groups, key_cols
            ).astype(np.int64, copy=False)
        else:
            series_bucket_id = np.zeros(n_series, dtype=np.int64)

        y_float = y_raw.astype(float)
        return cls(
            ga=ga,
            bucket_df=bucket_df,
            groups=groups,
            group_cols=group_cols_list,
            series_bucket_id=series_bucket_id,
            bucket_id=bid_arr,
            time=ts_raw,
            time_index=ord_arr,
            y=y_float,
            next_time_index_by_bucket=next_by_bucket,
            join_cols=join_cols,
            mode=mode,
            partition_cols=partition_cols_list,
            key_cols=key_cols,
            parent_scope_cols=parent_scope_cols,
            _parent_time_grids=parent_grids,
            _bucket_to_parent_id=bucket_to_parent,
            _parent_to_buckets=parent_to_buckets,
            _scope_key_to_parent_id=scope_key_to_parent,
            _ts_aggs=_build_ts_aggs(bid_arr, ord_arr, y_float),
        )

    def update_series_bucket_id(self, context_df, _id_col: str):
        """Recompute series_bucket_id from a context dataframe.

        This is used at prediction time when partition_by values are dynamic
        (e.g. the partition key comes from exogenous features that change
        each step). If new partition combinations appear, new buckets are
        created.

        Parameters
        ----------
        context_df : DataFrame
            Must contain ``id_col`` and all ``key_cols`` columns. One row per
            series, representing the current partition assignment.
        id_col : str
            The series identifier column.
        """
        if self.key_cols is None:
            return
        key_cols = self.key_cols
        context_with_bid = _attach_bucket_id(context_df, self.groups, key_cols)
        context_with_bid, self.groups = _extend_groups(
            context_with_bid, self.groups, key_cols
        )
        new_bid = context_with_bid["_bucket_id"].to_numpy().astype(np.int64)
        self.series_bucket_id = new_bid
        groups_nw = nw.from_native(self.groups)
        for bid in np.unique(new_bid):
            bid_int = int(bid)
            if bid_int not in self.next_time_index_by_bucket:
                pid = self._resolve_parent_for_bucket(bid_int, groups_nw=groups_nw)
                if pid is not None and self._parent_time_grids is not None:
                    self.next_time_index_by_bucket[bid_int] = len(
                        self._parent_time_grids[pid]
                    )
                else:
                    self.next_time_index_by_bucket[bid_int] = 0
            if bid_int not in self._ts_aggs:
                self._ts_aggs[bid_int] = _TimestampAggregates(
                    unique_times=np.array([], dtype=np.intp),
                    sums=np.array([], dtype=np.float64),
                    counts=np.array([], dtype=np.intp),
                    n_rows=np.array([], dtype=np.float64),
                    is_balanced=True,
                    sum_sq=np.array([], dtype=np.float64),
                    mins=np.array([], dtype=np.float64),
                    maxs=np.array([], dtype=np.float64),
                )

    def _resolve_parent_for_bucket(self, bid: int, groups_nw=None) -> Optional[int]:
        """Find or create the parent_id for a bucket from its scope columns.

        ``groups_nw`` is an optional pre-wrapped Narwhals view of ``self.groups``;
        callers that resolve many buckets in a loop pass it once to avoid
        re-wrapping per bucket. When omitted it is wrapped internally.
        """
        if (
            self._bucket_to_parent_id is None
            or self._parent_to_buckets is None
            or self._parent_time_grids is None
            or self._scope_key_to_parent_id is None
        ):
            return None

        if bid in self._bucket_to_parent_id:
            return self._bucket_to_parent_id[bid]

        if self.parent_scope_cols is None:
            scope_key = ()
        else:
            if groups_nw is None:
                groups_nw = nw.from_native(self.groups)
            row_nw = groups_nw.filter(nw.col("_bucket_id") == bid)
            if len(row_nw) == 0:
                return None
            scope_key = row_nw.select(self.parent_scope_cols).row(0)

        if scope_key in self._scope_key_to_parent_id:
            pid = self._scope_key_to_parent_id[scope_key]
            self._bucket_to_parent_id[bid] = pid
            self._parent_to_buckets[pid].append(bid)
            return pid

        pid = max(self._parent_time_grids.keys()) + 1 if self._parent_time_grids else 0
        dtype = (
            next(iter(self._parent_time_grids.values())).dtype
            if self._parent_time_grids
            else self.time.dtype
        )
        self._parent_time_grids[pid] = np.array([], dtype=dtype)
        self._bucket_to_parent_id[bid] = pid
        self._parent_to_buckets[pid] = [bid]
        self._scope_key_to_parent_id[scope_key] = pid
        return pid

    def _advance_parent_calendars(self, new_ts_val):
        """Advance all parent calendars and sync sibling bucket ordinals.

        When a prediction is appended at ``new_ts_val``, every parent
        calendar grows by that timestamp and ALL sibling buckets under
        each parent update their ``next_time_index_by_bucket`` to the
        new parent grid length.
        """
        if self._parent_time_grids is None or self._parent_to_buckets is None:
            return
        for pid, grid in self._parent_time_grids.items():
            if len(grid) == 0 or new_ts_val > grid[-1]:
                self._parent_time_grids[pid] = np.append(grid, new_ts_val)
            else:
                pos = np.searchsorted(grid, new_ts_val)
                if pos >= len(grid) or grid[pos] != new_ts_val:
                    self._parent_time_grids[pid] = np.insert(grid, pos, new_ts_val)
            new_len = len(self._parent_time_grids[pid])
            for bid in self._parent_to_buckets.get(pid, []):
                self.next_time_index_by_bucket[bid] = new_len

    def append_predictions(self, curr_dates, predictions, n_series):
        new_arr = np.asarray(predictions)
        new_ts_val = curr_dates[0]
        if self.groups is None:
            new_ts = np.full(n_series, new_ts_val)
            new_bid = np.zeros(n_series, dtype=np.int64)
            next_ord = self.next_time_index_by_bucket[0]
            new_ord = np.full(n_series, next_ord, dtype=np.int64)
            new_y = new_arr.astype(float)
            self.time = np.concatenate([self.time, new_ts])
            self.y = np.concatenate([self.y, new_y])
            self.bucket_id = np.concatenate([self.bucket_id, new_bid])
            self.time_index = np.concatenate([self.time_index, new_ord])
            self.next_time_index_by_bucket[0] = next_ord + 1
            if 0 in self._ts_aggs:
                agg = self._ts_aggs[0]
                valid = ~np.isnan(new_y)
                agg.unique_times = np.append(agg.unique_times, next_ord)
                agg.sums = np.append(agg.sums, np.sum(np.where(valid, new_y, 0.0)))
                agg.counts = np.append(agg.counts, np.sum(valid))
                nr = float(n_series)
                agg.n_rows = np.append(agg.n_rows, nr)
                agg.is_balanced = agg.is_balanced and (nr == agg.n_rows[0])
                agg.sum_sq = np.append(agg.sum_sq, np.sum(np.where(valid, new_y**2, 0.0)))
                valid_vals = new_y[valid]
                agg.mins = np.append(agg.mins, np.min(valid_vals) if len(valid_vals) > 0 else np.nan)
                agg.maxs = np.append(agg.maxs, np.max(valid_vals) if len(valid_vals) > 0 else np.nan)
            new_sizes = np.array([n_series], dtype=np.int32)
            new_values = new_arr.astype(self.ga.data.dtype)
            self.ga = self.ga.append_several(
                new_sizes=new_sizes,
                new_values=new_values,
                new_groups=np.array([False]),
            )
        else:
            sort_order = np.argsort(self.series_bucket_id, kind="stable")
            sorted_bids = self.series_bucket_id[sort_order]
            new_values = new_arr[sort_order].astype(self.ga.data.dtype, copy=False)
            ga_n_groups = len(self.ga.indptr) - 1
            n_groups = len(self.groups)
            new_sizes = np.zeros(n_groups, dtype=np.int32)
            np.add.at(new_sizes, self.series_bucket_id[sort_order], 1)
            new_groups_mask = np.arange(n_groups) >= ga_n_groups
            self.ga = self.ga.append_several(
                new_sizes=new_sizes,
                new_values=new_values,
                new_groups=new_groups_mask,
            )
            new_ts = np.full(n_series, new_ts_val)
            self.time = np.concatenate([self.time, new_ts[sort_order]])
            self.y = np.concatenate(
                [self.y, new_arr[sort_order].astype(float)]
            )
            self.bucket_id = np.concatenate([self.bucket_id, sorted_bids])
            new_ords = np.array(
                [self.next_time_index_by_bucket[int(gid)] for gid in sorted_bids],
                dtype=np.int64,
            )
            self.time_index = np.concatenate([self.time_index, new_ords])
            for bid in np.unique(sorted_bids):
                bid_int = int(bid)
                new_ord = self.next_time_index_by_bucket[bid_int]
                if bid_int in self._ts_aggs:
                    agg = self._ts_aggs[bid_int]
                    bid_mask = sorted_bids == bid
                    y_bid = new_arr[sort_order][bid_mask].astype(float)
                    valid = ~np.isnan(y_bid)
                    agg.unique_times = np.append(agg.unique_times, new_ord)
                    agg.sums = np.append(agg.sums, np.sum(np.where(valid, y_bid, 0.0)))
                    agg.counts = np.append(agg.counts, np.sum(valid))
                    nr = float(len(y_bid))
                    agg.n_rows = np.append(agg.n_rows, nr)
                    agg.is_balanced = agg.is_balanced and (nr == agg.n_rows[0])
                    agg.sum_sq = np.append(agg.sum_sq, np.sum(np.where(valid, y_bid**2, 0.0)))
                    valid_vals = y_bid[valid]
                    agg.mins = np.append(agg.mins, np.min(valid_vals) if len(valid_vals) > 0 else np.nan)
                    agg.maxs = np.append(agg.maxs, np.max(valid_vals) if len(valid_vals) > 0 else np.nan)
            if self._parent_time_grids is not None:
                self._advance_parent_calendars(new_ts_val)
            else:
                for bid in np.unique(sorted_bids):
                    self.next_time_index_by_bucket[int(bid)] += 1

    def append_observations(
        self,
        df,
        id_col: str,
        time_col: str,
        target_col: str,
        ga_data_dtype,
        static_features=None,
    ):
        if self.groups is None:
            new_df = df[_dedupe_preserve_order([id_col, time_col, target_col])]
            new_df = ufp.sort(new_df, by=[time_col, id_col])
            new_df = ufp.drop_index_if_pandas(new_df)
            new_ts = new_df[time_col].to_numpy()
            new_y = new_df[target_col].to_numpy().astype(float)
            new_bid = np.zeros(len(new_df), dtype=np.int64)
            old_ts = self.time
            all_ts = np.concatenate([old_ts, new_ts])
            unique_all = np.unique(all_ts)
            old_idx = np.searchsorted(unique_all, old_ts).astype(np.int64)
            new_idx = np.searchsorted(unique_all, new_ts).astype(np.int64)
            self.time = all_ts
            self.y = np.concatenate([self.y, new_y])
            self.bucket_id = np.concatenate([self.bucket_id, new_bid])
            self.time_index = np.concatenate([old_idx, new_idx])
            self.next_time_index_by_bucket[0] = len(unique_all)
            self._ts_aggs = _build_ts_aggs(self.bucket_id, self.time_index, self.y)
            new_values = new_df[target_col].to_numpy().astype(ga_data_dtype)
            new_sizes = np.array([len(new_values)], dtype=np.int32)
            self.ga = self.ga.append_several(
                new_sizes=new_sizes,
                new_values=new_values,
                new_groups=np.array([False]),
            )
            old_len = len(self.bucket_df)
            new_df_nw = nw.from_native(new_df)
            new_rows_nw = new_df_nw.with_row_index(name="_bucket_pos").with_columns(
                (nw.col("_bucket_pos") + old_len).cast(nw.Int64).alias("_bucket_pos")
            )
            new_rows = nw.to_native(new_rows_nw)
            cols = list(self.bucket_df.columns)
            self.bucket_df = ufp.vertical_concat([self.bucket_df, new_rows[cols]])
        else:
            group_cols_list = self.key_cols or self.group_cols
            assert group_cols_list is not None
            keep_cols = _dedupe_preserve_order(
                [id_col] + group_cols_list + [time_col, target_col]
            )
            bucket_df = df[keep_cols]
            bucket_df = ufp.sort(
                bucket_df, by=group_cols_list + [time_col, id_col]
            )
            bucket_df = ufp.drop_index_if_pandas(bucket_df)
            old_uids = self.group_uids
            groups = self.groups
            bucket_df = _attach_bucket_id(bucket_df, groups, group_cols_list)
            bucket_df, groups = _extend_groups(
                bucket_df, groups, group_cols_list
            )
            self.groups = groups
            id_counts = ufp.counts_by_id(bucket_df, "_bucket_id")
            uids = old_uids
            uids, new_ids = ufp.match_if_categorical(
                uids, bucket_df["_bucket_id"]
            )
            bucket_df = ufp.assign_columns(bucket_df, "_bucket_id", new_ids)
            bucket_df = ufp.sort(
                bucket_df, by=["_bucket_id", time_col, id_col]
            )
            values = bucket_df[target_col].to_numpy().astype(ga_data_dtype, copy=False)
            new_ts = bucket_df[time_col].to_numpy()
            new_y = values.astype(float)
            new_bid = bucket_df["_bucket_id"].to_numpy().astype(np.int64)
            try:
                sizes = ufp.join(
                    uids, id_counts, on="_bucket_id", how="outer_coalesce"
                )
            except (KeyError, ValueError):
                sizes = ufp.join(
                    uids, id_counts, on="_bucket_id", how="outer"
                )
            sizes = ufp.fill_null(sizes, {"counts": 0})
            sizes = ufp.sort(sizes, by="_bucket_id")
            new_groups_mask = ~ufp.is_in(sizes["_bucket_id"], uids)
            self.ga = self.ga.append_several(
                new_sizes=sizes["counts"].to_numpy().astype(np.int32),
                new_values=values,
                new_groups=new_groups_mask.to_numpy(),
            )
            self.time = np.concatenate([self.time, new_ts])
            self.y = np.concatenate([self.y, new_y])
            self.bucket_id = np.concatenate([self.bucket_id, new_bid])
            all_ts = self.time
            all_bid = self.bucket_id
            if (
                self._parent_time_grids is not None
                and self._bucket_to_parent_id is not None
                and self._parent_to_buckets is not None
            ):
                groups_nw = nw.from_native(self.groups)
                for bid in np.unique(new_bid):
                    bid_int = int(bid)
                    if bid_int not in self._bucket_to_parent_id:
                        self._resolve_parent_for_bucket(bid_int, groups_nw=groups_nw)
                affected_parents: set[int] = set()
                for bid in np.unique(new_bid):
                    pid = self._bucket_to_parent_id.get(int(bid))
                    if pid is not None:
                        affected_parents.add(pid)
                for pid in affected_parents:
                    sibling_bids = self._parent_to_buckets.get(pid, [])
                    all_new_for_parent = []
                    for sbid in sibling_bids:
                        mask = new_bid == sbid
                        if mask.any():
                            all_new_for_parent.append(new_ts[mask])
                    if all_new_for_parent:
                        combined_new = np.concatenate(all_new_for_parent)
                        grid = self._parent_time_grids[pid]
                        self._parent_time_grids[pid] = np.unique(
                            np.concatenate([grid, combined_new])
                        )
                per_bucket_grids = {
                    bid: self._parent_time_grids[self._bucket_to_parent_id[bid]]
                    for bid in self._bucket_to_parent_id
                }
                new_ord_arr, new_next = _compute_time_index_from_parent(
                    all_bid, all_ts, per_bucket_grids
                )
            else:
                new_ord_arr, new_next = _compute_time_index(all_bid, all_ts)
            self.time_index = new_ord_arr
            self.next_time_index_by_bucket = new_next
            self._ts_aggs = _build_ts_aggs(self.bucket_id, self.time_index, self.y)
            old_len = len(self.bucket_df)
            bucket_df_nw = nw.from_native(bucket_df)
            new_rows_nw = bucket_df_nw.with_row_index(name="_bucket_pos").with_columns(
                (nw.col("_bucket_pos") + old_len).cast(nw.Int64).alias("_bucket_pos")
            )
            new_rows = nw.to_native(new_rows_nw)
            cols = list(self.bucket_df.columns)
            self.bucket_df = ufp.vertical_concat([self.bucket_df, new_rows[cols]])
            if self._idsorted_to_bucket_pos is not None:
                self._idsorted_to_bucket_pos = _compute_idsorted_to_bucket_pos(
                    self.bucket_df, id_col, time_col,
                )
            if static_features is not None:
                lookup_cols = self.key_cols or group_cols_list
                sf_cols = set(static_features.columns)
                if all(c in sf_cols for c in lookup_cols):
                    self.series_bucket_id = lookup_bucket_ids(
                        static_features, groups, lookup_cols
                    ).astype(np.int64, copy=False)

    def build_query_arrays(self, _curr_dates, n_series):
        if self.groups is None:
            next_ord = self.next_time_index_by_bucket[0]
            tmp_y = np.concatenate([
                self.y,
                np.full(n_series, np.nan),
            ])
            tmp_bid = np.concatenate([
                self.bucket_id,
                np.zeros(n_series, dtype=np.int64),
            ])
            tmp_ord = np.concatenate([
                self.time_index,
                np.full(n_series, next_ord, dtype=np.int64),
            ])
            return tmp_bid, tmp_ord, tmp_y
        else:
            sort_order = np.argsort(self.series_bucket_id, kind="stable")
            new_bids = self.series_bucket_id[sort_order]
            new_ords = np.array(
                [self.next_time_index_by_bucket[int(gid)] for gid in new_bids],
                dtype=np.int64,
            )
            tmp_y = np.concatenate([
                self.y,
                np.full(n_series, np.nan),
            ])
            tmp_bid = np.concatenate([self.bucket_id, new_bids])
            tmp_ord = np.concatenate([self.time_index, new_ords])
            return tmp_bid, tmp_ord, tmp_y


def _attach_bucket_id(bucket_df, groups, group_cols_list):
    bucket_df_nw = nw.from_native(bucket_df)
    groups_nw = nw.from_native(groups)
    for col in group_cols_list:
        s1, s2 = ufp.match_if_categorical(
            nw.to_native(bucket_df_nw.get_column(col)),
            nw.to_native(groups_nw.get_column(col)),
        )
        bucket_df_nw = bucket_df_nw.with_columns(nw.from_native(s1, series_only=True))
        groups_nw = groups_nw.with_columns(nw.from_native(s2, series_only=True))
    return nw.to_native(bucket_df_nw.join(groups_nw, on=group_cols_list, how="left"))


def _reconcile_cats(df1_nw, df2_nw, cols):
    for col in cols:
        s1, s2 = ufp.match_if_categorical(
            df1_nw.get_column(col).to_native(),
            df2_nw.get_column(col).to_native(),
        )
        df1_nw = df1_nw.with_columns(nw.from_native(s1, series_only=True))
        df2_nw = df2_nw.with_columns(nw.from_native(s2, series_only=True))
    return df1_nw, df2_nw


def _extend_groups(bucket_df, groups, group_cols_list):
    bucket_df_nw = nw.from_native(bucket_df)
    groups_nw = nw.from_native(groups)
    missing = bucket_df_nw.get_column("_bucket_id").is_null()
    if missing.any():
        new_groups_nw = (
            bucket_df_nw.filter(missing)
            .select(group_cols_list)
            .unique(maintain_order=True)
        )
        start = len(groups_nw)
        new_groups_nw = new_groups_nw.with_row_index(name="_bucket_id").with_columns(
            (nw.col("_bucket_id") + start).cast(nw.Int64).alias("_bucket_id")
        )
        groups = ufp.vertical_concat([
            nw.to_native(groups_nw),
            nw.to_native(new_groups_nw),
        ])
        tmp_nw, groups_r_nw = _reconcile_cats(
            bucket_df_nw.drop("_bucket_id"),
            nw.from_native(groups),
            group_cols_list,
        )
        bucket_df = nw.to_native(tmp_nw.join(groups_r_nw, on=group_cols_list, how="left"))
    return bucket_df, groups


def compute_pooled_features(
    state: PooledState,
    transforms: Dict[str, _BaseLagTransform],
    query_arrays=None,
) -> Dict[str, np.ndarray]:
    if query_arrays is not None:
        bid_arr, idx_arr, y_arr = query_arrays
        ts_aggs = _build_ts_aggs(bid_arr, idx_arr, y_arr)
    else:
        bid_arr = state.bucket_id
        idx_arr = state.time_index
        y_arr = state.y
        ts_aggs = state._ts_aggs
    bucket_vals: Dict[str, np.ndarray] = {}
    for name, tfm in transforms.items():
        computed = tfm._compute_bucket_feature(
            bid_arr, idx_arr, y_arr, _ts_aggs=ts_aggs,
        )
        if computed is None:
            raise NotImplementedError(
                f"Transform {type(tfm).__name__!r} does not support pooled "
                f"(global/groupby/partition_by) computation. Implement "
                f"_compute_bucket_feature to use it with global_, groupby, "
                f"or partition_by."
            )
        bucket_vals[name] = computed
    return bucket_vals
