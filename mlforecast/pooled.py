__all__ = ["PooledState", "compute_pooled_features"]

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import utilsforecast.processing as ufp
from utilsforecast.compat import pl

from .grouped_array import GroupedArray
from .lag_transforms import _BaseLagTransform


def _dedupe_preserve_order(items):
    return list(dict.fromkeys(items))


def add_bucket_id(data, cols):
    cols = list(cols)
    if isinstance(data, pd.DataFrame):
        groups = data[cols].drop_duplicates().reset_index(drop=True)
        groups["_bucket_id"] = np.arange(len(groups), dtype=np.int64)
        merged = data.merge(groups, on=cols, how="left")
    else:
        groups = data.select(cols).unique(maintain_order=True)
        groups = groups.with_row_index(name="_bucket_id")
        merged = data.join(groups, on=cols, how="left")
    return ufp.drop_index_if_pandas(merged), ufp.drop_index_if_pandas(groups)


def lookup_bucket_ids(data, groups, cols):
    cols = list(cols)
    if isinstance(data, pd.DataFrame):
        data_slice = data[cols].copy()
        groups_copy = groups.copy()
        for col in cols:
            s1, s2 = ufp.match_if_categorical(data_slice[col], groups_copy[col])
            data_slice[col] = s1
            groups_copy[col] = s2
        joined = data_slice.merge(groups_copy, on=cols, how="left")
        return joined["_bucket_id"].to_numpy()
    data_slice = data.select(cols)
    groups_copy = groups
    for col in cols:
        s1, s2 = ufp.match_if_categorical(data_slice[col], groups_copy[col])
        data_slice = data_slice.with_columns(s1)
        groups_copy = groups_copy.with_columns(s2)
    joined = data_slice.join(groups_copy, on=cols, how="left")
    return joined["_bucket_id"].to_numpy()


def _compute_time_index(bid_arr, ts_arr):
    """Assign integer period coordinates over the validated regular time grid.

    For each bucket, maps raw timestamps to consecutive integers based on
    the bucket's sorted unique timestamps. Because the input grid is validated
    as regular (no gaps within a series), this is equivalent to SQL RANGE
    interval semantics. Series that start later simply have no rows at earlier
    periods -- no synthetic zeros are injected.
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
    # partition_by fields (defaults for backward compat with global/groupby)
    mode: str = "nonlocal"  # "local" or "nonlocal"
    partition_cols: Optional[List[str]] = None
    key_cols: Optional[List[str]] = None
    parent_scope_cols: Optional[List[str]] = None
    _parent_time_grids: Optional[Dict[int, np.ndarray]] = None
    _bucket_to_parent_id: Optional[Dict[int, int]] = None
    _parent_to_buckets: Optional[Dict[int, List[int]]] = None
    _scope_key_to_parent_id: Optional[Dict[tuple, int]] = None

    @property
    def group_uids(self):
        if self.groups is None:
            if isinstance(self.bucket_df, pd.DataFrame):
                return pd.Index([0], name="_bucket_id")
            return pl.Series("_bucket_id", [0])
        if isinstance(self.groups, pd.DataFrame):
            return pd.Index(sorted(self.groups["_bucket_id"].unique()), name="_bucket_id")
        return self.groups["_bucket_id"].sort()

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
        if isinstance(global_df, pd.DataFrame):
            global_df = global_df.copy()
            global_df["_bucket_pos"] = np.arange(len(global_df), dtype=np.int64)
            process_df = global_df[["_bucket_pos", target_col]].copy()
            process_df.insert(0, "_bucket_id", 0)
        else:
            global_df = global_df.with_row_index(name="_bucket_pos")
            process_df = global_df.select(
                [
                    pl.lit(0).alias("_bucket_id").cast(pl.Int64),
                    "_bucket_pos",
                    target_col,
                ]
            )
        processed = ufp.process_df(
            process_df,
            id_col="_bucket_id",
            time_col="_bucket_pos",
            target_col=target_col,
        )
        ga = GroupedArray(processed.data[:, 0], processed.indptr)
        unique_ts = np.unique(ts_raw)
        ord_raw = np.searchsorted(unique_ts, ts_raw).astype(np.int64)
        return cls(
            ga=ga,
            bucket_df=global_df,
            groups=None,
            group_cols=None,
            series_bucket_id=np.zeros(n_series, dtype=np.int64),
            bucket_id=np.zeros(len(global_df), dtype=np.int64),
            time=ts_raw,
            time_index=ord_raw,
            y=y_raw.astype(float),
            next_time_index_by_bucket={0: len(unique_ts)},
            join_cols=[id_col, time_col],
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
        if isinstance(bucket_df, pd.DataFrame):
            bucket_df = bucket_df.copy()
            bucket_df["_bucket_pos"] = (
                bucket_df.groupby("_bucket_id", sort=False)
                .cumcount()
                .astype(np.int64)
            )
            process_df = bucket_df[["_bucket_id", "_bucket_pos", target_col]]
        else:
            bucket_df = bucket_df.with_columns(
                pl.int_range(pl.len()).over("_bucket_id").alias("_bucket_pos")
            )
            process_df = bucket_df.select(
                ["_bucket_id", "_bucket_pos", target_col]
            )
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
        return cls(
            ga=ga,
            bucket_df=bucket_df,
            groups=groups,
            group_cols=group_cols_list,
            series_bucket_id=series_bucket_id,
            bucket_id=bid_arr,
            time=ts_raw,
            time_index=ord_arr,
            y=y_raw.astype(float),
            next_time_index_by_bucket=next_by_bucket,
            join_cols=[id_col, time_col],
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

        if isinstance(bucket_df, pd.DataFrame):
            bucket_df = bucket_df.copy()
            bucket_df["_bucket_pos"] = (
                bucket_df.groupby("_bucket_id", sort=False)
                .cumcount()
                .astype(np.int64)
            )
            process_df = bucket_df[["_bucket_id", "_bucket_pos", target_col]]
        else:
            bucket_df = bucket_df.with_columns(
                pl.int_range(pl.len()).over("_bucket_id").alias("_bucket_pos")
            )
            process_df = bucket_df.select(
                ["_bucket_id", "_bucket_pos", target_col]
            )

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

        # Determine parent scope for ordinal computation.
        # local: parent = individual series (id_col)
        # nonlocal + groupby: parent = groupby columns
        # nonlocal + global: parent = entire dataset (None)
        if mode == "local":
            parent_scope_cols = [id_col]
        elif group_cols_list:
            parent_scope_cols = list(group_cols_list)
        else:
            parent_scope_cols = None

        # Build parent time grids keyed by parent_id (not bucket_id).
        # Multiple sibling buckets under the same parent share one grid.
        parent_id_map: Dict[tuple, int] = {}
        bucket_to_parent: Dict[int, int] = {}
        parent_grids: Dict[int, np.ndarray] = {}
        next_pid = 0

        if parent_scope_cols is not None:
            for bid in np.unique(bid_arr):
                if isinstance(groups, pd.DataFrame):
                    row = groups[groups["_bucket_id"] == bid].iloc[0]
                    scope_key = tuple(row[c] for c in parent_scope_cols)
                else:
                    row = groups.filter(pl.col("_bucket_id") == bid).row(0, named=True)
                    scope_key = tuple(row[c] for c in parent_scope_cols)

                if scope_key not in parent_id_map:
                    pid = next_pid
                    next_pid += 1
                    parent_id_map[scope_key] = pid
                    if isinstance(sorted_df, pd.DataFrame):
                        mask = np.ones(len(sorted_df), dtype=bool)
                        for c, v in zip(parent_scope_cols, scope_key):
                            mask &= sorted_df[c].values == v
                        parent_ts = np.sort(np.unique(sorted_df[time_col].to_numpy()[mask]))
                    else:
                        expr = pl.lit(True)
                        for c, v in zip(parent_scope_cols, scope_key):
                            expr = expr & (pl.col(c) == v)
                        parent_ts = np.sort(sorted_df.filter(expr)[time_col].unique().to_numpy())
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

        # series_bucket_id: initial assignment from static features if possible
        sf_cols = set(
            static_features.columns
            if isinstance(static_features, pd.DataFrame)
            else static_features.columns
        )
        can_lookup = all(c in sf_cols for c in key_cols)
        if can_lookup:
            series_bucket_id = lookup_bucket_ids(
                static_features, groups, key_cols
            ).astype(np.int64, copy=False)
        else:
            series_bucket_id = np.zeros(n_series, dtype=np.int64)

        return cls(
            ga=ga,
            bucket_df=bucket_df,
            groups=groups,
            group_cols=group_cols_list,
            series_bucket_id=series_bucket_id,
            bucket_id=bid_arr,
            time=ts_raw,
            time_index=ord_arr,
            y=y_raw.astype(float),
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
        # Attach bucket IDs from the existing groups mapping
        context_with_bid = _attach_bucket_id(context_df, self.groups, key_cols)
        # Extend groups for any new key combinations
        context_with_bid, self.groups = _extend_groups(
            context_with_bid, self.groups, key_cols
        )
        new_bid = context_with_bid["_bucket_id"].to_numpy().astype(np.int64)
        self.series_bucket_id = new_bid
        # Ensure next_time_index_by_bucket has entries for new buckets
        for bid in np.unique(new_bid):
            bid_int = int(bid)
            if bid_int not in self.next_time_index_by_bucket:
                pid = self._resolve_parent_for_bucket(bid_int)
                if pid is not None and self._parent_time_grids is not None:
                    self.next_time_index_by_bucket[bid_int] = len(
                        self._parent_time_grids[pid]
                    )
                else:
                    self.next_time_index_by_bucket[bid_int] = 0

    def _resolve_parent_for_bucket(self, bid: int) -> Optional[int]:
        """Find or create the parent_id for a bucket from its scope columns."""
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
            if isinstance(self.groups, pd.DataFrame):
                row = self.groups[self.groups["_bucket_id"] == bid]
                if len(row) == 0:
                    return None
                row = row.iloc[0]
                scope_key = tuple(row[c] for c in self.parent_scope_cols)
            else:
                row = self.groups.filter(pl.col("_bucket_id") == bid)
                if len(row) == 0:
                    return None
                row = row.row(0, named=True)
                scope_key = tuple(row[c] for c in self.parent_scope_cols)

        if scope_key in self._scope_key_to_parent_id:
            pid = self._scope_key_to_parent_id[scope_key]
            self._bucket_to_parent_id[bid] = pid
            self._parent_to_buckets[pid].append(bid)
            return pid

        # New parent scope — create a new parent with empty grid
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

        This assumes all series predict at the same timestamp per step.
        For nonlocal transforms this is enforced by ``_check_aligned_ends()``
        at fit/predict time and by timestamp validation in ``update()``.
        For local partition transforms with staggered series this is
        structurally guaranteed by ``_update_features()`` advancing all
        series by the same freq step.
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
            # _extend_groups appends new bucket IDs contiguously starting
            # from the previous group count, so IDs >= ga_n_groups are new.
            ga_n_groups = len(self.ga.indptr) - 1
            n_groups = len(
                self.groups["_bucket_id"].unique()
                if isinstance(self.groups, pd.DataFrame)
                else self.groups["_bucket_id"]
            )
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
            new_values = new_df[target_col].to_numpy().astype(ga_data_dtype)
            new_sizes = np.array([len(new_values)], dtype=np.int32)
            self.ga = self.ga.append_several(
                new_sizes=new_sizes,
                new_values=new_values,
                new_groups=np.array([False]),
            )
            old_len = len(self.bucket_df)
            if isinstance(new_df, pd.DataFrame):
                new_rows = new_df.copy()
                new_rows["_bucket_pos"] = np.arange(
                    old_len, old_len + len(new_df), dtype=np.int64
                )
            else:
                pos_dtype = self.bucket_df["_bucket_pos"].dtype
                new_rows = new_df.with_columns(
                    pl.int_range(old_len, old_len + len(new_df))
                    .cast(pos_dtype)
                    .alias("_bucket_pos")
                )
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
            if isinstance(uids, pd.Index):
                uids = pd.Series(uids)
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
                # Register any new buckets from _extend_groups
                for bid in np.unique(new_bid):
                    bid_int = int(bid)
                    if bid_int not in self._bucket_to_parent_id:
                        self._resolve_parent_for_bucket(bid_int)
                # Update parent grids per-parent (not per-bucket)
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
            old_len = len(self.bucket_df)
            if isinstance(bucket_df, pd.DataFrame):
                new_rows = bucket_df.copy()
                new_rows["_bucket_pos"] = np.arange(
                    old_len, old_len + len(bucket_df), dtype=np.int64
                )
            else:
                pos_dtype = self.bucket_df["_bucket_pos"].dtype
                new_rows = bucket_df.with_columns(
                    pl.int_range(old_len, old_len + len(bucket_df))
                    .cast(pos_dtype)
                    .alias("_bucket_pos")
                )
            cols = list(self.bucket_df.columns)
            self.bucket_df = ufp.vertical_concat([self.bucket_df, new_rows[cols]])
            if static_features is not None:
                lookup_cols = self.key_cols or group_cols_list
                sf_cols = set(
                    static_features.columns
                    if isinstance(static_features, pd.DataFrame)
                    else static_features.columns
                )
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
    if isinstance(bucket_df, pd.DataFrame):
        bucket_df = bucket_df.copy()
        groups = groups.copy()
        for col in group_cols_list:
            s1, s2 = ufp.match_if_categorical(bucket_df[col], groups[col])
            bucket_df[col] = s1
            groups[col] = s2
        return bucket_df.merge(groups, on=group_cols_list, how="left")
    for col in group_cols_list:
        s1, s2 = ufp.match_if_categorical(bucket_df[col], groups[col])
        bucket_df = bucket_df.with_columns(s1)
        groups = groups.with_columns(s2)
    return bucket_df.join(groups, on=group_cols_list, how="left")


def _reconcile_cats(df1, df2, cols):
    if isinstance(df1, pd.DataFrame):
        df1 = df1.copy()
        df2 = df2.copy()
        for col in cols:
            s1, s2 = ufp.match_if_categorical(df1[col], df2[col])
            df1[col] = s1
            df2[col] = s2
    else:
        for col in cols:
            s1, s2 = ufp.match_if_categorical(df1[col], df2[col])
            df1 = df1.with_columns(s1)
            df2 = df2.with_columns(s2)
    return df1, df2


def _extend_groups(bucket_df, groups, group_cols_list):
    if isinstance(bucket_df, pd.DataFrame):
        missing = bucket_df["_bucket_id"].isna()
        if missing.any():
            new_groups = (
                bucket_df.loc[missing, group_cols_list].drop_duplicates()
            )
            new_groups = new_groups.reset_index(drop=True)
            start = len(groups)
            new_groups["_bucket_id"] = np.arange(
                start, start + len(new_groups), dtype=np.int64
            )
            groups = ufp.vertical_concat([groups, new_groups])
            tmp = bucket_df.drop(columns="_bucket_id")
            tmp, groups_r = _reconcile_cats(tmp, groups, group_cols_list)
            bucket_df = tmp.merge(groups_r, on=group_cols_list, how="left")
    else:
        missing = bucket_df["_bucket_id"].is_null()
        if missing.any():
            new_groups = (
                bucket_df.filter(missing)
                .select(group_cols_list)
                .unique(maintain_order=True)
            )
            start = groups.height
            new_groups = new_groups.with_row_index(
                name="_bucket_id", offset=start
            )
            groups = ufp.vertical_concat([groups, new_groups])
            tmp = bucket_df.drop("_bucket_id")
            tmp, groups_r = _reconcile_cats(tmp, groups, group_cols_list)
            bucket_df = tmp.join(groups_r, on=group_cols_list, how="left")
    return bucket_df, groups


def compute_pooled_features(
    state: PooledState,
    transforms: Dict[str, _BaseLagTransform],
    query_arrays=None,
) -> Dict[str, np.ndarray]:
    if query_arrays is not None:
        bid_arr, idx_arr, y_arr = query_arrays
    else:
        bid_arr = state.bucket_id
        idx_arr = state.time_index
        y_arr = state.y
    bucket_vals: Dict[str, np.ndarray] = {}
    for name, tfm in transforms.items():
        computed = tfm._compute_bucket_feature(bid_arr, idx_arr, y_arr)
        if computed is None:
            raise NotImplementedError(
                f"Transform {type(tfm).__name__!r} does not support pooled "
                f"(global/groupby/partition_by) computation. Implement "
                f"_compute_bucket_feature to use it with global_, groupby, "
                f"or partition_by."
            )
        bucket_vals[name] = computed
    return bucket_vals
