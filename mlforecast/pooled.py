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


@dataclass
class PooledState:
    """Holds all arrays and metadata for a pooled (global/groupby) bucket.

    ``time_index`` is an integer period coordinate over a validated regular
    time grid.  For global/group states (no partition), the input is validated
    to have no missing timestamps within any series.  Different series in the
    same bucket may have different start dates; only actual observations are
    included (no synthetic zeros).  Under this invariant, ``time_index``
    is equivalent to SQL ``RANGE BETWEEN … PRECEDING`` interval semantics.

    For PR 2 / partition_by buckets, ``time_index`` must be derived from
    the parent series/global calendar — not the bucket's observed timestamps —
    so that missing partition values leave holes and window bounds remain
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
            n_groups = len(
                self.groups["_bucket_id"].unique()
                if isinstance(self.groups, pd.DataFrame)
                else self.groups["_bucket_id"]
            )
            new_sizes = np.zeros(n_groups, dtype=np.int32)
            np.add.at(new_sizes, self.series_bucket_id[sort_order], 1)
            self.ga = self.ga.append_several(
                new_sizes=new_sizes,
                new_values=new_values,
                new_groups=np.zeros(n_groups, dtype=bool),
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
            assert self.group_cols is not None
            group_cols_list = self.group_cols
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
                self.series_bucket_id = lookup_bucket_ids(
                    static_features, groups, group_cols_list
                ).astype(np.int64, copy=False)

    def build_query_arrays(self, curr_dates, n_series):
        if self.groups is None:
            next_ord = self.next_time_index_by_bucket[0]
            tmp_ts = np.concatenate([
                self.time,
                np.full(n_series, curr_dates[0]),
            ])
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
            return tmp_bid, tmp_ts, tmp_y, tmp_ord
        else:
            sort_order = np.argsort(self.series_bucket_id, kind="stable")
            new_bids = self.series_bucket_id[sort_order]
            new_ords = np.array(
                [self.next_time_index_by_bucket[int(gid)] for gid in new_bids],
                dtype=np.int64,
            )
            tmp_ts = np.concatenate([
                self.time,
                np.full(n_series, curr_dates[0])[sort_order],
            ])
            tmp_y = np.concatenate([
                self.y,
                np.full(n_series, np.nan),
            ])
            tmp_bid = np.concatenate([self.bucket_id, new_bids])
            tmp_ord = np.concatenate([self.time_index, new_ords])
            return tmp_bid, tmp_ts, tmp_y, tmp_ord


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
        bid_arr, ts_arr, y_arr, idx_arr = query_arrays
    else:
        bid_arr = state.bucket_id
        ts_arr = state.time
        y_arr = state.y
        idx_arr = state.time_index
    bucket_vals: Dict[str, np.ndarray] = {}
    for name, tfm in transforms.items():
        computed = tfm._compute_bucket_feature(bid_arr, ts_arr, y_arr, idx_arr)
        if computed is None:
            raise NotImplementedError(
                f"Transform {type(tfm).__name__!r} does not support pooled "
                f"(global/groupby) computation. Implement "
                f"_compute_bucket_feature to use it with global_ or groupby."
            )
        bucket_vals[name] = computed
    return bucket_vals
