# Changes: Fix distributed predict with non-pandas X_df (Issue #507)

## Problem

`DistributedMLForecast.predict()` required `X_df` (future exogenous features) to be a
`pd.DataFrame`, raising a hard `ValueError` for any other type. For large datasets this
forced users to call `.toPandas()` or `.compute()` on the full exogenous DataFrame before
calling predict, which could cause out-of-memory errors.

## Solution

`X_df` is now processed **per partition** so that no single node ever holds the entire
exogenous DataFrame in memory. Dask, Spark, and Ray DataFrames are all accepted.

### `mlforecast/distributed/forecast.py`

- Added `import base64` (used to encode per-partition X_df payloads as ASCII strings,
  avoiding binary-column type issues in Dask/Spark joins).
- Extended `_partition_results` schema from `ts:binary,train:binary,valid:binary` to
  `ts:binary,train:binary,valid:binary,first_uid:str`. `first_uid` is the string
  representation of the first series ID in each partition and serves as the join key for
  attaching per-partition X_df slices. Updated in all four locations:
  `_preprocess_partition` (both return statements), `_load_ts`, `_update`, and the schema
  string literals in `_preprocess_partitions`, `load`, and `update`.
- Updated tuple unpackers in `_retrieve_df`, `_save_ts`, and `_update` to handle the new
  4-element rows.
- Added `_pack_x_df` static method. Called via `fa.transform` partitioned by
  `__partition_key__`; serializes each partition's X_df rows into a single
  base64-encoded cloudpickle blob.
- Added `_build_x_df_per_partition` instance method. Orchestrates the distributed join
  pipeline:
  1. Loads only the tiny `ts` and `first_uid` columns from `_partition_results` on the
     driver to build a `uid → first_uid` mapping (O(M × 2 strings) memory).
  2. Distributed join: tags every X_df row with its owning partition key.
  3. Distributed pack: serializes each partition's X_df rows into one base64 blob.
  4. Collects the N-row packed table to pandas (negligible) and joins it back into
     `partition_results` as an `x_df:str` column.
- Updated `_predict` static method: removed the `X_df` and `id_col` broadcast
  parameters. When a row has 5 elements the fifth is a base64 string; `_predict`
  decodes and deserializes it to recover the per-partition X_df.
- Updated `predict` instance method: changed `X_df` type hint to
  `Optional[fugue.AnyDataFrame]`, removed the old `isinstance` check and `ValueError`,
  and calls `_build_x_df_per_partition` when `X_df` is provided.

### `tests/test_distributed_forecast.py`

- Added `test_dask_distributed_forecast_with_x_df`: generates a series with an
  exogenous feature, fits a `DistributedMLForecast`, then verifies that calling
  `predict(h, X_df=dask_df)` produces identical results to `predict(h, X_df=pandas_df)`.
