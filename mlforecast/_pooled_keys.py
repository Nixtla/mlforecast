"""Null-safe bucket-key encoding and order-preserving joins for pooled states.

Shared by both pooled engines. Extracted unchanged from the original
``pooled.py``: this code is already narwhals-based and backend-correct, and it
survives the numpy engine's removal.
"""

import narwhals as nw
import utilsforecast.processing as ufp


def _dedupe_preserve_order(items):
    return list(dict.fromkeys(items))


# Sentinel string that every missing (null/NaN) key value is encoded to, so that
# missing matches missing (and only missing) across pandas/polars and all key
# dtypes. NUL-prefixed to make a collision with a real key value negligible.
_NULL_SENTINEL = "\x00__MLF_NULL__"


def _missing_expr(col, dtype):
    """Dtype-aware "is missing" predicate: null, plus NaN for float columns.

    narwhals ``is_null`` flags NaN on pandas but not float NaN on polars, so the
    ``is_nan`` arm is required (and is only valid on float dtypes).
    """
    expr = nw.col(col).is_null()
    if dtype.is_float():
        expr = expr | nw.col(col).is_nan()
    return expr


def _encode_col_expr(col, dtype, *, to_int=False):
    """Sentinel-encode one key column to a String expression.

    Missing values (null/NaN) become ``_NULL_SENTINEL``; non-missing values are
    cast to String. When ``to_int`` the *integral* non-missing values are routed
    through Int64 first so a float ``0.0`` matches an integer ``0`` (``"0"`` on
    both sides); a genuinely fractional float (e.g. ``1.5``) keeps its own float
    string form so it does not collide with an integer bucket. The int cast is
    applied only after missing values are masked out, since casting NaN/null
    through a non-nullable int raises on both engines.
    """
    miss = _missing_expr(col, dtype)
    if not to_int:
        return (
            nw.when(miss)
            .then(nw.lit(_NULL_SENTINEL))
            .otherwise(nw.col(col).cast(nw.String))
        )
    # replace missing with a safe placeholder before the (otherwise-raising) int
    # cast; placeholder rows are overwritten by the sentinel branch below.
    safe = nw.when(miss).then(nw.lit(0.0)).otherwise(nw.col(col))
    as_int = safe.cast(nw.Int64)
    is_integral = safe == as_int  # 1.0 -> True, 1.5 -> False
    # integral -> int string ("1"); fractional -> float string ("1.5", which
    # cannot collide with an integer bucket); missing -> sentinel.
    non_missing = (
        nw.when(is_integral)
        .then(as_int.cast(nw.String))
        .otherwise(safe.cast(nw.String))
    )
    return nw.when(miss).then(nw.lit(_NULL_SENTINEL)).otherwise(non_missing)


def _encode_keys(frame_nw, cols):
    """Add transient ``__enc_<col>`` string columns for bucket *creation*.

    Single-frame: no cross-frame dtype reconcile is needed because creation and
    its matching join operate on the same frame.
    """
    schema = frame_nw.schema
    return frame_nw.with_columns(
        [_encode_col_expr(c, schema[c]).alias(f"__enc_{c}") for c in cols]
    )


def _encode_join_keys(left_nw, right_nw, cols):
    """Sentinel-encode key ``cols`` on both frames for a null-equal *match*.

    Applies a narrow int<->float reconcile: when one side's key column is integer
    and the other float (e.g. a fit column contaminated to float by a NaN, matched
    against clean integer data), the float side's integral non-missing values are
    cast to int so they match. No blanket Float64 widening (that would corrupt
    large int keys above 2**53).
    """
    lschema, rschema = left_nw.schema, right_nw.schema
    left_exprs, right_exprs = [], []
    for c in cols:
        ldt, rdt = lschema[c], rschema[c]
        left_exprs.append(
            _encode_col_expr(c, ldt, to_int=ldt.is_float() and rdt.is_integer()).alias(
                f"__enc_{c}"
            )
        )
        right_exprs.append(
            _encode_col_expr(c, rdt, to_int=rdt.is_float() and ldt.is_integer()).alias(
                f"__enc_{c}"
            )
        )
    return left_nw.with_columns(left_exprs), right_nw.with_columns(right_exprs)


_ROW_ORDER_COL = "_mlf_row_order"


def _order_preserving_left_join(left_nw, right_nw, on):
    """Left join that preserves the left frame's row order.

    Results are consumed positionally against arrays aligned with the left
    frame, but narwhals exposes no ``maintain_order`` for joins and polars does
    not guarantee row order, so attach a row index to the left frame, sort on
    it after the join, and drop it. All call sites join against key-unique
    right frames, so the join never fans out.
    """
    left_idx = left_nw.with_row_index(name=_ROW_ORDER_COL)
    joined = left_idx.join(right_nw, on=list(on), how="left")
    return joined.sort(_ROW_ORDER_COL).drop(_ROW_ORDER_COL)


def _null_equal_left_join(left_nw, right_nw, cols, payload_cols):
    """Left-join ``left_nw`` to ``right_nw`` on ``cols`` treating missing as equal.

    ``right_nw`` contributes only ``payload_cols`` (e.g. ``["_bucket_id"]``); its
    original key columns are excluded so the join emits no ``*_right`` duplicates.
    Encoded columns are transient and dropped before returning.
    """
    enc_cols = [f"__enc_{c}" for c in cols]
    left_enc, right_enc = _encode_join_keys(left_nw, right_nw, cols)
    right_enc = right_enc.select(enc_cols + list(payload_cols))
    joined = _order_preserving_left_join(left_enc, right_enc, on=enc_cols)
    return joined.drop(enc_cols)


def add_bucket_id(data, cols):
    cols = list(cols)
    enc_cols = [f"__enc_{c}" for c in cols]
    data_enc = _encode_keys(nw.from_native(data), cols)
    # One representative original-key row per encoded key, so null and NaN (and
    # pandas-None) collapse into a single bucket and the join below cannot fan out.
    groups_enc = data_enc.select(cols + enc_cols).unique(
        subset=enc_cols, keep="first", maintain_order=True
    )
    groups_enc = groups_enc.with_row_index(name="_bucket_id").with_columns(
        nw.col("_bucket_id").cast(nw.Int64)
    )
    merged_nw = _order_preserving_left_join(
        data_enc, groups_enc.select(enc_cols + ["_bucket_id"]), on=enc_cols
    ).drop(enc_cols)
    groups_nw = groups_enc.drop(enc_cols)
    return (
        ufp.drop_index_if_pandas(nw.to_native(merged_nw)),
        ufp.drop_index_if_pandas(nw.to_native(groups_nw)),
    )


def lookup_bucket_ids(data, groups, cols):
    cols = list(cols)
    joined = _null_equal_left_join(
        nw.from_native(data).select(cols),
        nw.from_native(groups),
        cols,
        ["_bucket_id"],
    )
    return joined.get_column("_bucket_id").to_numpy()


def _attach_bucket_id(bucket_df, groups, group_cols_list):
    joined = _null_equal_left_join(
        nw.from_native(bucket_df),
        nw.from_native(groups),
        group_cols_list,
        ["_bucket_id"],
    )
    return nw.to_native(joined)


def _extend_groups(bucket_df, groups, group_cols_list):
    bucket_df_nw = nw.from_native(bucket_df)
    groups_nw = nw.from_native(groups)
    missing = bucket_df_nw.get_column("_bucket_id").is_null()
    if missing.any():
        enc_cols = [f"__enc_{c}" for c in group_cols_list]
        # Create new buckets on the *encoded* key so null/NaN/None collapse into a
        # single new bucket (polars .unique() keeps them as distinct raw rows).
        new_groups_nw = (
            _encode_keys(bucket_df_nw.filter(missing), group_cols_list)
            .select(group_cols_list + enc_cols)
            .unique(subset=enc_cols, keep="first", maintain_order=True)
            .drop(enc_cols)
        )
        start = len(groups_nw)
        new_groups_nw = new_groups_nw.with_row_index(name="_bucket_id").with_columns(
            (nw.col("_bucket_id") + start).cast(nw.Int64).alias("_bucket_id")
        )
        groups = ufp.vertical_concat(
            [
                nw.to_native(groups_nw),
                nw.to_native(new_groups_nw),
            ]
        )
        # Re-lookup every row (including pre-existing missing buckets) so an
        # existing null bucket is matched, not duplicated.
        joined = _null_equal_left_join(
            bucket_df_nw.drop("_bucket_id"),
            nw.from_native(groups),
            group_cols_list,
            ["_bucket_id"],
        )
        bucket_df = nw.to_native(joined)
    return bucket_df, groups
