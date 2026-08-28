"""Pooled lag-transform engine dispatch.

``MLFORECAST_POOLED_ENGINE`` selects the implementation:

- ``narwhals`` (default): the aggregate-table engine in this module.
- ``numpy``: the original engine in ``_pooled_legacy``, retained for the
  differential tests and the A/B benchmark.

Both the environment variable and the numpy engine are removed in iteration two.
"""

import os

from ._pooled_keys import (  # noqa: F401
    _attach_bucket_id,
    _extend_groups,
    _order_preserving_left_join,
    add_bucket_id,
    lookup_bucket_ids,
)
from ._pooled_legacy import (  # noqa: F401
    PooledState,
    _build_ts_aggs,
    _collapse_rows_by_time,
    _compute_idsorted_to_bucket_pos,
    _reaggregate_ts_aggs,
    compute_pooled_features,
)

__all__ = ["PooledState", "compute_pooled_features", "POOLED_ENGINE"]

_VALID_ENGINES = ("narwhals", "numpy")

POOLED_ENGINE = os.environ.get("MLFORECAST_POOLED_ENGINE", "numpy")
if POOLED_ENGINE not in _VALID_ENGINES:
    raise ValueError(
        f"MLFORECAST_POOLED_ENGINE must be one of {_VALID_ENGINES}; "
        f"got {POOLED_ENGINE!r}."
    )
