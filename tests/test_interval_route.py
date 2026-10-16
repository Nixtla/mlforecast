import numpy as np
import pytest

from mlforecast.conformal_prediction import TransferConformal as TC
from mlforecast.conformal_prediction import _resolve_interval_route

S = {"a", "b"}


def route(level=(90,), ids=None, new_ids=None, cs_ids=S, tc=None):
    return _resolve_interval_route(
        level=None if level is None else list(level),
        ids=ids,
        new_ids=new_ids,
        cs_ids=cs_ids,
        transfer_conformal=tc,
    )


def test_row0_tc_without_new_df():
    for level in ([90], None):
        with pytest.raises(ValueError, match="new_df"):
            route(level=level, tc=TC("weighted_conformal"))


def test_row1_source():
    assert route(ids=["a"]) == "source"
    assert route(ids=None) == "source"
    with pytest.raises(ValueError, match="Missing ids"):
        route(ids=["z"])


def test_row2_ids_with_tc():
    for method in ("scale_aligned", "recalibrate"):
        for level in ([90], None):
            with pytest.raises(ValueError, match="ids"):
                route(level=level, new_ids={"x"}, ids=["x"], tc=TC(method))


def test_row3_ids_seen():
    assert route(new_ids={"a", "x"}, ids=["a"]) == "source"


def test_row4_ids_unseen():
    with pytest.raises(ValueError, match="x"):
        route(new_ids={"a", "x"}, ids=["x"])
    assert route(level=None, new_ids={"a", "x"}, ids=["x"]) == "none"


def test_row5():
    assert route(new_ids={"a"}) == "source"
    assert route(new_ids={"a"}, tc=TC("recalibrate")) == "transfer"
    for method in (
        "scale_aligned",
        "error_scaled",
        "weighted_conformal",
        "scale_aligned_weighted",
    ):
        with pytest.raises(NotImplementedError, match=method):
            route(new_ids={"a"}, tc=TC(method))


def test_row6_unseen():
    assert route(new_ids={"x", "y"}) == "transfer"
    assert route(new_ids={"x", "y"}, tc=TC("scale_aligned")) == "transfer"


def test_row7_mixed():
    with pytest.raises(NotImplementedError, match="split"):
        route(new_ids={"a", "x"})
    assert route(level=None, new_ids={"a", "x"}) == "none"
    with pytest.raises(NotImplementedError, match="recalibrate"):
        route(new_ids={"a", "x"}, tc=TC("scale_aligned"))
    assert route(new_ids={"a", "x"}, tc=TC("recalibrate")) == "transfer"


def test_no_calibration():
    assert route(cs_ids=None) == "none"


def test_ids_as_array():
    assert route(ids=np.array(["a"])) == "source"


def test_level_new_df_without_calibration():
    with pytest.raises(ValueError, match="Prediction intervals require"):
        route(cs_ids=None, new_ids={"a"})
    assert route(level=None, cs_ids=None, new_ids={"a"}) == "none"


def test_row3_seen_id_not_in_new_df():
    with pytest.raises(ValueError, match="not in `new_df`"):
        route(ids=["a"], new_ids={"b"})
    assert route(level=None, ids=["a"], new_ids={"b"}) == "none"
