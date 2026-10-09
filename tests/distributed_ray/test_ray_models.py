import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import ray
import xgboost as xgb
from sklearn.base import clone

from mlforecast.distributed.models.ray.lgb import RayLGBMForecast, _lgb_train_loop
from mlforecast.distributed.models.ray.xgb import RayXGBForecast, _xgb_train_loop


@pytest.mark.ray
@pytest.mark.parametrize("model_cls", [RayLGBMForecast, RayXGBForecast])
def test_clone_preserves_booster_params(model_cls):
    """DistributedMLForecast._fit clones the model, so get_params has to be complete.

    sklearn's introspection reads the subclass' signature, which only names the
    ray arguments, so the booster's parameters have to come from the library's
    own get_params.
    """
    model = model_cls(num_workers=2, random_state=0, learning_rate=0.05, n_estimators=5)
    params = model.get_params()
    assert params["num_workers"] == 2
    assert params["random_state"] == 0
    assert params["learning_rate"] == 0.05
    assert params["n_estimators"] == 5

    cloned = clone(model).get_params()
    assert cloned["num_workers"] == 2
    assert cloned["random_state"] == 0
    assert cloned["learning_rate"] == 0.05
    assert cloned["n_estimators"] == 5


@pytest.mark.ray
def test_default_num_boost_round_matches_sklearn():
    """The estimators defaulted to 100 rounds; keep that.

    Each library resolves it its own way now: lightgbm reads `n_estimators`,
    xgboost leaves it None and falls back to 100 in get_num_boosting_rounds.
    """
    assert RayLGBMForecast().get_params()["n_estimators"] == 100
    assert RayXGBForecast().get_num_boosting_rounds() == 100


@pytest.mark.ray
# the thread method, since SIGALRM doesn't fire while lightgbm waits on a socket
@pytest.mark.timeout(300, method="thread")
def test_lgb_trains_on_the_full_dataset_across_workers(tmp_path):
    """Every worker only sees its shard, so lightgbm needs its network params.

    Without them each worker trains an independent model on 1/N of the data and
    rank 0's is the one that gets checkpointed, with no error anywhere. The
    feature is constant, so no split is possible and the prediction is exactly
    the mean of whatever the model actually saw.
    """
    df = pd.DataFrame(
        {"x": np.ones(200), "y": np.r_[np.zeros(100), np.full(100, 100.0)]}
    )
    model = RayLGBMForecast(
        num_workers=2, n_estimators=5, verbosity=-1, random_state=0, n_jobs=2
    )
    model.fit(ray.data.from_pandas(df), target_col="y")

    # 0.0 would mean rank 0 only ever saw the first shard
    np.testing.assert_allclose(
        model.model_.predict(pd.DataFrame({"x": [1.0]})), [50.0], atol=1e-6
    )
    # the clamp is the worker's business; model_ carries what was asked for
    assert model.model_.n_jobs == 2
    # and none of the worker's network params, which would hang a local refit
    assert "machines" not in model.model_.get_params()
    clone(model.model_).fit(df[["x"]], df["y"])
    model.model_.booster_.refit(df[["x"]], df["y"])
    model.model_.booster_.save_model(tmp_path / "model.txt")
    lgb.Booster(model_file=tmp_path / "model.txt").refit(df[["x"]], df["y"])


@pytest.mark.ray
@pytest.mark.parametrize(
    "model_cls,local_cls",
    [(RayLGBMForecast, lgb.LGBMRegressor), (RayXGBForecast, xgb.XGBRegressor)],
    ids=["lightgbm", "xgboost"],
)
def test_model_is_the_estimator_fitted_in_the_worker(model_cls, local_cls):
    """model_ is the estimator the worker fitted, not one rebuilt from params.

    Rebuilding it from the native params lost the user's arguments (n_estimators
    came back as the default) and the booster's scores.
    """
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "lag1": rng.normal(size=200),
            "lag2": rng.normal(size=200),
            "y": rng.normal(size=200),
        }
    )
    kwargs = {"verbosity": -1} if model_cls is RayLGBMForecast else {}
    model = model_cls(random_state=0, n_estimators=5, **kwargs)
    model.fit(ray.data.from_pandas(df), target_col="y")

    local = model.model_
    assert isinstance(local, local_cls)
    # the user's params survive the round trip
    assert local.n_estimators == 5
    assert local.random_state == 0
    # and so do the fitted attributes
    assert local.evals_result_
    assert local.feature_importances_.shape == (2,)

    X = df[["lag1", "lag2"]].head(10)
    booster = local.booster_ if model_cls is RayLGBMForecast else local.get_booster()
    if model_cls is RayLGBMForecast:
        np.testing.assert_allclose(local.predict(X), booster.predict(X))
        assert booster.num_trees() == 5
    else:
        np.testing.assert_allclose(
            local.predict(X), booster.predict(xgb.DMatrix(X)), rtol=1e-6
        )


@pytest.fixture
def run_train_loop(monkeypatch):
    """Runs a train loop in this process, recording its ray.train.report calls."""
    import ray.train
    import ray.train.lightgbm

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=100), "y": rng.random(size=100)})
    shard = SimpleNamespace(
        materialize=lambda: SimpleNamespace(to_pandas=lambda: df.copy())
    )
    monkeypatch.setattr(ray.train, "get_dataset_shard", lambda _name: shard)
    # the driver isn't a worker, so it has no assigned resources to read
    for module in ("lgb", "xgb"):
        monkeypatch.setattr(
            f"mlforecast.distributed.models.ray.{module}.worker_n_jobs",
            lambda _requested: 1,
        )
    monkeypatch.setattr(ray.train.lightgbm, "get_network_params", lambda: {})

    def run(train_loop, params, rank=0):
        reports = []

        def report(metrics, checkpoint=None):
            # the checkpoint's directory is removed once report returns
            files, model = None, None
            if checkpoint is not None:
                ckpt_dir = Path(checkpoint.path)
                files = sorted(p.name for p in ckpt_dir.iterdir())
                with open(ckpt_dir / "model.pkl", "rb") as f:
                    model = pickle.load(f)
            reports.append((metrics, files, model))

        context = SimpleNamespace(get_world_rank=lambda: rank)
        monkeypatch.setattr(ray.train, "get_context", lambda: context)
        monkeypatch.setattr(ray.train, "report", report)
        if train_loop is _lgb_train_loop:
            params = {"verbosity": -1, **params}
        train_loop({"params": params, "target_col": "y"})
        return reports

    return run


@pytest.mark.ray
@pytest.mark.parametrize(
    "train_loop,data,metric,booster_file",
    [
        (_lgb_train_loop, "train", "l2", "model.txt"),
        (_xgb_train_loop, "validation_0", "rmse", "model.ubj"),
    ],
    ids=["lightgbm", "xgboost"],
)
@pytest.mark.parametrize("rank", [0, 1])
def test_train_loop_reports_once(
    run_train_loop, train_loop, data, metric, booster_file, rank
):
    """Each worker reports once, with the final metrics; only rank 0 checkpoints."""
    reports = run_train_loop(train_loop, {"n_estimators": 5}, rank=rank)

    assert len(reports) == 1
    metrics, files, model = reports[0]
    assert list(metrics) == [f"{data}-{metric}"]
    if rank != 0:
        assert files is None
        return
    assert files == sorted(["model.pkl", booster_file])
    assert metrics[f"{data}-{metric}"] == model.evals_result_[data][metric][-1]


@pytest.mark.ray
def test_xgb_train_loop_without_evaluated_metrics(run_train_loop):
    """xgboost doesn't set evals_result_ when no metric is evaluated."""
    reports = run_train_loop(
        _xgb_train_loop, {"n_estimators": 2, "disable_default_eval_metric": True}
    )
    assert [metrics for metrics, *_ in reports] == [{}]


@pytest.mark.ray
@pytest.mark.parametrize(
    "n_jobs,user_nthread,nthread",
    [(None, None, 0), (2, None, 2), (None, 3, 3), (2, 3, 3)],
)
def test_xgb_train_loop_restores_the_thread_params_with_a_metric_list(
    run_train_loop, n_jobs, user_nthread, nthread
):
    """Restoring them mustn't push the eval_metric list into the booster."""
    params = {"n_estimators": 2, "eval_metric": ["rmse", "mae"], "n_jobs": n_jobs}
    if user_nthread is not None:
        params["nthread"] = user_nthread
    reports = run_train_loop(_xgb_train_loop, params)
    metrics, _, model = reports[0]
    assert set(metrics) == {"validation_0-rmse", "validation_0-mae"}
    assert model.n_jobs == n_jobs
    config = json.loads(model.get_booster().save_config())
    assert int(config["learner"]["generic_param"]["nthread"]) == nthread


@pytest.mark.ray
@pytest.mark.parametrize(
    "tree_param,tree_learner", [("tree_learner", "voting"), ("tree", "Voting")]
)
def test_lgb_train_loop_honors_the_user_tree_learner(
    run_train_loop, tree_param, tree_learner
):
    reports = run_train_loop(
        _lgb_train_loop, {"n_estimators": 2, tree_param: tree_learner}
    )
    _, _, model = reports[0]
    assert model.booster_.params["tree_learner"] == tree_learner
    assert model.get_params()[tree_param] == tree_learner


@pytest.mark.ray
@pytest.mark.parametrize("tree_learner", ["serial", "feature"])
def test_lgb_train_loop_replaces_an_unsupported_tree_learner(
    run_train_loop, tree_learner
):
    """Serial trains on the shard alone and feature parallel needs all the data."""
    with pytest.warns(UserWarning, match="tree_learner"):
        reports = run_train_loop(
            _lgb_train_loop, {"n_estimators": 2, "tree_learner": tree_learner}
        )
    _, _, model = reports[0]
    assert model.booster_.params["tree_learner"] == "data_parallel"


@pytest.mark.ray
def test_lgb_train_loop_ignores_user_network_params(run_train_loop, monkeypatch):
    """Ray sets up the network, so these would collide with or override its params."""
    import ray.train.lightgbm

    network = {"num_machines": 1, "local_listen_port": 12400}
    monkeypatch.setattr(ray.train.lightgbm, "get_network_params", lambda: network)
    trained_params = []
    lgb_fit = lgb.LGBMRegressor.fit

    def fit(self, *args, **kwargs):
        trained_params.append(self.get_params())
        return lgb_fit(self, *args, **kwargs)

    monkeypatch.setattr(lgb.LGBMRegressor, "fit", fit)
    with pytest.warns(UserWarning, match="will be ignored") as record:
        reports = run_train_loop(
            _lgb_train_loop, {"n_estimators": 2, "num_machines": 3, "port": 1234}
        )
    assert {"num_machines", "port"} <= {str(w.message).split()[1] for w in record}
    [params] = trained_params
    assert "port" not in params
    assert params.items() >= network.items()
    [(_, _, model)] = reports
    assert not {"num_machines", "port"} & model.get_params().keys()


@pytest.mark.ray
def test_lgb_model_keeps_only_the_user_params(run_train_loop, monkeypatch):
    """Worker only params on model_ make a local refit wait for gone peers."""
    import ray.train.lightgbm

    network = {
        "machines": "127.0.0.1:12400",
        "num_machines": 1,
        "local_listen_port": 12400,
    }
    monkeypatch.setattr(ray.train.lightgbm, "get_network_params", lambda: network)
    freed = []
    free_network = lgb.Booster.free_network

    def record_free_network(self):
        freed.append(self._network)
        return free_network(self)

    monkeypatch.setattr(lgb.Booster, "free_network", record_free_network)
    user_params = {"n_estimators": 2, "verbosity": -1, "nthread": 3}
    reports = run_train_loop(_lgb_train_loop, user_params)
    _, _, model = reports[0]
    assert model.get_params() == lgb.LGBMRegressor(**user_params).get_params()
    assert not {*network, "num_threads", "tree_learner"} & vars(model).keys()
    assert not network.keys() & model.booster_.params.keys()
    # the worker's booster joined the network, and it was freed
    assert freed == [True]
    # a booster loaded from the saved model would read them back
    model_str = model.booster_.model_to_string()
    assert not any(f"[{key}: " in model_str for key in network)


@pytest.mark.ray
@pytest.mark.parametrize(
    "train_loop,module,thread_param",
    [(_lgb_train_loop, "lgb", "num_threads"), (_xgb_train_loop, "xgb", "nthread")],
    ids=["lightgbm", "xgboost"],
)
def test_train_loop_clamps_thread_aliases(
    run_train_loop, monkeypatch, train_loop, module, thread_param
):
    """The alias the library prefers over n_jobs is what the clamp has to see."""
    requested = []
    monkeypatch.setattr(
        f"mlforecast.distributed.models.ray.{module}.worker_n_jobs",
        lambda r: requested.append(r) or 1,
    )
    trained_nthreads = []
    xgb_fit = xgb.XGBRegressor.fit

    def fit(self, *args, **kwargs):
        xgb_fit(self, *args, **kwargs)
        config = json.loads(self.get_booster().save_config())
        trained_nthreads.append(config["learner"]["generic_param"]["nthread"])
        return self

    monkeypatch.setattr(xgb.XGBRegressor, "fit", fit)
    reports = run_train_loop(train_loop, {"n_estimators": 2, thread_param: 8})
    _, _, model = reports[0]
    assert 8 in requested
    assert model.get_params()[thread_param] == 8
    if train_loop is _lgb_train_loop:
        assert model.booster_.params["num_threads"] == 1
    else:
        assert trained_nthreads == ["1"]


@pytest.mark.ray
def test_xgb_train_loop_caps_n_jobs_and_nthread_separately(run_train_loop, monkeypatch):
    """n_jobs builds the DMatrix and nthread trains, so neither takes the other's cap."""
    monkeypatch.setattr(
        "mlforecast.distributed.models.ray.xgb.worker_n_jobs", lambda r: min(r, 4)
    )
    fitted = []
    xgb_fit = xgb.XGBRegressor.fit

    def fit(self, *args, **kwargs):
        fitted.append((self.n_jobs, self.get_params()["nthread"]))
        return xgb_fit(self, *args, **kwargs)

    monkeypatch.setattr(xgb.XGBRegressor, "fit", fit)
    reports = run_train_loop(
        _xgb_train_loop, {"n_estimators": 2, "n_jobs": 1, "nthread": 8}
    )
    _, _, model = reports[0]
    assert fitted == [(1, 4)]
    assert (model.n_jobs, model.get_params()["nthread"]) == (1, 8)


@pytest.mark.ray
def test_lgb_honors_param_aliases():
    """The hand rolled translation dropped lightgbm's aliases; its own does not.

    `objective` has five aliases, and the previous `setdefault("objective", ...)`
    only checked the canonical name, so `application` was silently overridden
    with `regression`. `num_iterations` has eleven, of which three were covered.
    """
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=100), "y": rng.random(size=100)})
    model = RayLGBMForecast(application="poisson", num_iterations=7, verbosity=-1)
    model.fit(ray.data.from_pandas(df), target_col="y")

    assert model.model_.objective_ == "poisson"
    assert model.model_.booster_.num_trees() == 7


@pytest.mark.ray
def test_xgb_keeps_random_state_and_model_is_picklable():
    """xgb.train knows `random_state`, so there was never a `seed` to translate to."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=100), "y": rng.random(size=100)})
    model = RayXGBForecast(random_state=0, n_estimators=5)
    model.fit(ray.data.from_pandas(df), target_col="y")

    params = model.model_.get_params()
    assert params["random_state"] == 0
    assert "seed" not in params
    pickle.loads(pickle.dumps(model.model_))


@pytest.mark.ray
def test_reclaim_placement_groups_frees_a_leaked_group():
    """The conftest safety net that keeps a leak from hanging the next test."""
    from ray.util.placement_group import placement_group, placement_group_table

    from .conftest import _reclaim_placement_groups

    pg = placement_group([{"CPU": 1}])
    ray.get(pg.ready())
    assert any(info["state"] == "CREATED" for info in placement_group_table().values())

    _reclaim_placement_groups()

    assert all(info["state"] == "REMOVED" for info in placement_group_table().values())


@pytest.mark.ray
def test_reclaim_placement_groups_keeps_a_pre_existing_group():
    """A group a fixture legitimately holds across tests has to survive the cleanup."""
    from ray.util.placement_group import (
        placement_group,
        placement_group_table,
        remove_placement_group,
    )

    from .conftest import _reclaim_placement_groups

    pg = placement_group([{"CPU": 1}])
    ray.get(pg.ready())

    _reclaim_placement_groups(keep={pg.id.hex()})

    assert placement_group_table()[pg.id.hex()]["state"] == "CREATED"
    remove_placement_group(pg)


@pytest.mark.ray
def test_workers_get_a_share_of_the_cluster_cpus():
    """ScalingConfig assigns one CPU per worker, which would train single threaded.

    `xgboost_ray._autodetect_resources` split the cluster's CPUs across its
    actors instead, so `n_jobs` alone could never raise the thread count back up.
    """
    # the test cluster is a single node, so its CPUs are also the smallest node's
    cpus = int(ray.cluster_resources()["CPU"])
    assert RayLGBMForecast()._resources_per_worker() == {"CPU": cpus}
    assert RayLGBMForecast(num_workers=2)._resources_per_worker() == {"CPU": cpus // 2}
    # an explicit value wins
    assert RayXGBForecast(resources_per_worker={"CPU": 1})._resources_per_worker() == {
        "CPU": 1
    }


@pytest.mark.ray
@pytest.mark.timeout(300, method="thread")
def test_fit_executes_a_pending_dataset_before_taking_the_cpus():
    """A dataset with work left to do can't run once the workers hold the CPUs.

    Ray train hands `ScalingConfig.total_resources` to ray data as
    `exclude_resources`, so the default share (every CPU on the node) leaves data
    a budget of zero and `fit` blocks forever rather than failing.

    The timeout is what turns a regression here into a failure instead of a hung
    CI job, and it has to be the thread method: pytest-timeout's default raises
    from a SIGALRM handler, which never runs while the main thread sits in ray's
    C++ core worker. Measured, a plain `--timeout=90` didn't fire in 21 minutes.
    """
    df = pd.DataFrame(
        {"x": np.ones(200), "y": np.r_[np.zeros(100), np.full(100, 100.0)]}
    )
    # map_batches leaves the dataset pending, unlike a bare from_pandas
    dataset = ray.data.from_pandas(df).map_batches(lambda b: b, batch_size=50)
    model = RayLGBMForecast(n_estimators=5, verbosity=-1)
    # the whole cluster, so ray data would be left with nothing
    assert model._resources_per_worker() == {"CPU": int(ray.cluster_resources()["CPU"])}
    model.fit(dataset, target_col="y")

    np.testing.assert_allclose(
        model.model_.predict(pd.DataFrame({"x": [1.0]})), [50.0], atol=1e-6
    )


@pytest.mark.ray
@pytest.mark.parametrize(
    "nodes,num_workers,expected",
    [
        # the cluster has 4 CPUs but neither node can fit a 4 CPU bundle
        ([2.0, 2.0], 1, 2),
        # heterogeneous: 8 // 2 = 4, which only the larger node could take
        ([1.0, 7.0], 2, 1),
        # the node bound is what binds here, not the split
        ([8.0, 8.0], 1, 8),
    ],
    ids=["even", "heterogeneous", "bounded-by-node"],
)
def test_worker_share_fits_on_a_single_node(monkeypatch, nodes, num_workers, expected):
    """A placement group bundle has to be schedulable on one node.

    Sizing it from `cluster_resources` alone asks for more than any single node
    has, which the autoscaler can't fulfill, so `fit` hangs instead of failing.
    """
    node_table = [{"Alive": True, "Resources": {"CPU": cpus}} for cpus in nodes]
    # a stopped 1 CPU node shouldn't pull the bound down to 1
    node_table.append({"Alive": False, "Resources": {"CPU": 1.0}})
    monkeypatch.setattr(ray, "cluster_resources", lambda: {"CPU": sum(nodes)})
    monkeypatch.setattr(ray, "nodes", lambda: node_table)
    assert RayLGBMForecast(num_workers=num_workers)._resources_per_worker() == {
        "CPU": expected
    }


@pytest.mark.ray
def test_worker_n_jobs_is_capped_by_the_assigned_cpus():
    """The thread pool is sized from the CPUs the train worker was actually given.

    `model_` carries the requested `n_jobs` rather than the clamp, so asserting on
    it after a fit doesn't exercise this.
    """
    from mlforecast.distributed.models.ray._base import worker_n_jobs

    @ray.remote
    def assigned(requested):
        return worker_n_jobs(requested)

    # the session fixture starts the cluster with 2 CPUs
    assert ray.get(assigned.options(num_cpus=2).remote(8)) == 2
    # unset falls back to the whole share
    assert ray.get(assigned.options(num_cpus=2).remote(None)) == 2
    # a smaller explicit value is honoured
    assert ray.get(assigned.options(num_cpus=2).remote(1)) == 1


@pytest.mark.ray
def test_fit_does_not_write_to_the_default_storage_path(tmp_path):
    """The default RunConfig would grow ~/ray_results by a run per model per fit."""
    default_storage = Path("~/ray_results").expanduser()
    before = set(default_storage.iterdir()) if default_storage.exists() else set()

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=100), "y": rng.random(size=100)})
    dataset = ray.data.from_pandas(df)
    RayLGBMForecast(n_estimators=2, verbosity=-1).fit(dataset, target_col="y")

    after = set(default_storage.iterdir()) if default_storage.exists() else set()
    assert after == before

    # and an explicit path is honoured
    RayLGBMForecast(n_estimators=2, verbosity=-1, storage_path=str(tmp_path)).fit(
        dataset, target_col="y"
    )
    assert any(tmp_path.iterdir())
