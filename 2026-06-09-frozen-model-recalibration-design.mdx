# Design: Frozen-Model Recalibration for Transfer Conformal Prediction

**Date:** 2026-06-09
**Branch:** feat/transfer_cp
**Scope:** `findings_3_design_first.md` — item 1 (recalibration correctness bug)

## Problem

`_recalibrate_transfer` and `_error_scaled_transfer` call `cross_validation(refit=False)` on
`new_df` to obtain target-domain residuals. Despite `refit=False`, `cross_validation` always
fits fresh models on window 0 (`forecast.py`: `should_fit = i_window == 0 or ...`). The
residuals therefore come from a **target-trained model**, while point forecasts at serve time
come from the **source-trained model**. Intervals are calibrated for the wrong model —
systematically too narrow when the transferred model is worse on the target domain than a
locally-trained one.

## Goals

1. Replace the CV-based backtest with a frozen-model backtest that runs inference-only using
   the source-fitted `models_`.
2. Use signed residuals (`y − pred`) in the recalibrate path so systematic bias shifts the
   interval rather than just widening it.
3. Default to `step_size=1` for the frozen backtest — safe without refitting, maximises
   calibration points from short target histories.
4. Replace the current minimum-length validation with one based on actual feature-construction
   requirements (`max_lag`).

## Architecture

### Current flow (broken)

```
predict()
  → saves self.ts / models_ / _cs_df / pi / source_scales
  → _recalibrate_transfer(cv_fn=self.cross_validation)
      → cross_validation(new_df, refit=False)  ← fits fresh models on window 0
      → residuals from a target-trained model
  → restores saved state (also broken: cv mutates self.ts in-place so restore is a no-op)
```

### New flow

```
predict()
  → _frozen_backtest(fcst=self, new_df, ...)  ← inference only, source models_, no fit
      → for each window: self.predict(h, new_df=train_slice)
      → self.ts restored in finally block
      → returns backtest_results DataFrame (same format as cross_validation output)
  → _recalibrate_transfer(backtest_results)
      → signed residuals: y − pred
      → TransferResult(cs_df=signed_scores, signed=True)
  → asymmetric quantile step: lo = pred + q_lo, hi = pred + q_hi
```

**Files touched:**
- `mlforecast/forecast.py` — `_frozen_backtest`, dispatch changes, save/restore simplification,
  asymmetric quantile branch
- `mlforecast/conformal_prediction.py` — `compute_conformity_scores` signed param,
  `TransferResult.signed`, `_add_signed_transfer_intervals`, validation removal from
  `_recalibrate_transfer` / `_error_scaled_transfer`

## Component designs

### `TransferConformal` (updated, in `conformal_prediction.py`)

Add `step_size: Optional[int] = None`. When `None`, the frozen backtest defaults to
`step_size=1`. Validation: if set, must be >= 1.

### `_frozen_backtest` (new, in `forecast.py`)

```python
def _frozen_backtest(
    fcst,           # MLForecast instance
    new_df,
    n_windows: int,
    h: int,
    step_size: int = 1,
    max_lag: int = 0,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> DFType:
```

**Window splitting:**
1. Collect sorted unique timestamps from `new_df`: `times[0..T-1]`
2. For window `k` (k = 0..n_windows−1):
   - `cutoff = times[T − h − 1 − k * step_size]`
   - `train_slice` = rows where `time_col <= cutoff`
   - `valid_slice` = rows where `time_col` in `times[T−h−k*step_size : T−k*step_size]`
3. Call `fcst.predict(h=h, new_df=train_slice)` — uses source `models_`, no fit
4. Join predictions with `valid_slice[[id_col, time_col, target_col]]`
5. Add `cutoff` column; concatenate all windows

**State management:**
- Save `_original_ts = fcst.ts` before the loop
- `predict(new_df=...)` sets `fcst.ts = new_ts` on success (line 1378)
- `finally: fcst.ts = _original_ts` restores once after all windows complete

`predict` is called without `level` — only point forecasts are needed; passing `level` would
recurse into the interval computation.

**Minimum-length validation (replaces `_validate_transfer_df_length` for these methods):**

```
required = h + (n_windows − 1) * step_size + max_lag + 1
```

If any series has fewer than `required` unique timestamps, raise `ValueError` naming the
shortfall:
> "series X has 8 time steps; need 12 for n_windows=2, h=3, step_size=1, max_lag=5"

`max_lag` is read at the call site as `self.ts.keep_last_n` when set (already the max across
all lags and lag-transform windows, computed by `TimeSeries` at fit time), falling back to
`max(self.ts.lags)` if `keep_last_n` is `None`, or `0` if no lags are configured.

### `compute_conformity_scores` (updated, in `conformal_prediction.py`)

Add `signed: bool = False`. When `True`, return `y − pred` per model column instead of
`|y − pred|`. All existing callers use the default (`signed=False`).

### `TransferResult` (updated, in `conformal_prediction.py`)

Add `signed: bool = False` field. Set to `True` only by `_recalibrate_transfer`.

### `_recalibrate_transfer` (updated)

- Remove `cv_fn` parameter and call to `_validate_transfer_df_length` (validation now in
  `_frozen_backtest`)
- Accept `backtest_results` (pre-computed by `forecast.py`) instead of running its own backtest
- Call `compute_conformity_scores(backtest_results, model_names, target_col, signed=True)`
- Return `TransferResult(cs_df=signed_scores, signed=True)`

### `_error_scaled_transfer` (updated)

Same structural change — remove `cv_fn` and `_validate_transfer_df_length` call, accept
`backtest_results`. Keeps `signed=False` (needs unsigned residuals to estimate σ).

### Dispatch in `forecast.py`

For methods with `runs_target_cv=True`, run `_frozen_backtest` before calling `spec.fn`:

```python
if spec.runs_target_cv:
    backtest_results = _frozen_backtest(
        fcst=self,
        new_df=new_df,
        n_windows=effective_n,
        h=prediction_intervals.h,
        step_size=tc.step_size if tc.step_size is not None else 1,
        max_lag=self.ts.keep_last_n or (max(self.ts.lags) if self.ts.lags else 0),
        id_col=self.ts.id_col,
        time_col=self.ts.time_col,
        target_col=self.ts.target_col,
    )
else:
    backtest_results = None

_transfer_result = spec.fn(
    new_df=new_df,
    backtest_results=backtest_results,
    ...
)
```

### Save/restore simplification

For `runs_target_cv=True` methods, `_frozen_backtest` is the only state-touching operation and
it handles `self.ts` internally. The outer save/restore of `models_`, `_cs_df`,
`prediction_intervals`, and `_cs_source_scales_` is not needed for these methods.

`weighted_conformal` and `scale_aligned_weighted` (which call `preprocess`) still require the
full outer save/restore — unchanged.

### Asymmetric quantile branch (new, in `forecast.py`)

After `spec.fn` returns, before the existing `_add_conformal_*_intervals` call:

```python
if _transfer_result is not None and _transfer_result.signed:
    forecasts = _add_signed_transfer_intervals(
        forecasts, self._cs_df, level,
        id_col=self.ts.id_col,
        time_col=self.ts.time_col,
    )
else:
    # existing symmetric path unchanged
```

### `_add_signed_transfer_intervals` (new, in `conformal_prediction.py`)

For each level `L`:
- `α = 1 − L / 100`
- `q_lo = quantile(scores, α/2)` — will be negative when model over-predicts
- `q_hi = quantile(scores, 1 − α/2)`
- `lo = pred + q_lo`
- `hi = pred + q_hi`

**Bias warning:** emit `UserWarning` when `q_hi < 0` (interval entirely below point forecast)
or `q_lo > 0` (interval entirely above point forecast). Both indicate severe systematic bias
of the transferred model on the target domain. Do not clamp — clamping would break coverage.

`lo ≤ hi` is guaranteed by quantile monotonicity (`α/2 < 1 − α/2`) and cannot be violated.

## What is not in scope

- Adaptive conformal inference (`findings_3_design_first.md` item 2)
- Similarity-weighted source pooling (`findings_3_design_first.md` item 3)
- The broken save/restore for `weighted_conformal` / `scale_aligned_weighted` (pre-existing,
  separate concern)
