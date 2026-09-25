# PocketBot Architecture — What's In Place (as of 2026-09-24)

This document records the data-quality and calibration pipeline built
into `apps/manual_trading/`.  Future sessions should read it before
making changes to the signal path, so they know which components
exist and how they wire together.

## Data Layer

### The problem it solves

Pocket Option does not reliably return historical candles after
`changeSymbol`.  The bot builds OHLC candles live, tick-by-tick, from
`updateStream`, capped at 200 candles in memory, with no persistence
across restarts.  The old minimum-candle gate was timeframe-scaled
*down* (1m: 16, 5m: 10, 15m: 8) — 8 candles is not enough to
stabilize a 20-period Bollinger Band or a 26-period MACD.

### Components

- **`candle_store.py`** — `CandleStore` persists completed candles to
  Postgres keyed by `(symbol, timeframe_sec, timestamp)`.  New table
  `candles` created idempotently by `main.py`.  Methods:
  `insert_completed(symbol, timeframe_sec, candles)`,
  `load_history(symbol, timeframe_sec, limit=500)`.

- **`market_data.py`** — `_CandleBuilder` now tracks
  `ticks_in_candle` and pushes finalized candles into
  `_pending_candle_flush` via the `_on_push_to_flush` callback.  The
  callback is wired in `request_candles()` via `setattr(builder,
  "_on_push_to_flush", lambda ...)`.  `MarketDataCollector` has
  `set_candle_store(store)`, `flush_pending_candles(symbol|None)`,
  and `get_tick_history(symbol)` (returns per-candle tick counts for
  the DataSufficiencyGate liquidity check).

- **`main.py`** — instantiates `CandleStore(session_factory)`,
  wires it via `market_data.set_candle_store(candle_store)`, runs
  `_candle_flush_loop` every 30s to drain `_pending_candle_flush` to
  Postgres.  Cancels the flush task on shutdown.  Migrations:
  `candles` table, `predictions.feature_snapshot` column.

- **`constants.py`** — `MIN_CANDLES_BY_TIMEFRAME` now returns **40 for
  all timeframes** (was 16 / 10 / 8).  `ATR_SPIKE_MULTIPLIER` "always-signal"
  block marked **DEPRECATED** — kept for reference, not wired into the
  active signal path.

## Data Sufficiency Gate

**`data_sufficiency_gate.py`** — `DataSufficiencyGate.evaluate()` runs
five checks and returns a `DataQualityReport`:

1. Candle count ≥ 40 (MACD warm-up floor).
2. Staleness — last candle age ≤ 2× timeframe.
3. Timestamp gaps — max gap ≤ 1.5× timeframe (dead stream).
4. Flat candles — fraction of near-zero-range candles in recent window.
5. Tick liquidity — average ticks per candle ≥ 3.0 (skipped if
   `tick_counts` is None).

Both `_handle_quick_duration` and `_handle_ai_duration` in `handlers.py`
run the gate after waiting for candles.  If `report.is_sufficient` is
False, the handler returns "Not enough reliable data yet: {report.summary()}"
and does **not** generate a signal.

## Analysis Layer

### Regime classifier

**`regime_classifier.py`** — `RegimeClassifier.classify(df_with_indicators)`
uses ADX + Bollinger bandwidth percentile (both already computed by
`TechnicalIndicators`) to produce a `RegimeReading`:

- **TRENDING**: ADX ≥ 25 AND BB-width expanding (pctl ≥ 0.6).
- **RANGING**: ADX ≤ 18 AND BB-width squeezed (pctl ≤ 0.4).
- **UNDEFINED**: ADX in 18-25 no-man's-land, or ADX/BB disagree.

The reading carries `trend_weight` and `reversion_weight` multipliers
(default 1.6× tilt, 1.2× bonus on strong ADX+BB agreement).

### Signal generator

**`signal_generator.py`** — `generate_signal(df_with_indicators, regime,
quality_report)` splits 7 indicators into two groups:

- **Trend/momentum group** (MACD, EMA-cross, ROC) — weights scaled by
  `regime.trend_weight`.
- **Mean-reversion group** (RSI, BB %b, Stochastic) — weights scaled by
  `regime.reversion_weight`.
- **ATR volatility gate** — suppresses the signal when ATR% is in the
  bottom or top decile of the symbol's recent history (last 20 candles).
  When suppressed, `has_signal=False` and no forced direction.

When no indicator produces a usable vote or the gate is suppressed,
the output is `has_signal=False` with no forced direction.  Every
signal logs a `feature_snapshot` (rsi, macd_hist, ema_cross, bb_pct,
stoch_k, stoch_d, roc_5, atr_pct, adx, bb_width, zscore).

### Multi-timeframe confirmation

**`htf_confirmation.py`** — `HigherTimeframeConfirmation.check(df_htf,
direction, confidence)` compares the signal direction to the HTF
close-vs-recent-average.  Returns `HTFConfirmationResult` with a
disagreement type and confidence penalty (default 0.20).

In `handlers.py`, HTF disagreement downgrades both `trend_weight` and
`reversion_weight` by `(1 - penalty)`.  Run for timeframes ≥ 60s.

## Confidence Calibration

**`confidence_calibration.py`** — turns "how much the indicators agreed"
into "how often signals like this actually won."

- **`CalibrationCurve`** — buckets labeled predictions by raw confidence
  into 5 buckets (0.50-0.60, …, 0.90-1.00), computes empirical win rate
  per bucket, enforces monotonicity.  Falls back to raw confidence when
  < 30 labeled samples exist.
- **`ConfidenceCalibrator.calibrate(raw_confidence)`** — maps through the
  curve, clamped to [0.50, 0.95].
- **`CalibrationStore`** — wraps `PredictionStore.get_labeled_for_calibration()`
  and builds the curve.  `get_labeled_for_calibration()` is in
  `database.py`.

**Both signal paths calibrate**: quick path calls
`CalibrationStore(store).build_curve()` + `calibrator.calibrate(signal.confidence)`
after `generate_signal`; AI path does the same for `result_confidence`
before saving the prediction.

## DB / Model Changes

- `Prediction` model (`models.py`): added `candle_count`, `data_age_seconds`,
  `data_sufficiency_issues`, `feature_snapshot`.
- `database.py`: `PredictionStore.insert()` INSERT now includes all four new
  columns; `get_labeled_for_calibration()` returns `(symbol, direction,
  confidence, result)` for win/loss predictions.
- `main.py` migrations: `candles` table, `predictions.feature_snapshot` column,
  existing `candle_count` / `data_age_seconds` / `data_sufficiency_issues`
  columns.

## What's NOT in place yet

- **Confidence calibrator training loop**: the calibrator reads from
  `get_labeled_for_calibration()` each time it's called (lazy rebuild).
  A scheduled rebuild (cron / startup) that persists the curve is not
  wired — the lazy rebuild is fine for now but could be optimized once
  there are hundreds of labeled predictions.
- **Lower-timeframe HTF for 1-min signals**: 1-min signals skip HTF
  confirmation because there's no higher timeframe in the supported set.
  If a 30s or 15s timeframe is added later, HTF can be enabled for 1m.
- **Mean-reversion engine regime awareness**: `MeanReversionEngine` still
  uses its own standalone ADX gate (hardcoded `adx_trend_cutoff=25`).
  It is not wired to `RegimeClassifier`.  For 5-min OTC pairs it's used
  as-is; generalizing it to take a `RegimeReading` is a possible future
  enhancement, not a current gap.

## Files

| File | Role |
|---|---|
| `apps/manual_trading/candle_store.py` | Postgres candle persistence |
| `apps/manual_trading/data_sufficiency_gate.py` | Data quality gate + report |
| `apps/manual_trading/regime_classifier.py` | ADX + BB-width regime classifier |
| `apps/manual_trading/htf_confirmation.py` | Higher-timeframe confirmation |
| `apps/manual_trading/confidence_calibration.py` | Reliability curve + calibrator |
| `apps/manual_trading/signal_generator.py` | Regime-weighted rule-based signal |
| `apps/manual_trading/constants.py` | MIN_CANDLES=40, ATR_SPIKE deprecated |
| `apps/manual_trading/market_data.py` | Candle builder + collector + flush buffer |
| `apps/manual_trading/main.py` | CandleStore wiring + flush loop + migrations |
| `apps/manual_trading/handlers.py` | Gate + regime + HTF + calibrate in both paths |
| `apps/manual_trading/models.py` | Prediction model with new fields |
| `apps/manual_trading/database.py` | DB insert + get_labeled_for_calibration |
