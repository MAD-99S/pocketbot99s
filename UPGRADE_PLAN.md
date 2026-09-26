# PocketBot Upgrade Plan — Manual-Trade-Only, Self-Tuning Architecture
**Status:** Synced from review session, executing now.  
**Scope:** Consolidates findings from this conversation's own code investigation, commit c552609, and the investigation report into one plan.  
**Bot state at start of execution:** PID 407664 under pm2, broker connected (balance=48857.89), 9h uptime, no prediction activity in recent logs. 28 predictions in DB (15 wins / 13 losses, 53.6% win rate).

---

## 1. Decisions locked in (confirmed before writing)

| Decision | Answer |
|----------|--------|
| LLM / OpenRouter | Removed entirely. No `OPENROUTER_API_KEY`, no AI-analysis code path. |
| LightGBM ML model (`infrastructure/ml/`) | Kept in repo, dormant. Not wired into live predictions. Available later. Not part of "self-training" for this plan. |
| "AI Trade" mode | Removed as a separate mode. Only one trading mode remains: Manual Trade. |
| Multi-pair auto-scan ("best pick across all OTC pairs") | Kept, rebuilt without LLM, as a feature inside Manual Trade (not a separate mode). |
| Outcome labeling | Stays manual — Telegram button confirmation, as today. (Flagged as a throughput constraint on self-training — see §7.) |
| Self-training mechanism | The rule engine's own weights, thresholds, and confidence calibration adapt from logged outcomes. Architecture itself is what "trains." |
| Timeframes | 1-minute is primary, 5-minute is secondary. 15-minute support deprioritized from UI but code kept. |

---

## 2. Current state — verified findings

| Finding | Status | Note |
|---------|--------|------|
| `timestamp_gaps` blocking most predictions | Real, partially fixed | Commit c552609 (windowed gap-check + newer-than-newest-server-candle merge) reduces false positives, but it's a heuristic, not a structural fix. See §6.A. |
| No confidence floor exists | Confirmed | Any signal above the scorer's internal clamp (0.55–0.95) currently reaches the user, however weak. |
| 0.95+ confidence wins ~73%, below wins ~41% | **Verified from DB** | 11 predictions at conf=0.95: 8 wins = 72.7%. Below 0.95: 17 predictions, 7 wins = 41.2%. Exact, not "plausible." |
| `data_sufficiency_issues` "only populated on success" | Mischaracterized | Column is written on every Prediction row. Hard gate failures return an error before any row is created — so failure reasons aren't in DB because there's no row. Fix: log failures too (§6.C). |
| No trained ML model exists | Confirmed, expected | `infrastructure/ml/` scaffolded but idle. Per decision, stays that way. |
| AI path silently inert (no API key) | Confirmed | Moot — being removed. |
| Outcomes are self-reported only | Confirmed | Stays manual — noted as throughput limiter for self-tuning loop. |
| Bot runs four overlapping prediction systems | **Needs verification** | Claim: `generate_signal()` + MeanReversionEngine (5-min OTC, handlers.py:647) + LightGBM fallback + LLM path. **Unverified in this session** — grep handlers.py before accepting consolidation urgency. |

---

## 3. Target architecture

```
User request / auto-scan cycle
  → DataSufficiencyGate
    → fails → Log failure (status=no_signal_data) → reply: not enough data
    → passes → RegimeClassifier (ADX + BB bandwidth)
      → Timeframe / pair type
        → "5-min OTC" → MeanReversionEngine
        → "1-min / other" → Rule vote engine (generate_signal, regime-weighted)
          → ATR gate
            → suppressed → no_signal_other
            → passes → HTF confirmation (higher-TF agreement check)
              → ConfidenceCalibrator (raw score → empirical win rate)
                → Confidence floor met?
                  → no → no_signal_confidence
                  → yes → Signal delivered (single pair or scanner's top pick)
                    → User trades manually, confirms win/loss via button
                      → predictions table (outcome-labeled)
                        → Self-tuning job (calibration curve, floor, pair P&L)
                          → adjusts ConfidenceCalibrator, floor, pair list
```

One pipeline, two entry points (on-demand single pair, auto-scanner sweeping many pairs), both terminating in the same scoring path and outcome-feedback loop.

---

## 4. What gets removed

- `apps/manual_trading/strategies/ai_analysis/` (engine.py, openrouter_client.py, prompt_builder.py, response_parser.py, __init__.py) — delete outright.
- All `OPENROUTER_API_KEY` / related settings references in `config/settings.py` and `.env.example`.
- AI-first / ML-fallback branch in the handler that tries `AIAnalysisEngine` before falling back to LightGBM.
- The second, near-duplicate "AI duration" handler function in `handlers.py` — fold anything still needed into the single remaining manual-trade handler.
- `ai_signals` table: keep the table and historical data (don't drop), but stop writing new rows. Mark deprecated in code comment.

---

## 5. What gets consolidated

Today's prediction logic is split across two near-duplicate handler functions with slightly different wiring. Once AI trade is removed, there's no reason for two paths — merge into one manual-trade handler that always runs the full pipeline in §3.

**Open verification:** grep `handlers.py` for `MeanReversionEngine`, `LightGBM`, `TradingModel`, `ml_model` to confirm they're actually in the live path before consolidating.

---

## 6. Phase 1 — Fix the foundation (do first, before anything else)

### A. Finish the timestamp_gaps fix properly (structural)

Current fix (c552609) windows the gap-check to the most recent 100× timeframe_sec, which helps but can still be tripped if the boundary between old accumulated candles and a fresh batch falls inside that window.

**Structural fix:** When a full `loadHistoryPeriodFast` batch arrives, **replace** the stored candles for that `(symbol, timeframe)` key instead of `extend()`-and-cap. A fresh batch is self-consistent; there's no reason to keep merging it against arbitrarily old leftovers.

**File:** `apps/manual_trading/market_data.py`, `_handle_candle_history`.

**Add:** A log line whenever the gate's windowed-fallback path (`recent_df` falling back to `df.tail(min_candles)`) fires, so you can see how often the heuristic is still being leaned on after the structural fix.

### B. Add the confidence floor

**File:** `apps/manual_trading/signal_generator.py` (and `MeanReversionEngine`, same treatment).

**Start at 0.90** as a provisional constant — interpolated starting point, not directly supported by current data (the actual win-rate split was measured at the 0.95 boundary, since that's where the confidence clamp naturally clusters; 0.55–0.94 hasn't been cleanly tested as a range). Treat 0.90 as "conservative starting point, to be replaced by a calibration-derived floor once §7 is running" — not a permanent hand-tuned number.

Below the floor: return `has_signal=False` with a clear reason, same pattern as the data-sufficiency gate.

### C. Log gate failures, not just successes

Every time the gate returns `is_sufficient=False`, write a lightweight row capturing symbol, timeframe, issues, and detail.

**Implementation:** Add a `status` column to the existing `predictions` table (less migration than a new table):

- `delivered` — signal sent to user
- `no_signal_data` — gate failed (with issues in `data_sufficiency_issues`)
- `no_signal_confidence` — gate passed but confidence below floor
- `no_signal_other` — e.g. no model, no AI key, indicators inconclusive

Use the existing `data_sufficiency_issues` JSONB column for gate failure details (already there, just not populated on failures).

---

## 7. Phase 2 — The self-training mechanism (no model, no LLM)

### D. Scheduled self-tuning job

A background job (cron via pm2, or asyncio periodic task in the bot process) that runs on a cadence and:

1. **Rebuilds the confidence calibration curve** (`ConfidenceCalibrator` / `CalibrationStore` — already exists) from all outcome-labeled predictions to date. Formalize into a scheduled rebuild + cache (per-request rebuilding is wasted work when outcomes trickle in slowly).
2. **Re-derives the confidence floor** from the calibration curve instead of the fixed 0.90 constant — e.g. "the lowest raw-confidence bucket whose calibrated win rate, combined with that pair's payout %, is still +EV." This is the mechanism that literally makes the floor stricter or looser as the bot's own accuracy shifts.
3. **Recomputes per-symbol, payout-adjusted P&L** (win rate alone is misleading — 70% win rate on 80%-payout pair can be break-even) and maintains a "currently profitable pairs" list.
4. **Flags underperforming pairs/regimes for review** rather than silently continuing to trade them.

**EV formula (explicit):** `EV = (win_rate × payout_pct) - ((1 - win_rate) × 100)`. Floor = lowest confidence where EV > 0 (or EV > +5 minimum threshold). Payout-aware, pair-specific.

**Cadence:** Data-driven, not fixed daily. Run when N new outcomes have accumulated since last rebuild (e.g. N=10), or weekly, whichever comes first. Early on with manual labeling, daily rebuilds on 1 new data point are noise, not signal.

**First live run gate:** Wait for 50+ labeled outcomes before the first real calibration rebuild. The job can exist and be tested with synthetic data earlier.

### E. Manual outcome-labeling tradeoff

This loop is only as fast as the user tapping the confirm button. Worth revisiting the "automatic settlement" question later if the self-tuning loop feels too data-starved. Noted as a deliberate tradeoff, not an oversight.

---

## 8. Phase 3 — Rebuild the multi-pair auto-scanner (rule-based, no LLM)

As agreed, this comes back as a feature of Manual Trade, not a separate mode.

### Architecture

1. **Payout filter first** (cheap, no live data needed) — needs the payout % actually decoded from the broker's `updateAssets` payload. **This doesn't exist in the codebase yet** (both prior reports missed it; confirmed absent when checking `get_available_assets()`, which is currently a stub with no payout field). **Move payout-% decoding to Phase 1.5** — it's needed for Phase 2 before Phase 3.
2. For each pair clearing the payout filter, run the same consolidated pipeline from §3.
3. Rank by calibrated confidence; apply the confidence floor (now calibration-derived, per §7.D.2).
4. **Confirm before building:** Does concurrent live data collection across many OTC pairs actually work with the current broker client? `subscribe_candles()` only requests one-shot history, and the live-tick `changeSymbol` call lives in an unrelated method. **Test: subscribe to two pairs back to back, confirm both keep receiving live ticks** — before investing in the full scanner build. This determines whether it's a parallel scan or a rotating one.
5. Log every scanned candidate, not just the announced pick, so the ranking function can be validated over time.
6. No forced pick — if nothing clears the calibrated floor, say so rather than announcing the best of a weak field.

### Scanner defaults

- **Cadence:** Every 60s for 1-min trades (configurable). For 5-min trades, every 3-5 min.
- **Lead time:** ~20s between "picked" and announced entry time (still right for 1-min trades).
- **Underperforming-pair flags:** Flag visibly, don't auto-exclude. Auto-exclude only after larger sample + clear threshold (e.g. negative P&L over 20+ trades).

---

## 9. Open questions — resolved

| Question | Answer |
|----------|--------|
| 15-minute timeframe — remove or deprioritize? | Deprioritize from UI (pair-selection keyboard, auto-scanner). Keep code path available for manual selection. Don't delete support. |
| `gate_failures` logging — new table or status flag? | Status flag on existing `predictions` table. Less migration, keeps attempted + delivered rows together. |
| Auto-scanner cadence? | Every 60s for 1-min, every 3-5 min for 5-min. Configurable constant/setting. |
| Auto-scanner lead time? | ~20s between picked and announced entry. Still right. |
| Underperforming-pair flags — auto-exclude or visible flag? | Visible flag only. Auto-exclude after larger sample + clear threshold. |
| Payout-% decoding — when? | **Phase 1.5** — needed for Phase 2 (payout-adjusted P&L), not just Phase 3. Move earlier. |

---

## 10. Suggested order of work

1. **Phase 1 (A, B, C)** — foundation fixes. Nothing else is trustworthy until this lands.
2. **Phase 1.5** — payout-% decoding from `updateAssets` payload into a per-pair payout map.
3. **Phase 2 (D)** — self-tuning job, running against whatever labeled data exists (sparse early on; improves as data accumulates). First live run at 50+ labeled outcomes.
4. **Test the concurrency question** (§8.4) — before committing further engineering to the scanner.
5. **Phase 3 (§8)** — auto-scanner, once §8.4 confirms feasibility.
6. **Revisit manual-vs-automatic outcome-labeling** once you can see how much the pace of Phase 2 is actually limited by it.

---

*Plan synced from review session. Execution begins now — Phase 1A first.*
