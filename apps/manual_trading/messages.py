"""Formatted Telegram message builders."""
from __future__ import annotations

from apps.manual_trading.models import Signal, Prediction, DURATION_OPTIONS


def format_signal(symbol: str, signal: Signal, entry_price: float) -> str:
    """Format a signal into a clean Telegram message."""
    direction_label = "CALL (Up)" if signal.direction == "call" else "PUT (Down)"
    if signal.direction == "call":
        dir_icon = "\U0001f7e2"
        direction_emoji = "CALL"
    else:
        dir_icon = "\U0001f534"
        direction_emoji = "PUT"

    lines = [
        f"{dir_icon} {direction_emoji} {symbol.replace('_otc', ' (OTC)').replace('_', '/')}",
        "",
        f"Direction: {direction_label}",
        f"Confidence: {signal.confidence:.0%}",
        f"Entry Price: {entry_price:.5f}",
    ]

    # Extract trend info from reasoning bullets for the header
    trend_line = next(
        (b for b in signal.reasoning if "trend" in b.lower() or "EMA" in b),
        None,
    )
    if trend_line:
        lines.append(f"Trend: {trend_line}")

    # Extract entry zone from reasoning bullets for the header
    entry_line = next(
        (
            b
            for b in signal.reasoning
            if "RSI" in b and any(
                kw in b.lower() for kw in ("dip", "sell zone", "entry zone")
            )
        ),
        None,
    )
    if entry_line:
        lines.append(f"Entry Zone: {entry_line}")

    lines.append("")
    lines.append("Analysis:")

    for bullet in signal.reasoning:
        lines.append(f"  - {bullet}")

    return "\n".join(lines)


def format_no_signal(symbol: str, reason: str) -> str:
    """Format message when no trade signal is available."""
    symbol_display = symbol.replace("_otc", " (OTC)").replace("_", "/")
    return (
        f"No clear signal for {symbol_display}\n\n"
        f"Market conditions aren't favorable for a trade right now.\n\n"
        f"Reason: {reason}\n\n"
        f"Try again in a few minutes."
    )


def format_prediction_confirmed(prediction: Prediction, display_confidence: float | None = None) -> str:
    """Format confirmation after a prediction is saved.

    `display_confidence` is the calibrated confidence shown to the user.
    When None, falls back to the stored raw confidence.
    """
    symbol_display = prediction.symbol.replace("_otc", " (OTC)").replace("_", "/")
    direction_emoji = "CALL" if prediction.direction == "call" else "PUT"
    duration_label = _duration_label(prediction.timeframe_sec)
    conf = display_confidence if display_confidence is not None else prediction.confidence
    conf_pct = f"{conf:.0%}" if conf is not None else "N/A"

    return (
        f"\u23f0 Prediction Recorded\n\n"
        f"{direction_emoji} {symbol_display}\n"
        f"Entry: {float(prediction.entry_price):.5f}\n"
        f"Duration: {duration_label}\n"
        f"Expires: {prediction.expiry_time.strftime('%H:%M:%S UTC')}\n"
        f"Confidence: {conf_pct}\n\n"
        f"Tracking price movement..."
    )


def format_result_request(
    symbol: str,
    direction: str,
    entry_price: float,
    timeframe_sec: int,
) -> str:
    """Format the message asking the user to confirm trade outcome."""
    symbol_display = symbol.replace("_otc", " (OTC)").replace("_", "/")
    direction_label = "CALL" if direction == "call" else "PUT"

    return (
        f"\u23f0 Trade Expired\n\n"
        f"{direction_label} {symbol_display}\n"
        f"Entry: {entry_price:.5f}\n"
        f"Duration: {_duration_label(timeframe_sec)}\n\n"
        f"What was the result?"
    )


def format_result_recorded(result: str) -> str:
    """Format confirmation after the user submits a result."""
    if result == "win":
        return "\u2705 Result recorded: WIN"
    elif result == "loss":
        return "\u274c Result recorded: LOSS"
    else:
        return "\U0001f504 Result recorded: TIE"


def format_prediction_expired(
    prediction: Prediction,
    result: str,
    exit_price: float,
) -> str:
    """Format a prediction result message."""
    symbol_display = prediction.symbol.replace("_otc", " (OTC)").replace("_", "/")
    direction_emoji = "CALL" if prediction.direction == "call" else "PUT"

    if result == "win":
        status = "\U0001f3c6 WIN"
    elif result == "loss":
        status = "\U0001f4c9 LOSS"
    else:
        status = "\U0001f504 TIE"

    return (
        f"Prediction Result\n\n"
        f"{status} {direction_emoji} {symbol_display}\n"
        f"Entry: {float(prediction.entry_price):.5f}\n"
        f"Exit: {exit_price:.5f}\n"
        f"Confidence: {prediction.confidence:.0%}"
    )


def format_stats(stats: dict) -> str:
    """Format the /stats response."""
    total = stats["total"]
    wins = stats["wins"]
    losses = stats["losses"]
    win_rate = stats["win_rate"]

    lines = [
        "\U0001f4ca Trading Stats",
        "",
        f"Total Predictions: {total}",
        f"Wins: {wins} | Losses: {losses}",
        f"Win Rate: {win_rate:.1f}%",
    ]

    if stats["by_symbol"]:
        lines.append("")
        lines.append("By Symbol:")
        for s in stats["by_symbol"]:
            sym = s["symbol"].replace("_otc", " (OTC)").replace("_", "/")
            sr = (s["wins"] / s["total"] * 100) if s["total"] > 0 else 0
            lines.append(f"  {sym}: {s['wins']}/{s['total']} ({sr:.0f}%)")

    if stats["by_confidence"]:
        lines.append("")
        lines.append("By Confidence:")
        for c in stats["by_confidence"]:
            cr = (c["wins"] / c["total"] * 100) if c["total"] > 0 else 0
            lines.append(f"  {c['bucket']}: {c['wins']}/{c['total']} ({cr:.0f}%)")

    return "\n".join(lines)


def format_recent(predictions: list[dict]) -> str:
    """Format recent predictions."""
    if not predictions:
        return "No recent predictions."

    lines = ["\U0001f4cb Recent Predictions", ""]
    for p in predictions:
        sym = p["symbol"].replace("_otc", " (OTC)").replace("_", "/")
        d = "CALL" if p["direction"] == "call" else "PUT"
        result = (p.get("result") or "pending").upper()
        conf = f"{p.get('confidence', 0):.0%}" if p.get("confidence") else "N/A"
        lines.append(f"  [{result}] {d} {sym} | {conf}")

    return "\n".join(lines)


def _duration_label(seconds: int) -> str:
    for opt in DURATION_OPTIONS:
        if opt.seconds == seconds:
            return opt.label
    return f"{seconds}s"


def format_ai_signal(
    symbol: str,
    direction: str,
    win_probability: float,
    entry_price: float,
    model_version: str,
    top_features: list[tuple[str, float]] | None = None,
    indicator_snapshot: dict[str, float] | None = None,
    reasoning: str | None = None,
) -> str:
    """Format an AI analysis signal into a Telegram message."""
    if direction == "call":
        dir_icon = "\U0001f7e2"
        direction_emoji = "CALL"
        direction_label = "CALL (Up)"
    else:
        dir_icon = "\U0001f534"
        direction_emoji = "PUT"
        direction_label = "PUT (Down)"

    confidence_pct = win_probability if direction == "call" else 1 - win_probability

    lines = [
        f"{dir_icon} {direction_emoji} {symbol.replace('_otc', ' (OTC)').replace('_', '/')}",
        "",
        f"Direction: {direction_label}",
        f"Confidence: {confidence_pct:.0%}",
        f"Entry Price: {entry_price:.5f}",
        f"Model: {model_version}",
    ]

    if reasoning:
        lines.append("")
        lines.append("Reasoning:")
        lines.append(f"  {reasoning}")

    if top_features:
        lines.append("")
        lines.append("Key Features:")
        for feat_name, importance in top_features[:5]:
            feat_display = feat_name.replace("_", " ").title()
            lines.append(f"  - {feat_display}: {importance:.1%} importance")

    if indicator_snapshot:
        lines.append("")
        lines.append("Indicators:")
        key_indicators = ["rsi", "adx", "macd_hist", "bb_pct", "stoch_k", "roc_5"]
        for ind in key_indicators:
            val = indicator_snapshot.get(ind)
            if val is not None:
                lines.append(f"  - {ind.upper()}: {val:.4f}")

    return "\n".join(lines)


def format_no_model_available() -> str:
    """Format message when ML model is not yet trained."""
    return (
        "\U0001f916 AI Analysis - Model Not Available\n\n"
        "The ML model has not been trained yet.\n"
        "Use Quick Trade for rule-based signals.\n\n"
        "The model will be trained automatically once\n"
        "sufficient trade history is collected."
    )


def format_ai_model_info(model_metadata: dict | None) -> str:
    """Format ML model information for the user."""
    if model_metadata is None:
        return "No model trained yet."

    metrics = model_metadata.get("metrics")
    lines = [
        "\U0001f916 AI Model Info",
        "",
        f"Version: {model_metadata.get('version', 'N/A')}",
        f"Trained: {model_metadata.get('created_at', 'N/A')[:10]}",
    ]

    if metrics:
        lines.extend([
            "",
            "Performance:",
            f"  Accuracy: {metrics.get('accuracy', 0):.1%}",
            f"  Precision: {metrics.get('precision', 0):.1%}",
            f"  Recall: {metrics.get('recall', 0):.1%}",
            f"  F1 Score: {metrics.get('f1', 0):.1%}",
            f"  AUC: {metrics.get('auc', 0):.3f}",
            f"  Train samples: {metrics.get('train_samples', 0)}",
            f"  Test samples: {metrics.get('test_samples', 0)}",
        ])

    feature_names = model_metadata.get("feature_names", [])
    if feature_names:
        lines.append(f"\nFeatures: {len(feature_names)}")

    return "\n".join(lines)
