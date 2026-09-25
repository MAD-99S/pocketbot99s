"""Technical indicator computations — self-contained, no pandas-ta dependency."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
import pandas as pd
import numpy as np


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """Compute ADX, +DI, -DI directly (Wilder's smoothing).

    Returns a DataFrame with columns ADX_{period}, DMP_{period}, DMN_{period}
    matching the pandas-ta column-naming convention.
    """
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    p_dm = pd.concat([(high - high.shift(1)), tr2], axis=1).max(axis=1)
    n_dm = pd.concat([(low.shift(1) - low), tr3], axis=1).max(axis=1)
    p_dm = p_dm.where((p_dm > n_dm) & (p_dm > 0), 0.0)
    n_dm = n_dm.where((n_dm > p_dm) & (n_dm > 0), 0.0)

    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    p_di = 100 * (p_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan))
    n_di = 100 * (n_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan))

    dx = 100 * ((p_di - n_di).abs() / (p_di + n_di).replace(0, np.nan))
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    return pd.DataFrame(
        {f"ADX_{period}": adx, f"DMP_{period}": p_di, f"DMN_{period}": n_di}
    )


class IndicatorConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    ema_fast: int = 10
    ema_slow: int = 30
    atr_period: int = 14
    stoch_k: int = 14
    stoch_d: int = 3
    roc_period: int = 10
    volume_sma_period: int = 20


class TechnicalIndicators:
    def __init__(self, config: IndicatorConfig | None = None) -> None:
        self._config = config or IndicatorConfig()

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 2:
            return df.copy()

        out = df.copy()
        c = self._config

        out["rsi"] = self._rsi(out["close"], c.rsi_period)

        macd = self._macd(out["close"], c.macd_fast, c.macd_slow, c.macd_signal)
        out = pd.concat([out, macd], axis=1)

        bb = self._bollinger(out["close"], c.bb_period, c.bb_std)
        out = pd.concat([out, bb], axis=1)

        out["ema_fast"] = out["close"].ewm(span=c.ema_fast, adjust=False).mean()
        out["ema_slow"] = out["close"].ewm(span=c.ema_slow, adjust=False).mean()
        out["ema_cross"] = (out["ema_fast"] - out["ema_slow"]) / out["close"]

        out["atr"] = self._atr(out, c.atr_period)
        out["atr_pct"] = out["atr"] / out["close"]

        stoch = self._stochastic(out, c.stoch_k, c.stoch_d)
        out = pd.concat([out, stoch], axis=1)

        out["roc"] = out["close"].pct_change(periods=c.roc_period)
        out["roc_5"] = out["close"].pct_change(periods=5)

        if "volume" in out.columns:
            vol_sma = out["volume"].rolling(window=c.volume_sma_period).mean()
            out["volume_ratio"] = np.where(
                vol_sma > 0, out["volume"] / vol_sma, 1.0
            )
        else:
            out["volume_ratio"] = 1.0

        out["body_ratio"] = (out["close"] - out["open"]).abs() / (
            out["high"] - out["low"]
        ).replace(0, np.nan)
        out["body_ratio"] = out["body_ratio"].fillna(0.5)

        out["upper_shadow"] = (out["high"] - out[["open", "close"]].max(axis=1)) / (
            out["high"] - out["low"]
        ).replace(0, np.nan)
        out["lower_shadow"] = (out[["open", "close"]].min(axis=1) - out["low"]) / (
            out["high"] - out["low"]
        ).replace(0, np.nan)

        out["return_1"] = out["close"].pct_change(periods=1)
        out["return_3"] = out["close"].pct_change(periods=3)

        adx_df = _adx(out["high"], out["low"], out["close"], period=c.atr_period)
        adx_col = f"ADX_{c.atr_period}"
        out["adx"] = adx_df[adx_col]

        zscore_window = 20
        out["zscore"] = (
            out["close"] - out["close"].rolling(window=zscore_window).mean()
        ) / out["close"].rolling(window=zscore_window).std().replace(0, np.nan)

        return out

    def feature_columns(self) -> list[str]:
        return [
            "rsi",
            "macd",
            "macd_signal",
            "macd_hist",
            "bb_upper",
            "bb_lower",
            "bb_width",
            "bb_pct",
            "ema_fast",
            "ema_slow",
            "ema_cross",
            "atr",
            "atr_pct",
            "stoch_k",
            "stoch_d",
            "roc",
            "roc_5",
            "volume_ratio",
            "body_ratio",
            "upper_shadow",
            "lower_shadow",
            "return_1",
            "return_3",
            "adx",
            "zscore",
        ]

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _macd(
        close: pd.Series, fast: int, slow: int, signal: int
    ) -> pd.DataFrame:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return pd.DataFrame(
            {"macd": macd_line, "macd_signal": signal_line, "macd_hist": histogram}
        )

    @staticmethod
    def _bollinger(close: pd.Series, period: int, std: float) -> pd.DataFrame:
        sma = close.rolling(window=period).mean()
        rolling_std = close.rolling(window=period).std()
        upper = sma + std * rolling_std
        lower = sma - std * rolling_std
        width = (upper - lower) / sma.replace(0, np.nan)
        pct = (close - lower) / (upper - lower).replace(0, np.nan)
        return pd.DataFrame(
            {"bb_upper": upper, "bb_lower": lower, "bb_width": width, "bb_pct": pct}
        )

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        high = df["high"]
        low = df["low"]
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def _stochastic(df: pd.DataFrame, k_period: int, d_period: int) -> pd.DataFrame:
        low_min = df["low"].rolling(window=k_period).min()
        high_max = df["high"].rolling(window=k_period).max()
        denom = (high_max - low_min).replace(0, np.nan)
        k = 100 * (df["close"] - low_min) / denom
        d = k.rolling(window=d_period).mean()
        return pd.DataFrame({"stoch_k": k, "stoch_d": d})
