"""Data models for the manual trading prediction loop."""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class PredictionResult(str, Enum):
    WIN = "win"
    LOSS = "loss"
    TIE = "tie"
    PENDING = "pending"


class Prediction(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    telegram_id: int
    symbol: str
    timeframe_sec: int
    direction: str
    confidence: float | None = None
    reasoning: str | None = None
    indicators: dict
    entry_price: Decimal
    entry_time: datetime
    expiry_time: datetime
    exit_price: Decimal | None = None
    result: str | None = None
    result_requested_at: datetime | None = None
    created_at: datetime | None = None
    # Data quality metadata logged with the signal (for calibration auditing)
    candle_count: int | None = None
    data_age_seconds: float | None = None
    data_sufficiency_issues: list[str] | None = None
    feature_snapshot: dict | None = None


class Signal(BaseModel):
    model_config = ConfigDict(frozen=True)

    has_signal: bool
    direction: str
    confidence: float
    reasoning: list[str]
    indicators: dict


class DurationOption(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    seconds: int


DURATION_OPTIONS: list[DurationOption] = [
    DurationOption(label="1 min", seconds=60),
    DurationOption(label="5 min", seconds=300),
    DurationOption(label="15 min", seconds=900),
]
