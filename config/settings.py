from pathlib import Path
from decimal import Decimal

from pydantic import Field, PostgresDsn, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PostgresConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DATABASE_", env_file=".env", extra="ignore")

    url: PostgresDsn = "postgresql+asyncpg://trader:devpassword@localhost:5432/trading"
    pool_min: int = 5
    pool_max: int = 20
    connect_timeout: int = 30


class BrokerConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POCKET_OPTION_", env_file=".env", extra="ignore")

    ssid: str = ""
    url: str | None = None
    region: str = "eu"
    max_subscriptions: int = 4
    connection_timeout: int = 30
    reconnect_delay: int = 5


class TelegramConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TELEGRAM_", env_file=".env", extra="ignore")

    bot_token: SecretStr = SecretStr("")
    allowed_user_ids: list[int] = Field(default_factory=list)
    admin_user_ids: list[int] = Field(default_factory=list)
    polling: bool = True
    webhook_url: str | None = None
    webhook_port: int = 8443
    rate_limit_global: int = 30
    rate_limit_per_user: int = 10
    max_subscriptions_per_user: int = 5

    @field_validator("allowed_user_ids", "admin_user_ids", mode="before")
    @classmethod
    def _parse_comma_list(cls, v: str | int | list[int]) -> list[int]:
        if isinstance(v, int):
            return [v]
        if isinstance(v, str):
            return [int(x.strip()) for x in v.split(",") if x.strip()]
        return v


class TradingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRADING_", env_file=".env", extra="ignore")

    default_amount: Decimal = Decimal("10.0")
    default_timeframe: int = 60
    max_daily_trades: int = 50
    max_position_size: Decimal = Decimal("100.0")
    cooldown_seconds: int = 30
    max_daily_loss: Decimal = Decimal("50.0")
    max_consecutive_losses: int = 3
    base_stake: Decimal = Decimal("2.0")
    max_stake: Decimal = Decimal("10.0")


class SignalConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SIGNAL_", env_file=".env", extra="ignore")

    confidence_threshold: float = 0.65
    min_confidence: float = 0.6
    min_wins_to_activate: int = 10
    max_features: int = 50


class OpenRouterConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OPENROUTER_", env_file=".env", extra="ignore")

    api_key: str = ""
    request_timeout: float = 15.0
    max_retries: int = 3


class MLflowConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MLFLOW_", env_file=".env", extra="ignore")

    tracking_uri: str = "mlruns"
    experiment_name: str = "trading"
    model_registry_uri: str = "mlruns"


class LoggingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", extra="ignore")

    level: str = "DEBUG"
    environment: str = "development"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    debug: bool = True
    metrics_port: int = 9090
    health_check_port: int = 8080
    duckdb_path: Path = Path("storage/analytics.duckdb")

    postgres: PostgresConfig = PostgresConfig()
    broker: BrokerConfig = BrokerConfig()
    telegram: TelegramConfig = TelegramConfig()
    trading: TradingConfig = TradingConfig()
    signal: SignalConfig = SignalConfig()
    mlflow: MLflowConfig = MLflowConfig()
    openrouter: OpenRouterConfig = OpenRouterConfig()
    logging: LoggingConfig = LoggingConfig()


def load_settings() -> AppConfig:
    return AppConfig()
