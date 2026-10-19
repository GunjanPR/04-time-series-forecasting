"""Settings loaded from the environment and `.env` (see `.env.example`)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_parse_none_str="")

    data_dir: Path = Path("./data")
    results_dir: Path = Path("./results")

    dataset: str = "m4-monthly"
    # Blank = the dataset's default (see ts_forecasting.data.DATASETS)
    horizon: int | None = None
    n_windows: int | None = None
    max_series: int | None = 500
    random_seed: int = 42
    n_jobs: int = -1

    mlflow_tracking_uri: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
