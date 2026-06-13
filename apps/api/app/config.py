from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ResearchFlow"
    app_env: str = "development"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    web_origin: str = "http://localhost:3000"
    database_url: str = "sqlite:///./data/researchflow.db"
    redis_url: str = "redis://localhost:6379/0"
    storage_root: Path = Path("D:/ResearchFlow/data")
    secret_key: str = "local-development-secret-change-before-real-use"
    encryption_key: str = "local-development-encryption-key-change-me"
    task_mode: str = "local"
    semantic_scholar_api_key: str | None = None
    hf_token: str | None = None
    sandbox_base_image: str = "python:3.12-slim"

    model_config = SettingsConfigDict(
        env_file=("../../.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def artifact_root(self) -> Path:
        return self.storage_root / "artifacts"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.storage_root.mkdir(parents=True, exist_ok=True)
    settings.artifact_root.mkdir(parents=True, exist_ok=True)
    return settings
