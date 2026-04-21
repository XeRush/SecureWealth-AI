"""
Application configuration loaded from environment variables.
Uses pydantic-settings for type-safe, environment-based config.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM / AI provider keys
    openai_api_key: str = ""
    alpha_vantage_api_key: str = ""

    # Market data cache TTL in seconds (default: 15 minutes)
    market_data_cache_ttl_seconds: int = 900

    # Vector DB backend: "faiss" (default) or "chroma"
    vector_db_backend: str = "faiss"

    # Directory where trained ML models are persisted
    model_dir: str = "app/data/models"

    # Application settings
    app_name: str = "SecureWealth Twin AI"
    debug: bool = False


# Singleton instance — import this throughout the app
settings = Settings()
