from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Determine the base directory of the project for relative paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    # Configuration for Pydantic Settings
    # Reads environment variables from .env file (if present)
    # Allows for 'case_sensitive=False' if env vars might be different case
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        extra="ignore",  # Ignore extra env vars not defined here
        case_sensitive=True,
    )

    # --- Core Application Settings ---
    APP_NAME: str = "AshaVoice Backend"
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")
    LOG_FILE_NAME: str = Field("app.log", env="LOG_FILE_NAME")

    # --- Directory Settings ---
    DATA_DIR: str = Field("data", env="DATA_DIR")
    OUTPUT_DIR: str = Field("output", env="OUTPUT_DIR")
    LOG_DIR: str = Field("logs", env="LOG_DIR")

    @property
    def data_dir(self) -> Path:
        return Path.joinpath(BASE_DIR, self.DATA_DIR)

    @property
    def output_dir(self) -> Path:
        return Path.joinpath(self.data_dir, self.OUTPUT_DIR)

    @property
    def log_dir(self) -> Path:
        return Path.joinpath(self.data_dir, self.LOG_DIR)

    @property
    def log_file_path(self) -> Path:
        return Path.joinpath(self.log_dir, self.LOG_FILE_NAME)

    # --- Whisper Service Settings ---
    WHISPER_SERVICE_URL: str = Field("http://localhost:8001", env="WHISPER_SERVICE_URL")
    WHISPER_MAX_RETRIES: int = Field(3, env="WHISPER_MAX_RETRIES")
    WHISPER_TIMEOUT: float = Field(300.0, env="WHISPER_TIMEOUT")  # Seconds

    # --- Ollama (LLM) Service Settings ---
    LLM_PROVIDER: str = Field("ollama", env="LLM_PROVIDER")
    OLLAMA_HOST: str = Field("http://localhost:11434", env="OLLAMA_HOST")
    LLM_MODEL_NAME: str = Field("llama3", env="LLM_MODEL_NAME")
    LLM_MAX_RETRIES: int = Field(3, env="LLM_MAX_RETRIES")
    LLM_TEMPERATURE: float = Field(0.1, env="LLM_TEMPERATURE")
    LLM_TOP_P: float = Field(0.95, env="LLM_TOP_P")
    OLLAMA_TIMEOUT: float = Field(120.0, env="OLLAMA_TIMEOUT")  # Seconds


# Instantiate settings to be imported throughout the application
settings = Settings()

# Ensure directories exist during application startup (not just on instance creation)
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.output_dir.mkdir(parents=True, exist_ok=True)
settings.log_dir.mkdir(parents=True, exist_ok=True)
