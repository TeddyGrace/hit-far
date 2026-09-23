from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://postgres@localhost:5432/hitfar"

    # Auth: single user, single password.
    app_password: str = "changeme"
    secret_key: str = "dev-secret-change-me"
    session_max_age_s: int = 60 * 60 * 24 * 30
    cookie_secure: bool = False

    # Storage. "s3" for Railway Buckets / R2 / S3, "local" for dev + tests.
    storage_backend: Literal["s3", "local"] = "local"
    local_storage_dir: Path = Path("./.data/storage")
    s3_endpoint_url: str | None = Field(
        None, validation_alias=AliasChoices("S3_ENDPOINT_URL", "BUCKET_ENDPOINT")
    )
    s3_bucket: str | None = Field(None, validation_alias=AliasChoices("S3_BUCKET", "BUCKET_NAME"))
    s3_access_key_id: str | None = Field(
        None, validation_alias=AliasChoices("S3_ACCESS_KEY_ID", "BUCKET_ACCESS_KEY_ID")
    )
    s3_secret_access_key: str | None = Field(
        None,
        validation_alias=AliasChoices("S3_SECRET_ACCESS_KEY", "BUCKET_SECRET_ACCESS_KEY"),
    )
    s3_region: str = Field("auto", validation_alias=AliasChoices("S3_REGION", "BUCKET_REGION"))
    s3_addressing_style: Literal["virtual", "path", "auto"] = "virtual"
    # Public origin of the web app; used for bucket CORS so browsers can PUT presigned uploads.
    public_origin: str | None = None

    # Pipeline
    ffmpeg_bin: str | None = None  # defaults to the imageio-ffmpeg bundled binary
    pose_model_path: Path = Path("./.data/models/pose_landmarker_heavy.task")
    pose_model_url: str = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    )
    proxy_max_side: int = 1280
    golfer_handedness: Literal["right", "left"] = "right"

    # Diagnosis (Claude). The SDK reads ANTHROPIC_API_KEY from the environment.
    # Sonnet 5 at medium effort: vision + structured output at well under Opus cost. Set
    # DIAGNOSIS_MODEL=claude-opus-5 / DIAGNOSIS_EFFORT=high for the most thorough read.
    diagnosis_model: str = "claude-sonnet-5"
    diagnosis_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    diagnosis_timeout_s: float = 120.0

    # Training (event model). GolfDB: labels from GitHub, 160px clips from the authors' Drive link.
    golfdb_labels_url: str = "https://raw.githubusercontent.com/wmcnally/golfdb/master/data/golfDB.pkl"
    golfdb_videos_url: str = "https://drive.google.com/file/d/1uBwRxFxW04EqG87VCoX3l6vXeV5T5JYJ/view"
    train_workers: int | None = None  # pose-extraction processes; default = CPU count

    # Outcome models: retrain after this many outcome tags/edits. The LLM "explain in words" button
    # is off by default; the models, not the LLM, produce the findings.
    outcome_retrain_every: int = 5
    outcome_llm_explain: bool = False

    # On API start, queue housekeeping jobs: metric recompute after a formula change, and the first
    # event-model training run if no trained event model exists yet.
    auto_start_jobs: bool = True

    # Worker
    worker_poll_interval_s: float = 2.0
    job_lock_timeout_s: int = 60 * 30

    web_dist_dir: Path = Path(__file__).resolve().parents[2] / "web" / "dist"


@lru_cache
def get_settings() -> Settings:
    return Settings()
