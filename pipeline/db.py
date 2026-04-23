"""Single source of truth for project paths and the SQLite engine."""

from pathlib import Path

from sqlalchemy import create_engine


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    d = project_root() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def db_path() -> Path:
    return data_dir() / "sentiment.db"


def get_engine():
    return create_engine(f"sqlite:///{db_path().as_posix()}")
