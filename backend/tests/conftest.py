import os
import tempfile

# Configure before any app import (settings are cached).
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://postgres@localhost:5432/hitfar_test")
os.environ["STORAGE_BACKEND"] = "local"
os.environ["LOCAL_STORAGE_DIR"] = tempfile.mkdtemp(prefix="hitfar-test-storage-")
os.environ["APP_PASSWORD"] = "test-pw"
os.environ["SECRET_KEY"] = "test-secret"
os.environ["AUTO_START_JOBS"] = "false"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.db import get_engine  # noqa: E402
from app.models import Base  # noqa: E402


@pytest.fixture(scope="session")
def engine():
    eng = get_engine()
    Base.metadata.drop_all(eng)
    Base.metadata.create_all(eng)
    yield eng


@pytest.fixture
def client(engine):
    from app.main import app

    with TestClient(app) as c:
        yield c
    with engine.begin() as conn:
        names = ", ".join(t.name for t in Base.metadata.sorted_tables)
        conn.execute(text(f"TRUNCATE {names} CASCADE"))


@pytest.fixture
def authed(client):
    # API start-up created the "owner" user from APP_PASSWORD.
    r = client.post("/api/auth/login", json={"username": "owner", "password": "test-pw"})
    assert r.status_code == 200
    return client
