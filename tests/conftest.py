import os
import sys
from pathlib import Path

os.environ["GEMINI_API_KEY"] = "test-key-never-used"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "false"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


@pytest.fixture
def fake_embedding():
    return [0.1] * 768


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
