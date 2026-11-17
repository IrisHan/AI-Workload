from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def llama_cfg():
    return load_fixture("llama-3.2-1b.json")


@pytest.fixture
def qwen_cfg():
    return load_fixture("qwen2.5-1.5b.json")


@pytest.fixture
def v3_cfg():
    return load_fixture("deepseek-v3.json")


@pytest.fixture
def mixtral_cfg():
    return load_fixture("mixtral-8x7b.json")


@pytest.fixture
def qwen1_72b_cfg():
    return load_fixture("qwen-72b-chat.json")
