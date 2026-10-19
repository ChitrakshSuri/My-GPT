import os
import sys

import pytest
import torch

# make the backend modules importable, wherever pytest is started from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model as M  # noqa: E402  (this also moves into the repo root and imports gpt.py)


@pytest.fixture(scope="session")
def model():
    """A model with random weights, so the tests do not need the trained gpt.pt."""
    torch.manual_seed(0)
    m = M.gpt.GPTLanguageModel()
    m.eval()
    return m


@pytest.fixture()
def client(model, monkeypatch):
    from fastapi.testclient import TestClient
    import app as A

    monkeypatch.setattr(A, "MODEL", model)
    return TestClient(A.app)
