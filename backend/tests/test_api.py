import json
import os
import threading

import pytest
import torch

import app as A
import model as M

gpt = M.gpt


def test_health(client, model):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "vocab_size": 65, "parameters": sum(p.numel() for p in model.parameters())}


def test_vocab(client):
    d = client.get("/api/vocab").json()
    assert d["size"] == 65 == len(d["chars"])
    assert d["chars"][0] == "\n" and " " in d["chars"]


def test_token_table_default_and_custom_dims(client):
    rows = client.get("/api/token-table").json()["rows"]
    assert len(rows) == 65 and len(rows[0]) == 16
    assert len(client.get("/api/token-table?dims=4").json()["rows"][0]) == 4


def test_trace_endpoint(client):
    r = client.post("/api/trace", json={"text": "To be or"})
    assert r.status_code == 200
    d = r.json()
    assert d["T"] == 8 and [layer["index"] for layer in d["layers"]] == [0]
    assert "ffn_view" not in d["layers"][0]


def test_trace_with_hidden_and_other_layers(client):
    d = client.post("/api/trace", json={"text": "To be or", "layers": [3, 5], "include_hidden": True}).json()
    assert [layer["index"] for layer in d["layers"]] == [3, 5]
    assert "ffn_view" in d["layers"][0]


@pytest.mark.parametrize("body", [
    {"text": "hi", "layers": [6]},
    {"text": "hi", "layers": [-1]},
    {"text": "hi", "max_tokens": 0},
    {"text": "hi", "max_tokens": 33},
    {"text": "x" * 501},
    {},
])
def test_trace_rejects_bad_input(client, body):
    assert client.post("/api/trace", json=body).status_code == 422


def test_trace_of_unknown_characters_still_works(client):
    d = client.post("/api/trace", json={"text": "éé"}).json()
    assert d["tokens"] == ["\n"]


def test_generate_returns_the_requested_length(client):
    d = client.post("/api/generate", json={"text": "ROMEO:", "max_new_tokens": 25}).json()
    assert d["prompt"] == "ROMEO:" and len(d["continuation"]) == 25
    assert all(c in gpt.chars for c in d["continuation"])


def test_generate_works_without_a_prompt(client):
    assert len(client.post("/api/generate", json={"max_new_tokens": 5}).json()["continuation"]) == 5


def test_generate_is_repeatable_with_the_same_seed(client):
    body = {"text": "HAMLET:", "max_new_tokens": 30, "temperature": 0.8}
    torch.manual_seed(7)
    a = client.post("/api/generate", json=body).json()["continuation"]
    torch.manual_seed(7)
    b = client.post("/api/generate", json=body).json()["continuation"]
    assert a == b


@pytest.mark.parametrize("body", [
    {"text": "a", "temperature": 0.05},
    {"text": "a", "temperature": 2.5},
    {"text": "a", "max_new_tokens": 0},
    {"text": "a", "max_new_tokens": 501},
])
def test_generate_rejects_bad_input(client, body):
    assert client.post("/api/generate", json=body).status_code == 422


def test_generate_stream_sends_one_event_per_character(client):
    r = client.post("/api/generate/stream", json={"text": "ROMEO:", "max_new_tokens": 6})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = [e for e in r.text.split("\n\n") if e]
    assert events[-1] == "data: [DONE]"
    chars = [json.loads(e[len("data: "):]) for e in events[:-1]]
    assert len(chars) == 6 and all(c in gpt.chars for c in chars)


def test_endpoints_return_503_without_a_trained_model(client, monkeypatch):
    monkeypatch.setattr(A, "MODEL", None)
    assert client.get("/api/health").json()["ok"] is False
    assert client.get("/api/vocab").status_code == 200
    assert client.post("/api/trace", json={"text": "hi"}).status_code == 503
    assert client.post("/api/generate", json={"text": "hi"}).status_code == 503
    assert client.get("/api/token-table").status_code == 503


def test_website_is_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "GPT Anatomy" in page.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/fonts/Humor-Sans.ttf").status_code == 200


@pytest.mark.skipif(not os.path.exists(os.path.join(M.AI_DIR, "gpt.pt")), reason="trained gpt.pt not found")
def test_trained_weights_know_shakespeare():
    """A random model scores about 4.2 here. The trained one should be far lower."""
    trained = M.load_model()
    ids = torch.tensor([M.to_ids("First Citizen:\nBefore we proceed any further, hear me speak.")])
    with torch.no_grad():
        _, loss = trained(ids[:, :-1], ids[:, 1:])
    assert loss.item() < 2.5


def test_live_stream_sends_what_the_live_view_draws(client):
    r = client.post("/api/generate/stream", json={"text": "ROMEO:", "max_new_tokens": 4, "live": True})
    events = [e for e in r.text.split("\n\n") if e]
    assert events[-1] == "data: [DONE]"
    steps = [json.loads(e[len("data: "):]) for e in events[:-1]]
    assert len(steps) == 4
    for step in steps:
        assert step["c"] in gpt.chars and len(step["ctx"]) <= 12
        assert len(step["top"]) == 8
        probs = [p for _, p in step["top"]]
        assert probs == sorted(probs, reverse=True) and sum(probs) <= 1.0001
        assert 0 <= step["p"] <= 1
        assert len(step["attn"]) == 6 and all(len(block) == 6 for block in step["attn"])
        for block in step["attn"]:
            for head in block:
                assert len(head) == len(step["ctx"])      # one weight per letter shown
                assert all(w >= 0 for w in head) and sum(head) <= 1.01   # rounding to 3 decimals can add a hair


def test_live_context_grows_then_stops_at_twelve_letters(client):
    r = client.post("/api/generate/stream", json={"text": "ab", "max_new_tokens": 14, "live": True})
    steps = [json.loads(e[len("data: "):]) for e in r.text.split("\n\n") if e and e != "data: [DONE]"]
    assert [len(s["ctx"]) for s in steps][:3] == [2, 3, 4]
    assert len(steps[-1]["ctx"]) == 12


def test_chance_of_the_drawn_letter_matches_the_candidate_list(client):
    r = client.post("/api/generate/stream", json={"text": "ROMEO:", "max_new_tokens": 30, "live": True})
    steps = [json.loads(e[len("data: "):]) for e in r.text.split("\n\n") if e and e != "data: [DONE]"]
    listed = [s for s in steps if s["c"] in dict(s["top"])]
    assert listed                                          # at least some draws land in the top 8
    for s in listed:
        assert dict(s["top"])[s["c"]] == s["p"]            # same probability in both places


def test_the_lock_is_free_between_letters(model):
    """If a stream kept the lock while waiting for the browser, one slow reply would block everybody else."""
    events = A.generate_events(model, A.GenerateRequest(text="ROMEO:", max_new_tokens=3, live=True))
    for _ in range(3):
        next(events)
        assert M.LOCK.acquire(blocking=False), "the stream is still holding the lock"
        M.LOCK.release()


def run_within(seconds, fn):
    """Run fn in a thread and fail, instead of hanging the whole test run, if it is stuck waiting for the lock."""
    out = {}
    thread = threading.Thread(target=lambda: out.update(value=fn()), daemon=True)
    thread.start()
    thread.join(seconds)
    assert not thread.is_alive(), "stuck waiting for the lock"
    return out["value"]


def test_other_requests_can_run_in_the_middle_of_a_reply(client, model):
    events = A.generate_events(model, A.GenerateRequest(text="ROMEO:", max_new_tokens=4))
    run_within(30, lambda: next(events))                                              # one letter, then the reply pauses
    code = run_within(30, lambda: client.post("/api/trace", json={"text": "To be"}).status_code)
    assert code == 200                                                                # a trace fits in between
    assert client.get("/api/health").status_code == 200
    assert len(list(events)) == 4                                                     # the reply carries on: 3 letters and [DONE]
