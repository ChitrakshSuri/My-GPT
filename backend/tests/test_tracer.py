import pytest
import torch

import model as M
from tracer import trace

gpt = M.gpt
TEXT = "To be or"


@pytest.fixture(scope="module")
def traced(model):
    return trace(model, TEXT, layers=(0, 5), include_hidden=True)


def close(a, b, tol=2e-3):
    return torch.allclose(torch.tensor(a), torch.tensor(b), atol=tol)


def test_to_ids_skips_unknown_characters():
    assert M.to_ids("aéb") == [gpt.stoi["a"], gpt.stoi["b"]]
    assert M.to_ids("") == []


def test_start_context_defaults_to_newline():
    assert M.start_context("").tolist() == [[0]]
    assert M.start_context("éé").tolist() == [[0]]


def test_trace_basic_shape(traced):
    assert traced["text"] == TEXT
    assert traced["T"] == len(TEXT) == len(traced["ids"]) == len(traced["tokens"])
    cfg = traced["config"]
    assert (cfg["n_layer"], cfg["n_head"], cfg["n_embd"], cfg["head_size"]) == (6, 6, 384, 64)
    emb = traced["embeddings"]
    assert len(emb["token"]) == len(emb["position"]) == len(emb["sum"]) == traced["T"]
    assert len(emb["token"][0]) == cfg["n_embd"]


def test_only_requested_layers_are_returned(traced):
    assert [layer["index"] for layer in traced["layers"]] == [0, 5]


def test_hidden_slice_is_optional(model):
    plain = trace(model, TEXT, layers=(0,))
    assert "ffn_view" not in plain["layers"][0]


def test_max_tokens_keeps_the_last_characters(model):
    d = trace(model, "To be or not to be", max_tokens=5)
    assert d["text"] == "to be"
    assert d["T"] == 5


def test_empty_text_falls_back_to_newline(model):
    d = trace(model, "éé")
    assert d["ids"] == [0] and d["tokens"] == ["\n"]


def test_attention_weights_are_causal_and_sum_to_one(traced):
    for layer in traced["layers"]:
        assert len(layer["heads"]) == 6
        for head in layer["heads"]:
            w = torch.tensor(head["weights"])
            assert w.shape == (traced["T"], traced["T"])
            assert torch.allclose(w.sum(dim=1), torch.ones(traced["T"]), atol=5e-3)
            assert torch.all(torch.triu(w, diagonal=1) == 0)  # no peeking at the future


def test_weights_match_softmax_of_masked_scores(traced):
    head = traced["layers"][0]["heads"][0]
    s = torch.tensor(head["scores"])
    masked = s.masked_fill(torch.tril(torch.ones_like(s)) == 0, float("-inf"))
    assert close(torch.softmax(masked, dim=-1).tolist(), head["weights"])


def test_residual_connections_add_up(traced):
    for layer in traced["layers"]:
        assert close(layer["after_attn"], (torch.tensor(layer["in"]) + torch.tensor(layer["attn"])).tolist())
        assert close(layer["out"], (torch.tensor(layer["after_attn"]) + torch.tensor(layer["ffn"])).tolist())


def test_layer_norm_output_has_zero_mean_and_unit_spread(traced):
    ln = torch.tensor(traced["layers"][0]["ln1"])
    assert torch.allclose(ln.mean(dim=1), torch.zeros(traced["T"]), atol=1e-2)
    assert torch.allclose(ln.std(dim=1, unbiased=False), torch.ones(traced["T"]), atol=1e-2)


def test_feed_forward_slice(traced):
    fv = traced["layers"][0]["ffn_view"]
    T = traced["T"]
    assert len(fv["hidden"]) == T and len(fv["hidden"][0]) == 64
    assert len(fv["w1"]) == T and len(fv["w1"][0]) == 16 and len(fv["w1"][0][0]) == 64
    assert len(fv["w2"]) == T and len(fv["w2"][0]) == 64 and len(fv["w2"][0][0]) == 16
    assert fv["total"] == 4 * 384
    assert all(0 <= n <= fv["total"] for n in fv["active"])
    assert min(min(row) for row in fv["hidden"]) >= 0  # ReLU leaves no negative values


def test_trace_logits_match_a_plain_forward_pass(model, traced):
    with torch.no_grad():
        logits, _ = model(torch.tensor([traced["ids"]]))
    assert close(traced["final"]["logits"], logits[0].tolist())


def test_final_probabilities_sum_to_one(traced):
    probs = traced["final"]["probs"]
    assert len(probs) == gpt.vocab_size
    assert sum(probs) == pytest.approx(1.0, abs=1e-3)


def test_trace_does_not_leave_hooks_behind(model):
    before = sum(len(m._forward_hooks) for m in model.modules())
    trace(model, TEXT)
    assert sum(len(m._forward_hooks) for m in model.modules()) == before
