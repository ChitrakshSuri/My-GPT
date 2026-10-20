import os
import sys
import threading

import torch
from torch.nn import functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the repo folder
AI_DIR = os.path.join(ROOT, 'ai')  # gpt.py, gpt.pt and vocab.json live here
sys.path.insert(0, AI_DIR)
os.chdir(AI_DIR)  # gpt.py reads vocab.json from the current folder

import gpt  # noqa: E402  (importing it does not start training)

# one request at a time: the attention weights are stored on the model (Head.last_wei)
LOCK = threading.Lock()


def load_model():
    model = gpt.GPTLanguageModel()
    state = torch.load(os.path.join(AI_DIR, 'gpt.pt'), map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model


def to_ids(text):
    # keep only the characters the model knows
    return [gpt.stoi[c] for c in text if c in gpt.stoi]


@torch.no_grad()
def next_token(model, idx, temperature=1.0):
    logits, _ = model(idx[:, -gpt.block_size:])
    logits = logits[:, -1, :] / max(temperature, 0.05)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


LIVE_WINDOW = 12  # how many of the latest letters the live view shows attention over


@torch.no_grad()
def next_token_live(model, idx, temperature=1.0, top_k=8):
    """Like next_token, but also returns what the live view draws for this step.

    Everything comes from the one forward pass that picks the letter, so it costs almost nothing extra."""
    context = idx[:, -gpt.block_size:]
    logits, _ = model(context)
    probs = F.softmax(logits[:, -1, :] / max(temperature, 0.05), dim=-1)
    nxt = torch.multinomial(probs, num_samples=1)
    top_p, top_i = probs[0].topk(top_k)
    window = min(LIVE_WINDOW, context.shape[1])
    # how strongly the newest letter looked at each of the last few letters: [block][head][position]
    attn = [[[round(v, 3) for v in head.last_wei[0, -1, -window:].tolist()] for head in block.sa.heads]
            for block in model.blocks]
    info = {
        'ctx': gpt.decode(context[0, -window:].tolist()),               # the letters the attention numbers refer to
        'top': [[gpt.itos[i], round(p, 4)] for p, i in zip(top_p.tolist(), top_i.tolist())],
        'p': round(probs[0, nxt.item()].item(), 4),                     # chance of the letter that was drawn
        'attn': attn,
    }
    return nxt, info


def start_context(text):
    ids = to_ids(text)
    if not ids:
        ids = [0]  # same start as gpt.py: a newline
    return torch.tensor([ids], dtype=torch.long)
