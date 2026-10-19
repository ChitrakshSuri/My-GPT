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


def start_context(text):
    ids = to_ids(text)
    if not ids:
        ids = [0]  # same start as gpt.py: a newline
    return torch.tensor([ids], dtype=torch.long)
