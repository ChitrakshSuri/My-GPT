import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import model as M
from tracer import trace

gpt = M.gpt
app = FastAPI(title="My-GPT API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL = None
if os.path.exists(os.path.join(M.AI_DIR, 'gpt.pt')):
    MODEL = M.load_model()


def need_model():
    if MODEL is None:
        raise HTTPException(503, "gpt.pt not found. Train the model first (see train_colab.ipynb).")
    return MODEL


class TraceRequest(BaseModel):
    text: str = Field(max_length=500)
    max_tokens: int = Field(12, ge=1, le=32)
    layers: list[int] = [0]            # which blocks to return (0 to 5)
    include_hidden: bool = False       # a slice of the feed-forward layer for drawing


class GenerateRequest(BaseModel):
    text: str = Field("", max_length=500)
    max_new_tokens: int = Field(100, ge=1, le=500)
    temperature: float = Field(1.0, ge=0.1, le=2.0)
    live: bool = False  # stream mode only: also send what the live view draws for each letter


@app.get("/api/health")
def health():
    return {"ok": MODEL is not None, "vocab_size": gpt.vocab_size,
            "parameters": sum(p.numel() for p in MODEL.parameters()) if MODEL else None}


@app.get("/api/vocab")
def vocab():
    return {"size": gpt.vocab_size, "chars": gpt.chars}


@app.get("/api/token-table")
def token_table(dims: int = 16):
    # the first few numbers of every row of the token embedding table (65 rows)
    model = need_model()
    dims = max(1, min(dims, gpt.n_embd))
    return {"rows": [[round(v, 3) for v in row] for row in model.token_embedding_table.weight[:, :dims].tolist()]}


@app.post("/api/trace")
def trace_endpoint(req: TraceRequest):
    model = need_model()
    if any(not 0 <= i < gpt.n_layer for i in req.layers):
        raise HTTPException(422, f"layers must be between 0 and {gpt.n_layer - 1}")
    with M.LOCK:
        return trace(model, req.text, req.max_tokens, req.layers, req.include_hidden)


@app.post("/api/generate")
def generate(req: GenerateRequest):
    model = need_model()
    with M.LOCK:
        idx = M.start_context(req.text)
        new = []
        for _ in range(req.max_new_tokens):
            nxt = M.next_token(model, idx, req.temperature)
            idx = M.torch.cat((idx, nxt), dim=1)
            new.append(nxt.item())
    return {"prompt": req.text, "continuation": gpt.decode(new)}


@app.post("/api/generate/stream")
def generate_stream(req: GenerateRequest):
    model = need_model()

    def events():
        with M.LOCK:
            idx = M.start_context(req.text)
            for _ in range(req.max_new_tokens):
                if req.live:
                    nxt, payload = M.next_token_live(model, idx, req.temperature)
                    payload["c"] = gpt.itos[nxt.item()]
                else:
                    nxt = M.next_token(model, idx, req.temperature)
                    payload = gpt.itos[nxt.item()]  # just the letter
                idx = M.torch.cat((idx, nxt), dim=1)
                yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


# the website: must come last, so the /api routes above are matched first
app.mount("/", StaticFiles(directory=os.path.join(M.ROOT, "frontend"), html=True), name="frontend")
