# My-GPT

A character-level GPT trained on tiny Shakespeare, plus a 3D "dissection" website that shows what happens inside it,
step by step: tokens, embeddings, attention, feed-forward layers and the next-letter prediction.

Built by following Andrej Karpathy's ["Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY) lecture and [nanoGPT](https://github.com/karpathy/nanoGPT).
The commit history follows the order of the lecture.

The model is a **text completer**, not a chatbot: give it the start of a line and it continues it in Shakespeare's style.

![GPT Anatomy: a 3D view of the model with a flow chart above and an explanation panel on the right](image.png)

## Layout

```
ai/        the model: gpt.py (model + training), bigram.py (baseline), input.txt (data),
           train_colab.ipynb (train on a free GPU), gpt.pt + vocab.json (trained weights, not in git for gpt.pt)
backend/   FastAPI app: runs the model, returns every intermediate tensor, streams generated text, serves the website
frontend/  the website: plain HTML/JS with three.js (no build step)
```

## Run it

You need [uv](https://docs.astral.sh/uv/) and the trained `ai/gpt.pt` and `ai/vocab.json` (see below).

```bash
uv sync                                              # installs the exact versions from uv.lock
uv run uvicorn app:app --app-dir backend --port 8001
```

Open http://localhost:8001 for the website and http://localhost:8001/docs for the API.

Or with Docker (CPU only, same image that runs on Hugging Face):

```bash
docker compose up --build        # http://localhost:7860
```

## Train the model

Open `ai/train_colab.ipynb` in Google Colab with a **T4 GPU** runtime and run the cells in order
(about 50 minutes). Download `gpt.pt` and `vocab.json` into the `ai/` folder.
To train locally instead: `cd ai && python gpt.py` (slow without a GPU).

## Test it

```bash
uv run pytest backend/tests                # unit and API tests, about 15 seconds, no trained weights needed
uv run pytest backend/tests -m e2e         # browser tests with Playwright, about 2 minutes, needs Chrome
```

## Deploy to Google Cloud Run

The `Dockerfile` builds one container that serves the website and the API. Cloud Run runs it, scales it to zero when
nobody visits, and has a monthly free allowance (check Google's pricing page for the current numbers).
Hugging Face Docker Spaces are no longer free for new accounts, so they are not used here.

You need a Google Cloud project with billing turned on. Add a budget alert in the Billing page so a mistake cannot
become a surprise bill. Then, from this folder on your own machine (the upload must include `ai/gpt.pt`, which is not in git):

```bash
gcloud auth login
gcloud config set project <your-project-id>
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud run deploy gpt-anatomy --source . --region us-central1 --port 7860 \
  --cpu 1 --memory 1Gi --min-instances 0 --max-instances 1 --cpu-boost --allow-unauthenticated
```

It prints the public URL when it finishes. The first visit after a quiet period is slower while the container starts.
To take it down: `gcloud run services delete gpt-anatomy --region us-central1`.

## API

| Endpoint | What it does |
|---|---|
| `GET /api/health` | model loaded? number of parameters |
| `GET /api/vocab` | the 65 characters |
| `GET /api/token-table?dims=16` | first numbers of every token embedding |
| `POST /api/trace` | every intermediate tensor for a text (`layers`, `include_hidden` choose what to return) |
| `POST /api/generate` | continue a text |
| `POST /api/generate/stream` | the same, one character at a time (server-sent events) |

## Credits

Lettering: xkcd Script (CC BY-NC 3.0, non-commercial) and Humor Sans. See `frontend/fonts/README.md`.
