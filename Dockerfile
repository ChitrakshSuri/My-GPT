# One container: FastAPI runs the model and also serves the website.
# Built for Hugging Face Spaces (Docker SDK), which expects the app on port 7860.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.24 /uv /usr/local/bin/uv
# UV_NO_CACHE keeps uv's download cache out of the image (it would double the size)
ENV UV_PYTHON_DOWNLOADS=never UV_NO_CACHE=1 UV_COMPILE_BYTECODE=1 UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

# dependencies first, so this slow layer is cached until pyproject.toml or uv.lock change.
# --frozen installs exactly what uv.lock says, and the lock picks the CPU-only PyTorch.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
ENV PATH="/opt/venv/bin:$PATH"

# Spaces runs containers as user 1000
RUN useradd -m -u 1000 user

# only what is needed to run: no training data, notebook or tests
COPY --chown=user ai/gpt.py ai/vocab.json ai/gpt.pt ai/
COPY --chown=user backend/*.py backend/
COPY --chown=user frontend frontend

USER user
EXPOSE 7860
CMD ["uvicorn", "app:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "7860"]
