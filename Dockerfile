# One container: FastAPI runs the model and also serves the website.
# Built for Hugging Face Spaces (Docker SDK), which expects the app on port 7860.
FROM python:3.12-slim

# CPU-only PyTorch: the model is small, and this keeps the image far smaller than the CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY backend/requirements.txt /tmp/backend-requirements.txt
RUN pip install --no-cache-dir tqdm -r /tmp/backend-requirements.txt

# Spaces runs containers as user 1000
RUN useradd -m -u 1000 user
WORKDIR /app

# only what is needed to run: no training data, notebook or tests
COPY --chown=user ai/gpt.py ai/vocab.json ai/gpt.pt ai/
COPY --chown=user backend/*.py backend/
COPY --chown=user frontend frontend

USER user
EXPOSE 7860
CMD ["uvicorn", "app:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "7860"]
