# My-GPT

A character-level GPT trained on tiny Shakespeare, built step by step by following Andrej Karpathy's ["Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY) lecture and [nanoGPT](https://github.com/karpathy/nanoGPT).

The commit history follows the order of the lecture: data loading, a bigram baseline, then self-attention, multi-head attention, feedforward layers, residual connections, layer norm and dropout.

## Files

- `bigram.py` – the bigram language model baseline
- `gpt.py` – the Transformer language model
- `input.txt` – tiny Shakespeare dataset

## Run

```bash
pip install -r requirements.txt
python bigram.py
python gpt.py
```

With Docker:

```bash
docker compose up --build
```

`max_iters` in `gpt.py` is set low for quick runs. Raise it (the lecture uses 5000) for real training.
