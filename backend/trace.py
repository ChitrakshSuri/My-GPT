import torch
from torch.nn import functional as F

import model as M

gpt = M.gpt


def _t(x, digits=4):
    # tensor -> nested lists, rounded so the JSON stays small
    return x.round(decimals=digits).tolist()


@torch.no_grad()
def trace(model, text, max_tokens=12):
    """Run the model once and return every intermediate value the visualizer draws.

    Forward hooks copy the output of each module while the normal forward pass runs,
    so gpt.py itself stays untouched (except Head.last_wei for the attention weights)."""
    ids = M.to_ids(text)[-max_tokens:]
    if not ids:
        ids = [0]
    idx = torch.tensor([ids], dtype=torch.long)

    cap, hooks = {}, []

    def watch(module, name):
        def hook(mod, inp, out):
            out = out.detach()
            cap[name] = out[0] if out.dim() == 3 else out  # drop the batch dimension
        hooks.append(module.register_forward_hook(hook))

    watch(model.token_embedding_table, 'tok')
    watch(model.position_embedding_table, 'pos')
    watch(model.ln_f, 'ln_f')
    watch(model.lm_head, 'logits')
    for i, block in enumerate(model.blocks):
        watch(block.ln1, f'{i}.ln1')
        watch(block.sa, f'{i}.sa')
        watch(block.ln2, f'{i}.ln2')
        watch(block.ffwd.net[0], f'{i}.hidden_pre')
        watch(block.ffwd.net[1], f'{i}.hidden')
        watch(block.ffwd, f'{i}.ffn')
        watch(block, f'{i}.out')
        for j, head in enumerate(block.sa.heads):
            watch(head.query, f'{i}.{j}.q')
            watch(head.key, f'{i}.{j}.k')
            watch(head.value, f'{i}.{j}.v')
    try:
        model(idx)
    finally:
        for h in hooks:
            h.remove()

    layers = []
    x_in = cap['tok'] + cap['pos']  # input to the first block
    for i, block in enumerate(model.blocks):
        heads = []
        for j, head in enumerate(block.sa.heads):
            q, k, v = cap[f'{i}.{j}.q'], cap[f'{i}.{j}.k'], cap[f'{i}.{j}.v']
            scores = q @ k.T * k.shape[-1] ** -0.5          # before the mask
            weights = head.last_wei[0]                      # after mask + softmax
            heads.append({'q': _t(q), 'k': _t(k), 'v': _t(v), 'scores': _t(scores),
                          'weights': _t(weights), 'out': _t(weights @ v)})
        x_attn = x_in + cap[f'{i}.sa']                      # residual 1
        layers.append({
            'in': _t(x_in), 'ln1': _t(cap[f'{i}.ln1']), 'heads': heads,
            'attn': _t(cap[f'{i}.sa']), 'after_attn': _t(x_attn),
            'ln2': _t(cap[f'{i}.ln2']), 'hidden_pre': _t(cap[f'{i}.hidden_pre']),
            'hidden': _t(cap[f'{i}.hidden']), 'ffn': _t(cap[f'{i}.ffn']),
            'out': _t(cap[f'{i}.out']),
        })
        x_in = cap[f'{i}.out']

    probs = F.softmax(cap['logits'][-1], dim=-1)
    return {
        'text': gpt.decode(ids), 'ids': ids, 'tokens': [gpt.itos[i] for i in ids], 'T': len(ids),
        'config': {'n_layer': gpt.n_layer, 'n_head': gpt.n_head, 'n_embd': gpt.n_embd,
                   'head_size': gpt.n_embd // gpt.n_head, 'block_size': gpt.block_size,
                   'vocab_size': gpt.vocab_size},
        'embeddings': {'token': _t(cap['tok']), 'position': _t(cap['pos']),
                       'sum': _t(cap['tok'] + cap['pos'])},
        'layers': layers,
        'final': {'ln_f': _t(cap['ln_f']), 'logits': _t(cap['logits']), 'probs': _t(probs, 5)},
    }
