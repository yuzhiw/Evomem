#!/usr/bin/env python3
"""
embedding.py — Embedding functions (_load_embed, embed)
"""
from . import config

def _load_embed():
    if config._EMB: return
    from transformers import AutoModel, AutoTokenizer; import torch
    config._TOK = AutoTokenizer.from_pretrained(config.EMBED_PATH)
    config._EMB = AutoModel.from_pretrained(config.EMBED_PATH)
    _ = torch.cuda.is_available()
    config._EMB.eval(); config._EMB.to('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  [emb] BGE loaded on {config._EMB.device}', flush=True)

def embed(texts):
    import torch; _load_embed()
    # 分批嵌入，避免大 batch 峰值显存导致 CUDA OOM（并发多进程时尤其重要）
    BATCH = 128
    all_emb = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        enc = config._TOK(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
        enc = {k: v.to(config._EMB.device) for k, v in enc.items()}
        with torch.no_grad():
            out = config._EMB(**enc)
            mask = enc['attention_mask'].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
            e = torch.sum(out.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
            e = torch.nn.functional.normalize(e, p=2, dim=1)
        all_emb.append(e.cpu().numpy().tolist())
    # 展平
    return [v for chunk in all_emb for v in chunk]
