# pretraining/glove/glove_training.py
from __future__ import annotations
import os, math, json, pickle, random
from pathlib import Path
from typing import Dict, Tuple, Iterable, Iterator, List

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

SHUFFLE_BUFFER_SIZE = 100000


def pick_device(want: str = "auto") -> torch.device:
    if want == "cpu":
        return torch.device("cpu")
    if want == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if want == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_preproc(preproc_path):
    data = pickle.load(open(preproc_path,"rb"))

    # normalize naming from your cooccur preproc
    if "word2id" in data:
        vocab = data["word2id"]
    elif "vocab" in data:
        vocab = data["vocab"]
    else:
        raise ValueError("no vocab found in preproc")

    id2word = None
    if "id2word" in data:
        id2word = data["id2word"]
    else:
        # build from vocab
        id2word = [None]*len(vocab)
        for w,i in vocab.items():
            id2word[i] = w

    if "cooccur" in data:
        co = data["cooccur"]
    elif "co" in data:
        co = data["co"]
    else:
        raise ValueError("no co/cooccur found in preproc pickle")

    return vocab, id2word, co

def co_minibatches(co: Dict[Tuple[int,int], float], batch_size: int) -> Iterator[Tuple[np.ndarray,np.ndarray,np.ndarray]]:
    """
    Stream (i,j,Xij) in mini-batches without materializing a giant list.
    """
    # Buffer to hold (i, j, x) triplets
    buffer: List[Tuple[int, int, float]] = []
    i_buf: List[int] = []
    j_buf: List[int] = []
    x_buf: List[float] = []

    for (i, j), x in co.items():
        buffer.append((i, j, x))
        if len(buffer) >= SHUFFLE_BUFFER_SIZE:
            random.shuffle(buffer)
            for bi, bj, bx in buffer:
                i_buf.append(bi); j_buf.append(bj); x_buf.append(bx)
                if len(i_buf) >= batch_size:
                    yield (np.array(i_buf, dtype=np.int64),
                           np.array(j_buf, dtype=np.int64),
                           np.array(x_buf, dtype=np.float32))
                    i_buf.clear(); j_buf.clear(); x_buf.clear()
            buffer.clear()

    # Handle remaining items in buffer
    if buffer:
        random.shuffle(buffer)
        for bi, bj, bx in buffer:
            i_buf.append(bi); j_buf.append(bj); x_buf.append(bx)
            if len(i_buf) >= batch_size:
                yield (np.array(i_buf, dtype=np.int64),
                       np.array(j_buf, dtype=np.int64),
                       np.array(x_buf, dtype=np.float32))
                i_buf.clear(); j_buf.clear(); x_buf.clear()

    if i_buf:
        yield (np.array(i_buf, dtype=np.int64),
               np.array(j_buf, dtype=np.int64),
               np.array(x_buf, dtype=np.float32))


def glove_weight(x: torch.Tensor, xmax: float = 100.0, alpha: float = 0.75) -> torch.Tensor:
    # f(x) = (x/xmax)^alpha if x < xmax else 1
    return torch.where(x < xmax, torch.pow(x / xmax, alpha), torch.ones_like(x))

# -------------------------
# GloVe Model
# -------------------------
class GloveModel(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.wi = nn.Embedding(vocab_size, dim)  # W
        self.wj = nn.Embedding(vocab_size, dim)  # W'
        self.bi = nn.Embedding(vocab_size, 1)    # b
        self.bj = nn.Embedding(vocab_size, 1)    # b'

        bound = 0.5 / dim
        nn.init.uniform_(self.wi.weight, -bound, bound)
        nn.init.uniform_(self.wj.weight, -bound, bound)
        nn.init.zeros_(self.bi.weight)
        nn.init.zeros_(self.bj.weight)

    def forward(self, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        # returns dot + biases
        wi = self.wi(i)     # (B,d)
        wj = self.wj(j)     # (B,d)
        bi = self.bi(i)     # (B,1)
        bj = self.bj(j)     # (B,1)
        dot = torch.sum(wi * wj, dim=1, keepdim=True)  # (B,1)
        return dot + bi + bj

# -------------------------
# Training
# -------------------------
def train_glove(
    preproc_path: str = "datasets/processed/glove_preproc.pkl",
    out_dir: str = "results/glove",
    vector_size: int = 300,
    xmax: float = 100.0,
    alpha: float = 0.75,
    epochs: int = 5,
    lr: float = 0.05,
    batch_size: int = 65536,
    device_pref: str = "auto",
) -> None:
    """
    Memory-aware, streaming GloVe trainer.
    """
    vocab, id2word, co = load_preproc(preproc_path)
    V = len(vocab)
    device = pick_device(device_pref)
    print(f"[glove] V={V:,} | device={device} | entries≈{len(co):,}")

    model = GloveModel(V, vector_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # NOTE: We iterate in fixed order per epoch to keep memory down. If you want randomness,
    # add a one-pass shuffler: iterate keys once, buffer ~few million, shuffle buffer, yield, repeat.
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        n_items = 0

        pbar = tqdm(co_minibatches(co, batch_size), desc=f"[glove] epoch {ep}")
        for i_arr, j_arr, x_arr in pbar:
            i_t = torch.from_numpy(i_arr).to(device, non_blocking=True)
            j_t = torch.from_numpy(j_arr).to(device, non_blocking=True)
            x_t = torch.from_numpy(x_arr).to(device, non_blocking=True).view(-1, 1)

            opt.zero_grad(set_to_none=True)

            pred = model(i_t, j_t)                      # (B,1)
            target = torch.log(x_t)                     # (B,1)
            w = glove_weight(x_t, xmax=xmax, alpha=alpha)  # (B,1)

            diff = pred - target                         # (B,1)
            loss = (w * diff * diff).mean()              # mean over batch

            loss.backward()
            opt.step()

            bs = i_arr.shape[0]
            total_loss += float(loss.item()) * bs
            n_items += bs

            if n_items:
                pbar.set_postfix(loss=f"{(total_loss / n_items):.5f}")

        print(f"[glove] epoch {ep} mean_loss={total_loss / max(1,n_items):.5f}")

    # Export vectors = W + W'
    with torch.no_grad():
        W_in = model.wi.weight.detach().cpu().numpy()
        W_out = model.wj.weight.detach().cpu().numpy()
        W = (W_in + W_out)  # standard practice is sum; you can also average

    # Normalize rows (cosine-friendly)
    W /= (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)

    # Save in your downstream format
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # Build/repair vocab.json from id2word
    word2id = {w: idx for idx, w in enumerate(id2word)}
    with open(Path(out_dir) / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(word2id, f)

    np.savez_compressed(Path(out_dir) / "vectors.npz", W_in=W, W_out=W)  # keep both keys for uniformity
    print(f"[glove] saved → {out_dir}/vocab.json & {out_dir}/vectors.npz")

if __name__ == "__main__":
    train_glove()
