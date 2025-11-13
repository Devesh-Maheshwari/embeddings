# FastText training implementation
# Your implementation goes here
from __future__ import annotations
import argparse, json, math, os, pickle, random
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .subword_hashing import ngram_bucket_ids

# -------------------------
# Device selection
# -------------------------
def pick_device(want: str = "auto") -> torch.device:
    want = (want or "auto").lower()
    if want == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if want == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if want == "cpu":
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# -------------------------
# Minimal text8 loader (uses your cached file)
# -------------------------
def load_text8_sentences(path: str = "datasets/processed/text8_pretraining.pkl") -> List[List[str]]:
    with open(path, "rb") as f:
        sents = pickle.load(f)
    return sents  # List[List[str]]

# -------------------------
# Build vocab with min_count and subsampling
# -------------------------
def build_vocab(sentences: Iterable[List[str]], min_count: int) -> Tuple[Dict[str,int], List[str], np.ndarray, int]:
    freq: Dict[str, int] = {}
    total = 0
    for toks in sentences:
        for w in toks:
            freq[w] = freq.get(w, 0) + 1
            total += 1
    items = [(w, c) for (w, c) in freq.items() if c >= min_count]
    items.sort(key=lambda x: (-x[1], x[0]))
    id2word = [w for (w, _) in items]
    word2id = {w: i for i, w in enumerate(id2word)}
    counts = np.array([c for (_, c) in items], dtype=np.int64)
    return word2id, id2word, counts, total

def keep_token(count: int, total_tokens: int, subsample_t: float, rng: np.random.Generator) -> bool:
    if subsample_t <= 0:
        return True
    f = count / total_tokens
    p_keep = (math.sqrt(f / subsample_t) + 1.0) * (subsample_t / f)
    return rng.random() < p_keep

# -------------------------
# Negative sampling table
# -------------------------
def build_neg_table(counts: np.ndarray, table_size: int = 2_000_000) -> np.ndarray:
    pow_counts = np.power(counts, 0.75)
    probs = pow_counts / pow_counts.sum()
    cum = np.cumsum(probs)
    table = np.zeros(table_size, dtype=np.int32)
    j = 0
    for i in range(table_size):
        u = (i + 0.5) / table_size
        while u > cum[j]:
            j += 1
        table[i] = j
    return table

# -------------------------
# Pair iterator (skip-gram)
# -------------------------
def sgns_pairs(
    sentences: List[List[str]],
    word2id: Dict[str,int],
    counts: np.ndarray,
    total_tokens: int,
    window: int,
    subsample_t: float,
    rng: np.random.Generator,
):
    for toks in sentences:
        # subsample within sentence
        kept: List[int] = []
        for w in toks:
            i = word2id.get(w, -1)
            if i < 0:
                continue
            if keep_token(int(counts[i]), total_tokens, subsample_t, rng):
                kept.append(i)
        L = len(kept)
        for idx, center in enumerate(kept):
            # dynamic window like word2vec (sample radius 1..window)
            rad = rng.integers(1, window + 1)
            left = max(0, idx - rad)
            right = min(L, idx + rad + 1)
            for j in range(left, right):
                if j == idx:
                    continue
                yield center, kept[j]

# -------------------------
# FastText SGNS with hashed subwords (A_bucket)
# -------------------------
class FastTextTrainer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        vector_size: int,
        bucket: int,
        min_n: int,
        max_n: int,
        device: torch.device,
        seed: int = 42,
    ):
        super().__init__()
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)

        self.vocab_size = vocab_size
        self.vector_size = vector_size
        self.bucket = bucket
        self.min_n = min_n
        self.max_n = max_n

        # word embeddings (input) and output embeddings
        self.W_in = nn.Embedding(vocab_size, vector_size)
        self.W_out = nn.Embedding(vocab_size, vector_size)
        # subword buckets (input only)
        self.W_sub = nn.Embedding(bucket, vector_size)

        # init like word2vec
        bound = 0.5 / vector_size
        nn.init.uniform_(self.W_in.weight, -bound, bound, generator=g)
        nn.init.zeros_(self.W_out.weight)
        nn.init.uniform_(self.W_sub.weight, -bound, bound, generator=g)

        self.to(device)
        self.device = device

    def forward_input(self, center_ids: torch.Tensor, ngram_lists: List[List[int]]) -> torch.Tensor:
        """
        Build the input representation: word vector + sum of subword n-gram vectors.
        center_ids: (B,)
        ngram_lists: list length B; each element is a list[int] of bucket ids
        returns: (B, D)
        """
        wv = self.W_in(center_ids)  # (B, D)
        # sum subword vectors
        # For efficiency, pad ragged lists into a flat index + offsets
        all_idx = []
        offsets = [0]
        for grams in ngram_lists:
            all_idx.extend(grams)
            offsets.append(offsets[-1] + len(grams))
        if len(all_idx) == 0:
            return wv
        idx_t = torch.tensor(all_idx, dtype=torch.long, device=self.device)
        sub = self.W_sub(idx_t)  # (N_total, D)

        # segment sum
        out = torch.zeros_like(wv)
        start = 0
        for b in range(len(ngram_lists)):
            s, e = offsets[b], offsets[b+1]
            if e > s:
                out[b] = sub[s:e].sum(0)
        return wv + out

    @torch.no_grad()
    def export_word_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Export word vectors as (W_in + sum subwords) normalized; W_out kept separate.
        For downstream, we'll save W_in (composed) and W_out, both L2-normalized row-wise.
        """
        V = self.vocab_size
        D = self.vector_size
        W_in = self.W_in.weight.detach().cpu().numpy()
        W_out = self.W_out.weight.detach().cpu().numpy()
        # Note: to get composed vectors, caller must provide per-word n-grams. We’ll return raw matrices here.
        return W_in, W_out

# -------------------------
# Training loop
# -------------------------
def train_fasttext(
    vector_size: int = 300,
    window: int = 5,
    min_count: int = 5,
    negative: int = 10,
    subsample_t: float = 1e-5,
    epochs: int = 5,
    lr: float = 0.025,
    seed: int = 42,
    min_n: int = 3,
    max_n: int = 6,
    bucket: int = 2_000_000,
    batch_pairs: int = 4096,
    device_str: str = "auto",
    sentences_path: str = "datasets/processed/text8_pretraining.pkl",
    results_dir: str = "results/fasttext",
):
    rng = np.random.default_rng(seed)
    device = pick_device(device_str)
    print(f"[fasttext] device = {device}")

    # 1) load data + vocab
    sentences = load_text8_sentences(sentences_path)
    word2id, id2word, counts, total_tokens = build_vocab(sentences, min_count=min_count)
    V = len(word2id)
    print(f"[vocab] V={V:,} | total_tokens={total_tokens:,}")

    # 2) precompute ngram bucket ids for each word (for export & speed)
    word_ngrams: List[List[int]] = [None] * V
    for i, w in enumerate(id2word):
        word_ngrams[i] = ngram_bucket_ids(w, min_n, max_n, bucket)

    # 3) negative table
    neg_table = build_neg_table(counts, table_size=2_000_000)

    # 4) model
    model = FastTextTrainer(
        vocab_size=V, vector_size=vector_size, bucket=bucket,
        min_n=min_n, max_n=max_n, device=device, seed=seed
    )
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    # helper to draw negatives avoiding target
    def sample_negs(B: int, forbid: np.ndarray) -> np.ndarray:
        idx = rng.integers(0, len(neg_table), size=B*negative)
        neg = neg_table[idx].reshape(B, negative)
        if forbid is not None:
            mask = (neg == forbid[:, None])
            while mask.any():
                resample = rng.integers(0, len(neg_table), size=mask.sum())
                neg[mask] = neg_table[resample]
                mask = (neg == forbid[:, None])
        return neg

    # 5) training
    total_pairs_seen = 0
    for ep in range(1, epochs+1):
        pair_iter = sgns_pairs(sentences, word2id, counts, total_tokens, window, subsample_t, rng)
        loss_accum = 0.0
        batch_centers: List[int] = []
        batch_contexts: List[int] = []
        for c, t in pair_iter:
            batch_centers.append(c)
            batch_contexts.append(t)
            if len(batch_centers) >= batch_pairs:
                # Flush batch
                centers = np.array(batch_centers, dtype=np.int64)
                targets = np.array(batch_contexts, dtype=np.int64)
                batch_centers.clear()
                batch_contexts.clear()

                B = len(centers)
                negs = sample_negs(B, targets)

                # build input reps (word + sum subwords)
                ngram_lists = [word_ngrams[int(i)] for i in centers]
                centers_t = torch.from_numpy(centers).to(device)
                v = model.forward_input(centers_t, ngram_lists)     # (B, D)

                # positives
                targets_t = torch.from_numpy(targets).to(device)
                u_pos = model.W_out(targets_t)                      # (B, D)
                pos_logit = (v * u_pos).sum(1).clamp(-6, 6)
                sig_pos = torch.sigmoid(pos_logit)
                loss_pos = -torch.log(sig_pos + 1e-10).sum()

                # negatives
                negs_t = torch.from_numpy(negs).to(device)          # (B, K)
                u_neg = model.W_out(negs_t)                         # (B, K, D)
                neg_logit = torch.einsum("bd,bkd->bk", v, u_neg).clamp(-6, 6)
                sig_neg = torch.sigmoid(neg_logit)
                loss_neg = -torch.log(1.0 - sig_neg + 1e-10).sum()

                loss = (loss_pos + loss_neg) / B

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total_pairs_seen += B
                loss_accum += float(loss.item())

                if total_pairs_seen % (200_000) == 0:
                    print(f"[epoch {ep}] pairs={total_pairs_seen:,} loss={loss_accum:,.3f}")
                    loss_accum = 0.0

        print(f"[epoch {ep}] done.")

    # 6) export: compose final word vectors = word + sum(ngrams)
    with torch.no_grad():
        Ww = model.W_in.weight.detach().cpu().numpy()
        Ws = model.W_sub.weight.detach().cpu().numpy()
        M = np.zeros_like(Ww)
        for i in range(V):
            g = word_ngrams[i]
            if g:
                M[i] = Ww[i] + Ws[g].sum(axis=0)
            else:
                M[i] = Ww[i]

        # normalize rows for cosine stability
        M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        W_out = model.W_out.weight.detach().cpu().numpy()
        W_out /= (np.linalg.norm(W_out, axis=1, keepdims=True) + 1e-9)

    # 7) save in your standard format
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_json = {w: int(i) for i, w in enumerate(id2word)}
    with open(out_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)
    np.savez_compressed(out_dir / "vectors.npz", W_in=M.astype(np.float32), W_out=W_out.astype(np.float32))
    print(f"[fasttext] saved → {out_dir}")
    print(f"[fasttext] V={V:,} dim={vector_size} buckets={bucket} min_n={min_n} max_n={max_n}")
    print(f"[fasttext] examples: {len(sentences):,} pseudo-sentences from text8")
    return str(out_dir)

# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vector_size", type=int, default=300)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min_count", type=int, default=5)
    ap.add_argument("--negative", type=int, default=10)
    ap.add_argument("--subsample_t", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.025)
    ap.add_argument("--min_n", type=int, default=3)
    ap.add_argument("--max_n", type=int, default=6)
    ap.add_argument("--bucket", type=int, default=2_000_000)
    ap.add_argument("--batch_pairs", type=int, default=4096)
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cuda","mps","cpu"])
    ap.add_argument("--sentences_path", type=str, default="datasets/processed/text8_pretraining.pkl")
    ap.add_argument("--results_dir", type=str, default="results/fasttext")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_fasttext(
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        negative=args.negative,
        subsample_t=args.subsample_t,
        epochs=args.epochs,
        lr=args.lr,
        min_n=args.min_n,
        max_n=args.max_n,
        bucket=args.bucket,
        batch_pairs=args.batch_pairs,
        device_str=args.device,
        sentences_path=args.sentences_path,
        results_dir=args.results_dir,
    )
