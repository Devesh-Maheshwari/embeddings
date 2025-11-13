# pretraining/word2vec/base_sgns.py
from __future__ import annotations
import json, math
from pathlib import Path
from typing import Dict, List, Iterable, Tuple, Optional, Sequence, Any
import numpy as np

class BaseSGNS:
    """
    Base for Skip-gram / CBOW with Negative Sampling (Mikolov et al., 2013).
    Subclasses implement:
      - _pair_iterator(sentences): yields (input_repr, target_id)
        * Skip-gram: input_repr = int (center id)
        * CBOW:      input_repr = List[int] (context ids)
      - _make_input_vectors(batch_inputs): -> (B, D)
      - _apply_input_grad(batch_inputs, grad_v, lr)
    """

    def __init__(
        self,
        vector_size: int = 300,
        window: int = 5,
        min_count: int = 5,
        negative: int = 5,
        subsample_t: float = 1e-3,
        epochs: int = 5,
        lr: float = 0.025,
        seed: int = 42,
    ):
        self.vector_size = int(vector_size)
        self.window = int(window)
        self.min_count = int(min_count)
        self.negative = int(negative)
        self.subsample_t = float(subsample_t)
        self.epochs = int(epochs)
        self.lr0 = float(lr)
        self.rng = np.random.default_rng(int(seed))

        # filled after build_vocab
        self.word2id: Dict[str, int] = {}
        self.id2word: List[str] = []
        self.counts: Optional[np.ndarray] = None
        self.total_tokens: int = 0

        # parameters
        self.W_in: Optional[np.ndarray] = None   # (V, D)
        self.W_out: Optional[np.ndarray] = None  # (V, D)

        # neg sampling table
        self.neg_table: Optional[np.ndarray] = None

    # ---------- required subclass API ----------
    def _pair_iterator(self, tokenized_sentences: Iterable[List[str]]):
        raise NotImplementedError

    def _make_input_vectors(self, batch_inputs: Sequence[Any]) -> np.ndarray:
        raise NotImplementedError

    def _apply_input_grad(self, batch_inputs: Sequence[Any], grad_v: np.ndarray, lr: float):
        raise NotImplementedError

    # ---------- vocab / sampling ----------
    def build_vocab(self, tokenized_sentences: Iterable[List[str]]):
        freq: Dict[str, int] = {}
        total = 0
        for toks in tokenized_sentences:
            for w in toks:
                freq[w] = freq.get(w, 0) + 1
                total += 1
        self.total_tokens = total

        items = [(w, c) for (w, c) in freq.items() if c >= self.min_count]
        items.sort(key=lambda x: (-x[1], x[0]))

        self.id2word = [w for (w, _) in items]
        self.word2id = {w: i for i, w in enumerate(self.id2word)}
        self.counts = np.array([c for (_, c) in items], dtype=np.int64)

        V, D = len(self.id2word), self.vector_size
        bound = 0.5 / D
        self.W_in = self.rng.uniform(-bound, bound, size=(V, D)).astype(np.float32)
        self.W_out = np.zeros((V, D), dtype=np.float32)

        self._build_negative_table(self.counts)
        print(f"[BaseSGNS] Vocab size after prune: {V:,} | total tokens: {self.total_tokens:,}")

    def _build_negative_table(self, counts: np.ndarray, table_size: int = 2_000_000):
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
        self.neg_table = table

    def _keep_token(self, count: int) -> bool:
        if self.subsample_t <= 0:
            return True
        f = count / self.total_tokens
        p_keep = (math.sqrt(f / self.subsample_t) + 1.0) * (self.subsample_t / f)
        return self.rng.random() < p_keep

    # ---------- negatives ----------
    def _neg_samples(self, batch_size: int, forbid: Optional[np.ndarray] = None) -> np.ndarray:
        assert self.neg_table is not None
        K = self.negative
        idx = self.rng.integers(0, len(self.neg_table), size=batch_size * K)
        neg = self.neg_table[idx].reshape(batch_size, K)
        if forbid is not None:
            mask = (neg == forbid[:, None])
            while mask.any():
                resample = self.rng.integers(0, len(self.neg_table), size=mask.sum())
                neg[mask] = self.neg_table[resample]
                mask = (neg == forbid[:, None])
        return neg

    # ---------- training ----------
    def train(
        self,
        tokenized_sentences: Iterable[List[str]],
        batch_size: int = 1024,
        log_every_pairs: int = 200_000
    ):
        assert self.W_in is not None and self.W_out is not None and self.counts is not None

        for epoch in range(1, self.epochs + 1):
            lr = self.lr0 * (1.0 - (epoch - 1) / max(1, self.epochs))
            lr = max(lr, self.lr0 * 1e-4)

            total_pairs = 0
            loss_accum = 0.0

            inputs_batch: List[Any] = []
            targets_batch: List[int] = []

            def _flush():
                nonlocal inputs_batch, targets_batch, loss_accum, total_pairs
                if not inputs_batch:
                    return
                batch_inputs = inputs_batch
                t = np.array(targets_batch, dtype=np.int32)
                batch_loss = self._sgns_update(batch_inputs, t, lr)
                loss_accum += float(batch_loss)
                total_pairs += len(batch_inputs)
                inputs_batch = []
                targets_batch = []

            for inp, tgt in self._pair_iterator(tokenized_sentences):
                inputs_batch.append(inp)
                targets_batch.append(tgt)
                if len(inputs_batch) >= batch_size:
                    _flush()
                    if total_pairs and (total_pairs % log_every_pairs == 0):
                        print(f"[epoch {epoch}] pairs={total_pairs:,}")

            _flush()
            avg_loss = loss_accum / max(1, total_pairs)
            # self.W_in /= np.linalg.norm(self.W_in, axis=1, keepdims=True) + 1e-9
            # self.W_out /= np.linalg.norm(self.W_out, axis=1, keepdims=True) + 1e-9
            print(f"Epoch {epoch}/{self.epochs} | pairs={total_pairs:,} | lr={lr:.5f} | loss={avg_loss:.4f}")

    def _sgns_update(self, batch_inputs: Sequence[Any], target_ids: np.ndarray, lr: float) -> float:
        assert self.W_in is not None and self.W_out is not None
        B = len(batch_inputs)

        # input vectors (B, D)
        v = self._make_input_vectors(batch_inputs)               # (B, D)

        # positives
        u_pos = self.W_out[target_ids]                           # (B, D)
        pos_logit = np.sum(v * u_pos, axis=1)                    # (B,)
        pos_logit = np.clip(pos_logit, -6, 6)
        sig_pos = 1.0 / (1.0 + np.exp(-pos_logit))               # (B,)
        g_pos = (sig_pos - 1.0).astype(np.float32)               # (B,)

        # negatives
        neg_ids = self._neg_samples(B, forbid=target_ids)        # (B, K)
        u_neg = self.W_out[neg_ids]                              # (B, K, D)
        neg_logit = np.einsum("bd,bkd->bk", v, u_neg)            # (B, K)
        neg_logit = np.clip(neg_logit, -6, 6)
        sig_neg = 1.0 / (1.0 + np.exp(-neg_logit))               # (B, K)
        g_neg = sig_neg.astype(np.float32)                       # (B, K)

        # gradients wrt v
        neg_term = np.einsum("bk,bkd->bd", g_neg, u_neg)         # (B, D)
        grad_v = (g_pos[:, None] * u_pos) + neg_term             # (B, D)

        # gradients wrt u_pos / u_neg
        grad_u_pos = g_pos[:, None] * v                          # (B, D)
        grad_u_neg = g_neg[..., None] * v[:, None, :]            # (B, K, D)

        # apply to W_out
        self.W_out[target_ids] -= lr * grad_u_pos
        for b in range(B):
            self.W_out[neg_ids[b]] -= lr * grad_u_neg[b]

        # scatter grad back to W_in via subclass rule
        self._apply_input_grad(batch_inputs, grad_v, lr)

        # loss
        loss_pos = -np.log(sig_pos + 1e-10).sum()
        loss_neg = -np.log(1.0 - sig_neg + 1e-10).sum()
        return float(loss_pos + loss_neg)

    # ---------- IO ----------
    def save(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.word2id, f, ensure_ascii=False, indent=2)
        assert self.W_in is not None and self.W_out is not None
        np.savez_compressed(out_dir / "vectors.npz", W_in=self.W_in, W_out=self.W_out)
        print(f"[BaseSGNS] Saved to {out_dir}")

    @classmethod
    def load(cls, in_dir: Path) -> BaseSGNS:
        in_dir = Path(in_dir)
        with open(in_dir / "vocab.json", "r", encoding="utf-8") as f:
            word2id = json.load(f)
        arrs = np.load(in_dir / "vectors.npz")
        model = cls()
        model.word2id = {k: int(v) for k, v in word2id.items()}
        model.id2word = [None] * len(model.word2id)
        for w, i in model.word2id.items():
            model.id2word[i] = w
        model.W_in = arrs["W_in"]
        model.W_out = arrs["W_out"]
        return model

    # ---------- export ----------
    def export_word_vectors(self, which: str = "in") -> Dict[str, np.ndarray]:
        """
        which ∈ {"in","out","combined"}
        - "in":       W_in (standard)
        - "out":      W_out
        - "combined": (W_in + W_out)/2
        """
        assert self.W_in is not None and self.W_out is not None
        if which == "in":
            M = self.W_in
        elif which == "out":
            M = self.W_out
        elif which == "combined":
            M = (self.W_in + self.W_out) / 2.0
        else:
            raise ValueError("which must be one of {'in','out','combined'}")
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        return {w: M[i] for i, w in enumerate(self.id2word)}
