# Skip-gram implementation from scratch
# Your implementation goes here
from __future__ import annotations
from operator import neg
import json, math, random
from pathlib import Path
from typing import Dict, List, Iterable, Tuple, Optional
import numpy as np
from pathlib import Path

class SkipGramModelNS:
    """
    Skip-gram with Negative Sampling (Mikolov et al., 2013)
    - vocab built from tokenized sentences (list[list[str]])
    - trains input/output embeddings with negative sampling
    - supports subsampling, dynamic window, min_count pruning
    - saves embeddings as npz + vocab as json
    """

    def __init__(
        self,
        vector_size: int = 300,
        window: int = 5,
        min_count: int = 5,
        negative: int = 5,
        subsample_t: float = 1e-3,
        epochs: int = 15,
        lr: float = 0.025,
        seed: int = 42,
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.negative = negative
        self.subsample_t = subsample_t
        self.epochs = epochs
        self.lr0 = lr
        self.rng = np.random.default_rng(seed)

        self.word2id: Dict[str, int] = {}
        self.id2word: List[str] = []
        self.counts: np.ndarray | None = None

        self.W_in: np.ndarray | None = None   # V x D
        self.W_out: np.ndarray | None = None  # V x D
        self.neg_table: np.ndarray | None = None
        self.total_tokens: int = 0

    def build_vocab(self, tokenized_sentences: Iterable[List[str]]):
        """Build vocabulary from tokenized sentences."""
        word_freq: Dict[str, int] = {}
        for sentence in tokenized_sentences:
            for word in sentence:
                word_freq[word] = word_freq.get(word, 0) + 1
                self.total_tokens += 1

        # Prune infrequent words
        pruned_words = [(word, freq) for word, freq in word_freq.items() if freq >= self.min_count]
        # sort by frequency descending then lexicographically for determinism
        pruned_words.sort(key=lambda x: (-x[1], x[0]))
        self.id2word = [word for word, _ in pruned_words]
        self.word2id = {word: idx for idx, word in enumerate(self.id2word)}
        self.counts = np.array([freq for _, freq in pruned_words], dtype=np.int64)
        
        # init embeddings
        V = len(self.id2word)
        D = self.vector_size
        bound = 0.5 / D
        self.W_in = self.rng.uniform(-bound, bound, (V, D)).astype(np.float32)
        self.W_out = np.zeros((V, D), dtype=np.float32) # output embeddings initialized to zero

        # negative sampling table (unigram distribution ^0.75)
        self._build_negative_table(self.counts)

    # build negative sampling table
    def _build_negative_table(self, counts: np.ndarray, table_size: int = 2_000_000):
        pow_freq = counts ** 0.75
        probs = pow_freq / pow_freq.sum()
        cum_probs = np.cumsum(probs)
        neg_table = np.zeros(table_size, dtype=np.int32)
        j = 0
        for i in range(table_size):
            u = (i+ 0.5) / table_size
            while u > cum_probs[j]:
                j += 1
            neg_table[i] = j
        self.neg_table = neg_table

    def _keep_token(self, count: int) -> bool:
        """Decide whether to keep a token based on subsampling."""
        if self.subsample_t <= 0:
            return True
        f = count / self.total_tokens
        # Mikolov subsampling keep prob:
        p_keep = (math.sqrt(f / self.subsample_t) + 1) * (self.subsample_t / f)
        return self.rng.random() < p_keep
    
    def _iter_pairs(self, tokenized_sentences: Iterable[List[str]]):
        """Generate (input_word_id, context_word_id) pairs with subsampling and dynamic window."""
        assert self.word2id is not None and self.counts is not None
        for sentence in tokenized_sentences:
            word_ids = [self.word2id[w] for w in sentence if w in self.word2id]
            filtered_ids = [wid for wid in word_ids if self._keep_token(self.counts[wid])]
            L = len(filtered_ids)
            for i, input_id in enumerate(filtered_ids):
                win_size = self.rng.integers(1, self.window + 1)
                start = max(0, i - win_size)
                end = min(L, i + win_size + 1)
                for j in range(start, end):
                    if j != i:
                        context_id = filtered_ids[j]
                        yield input_id, context_id

    def _negative_samples(self, batch_size: int, forbid: Optional[np.ndarray] = None) -> np.ndarray:
        """Draw negative samples from the unigram table. If forbid is provided
        (target ids), avoid collisions by simple resampling (cheap here).
        """
        assert self.neg_table is not None
        idx = self.rng.integers(0, len(self.neg_table), size=batch_size * self.negative)
        neg_samples = self.neg_table[idx].reshape(batch_size, self.negative)
        if forbid is not None:
            # avoid rare collision of negatives == targets
            mask = (neg_samples == forbid[:, None])
            while mask.any():
                resample = self.rng.integers(0, len(self.neg_table), size=mask.sum())
                neg_samples[mask] = self.neg_table[resample]
                mask = (neg_samples == forbid[:, None])
        return neg_samples

    
    def train(
        self,
        tokenized_sentences: Iterable[List[str]],
        batch_size: int = 1024,
        print_every: int = 20000
    ):
        """Train the Skip-gram model with negative sampling."""
        assert self.W_in is not None and self.W_out is not None and self.counts is not None
        for epoch in range(1, self.epochs+1):
            lr = self.lr0 * (1.0 - (epoch-1) / max(1,self.epochs))
            lr = max(lr, self.lr0 * 0.0001)
            total =0
            loss_accum = 0.0

            # mini-batch buffers
            centers = []
            targets = []

            def _flush_batch():
                nonlocal centers, targets, loss_accum, total
                if not centers:
                    return
                c = np.array(centers, dtype=np.int32)
                t = np.array(targets, dtype=np.int32)
                batch_loss = self._sgns_update(c, t, lr)
                loss_accum += batch_loss
                total += len(c)
                centers.clear()
                targets.clear()

            for center, ctx in self._iter_pairs(tokenized_sentences):
                centers.append(center)
                targets.append(ctx)
                if len(centers) >= batch_size:
                    _flush_batch()
                    if (total // batch_size) % (print_every // max(1, batch_size)) == 0:
                        pass  # keep stdout quiet by default
            _flush_batch()
            print(f"Epoch {epoch}/{self.epochs} completed. Total pairs: {total}")
            print(f"Learning rate: {lr}")
            print(f"Loss: {loss_accum / total if total > 0 else 0:.4f}")
            
    def _sgns_update(self, center_ids: np.ndarray, target_ids: np.ndarray, lr: float):
        """Perform SGNS update for a batch of center and target word ids."""
        assert self.W_in is not None and self.W_out is not None
        batch_size = center_ids.shape[0]
        k=self.negative
        D = self.vector_size

        neg_sample_ids = self._negative_samples(batch_size, forbid=target_ids)

        # Fetch embeddings
        v_c = self.W_in[center_ids]          # B x D
        v_o = self.W_out[target_ids]         # B x D
        v_n = self.W_out[neg_sample_ids]     # B x K x D

        # Positive samples
        score_pos = np.sum(v_c * v_o, axis=1)  # (B,)
        sig_pos = 1 / (1 + np.exp(-score_pos))  # σ(v·u)
        grad_pos = (sig_pos - 1)            # (B,)

        # Negative samples
        neg_logit = np.einsum("bd,bkd->bk", v_c, v_n)  
        neg_sig = 1.0 / (1.0 + np.exp(-neg_logit))       # σ(v·u_neg)
        # grad wrt logits for negative: σ (since label=0, derivative is σ-0)
        g_neg = neg_sig.astype(np.float32)               # (B, K)

        loss_pos = -np.log(sig_pos + 1e-10).sum()
        loss_neg = -np.log(1 - neg_sig + 1e-10).sum()
        # total loss for monitoring
        batch_loss = loss_pos + loss_neg

        # gradients wrt parameters
        # dL/dv = g_pos * u_pos + Σ_k (g_neg_k * u_neg_k
        grad_v_c = (grad_pos[:, None] * v_o) + np.einsum("bk,bkd->bd", g_neg, v_n)
        # dL/dv_o
        grad_v_o = grad_pos[:, None] * v_c
        # dL/dv_n_k = g_neg_k * v_c
        grad_v_n = g_neg[:, :, None] * v_c[:, None, :]
        # SGD updates
        self.W_in[center_ids] -= lr * grad_v_c
        self.W_out[target_ids] -= lr * grad_v_o
        # accumulate into the rows of W_out for negative samples
        for b in range(batch_size):
            self.W_out[neg_sample_ids[b]] -= lr * grad_v_n[b]
        return batch_loss

    def save_embeddings(self, filepath: str):
        """Save embeddings and vocabulary to disk."""
        assert self.W_in is not None
        filepath.mkdir(parents=True, exist_ok=True)
        with open(filepath / "skipgram_vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.word2id, f, ensure_ascii=False, indent=2)

        np.savez_compressed(filepath / "skipgram_embeddings.npz", W_in=self.W_in, W_out=self.W_out)
        print(f"💾 Saved embeddings and vocab to {filepath}")

    @classmethod
    def load_embeddings(cls, filepath: str) -> SkipGramModelNS:
        """Load embeddings and vocabulary from disk."""
        model = cls()
        filepath = Path(filepath)
        with open(filepath / "skipgram_vocab.json", "r", encoding="utf-8") as f:
            model.word2id = json.load(f)
        model.id2word = [None] * len(model.word2id)
        for word, idx in model.word2id.items():
            model.id2word[idx] = word
        data = np.load(filepath / "skipgram_embeddings.npz")
        model.W_in = data["W_in"]
        model.W_out = data["W_out"]
        print(f"💾 Loaded embeddings and vocab from {filepath}")
        return model
    
    def export_word_vectors(self)-> Dict[str, np.ndarray]:
        """Export word vectors as a dictionary {word: vector}."""
        assert self.W_in is not None
        return {word: self.W_in[idx] for word, idx in self.word2id.items()}

