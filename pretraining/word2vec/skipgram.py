# pretraining/word2vec/skipgram.py
from __future__ import annotations
from typing import Iterable, List, Sequence, Any
import numpy as np
from pretraining.word2vec.base_sgns import BaseSGNS

class SkipGramSGNS(BaseSGNS):
    """
    Skip-gram with Negative Sampling:
      center id -> predict each context id
    """

    def _pair_iterator(self, tokenized_sentences: Iterable[List[str]]):
        assert self.word2id is not None and self.counts is not None
        for toks in tokenized_sentences:
            ids = [self.word2id[w] for w in toks if w in self.word2id]
            ids = [wid for wid in ids if self._keep_token(self.counts[wid])]
            L = len(ids)
            for i, center in enumerate(ids):
                win = self.rng.integers(1, self.window + 1)
                left = max(0, i - win)
                right = min(L, i + win + 1)
                for j in range(left, right):
                    if j == i:
                        continue
                    yield center, ids[j]

    def _make_input_vectors(self, batch_inputs: Sequence[Any]) -> np.ndarray:
        centers = np.asarray(batch_inputs, dtype=np.int32)
        assert self.W_in is not None
        return self.W_in[centers]  # (B, D)

    def _apply_input_grad(self, batch_inputs: Sequence[Any], grad_v: np.ndarray, lr: float):
        centers = np.asarray(batch_inputs, dtype=np.int32)
        assert self.W_in is not None
        self.W_in[centers] -= lr * grad_v
