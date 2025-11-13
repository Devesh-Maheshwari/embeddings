# pretraining/word2vec/cbow.py
from __future__ import annotations
from typing import Iterable, List, Sequence, Any
import numpy as np
from pretraining.word2vec.base_sgns import BaseSGNS

class CBOWSGNS(BaseSGNS):
    """
    CBOW with Negative Sampling:
      average(context ids) -> predict center id
    """

    def _pair_iterator(self, tokenized_sentences: Iterable[List[str]]):
        assert self.word2id is not None and self.counts is not None
        for toks in tokenized_sentences:
            ids = [self.word2id[w] for w in toks if w in self.word2id]
            ids = [wid for wid in ids if self._keep_token(self.counts[wid])]
            L = len(ids)
            for i, target in enumerate(ids):
                win = self.rng.integers(1, self.window + 1)
                left = max(0, i - win)
                right = min(L, i + win + 1)
                ctx = [ids[j] for j in range(left, right) if j != i]
                if ctx:
                    yield ctx, target

    def _make_input_vectors(self, batch_inputs: Sequence[Any]) -> np.ndarray:
        # batch_inputs = List[List[int]] (context id lists)
        assert self.W_in is not None
        D = self.vector_size
        B = len(batch_inputs)
        out = np.zeros((B, D), dtype=np.float32)
        for b, ctx_ids in enumerate(batch_inputs):
            ctx_ids = np.asarray(ctx_ids, dtype=np.int32)
            vecs = self.W_in[ctx_ids]             # (len(ctx), D)
            out[b] = vecs.mean(axis=0)            # average
        return out

    def _apply_input_grad(self, batch_inputs: Sequence[Any], grad_v: np.ndarray, lr: float):
        # distribute grad_v[b] equally to each context word in sample b
        assert self.W_in is not None
        for b, ctx_ids in enumerate(batch_inputs):
            ctx_ids = np.asarray(ctx_ids, dtype=np.int32)
            if len(ctx_ids) == 0:
                continue
            self.W_in[ctx_ids] -= (lr / len(ctx_ids)) * grad_v[b]
