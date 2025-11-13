# pretraining/evaluation/eval_embeddings.py
from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
import json, math
import numpy as np
from pathlib import Path

# ---- small helpers ----
def _l2_normalize(M: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    return M / denom

def _cosine_sim_matrix(vec: np.ndarray, M: np.ndarray) -> np.ndarray:
    # vec: (D,) already normalized; M: (V, D) normalized
    return M @ vec 

def _safe_get(word_to_idx: Dict[str,int], w: str) -> Optional[int]:
    return word_to_idx.get(w, None)

def _rank_and_pick(sim: np.ndarray, forbid: List[int], topn: int = 1) -> List[int]:
    # sim: (V,); forbid: indices to exclude from ranking (a,b,c words)
    sim = sim.copy()
    for idx in forbid:
        if idx is not None:
            sim[idx] = -1.0  # exclude
    order = np.argsort(-sim)  # descending
    return order[:topn].tolist()

# ---- 3CosAdd & 3CosMul ----
def _predict_analogy_add(a: np.ndarray, b: np.ndarray, c: np.ndarray, M_norm: np.ndarray) -> np.ndarray:
    # v = b - a + c; normalize for cosine against normalized M
    v = b - a + c
    v /= (np.linalg.norm(v) + 1e-9)
    return _cosine_sim_matrix(v, M_norm)

def _predict_analogy_mul(a: np.ndarray, b: np.ndarray, c: np.ndarray, M_norm: np.ndarray) -> np.ndarray:
    # Levy & Goldberg 2014: 3CosMul ~ maximize cos(x,b)*cos(x,c) / (cos(x,a)+eps)
    # work in log-space to avoid underflow: log cos(x,b) + log cos(x,c) - log(max(eps, cos(x,a)))
    eps = 1e-9
    sb = _cosine_sim_matrix(b, M_norm) + eps
    sc = _cosine_sim_matrix(c, M_norm) + eps
    sa = _cosine_sim_matrix(a, M_norm) + eps
    score = np.log(sb) + np.log(sc) - np.log(sa)
    return score  # higher is better

# ---- public API ----
def evaluate_embeddings(
    word_vectors: Dict[str, np.ndarray],
    similarity_pairs: List[Tuple[str, str, float]],
    analogy_quads: List[Tuple[str, str, str, str]],
    topk: int = 1,
    lowercase: bool = True,
    save_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    word_vectors: dict word -> vector (same dim)
    similarity_pairs: [(w1, w2, gold_score in [0,1] or any scale), ...]
    analogy_quads: [(a,b,c,d), ...] meaning a:b :: c:d
    Returns a metrics dict; optionally saves JSON.
    """
    # build matrices
    vocab = sorted(word_vectors.keys())
    if lowercase:
        # map by lowercase if keys are not already lowercase
        # (assume provided dict already aligned; just be permissive on lookup below)
        pass
    W = np.stack([word_vectors[w] for w in vocab], axis=0).astype(np.float32)
    W_norm = _l2_normalize(W)
    word2idx = {w: i for i, w in enumerate(vocab)}
    dim = W.shape[1]

    # ---- similarity: Spearman-style proxy (Pearson on z-scored since scales small) ----
    # We’ll compute cosine(w1,w2) and Pearson correlation with provided gold scores.
    sim_preds, sim_gold = [], []
    for w1, w2, gold in similarity_pairs:
        a = _safe_get(word2idx, w1.lower() if lowercase else w1)
        b = _safe_get(word2idx, w2.lower() if lowercase else w2)
        if a is None or b is None:
            continue
        sim = (W_norm[a] @ W_norm[b]).item()
        sim_preds.append(sim)
        sim_gold.append(float(gold))
    if len(sim_preds) >= 2:
        x = np.asarray(sim_preds)
        y = np.asarray(sim_gold)
        # Pearson
        x = (x - x.mean()) / (x.std() + 1e-9)
        y = (y - y.mean()) / (y.std() + 1e-9)
        similarity_corr = float((x * y).mean())
    else:
        similarity_corr = float("nan")

    # ---- analogies: 3CosAdd & 3CosMul top-1 accuracy ----
    total_add = total_mul = 0
    correct_add = correct_mul = 0

    # optional samples for logging
    sample_add, sample_mul = [], []

    for a_w, b_w, c_w, d_w in analogy_quads:
        a_i = _safe_get(word2idx, a_w.lower() if lowercase else a_w)
        b_i = _safe_get(word2idx, b_w.lower() if lowercase else b_w)
        c_i = _safe_get(word2idx, c_w.lower() if lowercase else c_w)
        d_i = _safe_get(word2idx, d_w.lower() if lowercase else d_w)
        if None in (a_i, b_i, c_i, d_i):
            continue

        a, b, c = W_norm[a_i], W_norm[b_i], W_norm[c_i]

        # 3CosAdd
        score_add = _predict_analogy_add(a, b, c, W_norm)
        pred_add = _rank_and_pick(score_add, forbid=[a_i, b_i, c_i], topn=topk)[0]
        total_add += 1
        ok_add = int(pred_add == d_i)
        correct_add += ok_add
        if len(sample_add) < 10:
            sample_add.append({
                "q": [a_w, b_w, c_w, d_w],
                "pred": vocab[pred_add],
                "ok": bool(ok_add)
            })

        # 3CosMul
        score_mul = _predict_analogy_mul(a, b, c, W_norm)
        pred_mul = _rank_and_pick(score_mul, forbid=[a_i, b_i, c_i], topn=topk)[0]
        total_mul += 1
        ok_mul = int(pred_mul == d_i)
        correct_mul += ok_mul
        if len(sample_mul) < 10:
            sample_mul.append({
                "q": [a_w, b_w, c_w, d_w],
                "pred": vocab[pred_mul],
                "ok": bool(ok_mul)
            })

    analogy_add_acc = float(correct_add) / max(1, total_add)
    analogy_mul_acc = float(correct_mul) / max(1, total_mul)

    out = {
        "similarity_pearson": similarity_corr,
        "analogy_top1": {
            "3CosAdd": analogy_add_acc,
            "3CosMul": analogy_mul_acc
        },
        "counts": {
            "similarity_evaluated": len(sim_preds),
            "analogies_evaluated": int(max(total_add, total_mul))
        },
        "samples": {
            "analogy_add": sample_add,
            "analogy_mul": sample_mul
        },
        "dim": int(dim),
        "vocab_size": int(len(vocab))
    }

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(out, f, indent=2)

    return out
