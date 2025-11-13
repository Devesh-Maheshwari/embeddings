# Similarity demo utilities
# Your implementation goes here

import json
from pathlib import Path
import numpy as np
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data(show_spinner=False)
def load_tfidf(ROOT: Path):
    idf_path = ROOT / "results" / "tfidf.idf.npy"
    vocab_path = ROOT / "results" / "tfidf.vocab.json"
    if idf_path.exists() and vocab_path.exists():
        idf = np.load(idf_path)
        vocab = json.loads(vocab_path.read_text())
        token2id = vocab if isinstance(vocab, dict) else {t:i for i,t in enumerate(vocab)}
        return idf, token2id
    return None


@st.cache_data(show_spinner=False)
def load_vectors(vec_path: Path, vocab_path: Path):
    arr = np.load(vec_path)
    key = list(arr.files)[0]
    E = arr[key]
    vocab = json.loads(vocab_path.read_text())
    if isinstance(vocab, dict) and 'itos' in vocab:
        itos = vocab['itos']; token2id = {t:i for i,t in enumerate(itos)}
    elif isinstance(vocab, dict):
        token2id = vocab
    else:
        token2id = {t:i for i,t in enumerate(vocab)}
    return E, token2id

def cosine(a, b):
    den = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / den) if den else 0.0

def topk_neighbors(query, E, V, k=10):
    if query not in V: return []
    q = E[V[query]]
    sims = E @ q / (np.linalg.norm(E, axis=1) * (np.linalg.norm(q) or 1))
    order = np.argsort(-sims)[:k+1]
    id2tok = {i:t for t,i in V.items()}
    out = []
    for i in order:
        t = id2tok.get(i)
        if t == query: continue
        out.append((t, float(sims[i])))
        if len(out) >= k: break
    return out

def sentence_embed_avg(sents, E, V, pool="avg"):
    id2tok = {i: t for t, i in V.items()}
    tok2vec = {t: E[V[t]] for t in V}
    mats = []
    for s in sents:
        words = [w for w in s.lower().split() if w in tok2vec]
        if not words:
            mats.append(np.zeros(E.shape[1]))
            continue
        mats.append(np.mean([tok2vec[w] for w in words], axis=0))
    return np.vstack(mats)

def sentence_embed_tfidf(sents, vocab, idf):
    mats = []
    for s in sents:
        words = [w for w in s.lower().split() if w in vocab]
        vec = np.zeros(len(vocab))
        for w in words:
            vec[vocab[w]] += idf[vocab[w]]
        mats.append(vec)
    return np.vstack(mats)

def cosine_matrix(mat):
    return cosine_similarity(mat)
