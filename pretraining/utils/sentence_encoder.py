# pretraining/utils/sentence_encoder.py
from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Tuple, Union
import json
import numpy as np
import re
import torch

# ---------- tokenization ----------
# Reuse the same token rule your TF-IDF used (fallback if not importable)
try:
    from ..tfidf.tfidf_vectorizer import _TOKEN_RE  # compiled regex
except Exception:
    # alnum+apostrophe tokens
    _TOKEN_RE = re.compile(r"[a-z0-9']+")

def simple_tokenize(text: str, lowercase: bool = True) -> List[str]:
    if lowercase:
        text = text.lower()
    return _TOKEN_RE.findall(text)

# ---------- device helpers ----------
def pick_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ---------- main encoder ----------
class SentenceEncoder:
    """
    Unified sentence encoder supporting:
      - Static embeddings: average pooling
      - Static embeddings: TF-IDF weighted pooling (Option B)
      - BERT [CLS] embeddings (optional, GPU/MPS-aware)

    Static embeddings:
      word_vectors: Dict[str, np.ndarray]  (required for avg/tfidf)
      vector_dim: inferred from any vector in word_vectors

    TF-IDF weighting:
      Provide either:
        - idf_array: np.ndarray of shape (V,), and tfidf_vocab: Dict[str, int] (term -> index)
          (this matches your text8 TF-IDF artifacts: tfidf.idf.npy + tfidf.vocab.json)
        - OR idf_map: Dict[str, float]
      For words missing in TF-IDF resources, weight defaults to 1.0.

    BERT:
      Provide bert_model and bert_tokenizer (HuggingFace), and device="auto"/"mps"/"cuda"/"cpu".
      Uses the [CLS] embedding from the last hidden state, with no gradients.
    """

    def __init__(
        self,
        word_vectors: Optional[Dict[str, np.ndarray]] = None,
        idf_array: Optional[np.ndarray] = None,
        tfidf_vocab: Optional[Dict[str, int]] = None,
        idf_map: Optional[Dict[str, float]] = None,
        bert_model: Optional[object] = None,
        bert_tokenizer: Optional[object] = None,
        device: str = "auto",
        lowercase: bool = True,
    ):
        self.word_vectors = word_vectors or {}
        self.lowercase = lowercase

        # infer dimension from static vectors if available
        self.vector_dim: Optional[int] = None
        if self.word_vectors:
            any_vec = next(iter(self.word_vectors.values()))
            self.vector_dim = int(any_vec.shape[0])

        # TF-IDF pieces (either array+vocab or dict map)
        self.idf_array = idf_array
        self.tfidf_vocab = tfidf_vocab
        self.idf_map = idf_map

        # BERT parts
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.device = pick_device(device)

        if self.bert_model is not None:
            self.bert_model.to(self.device)
            self.bert_model.eval()

        # Normalize static vectors once for cosine-friendly behavior
        if self.word_vectors:
            for k, v in self.word_vectors.items():
                n = np.linalg.norm(v)
                if n > 0:
                    self.word_vectors[k] = (v / n).astype(np.float32)

    # --------- static encoders ---------
    def encode_sentence_avg(self, text: str) -> np.ndarray:
        """
        Average pooling of static word vectors. Returns zero vector if nothing matches.
        """
        if not self.word_vectors:
            raise ValueError("word_vectors is required for avg pooling.")
        tokens = simple_tokenize(text, self.lowercase)
        if not tokens:
            return np.zeros(self.vector_dim, dtype=np.float32)

        acc = np.zeros(self.vector_dim, dtype=np.float32)
        count = 0
        for t in tokens:
            vec = self.word_vectors.get(t)
            if vec is not None:
                acc += vec
                count += 1
        if count == 0:
            return np.zeros(self.vector_dim, dtype=np.float32)
        return acc / float(count)

    def _idf_weight(self, token: str) -> float:
        # prefer array+vocab (fast), else map, else 1.0
        if self.idf_array is not None and self.tfidf_vocab is not None:
            idx = self.tfidf_vocab.get(token)
            if idx is not None and 0 <= idx < len(self.idf_array):
                return float(self.idf_array[idx])
            return 1.0
        if self.idf_map is not None:
            return float(self.idf_map.get(token, 1.0))
        return 1.0

    def encode_sentence_tfidf(self, text: str, smooth: float = 1e-12) -> np.ndarray:
        """
        TF-IDF weighted pooling over static embeddings.
        Uses global IDF (from text8 in your case) to reweight any static vectors.
        """
        if not self.word_vectors:
            raise ValueError("word_vectors is required for tfidf pooling.")
        tokens = simple_tokenize(text, self.lowercase)
        if not tokens:
            return np.zeros(self.vector_dim, dtype=np.float32)

        tf: Dict[str, int] = {}
        for t in tokens:
            if t in self.word_vectors:
                tf[t] = tf.get(t, 0) + 1

        if not tf:
            return np.zeros(self.vector_dim, dtype=np.float32)

        acc = np.zeros(self.vector_dim, dtype=np.float32)
        wsum = 0.0
        for t, f in tf.items():
            w = float(f) * self._idf_weight(t)
            vec = self.word_vectors.get(t)
            if vec is None:
                continue
            acc += w * vec
            wsum += w
        if wsum == 0.0:
            return np.zeros(self.vector_dim, dtype=np.float32)
        return acc / (wsum + smooth)

    def batch_encode_avg(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        out = np.zeros((len(texts), self.vector_dim), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i] = self.encode_sentence_avg(t)
        return out

    def batch_encode_tfidf(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        out = np.zeros((len(texts), self.vector_dim), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i] = self.encode_sentence_tfidf(t)
        return out

    # --------- BERT encoders ---------
    @torch.inference_mode()
    def encode_sentence_bert(self, text: str, max_length: int = 512) -> np.ndarray:
        if self.bert_model is None or self.bert_tokenizer is None:
            raise ValueError("BERT model/tokenizer not provided.")
        enc = self.bert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.bert_model(**enc)
        # last hidden state, CLS = position 0: shape (1, seq_len, hidden)
        cls = out.last_hidden_state[:, 0, :]  # (1, H)
        v = cls[0].detach().cpu().numpy()
        # L2 normalize for cosine stability
        n = np.linalg.norm(v)
        return (v / (n + 1e-9)).astype(np.float32)

    @torch.inference_mode()
    def batch_encode_bert(self, texts: Iterable[str], max_length: int = 512, batch_size: int = 16) -> np.ndarray:
        if self.bert_model is None or self.bert_tokenizer is None:
            raise ValueError("BERT model/tokenizer not provided.")
        texts = list(texts)
        reps: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = self.bert_tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.bert_model(**enc)
            cls = out.last_hidden_state[:, 0, :]               # (B, H)
            v = cls.detach().cpu().numpy()
            # L2 normalize row-wise
            v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
            reps.append(v.astype(np.float32))
        return np.concatenate(reps, axis=0)

# ---------- helpers to load TF-IDF artifacts ----------
def load_tfidf_artifacts(idf_npy_path: str, vocab_json_path: str) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Load IDF array and vocab map (term -> index) as produced by your TF-IDF trainer.
    """
    idf = np.load(idf_npy_path)
    with open(vocab_json_path, "r") as f:
        vocab = json.load(f)  # token -> index
    # defensive: ensure indices are ints
    vocab = {k: int(v) for k, v in vocab.items()}
    return idf, vocab
# ---------- end of file ----------


# alias mapping for downstream evaluator compatibility
def average_pool(text: str, encoder: "SentenceEncoder") -> np.ndarray:
    return encoder.encode_sentence_avg(text)

def tfidf_weighted_average(text: str, encoder: "SentenceEncoder") -> np.ndarray:
    return encoder.encode_sentence_tfidf(text)

def batch_average_pool(texts: Iterable[str], encoder: "SentenceEncoder") -> np.ndarray:
    return encoder.batch_encode_avg(texts)

def batch_tfidf_weighted_average(texts: Iterable[str], encoder: "SentenceEncoder") -> np.ndarray:
    return encoder.batch_encode_tfidf(texts)
