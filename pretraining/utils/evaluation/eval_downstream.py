# pretraining/utils/evaluation/eval_downstream.py
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pretraining.utils.sentence_encoder import SentenceEncoder,simple_tokenize
# --- local imports (package-friendly) ---
# If executed via -m, these work; if from a notebook, ensure project root is on sys.path.
from pretraining.utils.benchmark_datasets import EmbeddingBenchmarkDatasets
from pretraining.word2vec.base_sgns import BaseSGNS  # for loading saved vectors
from sklearn.feature_extraction.text import TfidfVectorizer


# Torch is optional (only for the MLP)
try:
    import torch
    import torch.nn as nn
    import torch.utils.data as tdata
    TORCH_OK = True
except Exception:
    TORCH_OK = False


# --------------------------
# IDF on IMDB train (aligned)
# --------------------------
def compute_idf_on_corpus(texts):
    vec = TfidfVectorizer(min_df=5, max_df=0.95, token_pattern=r"[a-z0-9']+")
    vec.fit(texts)
    return vec.idf_.astype(np.float32), vec.vocabulary_



# --------------------------
# Helpers: load word vectors
# --------------------------
def _load_word_vectors_from_dir(vectors_dir: Path, which: str = "combined") -> Dict[str, np.ndarray]:
    """
    Load vectors saved by BaseSGNS.save(...) OR any (vocab.json + vectors.npz) pair.
    which ∈ {"in","out","combined"}; for glove/fastText you likely only have a single matrix -> use "in".
    """
    vectors_dir = Path(vectors_dir)
    vocab_path = vectors_dir / "vocab.json"
    vec_path = vectors_dir / "vectors.npz"
    if not vocab_path.exists() or not vec_path.exists():
        raise FileNotFoundError(f"Expected {vocab_path} and {vec_path}")

    with open(vocab_path, "r", encoding="utf-8") as f:
        word2id = {k: int(v) for k, v in json.load(f).items()}
    arrs = np.load(vec_path)
    if which == "in":
        M = arrs["W_in"]
    elif which == "out":
        M = arrs["W_out"]
    else:
        # combined by default if available
        if "W_in" in arrs.files and "W_out" in arrs.files:
            M = (arrs["W_in"] + arrs["W_out"]) / 2.0
        elif "W" in arrs.files:
            M = arrs["W"]
        else:
            # fallback
            M = arrs[arrs.files[0]]

    # L2 normalize rows for stable cosine geometry
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    id2word = [None] * len(word2id)
    for w, i in word2id.items():
        id2word[i] = w
    return {w: M[i] for i, w in enumerate(id2word)}


# --------------------------
# Encoders
# --------------------------
def build_static_encoder(
    encoder_name: str,
    results_dir: Path,
    pooling: str,
    vectors_choice: str = "combined",
) -> Tuple[Dict[str, np.ndarray], str]:
    """
    Load static word vectors for: skipgram / cbow / fasttext / glove
    Return (word_vectors_dict, pooling_mode)
    """
    encoder_name = encoder_name.lower()
    if encoder_name == "skipgram":
        vec_dir = results_dir / "skipgram"
    elif encoder_name == "cbow":
        vec_dir = results_dir / "cbow"
    elif encoder_name == "fasttext":
        vec_dir = results_dir / "fasttext"
    elif encoder_name == "glove":
        vec_dir = results_dir / "glove"
    else:
        raise ValueError(f"Unknown static encoder: {encoder_name}")

    wv = _load_word_vectors_from_dir(vec_dir, which=vectors_choice)
    return wv, pooling

def encode_texts_static(
    texts: List[str],
    wv: Dict[str, np.ndarray],
    idf: Optional[Dict[str, float]],
    pooling: str = "avg",
) -> np.ndarray:
    if isinstance(idf, tuple):
        idf_array, tfidf_vocab = idf
        encoder = SentenceEncoder(word_vectors=wv, idf_array=idf_array, tfidf_vocab=tfidf_vocab)
    else:
        encoder = SentenceEncoder(word_vectors=wv, idf_map=idf)
    if pooling == "avg":
        return encoder.batch_encode_avg(texts)
    elif pooling == "tfidf":
        return encoder.batch_encode_tfidf(texts)
    else:
        raise ValueError(f"Unknown pooling {pooling}")

# --------------------------
# Torch MLP Classifier
# --------------------------
class TextDataset(tdata.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2),   # second projection layer
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, 2),
        )
    def forward(self, x):
        return self.net(x)

def torch_device_from_config(hw_cfg: dict) -> torch.device:
    # mps > cuda > cpu (if available)
    want = (hw_cfg or {}).get("device", "mps").lower()
    if want == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if want == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def fit_mlp_classifier(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    hw_cfg: dict, epochs: int = 10, lr: float = 1e-3, batch_size: int = 256,
) -> Tuple[np.ndarray, Dict[str, float]]:
    device = torch_device_from_config(hw_cfg)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    train_ds = TextDataset(X_train, y_train)
    val_ds   = TextDataset(X_val, y_val)
    train_loader = tdata.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = tdata.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MLP(X_train.shape[1], hidden=256, dropout=0.1).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr)
    crit  = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

        # quick val
        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                all_pred.append(pred)
                all_true.append(yb.numpy())
        y_pred = np.concatenate(all_pred)
        y_true = np.concatenate(all_true)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_val_acc:
            best_val_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[MLP] epoch {ep} val_acc={acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # return a callable like predict_proba
    def predict_proba(X: np.ndarray) -> np.ndarray:
        model.eval()
        device_local = device
        logits_all = []
        X = scaler.transform(X)
        with torch.no_grad():
            for i in range(0, len(X), 1024):
                xb = torch.from_numpy(X[i:i+1024].astype(np.float32)).to(device_local)
                logits = model(xb).cpu().numpy()
                logits_all.append(logits)
        logits_all = np.vstack(logits_all)
        # softmax
        e = np.exp(logits_all - logits_all.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        return p

    return predict_proba, {"val_acc": float(best_val_acc)}

# --------------------------
# Evaluation pipeline
# --------------------------
def evaluate_downstream(
    cfg_path: str,
    encoder: str,
    classifier: str,
    pooling: str = "avg",
    vectors_choice: str = "combined",
    results_dir: str = "results",
    save_name: Optional[str] = None,
) -> Dict[str, float]:
    """
    encoder: tfidf | skipgram | cbow | glove | fasttext
    classifier: logistic | mlp
    pooling: avg | tfidf   (only relevant for static encoders)
    """
    # 1) load IMDB
    loader = EmbeddingBenchmarkDatasets(data_dir="datasets")
    imdb = loader.load_imdb_real()  # cached pickle if present
    X_train_texts = imdb["train_texts"]
    y_train = np.array(imdb["train_labels"])
    X_test_texts  = imdb["test_texts"]
    y_test = np.array(imdb["test_labels"])

    # simple split train → (train,val)
    n = len(X_train_texts)
    val_cut = int(0.9 * n)
    train_texts, val_texts = X_train_texts[:val_cut], X_train_texts[val_cut:]
    y_tr, y_val = y_train[:val_cut], y_train[val_cut:]

    # 2) build encodings
    encoder = encoder.lower()
    classifier = classifier.lower()
    pooling = pooling.lower()
    results_dir = Path(results_dir)

    if encoder == "tfidf":
        # Use TF-IDF vectorizer as sentence embedding (bag-of-ngrams).
        # We reuse your existing TF-IDF pipeline from pretraining/tfidf if desired.
        # Here, for simplicity, create a fresh sklearn TF-IDF to keep the file self-contained.
        vec = TfidfVectorizer(
            max_features=10000, ngram_range=(1,2), lowercase=True, min_df=5, max_df=0.95
        )
        X_tr = vec.fit_transform(train_texts)
        X_val = vec.transform(val_texts)
        X_te = vec.transform(X_test_texts)

        # To keep shapes dense for MLP, we densify; for logistic, sparse is fine
        is_sparse = True

    elif encoder in {"skipgram", "cbow", "glove", "fasttext"}:
        # Load static word vectors saved in results/<encoder>/{vocab.json,vectors.npz}
        wv, _ = build_static_encoder(encoder, results_dir, pooling, vectors_choice=vectors_choice)

        # IDF on IMDB train (aligned!)
        if pooling == "tfidf":
            idf_array, vocab_map = compute_idf_on_corpus(train_texts)
            idf = (idf_array, vocab_map)
            X_tr = encode_texts_static(train_texts, wv, idf, pooling="tfidf")
            X_val = encode_texts_static(val_texts,   wv, idf, pooling="tfidf")
            X_te = encode_texts_static(X_test_texts, wv, idf, pooling="tfidf")
        else:
            X_tr = encode_texts_static(train_texts, wv, None, pooling="avg")
            X_val = encode_texts_static(val_texts,   wv, None, pooling="avg")
            X_te = encode_texts_static(X_test_texts, wv, None, pooling="avg")

        is_sparse = False

    else:
        raise ValueError(f"Unknown encoder: {encoder}")

    # 3) fit classifier
    if classifier == "logistic":
        if is_sparse:
            # sparse ok; add a standard scaler? Not necessary for sparse TF-IDF + liblinear
            clf = LogisticRegression(
                max_iter=2000, n_jobs=-1, solver="liblinear"
            )
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
        else:
            # dense: scale then LR
            pipe = Pipeline([
                ("scale", StandardScaler(with_mean=True, with_std=True)),
                ("lr", LogisticRegression(max_iter=2000, n_jobs=-1, solver="lbfgs"))
            ])
            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict(X_te)

    elif classifier == "mlp":
        if not TORCH_OK:
            raise RuntimeError("PyTorch not available; cannot run MLP classifier.")
        if is_sparse:
            # densify for MLP (memory okay for 25k examples * 10k feats ~ fits on CPU; chunk if needed)
            X_tr_d = X_tr.astype(np.float32).toarray()
            X_val_d = X_val.astype(np.float32).toarray()
            X_te_d  = X_te.astype(np.float32).toarray()
        else:
            X_tr_d, X_val_d, X_te_d = X_tr, X_val, X_te

        # small training
        # pull hardware cfg if present
        hw_cfg = {}
        # Fit MLP with quick early stop via val_acc tracking
        predict_proba, hist = fit_mlp_classifier(
            X_tr_d, y_tr, X_val_d, y_val, hw_cfg=hw_cfg,
            epochs=8, lr=1e-3, batch_size=256
        )
        proba = predict_proba(X_te_d)
        y_pred = np.argmax(proba, axis=1)

    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    # 4) metrics
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)

    metrics = {
        "encoder": encoder,
        "pooling": pooling,
        "classifier": classifier,
        "test_accuracy": float(acc),
        "test_f1": float(f1),
        "test_precision": float(prec),
        "test_recall": float(rec),
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
    }

    # 5) save
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = save_name or f"downstream_{encoder}_{pooling}_{classifier}.json"
    with open(out_dir / tag, "w") as f:
        json.dump(metrics, f, indent=2)
    print("[downstream] ", json.dumps(metrics, indent=2))
    print(f"[downstream] saved -> {out_dir / tag}")
    return metrics

# --------------------------
# CLI
# --------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="pretraining/config.yaml", help="(kept for symmetry; dataset loader uses its own caching)")
    ap.add_argument("--encoder", type=str, required=True,
                    choices=["tfidf","skipgram","cbow","glove","fasttext"])
    ap.add_argument("--classifier", type=str, required=True,
                    choices=["logistic","mlp"])
    ap.add_argument("--pooling", type=str, default="avg", choices=["avg","tfidf"],
                    help="Pooling for static encoders; ignored for pure tfidf encoder.")
    ap.add_argument("--vectors_choice", type=str, default="combined", choices=["in","out","combined"])
    ap.add_argument("--results_dir", type=str, default="results")
    ap.add_argument("--save_name", type=str, default=None)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluate_downstream(
        cfg_path=args.config,
        encoder=args.encoder,
        classifier=args.classifier,
        pooling=args.pooling,
        vectors_choice=args.vectors_choice,
        results_dir=args.results_dir,
        save_name=args.save_name,
    )

