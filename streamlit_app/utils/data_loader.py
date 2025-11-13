"""
Centralized data loading utilities
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import streamlit as st

@st.cache_data(show_spinner=False)
def load_model_vectors(model_name: str, results_dir: Path) -> Optional[Tuple[np.ndarray, Dict[str, int]]]:
    """
    Load embeddings and vocabulary for a model.

    Returns:
        (embeddings, word2id) or None if not found
    """
    model_dir = results_dir / model_name
    vec_path = model_dir / "vectors.npz"
    vocab_path = model_dir / "vocab.json"

    if not vec_path.exists() or not vocab_path.exists():
        return None

    try:
        # Load embeddings
        arr = np.load(vec_path)
        key = list(arr.files)[0]
        embeddings = arr[key]

        # Load vocabulary
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)

        # Ensure vocab is dict format
        if isinstance(vocab, dict):
            word2id = {str(k): int(v) for k, v in vocab.items()}
        else:
            word2id = {str(w): i for i, w in enumerate(vocab)}

        return embeddings, word2id
    except Exception as e:
        st.error(f"Error loading {model_name}: {e}")
        return None

@st.cache_data(show_spinner=False)
def load_tfidf_data(results_dir: Path) -> Optional[Tuple[np.ndarray, Dict[str, int]]]:
    """Load TF-IDF IDF values and vocabulary"""
    idf_path = results_dir / "tfidf.idf.npy"
    vocab_path = results_dir / "tfidf.vocab.json"

    if not idf_path.exists() or not vocab_path.exists():
        return None

    try:
        idf = np.load(idf_path)
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)

        # Ensure correct format
        if isinstance(vocab, dict):
            word2id = {str(k): int(v) for k, v in vocab.items()}
        else:
            word2id = {str(w): i for i, w in enumerate(vocab)}

        return idf, word2id
    except Exception as e:
        st.error(f"Error loading TF-IDF: {e}")
        return None

@st.cache_data(show_spinner=False)
def load_intrinsic_results(model_name: str, results_dir: Path) -> Optional[Dict]:
    """Load intrinsic evaluation results (analogies, similarity)"""
    model_dir = results_dir / model_name
    eval_path = model_dir / "intrinsic_eval.json"

    if not eval_path.exists():
        return None

    try:
        with open(eval_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading intrinsic eval for {model_name}: {e}")
        return None

@st.cache_data(show_spinner=False)
def load_downstream_results(results_dir: Path) -> Dict[str, Dict]:
    """
    Load all downstream evaluation results.

    Returns dict with structure:
        {
            "skipgram": {"avg_logistic": {...}, "tfidf_mlp": {...}},
            "cbow": {...},
            ...
        }
    """
    results = {}

    # Pattern: downstream_{model}_{pooling}_{classifier}.json
    for json_file in results_dir.glob("downstream_*.json"):
        try:
            # Parse filename
            parts = json_file.stem.split("_")
            if len(parts) < 4:
                continue

            model = parts[1]  # skipgram, cbow, glove, tfidf
            pooling = parts[2]  # avg, tfidf
            classifier = parts[3]  # logistic, mlp

            # Load data
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Store in nested dict
            if model not in results:
                results[model] = {}

            key = f"{pooling}_{classifier}"
            results[model][key] = data

        except Exception as e:
            st.warning(f"Skipping {json_file.name}: {e}")

    return results

@st.cache_data(show_spinner=False)
def get_all_available_models(results_dir: Path) -> List[str]:
    """Get list of models that have been trained"""
    available = []

    for model in ["skipgram", "cbow", "glove", "tfidf"]:
        if model == "tfidf":
            if load_tfidf_data(results_dir) is not None:
                available.append(model)
        else:
            if load_model_vectors(model, results_dir) is not None:
                available.append(model)

    return available
