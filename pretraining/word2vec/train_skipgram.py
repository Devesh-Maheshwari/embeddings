# pretraining/word2vec/train_skipgram.py
import argparse, json, yaml
from pathlib import Path
from typing import List, Union

from pretraining.utils.benchmark_datasets import EmbeddingBenchmarkDatasets
from pretraining.utils.tokenizer import simple_tokenizer
from pretraining.word2vec.skipgram import SkipGramSGNS
import numpy as np
from pretraining.utils.evaluation.eval_intrinsic import evaluate_embeddings




# --- normalize inputs to List[List[str]] (works for text8 and wikitext2) ---
def _normalize_sentences(raw: Union[List[str], List[List[str]]], chunk_size: int = 256) -> List[List[str]]:
    if not raw:
        return []
    # if already token lists (text8)
    if isinstance(raw[0], list):
        return raw  # assume tokens already
    # else raw is list[str] documents/lines → tokenize + chunk
    sents: List[List[str]] = []
    for doc in raw:
        toks = simple_tokenizer(doc)
        for i in range(0, len(toks), chunk_size):
            seg = toks[i:i+chunk_size]
            if len(seg) >= 4:
                sents.append(seg)
    return sents

def main(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===== datasets =====
    loader = EmbeddingBenchmarkDatasets(data_dir="datasets")
    ds_name = cfg["datasets"]["pretraining"]["name"].lower()
    print(f"Pretraining dataset: {ds_name}")
    if ds_name == "text8":
        raw_pretrain = loader.load_text8(chunk_size=256)
    elif ds_name in ("wikitext2", "wikitext-2", "wikitext"):
        raw_pretrain = loader.load_wikitext2()
    else:
        raise ValueError(f"Unknown pretraining dataset name: {ds_name}")

    sentences = _normalize_sentences(raw_pretrain, chunk_size=256)
    print(f"Number of sentences for pretraining: {len(sentences)} for dataset {ds_name}")

    # ===== hyperparams =====
    w2v_cfg = cfg["embeddings"]["word2vec_skipgram"]
    model = SkipGramSGNS(
        vector_size=int(w2v_cfg.get("vector_size", 300)),
        window=int(w2v_cfg.get("window", 5)),
        min_count=int(w2v_cfg.get("min_count", 5)),
        negative=int(w2v_cfg.get("negative", 5)),
        subsample_t=float(w2v_cfg.get("subsample_t", 1e-3)),
        epochs=int(w2v_cfg.get("epochs", 5)),
        lr=float(w2v_cfg.get("learning_rate", 0.015)),
        seed=42,
    )

    model.build_vocab(sentences)
    model.train(sentences, batch_size=1024)

    # save + quick demo nearest-neighbors (optional small sanity print)
    model_dir = out_dir / "skipgram"
    model.save(model_dir)

    # Export vectors (default W_in)
    word_vecs = model.export_word_vectors(which="combined")
    loader = EmbeddingBenchmarkDatasets(data_dir="datasets")
    analogy_pairs = loader.get_word_analogy_pairs()
    similarity_pairs = loader.get_word_similarity_pairs()

    # convert our (word -> vector) dict form from your export_word_vectors()
    # NOTE: this returns dict[str, np.ndarray]
    intrinsic_results = evaluate_embeddings(
        word_vectors=word_vecs,
        similarity_pairs=[(*p, 0.0) if len(p)==2 else p for p in similarity_pairs],  # our similarity had gold score included already in your new updated pairs
        analogy_quads=[tuple(a) for a in analogy_pairs],
        topk=1,
        lowercase=True
    )

    # save
    intrinsic_path = model_dir / "intrinsic_eval.json"
    with open(intrinsic_path, "w") as f:
        json.dump(intrinsic_results, f, indent=2)
    print("[SkipGram] Intrinsic Eval:", intrinsic_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="pretraining/config.yaml")
    args = parser.parse_args()
    main(args.config)
