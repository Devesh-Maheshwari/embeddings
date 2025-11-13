# Co-occurrence matrix calculation for GloVe
# Your implementation goes here

from __future__ import annotations
import pickle
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
import numpy as np

from pretraining.utils.benchmark_datasets import EmbeddingBenchmarkDatasets


WINDOW_SIZE = 10
MIN_COUNT = 5
SAVE_PATH = Path("datasets/processed/glove_preproc.pkl")


def build_cooccurrence():
    loader = EmbeddingBenchmarkDatasets()
    sentences = loader.load_text8()   # returns list[list[str]]

    print(f"[text8] loaded sentences = {len(sentences):,}")

    vocab_counter = Counter()
    for toks in sentences:
        vocab_counter.update(toks)

    vocab_counter = Counter({w:c for w,c in vocab_counter.items() if c >= MIN_COUNT})
    vocab_items = sorted(vocab_counter.items(), key=lambda x: x[1], reverse=True)
    final_word2id = {w:i for i,(w,_) in enumerate(vocab_items)}
    final_id2word = [w for w,_ in vocab_items]

    print(f"[vocab] after MIN_COUNT={MIN_COUNT} -> {len(vocab_counter):,}")

    # initial indexing (temporary)
    tmp_id2word = list(vocab_counter.keys())
    tmp_word2id = {w:i for i,w in enumerate(tmp_id2word)}

    cooccur = defaultdict(float)

    for toks in tqdm(sentences, total=len(sentences), desc="cooccur"):
        ids = [tmp_word2id[w] for w in toks if w in tmp_word2id]
        for center_pos, center in enumerate(ids):
            left = max(0, center_pos - WINDOW_SIZE)
            right = min(len(ids), center_pos + WINDOW_SIZE + 1)
            for pos in range(left, right):
                if pos == center_pos: continue
                dist = abs(pos - center_pos)
                if dist <= WINDOW_SIZE:
                    cooccur[(center, ids[pos])] += 1.0 / dist


    payload = {
        "word2id": final_word2id,
        "id2word": final_id2word,
        "cooccur": cooccur
    }

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SAVE_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"[SAVE] glove_preproc.pkl written -> {SAVE_PATH}")
    print(f"[STATS] vocab={len(final_id2word):,} | cooccur entries={len(new_co):,}")


if __name__ == "__main__":
    build_cooccurrence()
