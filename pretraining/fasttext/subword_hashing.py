# Subword tokenizer for FastText
# Your implementation goes here
from __future__ import annotations
from typing import List, Iterable
import hashlib

def add_bow(word: str) -> str:
    return f"<{word}>"

def char_ngrams(word: str, min_n: int, max_n: int) -> List[str]:
    w = add_bow(word)
    out: List[str] = []
    L = len(w)
    for n in range(min_n, max_n + 1):
        if n > L: 
            continue
        for i in range(L - n + 1):
            out.append(w[i:i+n])
    return out

def fasttext_hash(s: str, bucket: int) -> int:
    # stable 32-bit hash → bucket
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    # take lower 32 bits
    v = int(h[:8], 16)
    return v % bucket

def ngram_bucket_ids(word: str, min_n: int, max_n: int, bucket: int) -> List[int]:
    grams = char_ngrams(word, min_n, max_n)
    return [fasttext_hash(g, bucket) for g in grams]
