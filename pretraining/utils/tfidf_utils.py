from __future__ import annotations
from typing import List, Dict, Iterable
import numpy as np
from ..tfidf.tfidf_vectorizer import TfidfVectorizer

def idf_dict_from_vectorizer(vectorizer: TfidfVectorizer) -> Dict[str, float]:
    """
    Extract IDF values from a fitted TfidfVectorizer into a dictionary.
    
    Args:
        vectorizer (TfidfVectorizer): A fitted TfidfVectorizer instance.

    Returns:
        Dict[str, float]: A dictionary mapping terms to their IDF values.
    """
    assert vectorizer.vocabulary_ is not None and vectorizer.idf_ is not None and vectorizer.feature_names_ is not None
    idf_map = {}
    for term, idx in vectorizer.vocabulary_.items():
        idf_map[term] = float(vectorizer.idf_[idx])
    return idf_map