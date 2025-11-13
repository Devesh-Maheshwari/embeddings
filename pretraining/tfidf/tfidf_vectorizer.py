import re 
import json
from collections import defaultdict, Counter
from math import log
from typing import Tuple
from typing import List, Tuple, Optional,Dict, Iterable
import numpy as np
from scipy.sparse import csr_matrix

_TOKEN_RE = re.compile(r"\b\w\w+\b", flags=re.UNICODE)

def _generate_ngrams(tokens: List[str], ngram_range: Tuple[int, int]) -> List[str]:
    min_n, max_n = ngram_range
    ngrams = []
    for n in range(min_n, max_n + 1):
        if n == 1:
            ngrams.extend(tokens)
        else:           
            ngrams.extend([' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)])
    return ngrams

class TfidfVectorizer:
    def __init__(
            self,
            max_features: Optional[int] = None,
            min_df: int | float = 1,
            max_df: int | float = 1.0,
            ngram_range: Tuple[int, int] = (1, 1),
            smooth_idf: bool = True,
            sublinear_tf: bool = False,
            norm: Optional[str] = 'l2',
            stop_words: Optional[List[str]] = None,
            lowercase: bool = True,
            binary: bool = False
    ):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.stop_words = stop_words
        self.lowercase = lowercase
        self.binary = binary
    
        self.vocabulary_: Optional[Dict[str, int]] = None
        self.idf_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None
        self._n_docs: int = 0
    
    def _tokenize(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()
        tokens = _TOKEN_RE.findall(text)
        if self.stop_words:
            tokens = [t for t in tokens if t not in self.stop_words]
        return tokens

    def _iter_ngram_docs(self, docs: Iterable[str]) -> List[List[str]]:
        out=[]
        for doc in docs:
            tokens = self._tokenize(doc)
            ngrams = _generate_ngrams(tokens, self.ngram_range)
            out.append(ngrams)
        return out
    
    def _df_prune_and_build(self, docs_ngram: Iterable[str]) -> List[List[str]]:
        df_counter = Counter()
        for ngrams in docs_ngram:
            unique_ngrams = set(ngrams)
            df_counter.update(unique_ngrams)
        
        n_docs = len(docs_ngram)
        pruned_ngrams = set()
        min_df_abs = int(self.min_df*n_docs) if isinstance(self.min_df, float) else int(self.min_df)
        max_df_abs = int(self.max_df*n_docs) if isinstance(self.max_df, float) else int(self.max_df if self.max_df >=1 else self.max_df*n_docs)

        items = [(ngram, df) for ngram, df in df_counter.items() if df >= min_df_abs and df <= max_df_abs]
        items.sort(key=lambda x: x[1], reverse=True)
        if self.max_features:
            items = items[:self.max_features]

        vocabulary = {ngram: idx for idx, (ngram, _) in enumerate(items)}
        return vocabulary
    
    def fit(self, raw_documents: Iterable[str]):
        docs_ngrams = self._iter_ngram_docs(list(raw_documents))
        self.vocabulary_ = self._df_prune_and_build(docs_ngrams )
        self.feature_names_ = [None]*len(self.vocabulary_)
        for ngram, idx in self.vocabulary_.items():
            self.feature_names_[idx] = ngram
        
        self._n_docs = len(docs_ngrams)
        # compute document frequency for terms in vocab
        df = np.zeros(len(self.vocabulary_), dtype=np.int32)
        for ngrams in docs_ngrams:
            seen = set()
            for term in ngrams:
                j = self.vocabulary_.get(term)
                if j is not None and j not in seen:
                    df[j] += 1
                    seen.add(j)

        # IDF
        if self.smooth_idf:
            idf = np.log((1.0 + self._n_docs) / (1.0 + df)) + 1.0
        else:
            idf = np.log(self._n_docs / np.maximum(df, 1)) + 1.0
        self.idf_ = idf.astype(np.float32)
        return self
    
    def transform(self, raw_documents: Iterable[str]) -> csr_matrix:
        assert self.vocabulary_ is not None and self.idf_ is not None
        docs_ngrams = self._iter_ngram_docs(list(raw_documents))
        indexptr = [0]
        indices = []
        data = []   
        V=len(self.vocabulary_)
        for terms in docs_ngrams:
            counts =defaultdict(int)
            for term in terms:
                idx = self.vocabulary_.get(term)
                if idx is not None:
                    counts[idx] += 1
            if self.binary:
                for j in counts.keys():
                    counts[j] = 1

            if self.sublinear_tf:
                row_data = [ 1 + log(v) for v in counts.values()]
            else:
                row_data = list(map(float, counts.values()))


            indices.extend(counts.keys())
            data.extend(row_data)
            indexptr.append(len(indices))

        X = csr_matrix((np.array(data,dtype=np.float32), np.array(indices, dtype=np.int32),
                         np.array(indexptr, dtype=np.int32)),
                       shape=(len(docs_ngrams), V), dtype=np.float32)

        # Apply IDF scaling
        X = X.multiply(self.idf_)

        # Apply normalization
        if self.norm is None:
            return X
        if self.norm not in ('l1', 'l2'):
            raise ValueError("Unsupported norm type. Use 'l1', 'l2', or None.")
        
        if self.norm == "l2":
            row_sq_sum = np.asarray(X.power(2).sum(axis=1)).ravel()
            denom = np.sqrt(np.maximum(row_sq_sum, 1e-12))[:, None]
        else:  # l1
            row_abs_sum = np.asarray(np.abs(X).sum(axis=1)).ravel()
            denom = np.maximum(row_abs_sum, 1e-12)[:, None]
        X = X.multiply(1.0 / denom)
        return X

    def fit_transform(self, raw_documents: Iterable[str]) -> csr_matrix:
        self.fit(raw_documents)
        return self.transform(raw_documents)

    def save(self, filepath: str):
        assert self.vocabulary_ is not None and self.idf_ is not None
        with open(f"{filepath}.vocab.json", 'w', encoding='utf-8') as f:
            json.dump(self.vocabulary_, f, ensure_ascii=False, indent=4)
        np.save(f"{filepath}.idf.npy", self.idf_)
        meta = {
            "max_features": self.max_features,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "ngram_range": self.ngram_range,
            "smooth_idf": self.smooth_idf,
            "sublinear_tf": self.sublinear_tf,
            "norm": self.norm,
            "stop_words": self.stop_words,
            "lowercase": self.lowercase,
            "binary": self.binary
        }
        with open(f"{filepath}.meta.json", 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=4)
    @classmethod
    def load(cls, filepath: str) -> 'TfidfVectorizer':
        with open(f"{filepath}.vocab.json", 'r', encoding='utf-8') as f:
            vocabulary = json.load(f)
        idf = np.load(f"{filepath}.idf.npy")
        with open(f"{filepath}.meta.json", 'r', encoding='utf-8') as f:
            meta = json.load(f)

        vec = cls(**meta)
        vec.vocabulary_ = {k:int(v) for k,v in vocabulary.items()}
        vec.feature_names_ = [None]*len(vec.vocabulary_)
        for t,i in vec.vocabulary_.items():
            vec.feature_names_[i] = t 
        vec.idf_ = idf
        return vec