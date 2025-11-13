import pandas as pd
import numpy as np
import re

class TfidfVectorizer:
    def __init__(self, input='content', encoding='utf-8', decode_error='strict',
                norm='l2', smooth_idf=True, sublinear_tf=False):
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.norm = norm
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.vocab = None
        self.vocab_index = None
        self.doc_freq = None
        self.tfidf_matrix = None
        self.idf_ = None
        self._token_re = re.compile(r'\b\w\w+\b', flags=re.UNICODE)



    def _prepare_documents(self, raw_documents):
        documents = []
        if self.input == 'filename':
            for filename in raw_documents:
                with open(filename, encoding=self.encoding, errors=self.decode_error) as f:
                    documents.append(f.read())
        elif self.input == 'file':
            for f in raw_documents:
                documents.append(f.read())
        elif self.input == 'content':
            documents = [doc if isinstance(doc, str) else doc.decode(self.encoding, errors=self.decode_error) for doc in raw_documents]
        else:
            raise ValueError("Invalid input type. Expected 'filename', 'file', or 'content'.")
        return documents

    def _tokenize(self, text):
        text = text.lower()
        return self._token_re.findall(text)

    def fit(self, raw_documents):
        documents = self._prepare_documents(raw_documents)
        tokenized_docs = [self._tokenize(doc) for doc in documents]
        self.vocab = sorted(set(word for doc in tokenized_docs for word in doc))
        self.vocab_index = {word: idx for idx, word in enumerate(self.vocab)}
        # Calculate term frequencies
        term_freq = np.zeros((len(documents), len(self.vocab)), dtype=float)
        for doc_idx, doc in enumerate(tokenized_docs):
            for word in doc:
                if word in self.vocab_index:
                    term_freq[doc_idx, self.vocab_index[word]] += 1
        
        # Calculate document frequencies
        self.doc_freq = np.sum(term_freq > 0, axis=0)
        if self.smooth_idf:
            idf = np.log((1 + len(documents)) / (1 + self.doc_freq)) + 1
        else:
            idf = np.log(len(documents) / np.maximum(self.doc_freq, 1)) + 1
        self.idf_ = idf
        return self
    
    def transform(self, raw_documents=None):
        if self.vocab is None:
            raise ValueError("The model has not been fitted yet. Please call 'fit' with training data before using 'transform'.")
        documents =  self._prepare_documents(raw_documents) 
        tokenized_docs = [self._tokenize(doc) for doc in documents]

        term_freq = np.zeros((len(documents), len(self.vocab)), dtype=float)
        for doc_idx, doc in enumerate(tokenized_docs):
            for word in doc:
                if word in self.vocab_index:
                    term_freq[doc_idx, self.vocab_index[word]] += 1

        tf = term_freq / np.maximum(term_freq.sum(axis=1, keepdims=True), 1)
        # idf = np.log((len(documents) + 1) / (self.doc_freq + 1)) + 1
        tfidf = tf * self.idf_

        # Normalize if required
        if self.norm == 'l2':
            norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
            tfidf = tfidf / np.maximum(norms, 1e-10)
        elif self.norm == 'l1':
            norms = np.sum(np.abs(tfidf), axis=1, keepdims=True)
            tfidf = tfidf / np.maximum(norms, 1e-10)
        
        self.tfidf_matrix = tfidf
        return pd.DataFrame(self.tfidf_matrix, columns=self.vocab)
    
    def fit_transform(self, raw_documents):
        self.fit(raw_documents)
        return self.transform(raw_documents)
    
# Example usage
if __name__ == "__main__":
    documents = [
        'The sun is shining brightly',
        'The weather is sweet today',
        'The sun and the weather are a sweet combination'
    ]
    train_docs = ["the sun is bright", "the weather is sweet"]
    test_docs = ["the sun is shining", "it is raining heavily"]
    vectorizer = TfidfVectorizer(input='content')
    tfidf_df = vectorizer.fit_transform(train_docs)

    print("--- TF-IDF Matrix ---")
    print(tfidf_df)
    print("\n ----Learned Vocabulary ----")
    print(vectorizer.vocab)
    print("\n ----correcct tf-idf for test docs ----")
    test_tfidf_df = vectorizer.transform(test_docs)
    print(pd.DataFrame(test_tfidf_df, columns=vectorizer.vocab))
    from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidf
    print("---- Check the results with sklearn ----")
    sklearn_vectorizer = SklearnTfidf()
    sklearn_tfidf = sklearn_vectorizer.fit_transform(train_docs)
    print('----vocabulary from sklearn----')
    print(sklearn_vectorizer.vocabulary_)
    print('\n ----- correct tf-idf for test docs from sklearn----')
    test_sklearn_tfidf = sklearn_vectorizer.transform(test_docs)
    print(pd.DataFrame(test_sklearn_tfidf.toarray(), columns=sklearn_vectorizer.get_feature_names_out()))
