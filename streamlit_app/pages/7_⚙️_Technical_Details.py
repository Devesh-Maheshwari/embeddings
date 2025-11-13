"""
Technical Details Page - Implementation and reproducibility
For developers and researchers
"""
import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PAGE_CONFIG, RESULTS_DIR

# Page configuration
st.set_page_config(**PAGE_CONFIG)

# Custom CSS
st.markdown("""
<style>
    .code-block {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #4CAF50;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
    }
    .tech-spec {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("⚙️ Technical Details")
st.markdown("**Implementation details and reproducibility guide** for developers and researchers")

st.divider()

# Project structure
st.markdown("## 📁 Project Structure")

st.code("""
embeddings/
├── pretraining/
│   ├── tfidf/
│   │   ├── tfidf_vectorizer.py      # TF-IDF from scratch
│   │   └── train.py                 # TF-IDF training script
│   ├── word2vec/
│   │   ├── base_sgns.py             # Shared SGNS components
│   │   ├── skipgram_from_scratch.py # Skip-gram implementation
│   │   ├── cbow_from_scratch.py     # CBOW implementation
│   │   └── train.py                 # Word2Vec training script
│   ├── glove/
│   │   ├── glove_model.py           # GloVe PyTorch implementation
│   │   └── train.py                 # GloVe training script
│   ├── fasttext/
│   │   ├── fasttext_model.py        # FastText PyTorch implementation
│   │   └── train.py                 # FastText training script
│   └── utils/
│       ├── sentence_encoder.py      # Sentence pooling utilities
│       └── data_utils.py            # Data loading helpers
├── intrinsic/
│   └── evaluate.py                  # Word analogies & similarity
├── downstream/
│   └── evaluate_sentiment.py        # IMDB sentiment classification
├── results/
│   ├── skipgram/
│   │   ├── vectors.npz              # Trained embeddings
│   │   ├── vocab.json               # Vocabulary mapping
│   │   └── intrinsic_eval.json      # Evaluation results
│   ├── cbow/
│   ├── glove/
│   └── tfidf/
│       ├── tfidf.idf.npy            # IDF values
│       └── tfidf.vocab.json         # Vocabulary
├── streamlit_app/
│   ├── Home.py                      # Landing page
│   ├── pages/
│   │   ├── 1_📊_Executive_Summary.py
│   │   ├── 2_🔬_Model_Comparison.py
│   │   ├── 3_🔤_Word_Analysis.py
│   │   ├── 4_📝_Sentence_Analysis.py
│   │   ├── 6_🧪_Failure_Analysis.py
│   │   └── 7_⚙️_Technical_Details.py (this page)
│   ├── config.py                    # Centralized configuration
│   └── utils/
│       └── data_loader.py           # Cached data loading
└── data/
    └── text8                        # Training corpus (17M tokens)
""", language="text")

st.divider()

# Implementation details
st.markdown("## 🔧 Implementation Details")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["TF-IDF", "Skip-gram", "CBOW", "GloVe", "FastText"])

with tab1:
    st.markdown("### 📊 TF-IDF Implementation")

    st.markdown("""
    **Algorithm**: Term Frequency-Inverse Document Frequency

    **Formula**:
    """)

    st.latex(r"""
    \text{tfidf}(t, d) = \text{tf}(t, d) \times \text{idf}(t)
    """)

    st.latex(r"""
    \text{idf}(t) = \log\left(\frac{N}{\text{df}(t)}\right)
    """)

    st.markdown("**Implementation**: Pure NumPy (no sklearn)")

    st.code("""
# Core TF-IDF computation
class TfidfVectorizer:
    def fit(self, documents):
        # Build vocabulary
        self.vocab = {word: idx for idx, word in enumerate(unique_words)}

        # Compute document frequency
        df = np.zeros(len(self.vocab))
        for doc in documents:
            unique_in_doc = set(doc)
            for word in unique_in_doc:
                if word in self.vocab:
                    df[self.vocab[word]] += 1

        # Compute IDF
        N = len(documents)
        self.idf = np.log(N / df)

    def transform(self, documents):
        # Compute TF for each document
        X = np.zeros((len(documents), len(self.vocab)))
        for i, doc in enumerate(documents):
            tf = Counter(doc)
            for word, count in tf.items():
                if word in self.vocab:
                    j = self.vocab[word]
                    X[i, j] = count * self.idf[j]
        return X
    """, language="python")

    st.markdown("**Training Command**:")
    st.code("python pretraining/tfidf/train.py --data data/text8", language="bash")

with tab2:
    st.markdown("### 🎯 Skip-gram Implementation")

    st.markdown("""
    **Algorithm**: Skip-gram with Negative Sampling (SGNS)

    **Objective**: Predict context words given center word
    """)

    st.latex(r"""
    J(\theta) = -\sum_{(w,c)\in D} \left[\log\sigma(v_w^\top v_c) + \sum_{i=1}^k \mathbb{E}_{c_n\sim P_n}[\log\sigma(-v_w^\top v_{c_n})]\right]
    """)

    st.markdown("""
    **Where**:
    - `w` = center word
    - `c` = context word
    - `k` = number of negative samples
    - `P_n` = noise distribution (unigram^0.75)
    """)

    st.code("""
# Skip-gram forward pass
def forward(self, center_word, context_word, negative_samples):
    # Get embeddings
    center_vec = self.embeddings[center_word]          # (D,)
    context_vec = self.context_weights[context_word]   # (D,)
    neg_vecs = self.context_weights[negative_samples]  # (K, D)

    # Positive score
    pos_score = np.dot(center_vec, context_vec)
    pos_loss = -np.log(sigmoid(pos_score))

    # Negative scores
    neg_scores = np.dot(neg_vecs, center_vec)
    neg_loss = -np.sum(np.log(sigmoid(-neg_scores)))

    return pos_loss + neg_loss
    """, language="python")

    st.markdown("**Training Command**:")
    st.code("python pretraining/word2vec/train.py --model skipgram --epochs 5 --dim 100", language="bash")

with tab3:
    st.markdown("### 🎯 CBOW Implementation")

    st.markdown("""
    **Algorithm**: Continuous Bag of Words with Negative Sampling

    **Objective**: Predict center word given context words
    """)

    st.latex(r"""
    J(\theta) = -\sum_{(w,C)\in D} \left[\log\sigma(\bar{v}_C^\top v_w) + \sum_{i=1}^k \mathbb{E}_{w_n\sim P_n}[\log\sigma(-\bar{v}_C^\top v_{w_n})]\right]
    """)

    st.markdown("""
    **Where**:
    - `w` = center word
    - `C` = context words
    - `v̄_C` = average of context word embeddings
    """)

    st.code("""
# CBOW forward pass
def forward(self, context_words, center_word, negative_samples):
    # Average context embeddings
    context_vecs = self.embeddings[context_words]      # (|C|, D)
    context_avg = np.mean(context_vecs, axis=0)        # (D,)

    # Get center word embedding
    center_vec = self.context_weights[center_word]     # (D,)

    # Positive score
    pos_score = np.dot(context_avg, center_vec)
    pos_loss = -np.log(sigmoid(pos_score))

    # Negative scores
    neg_vecs = self.context_weights[negative_samples]  # (K, D)
    neg_scores = np.dot(neg_vecs, context_avg)
    neg_loss = -np.sum(np.log(sigmoid(-neg_scores)))

    return pos_loss + neg_loss
    """, language="python")

    st.markdown("**Training Command**:")
    st.code("python pretraining/word2vec/train.py --model cbow --epochs 5 --dim 100", language="bash")

with tab4:
    st.markdown("### 🌐 GloVe Implementation")

    st.markdown("""
    **Algorithm**: Global Vectors for Word Representation

    **Objective**: Factorize word co-occurrence matrix
    """)

    st.latex(r"""
    J = \sum_{i,j=1}^V f(X_{ij})(w_i^\top \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2
    """)

    st.markdown("""
    **Where**:
    - `X_ij` = co-occurrence count (word i, context j)
    - `f(x)` = weighting function = (x/x_max)^α if x < x_max else 1
    - `w_i` = word embedding
    - `w̃_j` = context embedding
    """)

    st.code("""
# GloVe PyTorch model
class GloVeModel(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, dim)
        self.context_embeddings = nn.Embedding(vocab_size, dim)
        self.word_biases = nn.Embedding(vocab_size, 1)
        self.context_biases = nn.Embedding(vocab_size, 1)

    def forward(self, i_idx, j_idx, cooc):
        # Get embeddings and biases
        w_i = self.word_embeddings(i_idx)
        w_j = self.context_embeddings(j_idx)
        b_i = self.word_biases(i_idx).squeeze()
        b_j = self.context_biases(j_idx).squeeze()

        # Compute loss
        dot_product = (w_i * w_j).sum(dim=1)
        log_cooc = torch.log(cooc)
        diff = dot_product + b_i + b_j - log_cooc

        # Weighting function
        weight = torch.clamp(cooc / 100, max=1.0) ** 0.75

        return (weight * diff ** 2).mean()
    """, language="python")

    st.markdown("**Training Command**:")
    st.code("python pretraining/glove/train.py --epochs 15 --dim 100", language="bash")

with tab5:
    st.markdown("### ⚡ FastText Implementation")

    st.markdown("""
    **Algorithm**: Skip-gram with character n-grams

    **Objective**: Same as Skip-gram, but words represented as sum of character n-gram embeddings
    """)

    st.markdown("""
    **Key Difference from Skip-gram**:
    - Each word decomposed into character n-grams (e.g., "where" → <wh, whe, her, ere, re>)
    - Word embedding = sum of n-gram embeddings + whole word embedding
    - Handles OOV words by summing n-gram embeddings
    """)

    st.code("""
# FastText character n-grams
def get_ngrams(word, min_n=3, max_n=6):
    word = f"<{word}>"  # Add boundary markers
    ngrams = []
    for n in range(min_n, max_n + 1):
        for i in range(len(word) - n + 1):
            ngrams.append(word[i:i+n])
    return ngrams

# FastText forward pass
def forward(self, center_word, context_word, negative_samples):
    # Get character n-grams for center word
    ngrams = get_ngrams(center_word)

    # Sum n-gram embeddings
    center_vec = self.word_embeddings[center_word]  # Whole word
    for ng in ngrams:
        if ng in self.ngram_vocab:
            center_vec += self.ngram_embeddings[self.ngram_vocab[ng]]

    # Rest is same as Skip-gram
    ...
    """, language="python")

    st.markdown("**Training Command**:")
    st.code("python pretraining/fasttext/train.py --epochs 5 --dim 100 --min-n 3 --max-n 6", language="bash")

st.divider()

# Hyperparameters
st.markdown("## 🎛️ Hyperparameters")

st.markdown("### Training Configuration")

hyperparam_data = {
    "Parameter": ["Embedding Dimension", "Context Window", "Epochs", "Negative Samples",
                  "Learning Rate (Word2Vec)", "Learning Rate (GloVe)", "Min Count",
                  "Batch Size (GloVe)", "X_max (GloVe)", "Alpha (GloVe)"],
    "Value": [100, 5, 5, 5, 0.025, 0.05, 5, 512, 100, 0.75],
    "Description": [
        "Size of word embeddings",
        "Words on each side of center word",
        "Passes through training corpus",
        "Negative samples per positive example",
        "Initial learning rate (decays to 0.0001)",
        "Adam optimizer learning rate",
        "Minimum word frequency threshold",
        "Co-occurrence pairs per batch",
        "Weighting function cutoff",
        "Weighting function exponent"
    ]
}

import pandas as pd
st.dataframe(pd.DataFrame(hyperparam_data), use_container_width=True, hide_index=True)

st.divider()

# Evaluation methodology
st.markdown("## 📏 Evaluation Methodology")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🎯 Intrinsic Evaluation")
    st.markdown("""
    **Word Analogies** (Google analogy dataset subset)

    **Format**: `a - b + c = ?`

    **Examples**:
    - king - man + woman = queen
    - paris - france + london = england

    **Metrics**:
    - Top-1 accuracy (3CosAdd & 3CosMul)
    - Coverage (% of questions answerable)

    **Implementation**:
    ```python
    # 3CosAdd method
    target_vec = v_a - v_b + v_c
    similarities = cosine_similarity(target_vec, all_embeddings)
    prediction = argmax(similarities, excluding=[a,b,c])
    ```

    **Evaluation Script**:
    ```bash
    python intrinsic/evaluate.py --model skipgram
    ```
    """)

with col2:
    st.markdown("### 🎬 Extrinsic Evaluation")
    st.markdown("""
    **Downstream Task**: IMDB Sentiment Classification

    **Dataset**:
    - 25,000 movie reviews (train)
    - Binary labels (positive/negative)
    - Split: 15k train, 5k val, 5k test

    **Sentence Pooling Strategies**:
    1. **Average Pooling**: Mean of word vectors
    2. **TF-IDF Weighted**: Weight by informativeness

    **Classifiers**:
    1. **Logistic Regression**: Linear baseline
    2. **MLP**: 2-layer neural network (100 → 50 → 2)

    **Metrics**:
    - Train/Val/Test accuracy
    - Confusion matrix
    - F1 score

    **Evaluation Script**:
    ```bash
    python downstream/evaluate_sentiment.py \\
        --model skipgram \\
        --pooling avg \\
        --classifier logistic
    ```
    """)

st.divider()

# Reproducibility
st.markdown("## 🔄 Reproducibility Guide")

with st.container(border=True):
    st.markdown("### Step-by-Step Setup")

    st.markdown("#### 1. Clone Repository")
    st.code("""
git clone https://github.com/patron02/embeddings.git
cd embeddings
    """, language="bash")

    st.markdown("#### 2. Create Environment")
    st.code("""
# Using conda
conda create -n embeddings python=3.11
conda activate embeddings

# Install dependencies
pip install -r requirements.txt
    """, language="bash")

    st.markdown("#### 3. Download Data")
    st.code("""
# Download Text8 corpus (17MB)
wget http://mattmahoney.net/dc/text8.zip
unzip text8.zip -d data/

# Download IMDB dataset (optional, for downstream eval)
# Will be auto-downloaded by evaluation script
    """, language="bash")

    st.markdown("#### 4. Train Models")
    st.code("""
# Train TF-IDF
python pretraining/tfidf/train.py --data data/text8

# Train Skip-gram
python pretraining/word2vec/train.py \\
    --model skipgram \\
    --data data/text8 \\
    --epochs 5 \\
    --dim 100

# Train CBOW
python pretraining/word2vec/train.py \\
    --model cbow \\
    --data data/text8 \\
    --epochs 5 \\
    --dim 100

# Train GloVe
python pretraining/glove/train.py \\
    --data data/text8 \\
    --epochs 15 \\
    --dim 100

# Train FastText
python pretraining/fasttext/train.py \\
    --data data/text8 \\
    --epochs 5 \\
    --dim 100
    """, language="bash")

    st.markdown("#### 5. Run Intrinsic Evaluation")
    st.code("""
# Evaluate each model on word analogies
python intrinsic/evaluate.py --model skipgram
python intrinsic/evaluate.py --model cbow
python intrinsic/evaluate.py --model glove
    """, language="bash")

    st.markdown("#### 6. Run Downstream Evaluation")
    st.code("""
# Evaluate all configurations (pooling × classifier)
for model in skipgram cbow glove tfidf; do
    for pooling in avg tfidf; do
        for classifier in logistic mlp; do
            python downstream/evaluate_sentiment.py \\
                --model $model \\
                --pooling $pooling \\
                --classifier $classifier
        done
    done
done
    """, language="bash")

    st.markdown("#### 7. Launch Streamlit UI")
    st.code("""
cd streamlit_app
streamlit run Home.py
    """, language="bash")

st.divider()

# System requirements
st.markdown("## 💻 System Requirements")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Minimum Requirements")
    st.markdown("""
    - **CPU**: Dual-core 2.0 GHz
    - **RAM**: 8 GB
    - **Storage**: 2 GB free space
    - **Python**: 3.8+
    - **OS**: Linux, macOS, Windows

    **Training Time** (Text8, 5 epochs):
    - Skip-gram: ~10 min (CPU)
    - CBOW: ~8 min (CPU)
    - GloVe: ~15 min (CPU)
    - TF-IDF: ~2 min (CPU)
    """)

with col2:
    st.markdown("### Recommended Requirements")
    st.markdown("""
    - **CPU**: Quad-core 3.0 GHz or GPU
    - **RAM**: 16 GB
    - **Storage**: 10 GB free space
    - **Python**: 3.11+
    - **GPU**: CUDA-compatible (optional, for GloVe/FastText)

    **Training Time** (Text8, 5 epochs, GPU):
    - Skip-gram: ~3 min
    - CBOW: ~2 min
    - GloVe: ~5 min
    - FastText: ~8 min
    """)

st.divider()

# Dependencies
st.markdown("## 📦 Dependencies")

st.markdown("### Core Libraries")

deps_data = {
    "Library": ["numpy", "pandas", "torch", "scikit-learn", "streamlit", "matplotlib"],
    "Version": ["≥1.24.0", "≥2.0.0", "≥2.0.0", "≥1.3.0", "≥1.28.0", "≥3.7.0"],
    "Purpose": [
        "Numerical computations, embedding operations",
        "Data manipulation, evaluation results",
        "GloVe, FastText (PyTorch implementations)",
        "Logistic regression, MLP, evaluation metrics",
        "Interactive UI for exploration",
        "Visualization (optional)"
    ]
}

st.dataframe(pd.DataFrame(deps_data), use_container_width=True, hide_index=True)

st.code("""
# requirements.txt
numpy>=1.24.0
pandas>=2.0.0
torch>=2.0.0
scikit-learn>=1.3.0
streamlit>=1.28.0
matplotlib>=3.7.0
""", language="text")

st.divider()

# Testing
st.markdown("## 🧪 Testing & Validation")

st.markdown("""
### Unit Tests (Coming Soon)

```python
# tests/test_skipgram.py
def test_skipgram_training():
    model = SkipGram(vocab_size=1000, dim=50)
    loss = model.train(corpus, epochs=1)
    assert loss > 0

def test_analogy_evaluation():
    embeddings, vocab = load_vectors("skipgram")
    result = evaluate_analogies(embeddings, vocab)
    assert 0 <= result["accuracy"] <= 1
```

### Continuous Integration

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.11
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/
```
""")

st.divider()

# Citation
st.markdown("## 📚 Citation")

st.markdown("""
If you use this code in your research, please cite:

```bibtex
@software{embeddings2025,
  title={Word Embeddings from Scratch: TF-IDF, Skip-gram, CBOW, GloVe, FastText},
  author={Your Name},
  year={2025},
  url={https://github.com/patron02/embeddings}
}
```

### Referenced Papers

**Skip-gram & CBOW**:
```bibtex
@article{mikolov2013efficient,
  title={Efficient estimation of word representations in vector space},
  author={Mikolov, Tomas and Chen, Kai and Corrado, Greg and Dean, Jeffrey},
  journal={arXiv preprint arXiv:1301.3781},
  year={2013}
}
```

**GloVe**:
```bibtex
@inproceedings{pennington2014glove,
  title={Glove: Global vectors for word representation},
  author={Pennington, Jeffrey and Socher, Richard and Manning, Christopher D},
  booktitle={EMNLP},
  year={2014}
}
```

**FastText**:
```bibtex
@article{bojanowski2017enriching,
  title={Enriching word vectors with subword information},
  author={Bojanowski, Piotr and Grave, Edouard and Joulin, Armand and Mikolov, Tomas},
  journal={TACL},
  year={2017}
}
```
""")

st.divider()

# Footer
st.markdown("### 📚 Related Pages")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📊 Executive Summary", use_container_width=True):
        st.switch_page("pages/1_📊_Executive_Summary.py")

with col2:
    if st.button("🔬 Model Comparison", use_container_width=True):
        st.switch_page("pages/2_🔬_Model_Comparison.py")

with col3:
    if st.button("🧪 Failure Analysis", use_container_width=True):
        st.switch_page("pages/6_🧪_Failure_Analysis.py")

with col4:
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("Home.py")

# Footer note
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>Built with ❤️ using Streamlit |
    <a href="https://github.com/patron02/embeddings">GitHub</a> |
    <a href="https://github.com/patron02/embeddings/issues">Report Issue</a></p>
</div>
""", unsafe_allow_html=True)
