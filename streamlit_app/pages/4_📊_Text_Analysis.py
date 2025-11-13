"""
Text Analysis Page - Phrase, Sentence, and Document Embeddings
Shows the complete embedding hierarchy
"""
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PAGE_CONFIG, RESULTS_DIR, MODELS, MODEL_DISPLAY_NAMES, COLORS
from utils.data_loader import load_model_vectors, load_tfidf_data, get_all_available_models

# Page configuration
st.set_page_config(**PAGE_CONFIG)

# Custom CSS
st.markdown("""
<style>
    .hierarchy-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    .similarity-high {
        background-color: #d4edda;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .similarity-low {
        background-color: #f8d7da;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("📊 Text Analysis: The Complete Embedding Hierarchy")
st.markdown("**Explore how embeddings work at different granularities**: Phrases → Sentences → Documents")

st.divider()

# Educational: Embedding Hierarchy
with st.expander("📚 **Understanding the Embedding Hierarchy**", expanded=False):
    st.markdown("""
    Text can be embedded at different granularities, each with different use cases:

    ### 🔤 **Word Embeddings** (see Word Analysis page)
    - **Granularity**: Single tokens ("king", "movie")
    - **Models**: Skip-gram, CBOW, GloVe
    - **Use cases**: Word similarity, analogies

    ### 📝 **Phrase Embeddings** (2-5 words)
    - **Granularity**: Multi-word expressions ("New York", "machine learning", "very good")
    - **Key insight**: Compositional vs non-compositional
      - Compositional: "big house" ≈ mean("big", "house") ✅
      - Non-compositional: "kick the bucket" ≠ mean("kick", "the", "bucket") ❌
    - **Use cases**: Named entity recognition, collocation detection

    ### 📄 **Sentence Embeddings** (5-20 words)
    - **Granularity**: Single sentences ("The movie was great")
    - **Pooling strategies**: Average, TF-IDF weighted
    - **Use cases**: Sentence similarity, semantic search

    ### 📚 **Document Embeddings** (50-500+ words)
    - **Granularity**: Paragraphs, full reviews, articles
    - **Key difference**: Multi-sentence discourse structure
    - **Use cases**: Document classification, clustering, retrieval

    ---

    ### 🎯 Why This Hierarchy Matters:

    **Skip-gram/GloVe** capture phrase-level associations through context windows (5 words each side)

    **BERT** (512 tokens) is designed for sentences AND documents

    **Your downstream task** (IMDB reviews) uses document-level embeddings (200-300 words)!
    """)

st.divider()

# Load models
available_models = get_all_available_models(RESULTS_DIR)

if not available_models:
    st.error("❌ No trained models found. Please train models first!")
    st.stop()

# Model selection (shared across tabs)
col1, col2 = st.columns(2)

with col1:
    selected_model = st.selectbox(
        "Select Model",
        available_models,
        format_func=lambda x: MODEL_DISPLAY_NAMES.get(x, x.upper()),
        key="text_model_select"
    )

with col2:
    if selected_model == "tfidf":
        pooling_strategy = "tfidf"
        st.info("TF-IDF model only supports TF-IDF pooling")
    else:
        pooling_strategy = st.selectbox(
            "Pooling Strategy",
            ["avg", "tfidf"],
            format_func=lambda x: "Average Pooling" if x == "avg" else "TF-IDF Weighted Pooling",
            key="text_pooling_select"
        )

st.divider()

# Helper functions
def simple_tokenize(text: str) -> list[str]:
    """Simple tokenization matching training code"""
    import re
    text = text.lower()
    return re.findall(r"[a-z0-9']+", text)

def encode_text_avg(text: str, embeddings: np.ndarray, word2id: dict) -> np.ndarray:
    """Average pooling of word vectors"""
    tokens = simple_tokenize(text)
    if not tokens:
        return np.zeros(embeddings.shape[1], dtype=np.float32)

    vectors = []
    for token in tokens:
        if token in word2id:
            idx = word2id[token]
            vectors.append(embeddings[idx])

    if not vectors:
        return np.zeros(embeddings.shape[1], dtype=np.float32)

    return np.mean(vectors, axis=0)

def encode_text_tfidf(text: str, embeddings: np.ndarray, word2id: dict,
                      idf_values: np.ndarray, tfidf_vocab: dict) -> np.ndarray:
    """TF-IDF weighted pooling of word vectors"""
    tokens = simple_tokenize(text)
    if not tokens:
        return np.zeros(embeddings.shape[1], dtype=np.float32)

    # Calculate term frequencies
    tf = {}
    for token in tokens:
        if token in word2id:
            tf[token] = tf.get(token, 0) + 1

    if not tf:
        return np.zeros(embeddings.shape[1], dtype=np.float32)

    # Weighted average
    weighted_sum = np.zeros(embeddings.shape[1], dtype=np.float32)
    total_weight = 0.0

    for token, freq in tf.items():
        # Get IDF weight
        idf_weight = 1.0
        if token in tfidf_vocab:
            idf_idx = tfidf_vocab[token]
            if 0 <= idf_idx < len(idf_values):
                idf_weight = idf_values[idf_idx]

        # Get word vector
        if token in word2id:
            idx = word2id[token]
            weight = freq * idf_weight
            weighted_sum += weight * embeddings[idx]
            total_weight += weight

    if total_weight == 0:
        return np.zeros(embeddings.shape[1], dtype=np.float32)

    return weighted_sum / total_weight

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

# Load model data
if selected_model == "tfidf":
    result = load_tfidf_data(RESULTS_DIR)
    if result is None:
        st.error(f"❌ Could not load TF-IDF data")
        st.stop()
    idf_values, tfidf_vocab = result
    embeddings, word2id = None, None
else:
    result = load_model_vectors(selected_model, RESULTS_DIR)
    if result is None:
        st.error(f"❌ Could not load {selected_model} model")
        st.stop()
    embeddings, word2id = result

    # Load TF-IDF data for weighted pooling (if needed)
    idf_result = load_tfidf_data(RESULTS_DIR)
    if idf_result is None and pooling_strategy == "tfidf":
        st.error("❌ TF-IDF data not found. Cannot use TF-IDF weighted pooling.")
        st.info("Train TF-IDF model first or use average pooling.")
        st.stop()

    if pooling_strategy == "tfidf":
        idf_values, tfidf_vocab = idf_result

# Three tabs for hierarchy
tab1, tab2, tab3 = st.tabs(["🔤 Phrase Embeddings", "📝 Sentence Embeddings", "📄 Document Embeddings"])

# ============================================
# TAB 1: PHRASE EMBEDDINGS
# ============================================
with tab1:
    st.markdown("## 🔤 Phrase Embeddings (2-5 words)")
    st.markdown("*Understand how multi-word expressions are composed*")

    with st.expander("💡 **Compositional vs Non-Compositional Phrases**"):
        st.markdown("""
        ### Compositional Phrases (Additive)
        Meaning can be derived from component words:
        - "big house" ≈ mean("big", "house")
        - "machine learning" ≈ mean("machine", "learning")
        - "very good" ≈ mean("very", "good")

        ### Non-Compositional Phrases (Idiomatic)
        Meaning CANNOT be derived from components:
        - "kick the bucket" = die (not literal kicking!)
        - "piece of cake" = easy (not about dessert!)
        - "break a leg" = good luck (not injury!)

        ### How Skip-gram/GloVe Handle Phrases:
        - **Implicit**: Context window (5 words) captures co-occurrence
        - **Explicit**: Train with phrase detection (bigrams/trigrams as single tokens)
        - **Limitation**: Cannot distinguish compositional vs idiomatic without phrase detection
        """)

    st.divider()

    # Example phrases
    st.markdown("### 📝 Example Phrases to Explore")

    example_phrases = {
        "Compositional": [
            "machine learning",
            "deep neural network",
            "natural language processing",
            "big house",
            "very good"
        ],
        "Named Entities": [
            "new york",
            "united states",
            "san francisco",
            "los angeles"
        ],
        "Simple Combinations": [
            "good movie",
            "bad acting",
            "great film"
        ]
    }

    selected_category = st.selectbox(
        "Choose phrase category:",
        list(example_phrases.keys()),
        key="phrase_category"
    )

    cols = st.columns(min(3, len(example_phrases[selected_category])))
    for idx, phrase in enumerate(example_phrases[selected_category][:3]):
        with cols[idx]:
            if st.button(f'"{phrase}"', key=f"phrase_ex_{idx}", use_container_width=True):
                st.session_state.phrase_input = phrase
                st.rerun()

    st.divider()

    # Phrase similarity
    st.markdown("### 🔍 Phrase Similarity Analysis")

    phrase_input = st.text_area(
        "Enter phrases (one per line):",
        value=st.session_state.get("phrase_input", "machine learning\ndeep learning\nartificial intelligence"),
        height=100,
        key="phrase_text_area"
    )

    if st.button("🔍 Analyze Phrases", type="primary", use_container_width=True, key="phrase_analyze"):
        phrases = [p.strip() for p in phrase_input.split("\n") if p.strip()]

        if len(phrases) < 2:
            st.warning("⚠️ Enter at least 2 phrases")
            st.stop()

        # Encode phrases
        phrase_vectors = []

        if selected_model == "tfidf":
            for phrase in phrases:
                tokens = simple_tokenize(phrase)
                tf = {}
                for token in tokens:
                    tf[token] = tf.get(token, 0) + 1

                vec = np.zeros(len(idf_values), dtype=np.float32)
                for token, freq in tf.items():
                    if token in tfidf_vocab:
                        idx = tfidf_vocab[token]
                        if 0 <= idx < len(idf_values):
                            vec[idx] = freq * idf_values[idx]
                phrase_vectors.append(vec)
        else:
            for phrase in phrases:
                if pooling_strategy == "avg":
                    vec = encode_text_avg(phrase, embeddings, word2id)
                else:
                    vec = encode_text_tfidf(phrase, embeddings, word2id, idf_values, tfidf_vocab)
                phrase_vectors.append(vec)

        # Compute similarity matrix
        n = len(phrases)
        sim_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                sim_matrix[i, j] = cosine_similarity(phrase_vectors[i], phrase_vectors[j])

        # Display
        st.markdown("#### 📊 Phrase Similarity Matrix")

        df_sim = pd.DataFrame(
            sim_matrix,
            index=[f"P{i+1}" for i in range(n)],
            columns=[f"P{i+1}" for i in range(n)]
        ).round(3)

        st.dataframe(
            df_sim.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1),
            use_container_width=True
        )

        # Legend
        st.markdown("#### 📝 Phrase Legend")
        for i, phrase in enumerate(phrases):
            # Analyze compositionality
            tokens = simple_tokenize(phrase)
            if len(tokens) >= 2 and embeddings is not None:
                # Compare phrase embedding to sum/average of word embeddings
                word_vecs = [embeddings[word2id[t]] for t in tokens if t in word2id]
                if word_vecs:
                    avg_vec = np.mean(word_vecs, axis=0)
                    compositional_score = cosine_similarity(phrase_vectors[i], avg_vec)

                    if compositional_score > 0.9:
                        comp_label = "🟢 Highly compositional"
                    elif compositional_score > 0.7:
                        comp_label = "🟡 Moderately compositional"
                    else:
                        comp_label = "🔴 Non-compositional (may be idiomatic)"

                    st.markdown(f"**P{i+1}**: {phrase} ({len(tokens)} words) - {comp_label} ({compositional_score:.3f})")
                else:
                    st.markdown(f"**P{i+1}**: {phrase} (OOV words)")
            else:
                st.markdown(f"**P{i+1}**: {phrase}")

# ============================================
# TAB 2: SENTENCE EMBEDDINGS
# ============================================
with tab2:
    st.markdown("## 📝 Sentence Embeddings (5-20 words)")
    st.markdown("*Single-sentence semantic similarity*")

    with st.expander("💡 **Pooling Strategies for Sentences**"):
        st.markdown("""
        ### Average Pooling
        - Simple mean of word vectors
        - Treats all words equally
        - Fast and works well for most cases

        ### TF-IDF Weighted Pooling
        - Weight words by informativeness (IDF score)
        - Rare/informative words get higher weight
        - Common/stopwords get lower weight
        - Better for noisy text with many stopwords
        """)

    st.divider()

    # Example sentences
    st.markdown("### 🎯 Example Sentence Sets")

    sentence_examples = {
        "Movie Reviews (Similar)": [
            "The movie was great and entertaining",
            "The film was fantastic and enjoyable",
            "The acting was terrible and boring"
        ],
        "Movie Reviews (Mixed)": [
            "This movie is absolutely brilliant",
            "This film is completely awful",
            "The weather is sunny today"
        ],
        "Semantic Similarity": [
            "A man is playing guitar",
            "A woman is playing violin",
            "The sun is shining brightly"
        ]
    }

    cols = st.columns(len(sentence_examples))
    for idx, (name, sentences) in enumerate(sentence_examples.items()):
        with cols[idx]:
            if st.button(name, key=f"sent_ex_{idx}", use_container_width=True):
                st.session_state.sentence_input = "\n".join(sentences)
                st.rerun()

    st.divider()

    # Sentence similarity
    sentence_input = st.text_area(
        "Enter sentences (one per line):",
        value=st.session_state.get("sentence_input", "The movie was great\nThe film was fantastic\nThe acting was terrible"),
        height=120,
        key="sentence_text_area"
    )

    if st.button("🔍 Compute Sentence Similarity", type="primary", use_container_width=True, key="sent_compute"):
        sentences = [s.strip() for s in sentence_input.split("\n") if s.strip()]

        if len(sentences) < 2:
            st.warning("⚠️ Enter at least 2 sentences")
            st.stop()

        # Encode sentences (same as phrases, just different length)
        sentence_vectors = []

        if selected_model == "tfidf":
            for sent in sentences:
                tokens = simple_tokenize(sent)
                tf = {}
                for token in tokens:
                    tf[token] = tf.get(token, 0) + 1

                vec = np.zeros(len(idf_values), dtype=np.float32)
                for token, freq in tf.items():
                    if token in tfidf_vocab:
                        idx = tfidf_vocab[token]
                        if 0 <= idx < len(idf_values):
                            vec[idx] = freq * idf_values[idx]
                sentence_vectors.append(vec)
        else:
            for sent in sentences:
                if pooling_strategy == "avg":
                    vec = encode_text_avg(sent, embeddings, word2id)
                else:
                    vec = encode_text_tfidf(sent, embeddings, word2id, idf_values, tfidf_vocab)
                sentence_vectors.append(vec)

        # Compute similarity
        n = len(sentences)
        sim_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                sim_matrix[i, j] = cosine_similarity(sentence_vectors[i], sentence_vectors[j])

        # Display
        st.markdown("#### 📊 Sentence Similarity Matrix")

        df_sim = pd.DataFrame(
            sim_matrix,
            index=[f"S{i+1}" for i in range(n)],
            columns=[f"S{i+1}" for i in range(n)]
        ).round(3)

        st.dataframe(
            df_sim.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1),
            use_container_width=True
        )

        # Legend
        st.markdown("#### 📝 Sentence Legend")
        for i, sent in enumerate(sentences):
            word_count = len(simple_tokenize(sent))
            st.markdown(f"**S{i+1}**: {sent} ({word_count} words)")

        # Find most/least similar
        max_sim = -1
        max_pair = None
        for i in range(n):
            for j in range(i+1, n):
                if sim_matrix[i, j] > max_sim:
                    max_sim = sim_matrix[i, j]
                    max_pair = (i, j)

        min_sim = 2
        min_pair = None
        for i in range(n):
            for j in range(i+1, n):
                if sim_matrix[i, j] < min_sim:
                    min_sim = sim_matrix[i, j]
                    min_pair = (i, j)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🟢 Most Similar Pair")
            if max_pair:
                i, j = max_pair
                st.markdown(f"**S{i+1}** ↔ **S{j+1}** (similarity: **{max_sim:.3f}**)")
                st.markdown(f'<div class="similarity-high">"{sentences[i]}"<br>"<strong>{sentences[j]}</strong>"</div>',
                           unsafe_allow_html=True)

        with col2:
            st.markdown("#### 🔴 Least Similar Pair")
            if min_pair:
                i, j = min_pair
                st.markdown(f"**S{i+1}** ↔ **S{j+1}** (similarity: **{min_sim:.3f}**)")
                st.markdown(f'<div class="similarity-low">"{sentences[i]}"<br>"<strong>{sentences[j]}</strong>"</div>',
                           unsafe_allow_html=True)

# ============================================
# TAB 3: DOCUMENT EMBEDDINGS
# ============================================
with tab3:
    st.markdown("## 📄 Document Embeddings (50-500+ words)")
    st.markdown("*Multi-sentence documents like movie reviews, articles, papers*")

    with st.expander("💡 **Document vs Sentence Embeddings**"):
        st.markdown("""
        ### Key Differences:

        **Length**:
        - Sentences: 5-20 words (single thought)
        - Documents: 50-500+ words (multiple sentences, paragraphs)

        **Structure**:
        - Sentences: Single grammatical unit
        - Documents: Discourse structure, topic flow, multi-paragraph coherence

        **Pooling Challenges**:
        - Average pooling still works but may lose information
        - TF-IDF pooling becomes MORE important (filters stopwords in long text)
        - Document-level models (Doc2Vec) capture inter-sentence relationships

        ### Why This Matters for IMDB:
        Your downstream evaluation uses **full movie reviews** (100-300 words), not sentences!

        BERT can handle up to **512 tokens** (~400 words) - perfect for documents!
        """)

    st.divider()

    # Example documents (simulated IMDB reviews)
    st.markdown("### 📚 Example Documents (Movie Reviews)")

    example_docs = {
        "Positive Review": """This movie was absolutely fantastic! The acting was superb,
        with brilliant performances from the entire cast. The storyline kept me engaged from
        start to finish, and the cinematography was breathtaking. I especially loved the
        character development and the unexpected plot twists. The director did an amazing job
        bringing this story to life. Highly recommended for anyone who enjoys quality cinema.
        I'll definitely be watching it again and recommending it to all my friends.
        A true masterpiece of modern filmmaking!""",

        "Negative Review": """I was extremely disappointed with this film. The plot was
        predictable and boring, with cardboard characters that I couldn't care less about.
        The dialogue felt forced and unnatural throughout. The special effects were the only
        saving grace, but even those couldn't save this trainwreck. I found myself checking
        my watch multiple times during the screening. The pacing was terrible, dragging in
        the middle with unnecessary scenes. Save your money and skip this one. There are far
        better movies out there worth your time. A complete waste of two hours.""",

        "Mixed Review": """This movie had its moments but ultimately fell short of
        expectations. While the cinematography was gorgeous and the soundtrack memorable,
        the plot had several gaping holes that were hard to ignore. Some actors delivered
        strong performances, but others seemed to be phoning it in. The first half showed
        promise with interesting character setups, but the second half rushed through
        important plot points. It's not terrible, but it's not great either. Worth a watch
        if you're a fan of the genre, but don't expect a masterpiece."""
    }

    cols = st.columns(len(example_docs))
    for idx, (name, doc) in enumerate(example_docs.items()):
        with cols[idx]:
            if st.button(name, key=f"doc_ex_{idx}", use_container_width=True):
                st.session_state.doc_input = "\n\n".join(list(example_docs.values())[:3])
                st.rerun()

    st.divider()

    # Document similarity
    st.markdown("### 🔍 Document Similarity Analysis")
    st.markdown("*Enter 2-5 documents (separate with blank lines)*")

    doc_input = st.text_area(
        "Enter documents:",
        value=st.session_state.get("doc_input", "\n\n".join(list(example_docs.values())[:2])),
        height=300,
        key="doc_text_area"
    )

    if st.button("🔍 Compute Document Similarity", type="primary", use_container_width=True, key="doc_compute"):
        # Split by double newlines
        documents = [d.strip() for d in doc_input.split("\n\n") if d.strip()]

        if len(documents) < 2:
            st.warning("⚠️ Enter at least 2 documents (separated by blank lines)")
            st.stop()

        if len(documents) > 5:
            st.warning("⚠️ Maximum 5 documents for readability")
            documents = documents[:5]

        # Encode documents (same pooling, just longer)
        doc_vectors = []

        if selected_model == "tfidf":
            for doc in documents:
                tokens = simple_tokenize(doc)
                tf = {}
                for token in tokens:
                    tf[token] = tf.get(token, 0) + 1

                vec = np.zeros(len(idf_values), dtype=np.float32)
                for token, freq in tf.items():
                    if token in tfidf_vocab:
                        idx = tfidf_vocab[token]
                        if 0 <= idx < len(idf_values):
                            vec[idx] = freq * idf_values[idx]
                doc_vectors.append(vec)
        else:
            for doc in documents:
                if pooling_strategy == "avg":
                    vec = encode_text_avg(doc, embeddings, word2id)
                else:
                    vec = encode_text_tfidf(doc, embeddings, word2id, idf_values, tfidf_vocab)
                doc_vectors.append(vec)

        # Compute similarity
        n = len(documents)
        sim_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                sim_matrix[i, j] = cosine_similarity(doc_vectors[i], doc_vectors[j])

        # Display
        st.markdown("#### 📊 Document Similarity Matrix")

        df_sim = pd.DataFrame(
            sim_matrix,
            index=[f"D{i+1}" for i in range(n)],
            columns=[f"D{i+1}" for i in range(n)]
        ).round(3)

        st.dataframe(
            df_sim.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1),
            use_container_width=True
        )

        # Document statistics
        st.markdown("#### 📝 Document Statistics")

        doc_stats = []
        for i, doc in enumerate(documents):
            tokens = simple_tokenize(doc)
            sentences = [s.strip() for s in doc.split('.') if s.strip()]

            doc_stats.append({
                "Doc": f"D{i+1}",
                "Words": len(tokens),
                "Sentences": len(sentences),
                "Avg Words/Sent": f"{len(tokens)/max(len(sentences), 1):.1f}",
                "Unique Words": len(set(tokens)),
                "Preview": doc[:80] + "..." if len(doc) > 80 else doc
            })

        st.dataframe(pd.DataFrame(doc_stats), use_container_width=True, hide_index=True)

        # Insights
        st.markdown("#### 💡 Insights")

        col1, col2 = st.columns(2)

        with col1:
            avg_words = np.mean([s["Words"] for s in doc_stats])
            st.metric("Average Document Length", f"{avg_words:.0f} words")

            if avg_words > 100:
                st.info("✅ These are proper **documents** (>100 words) - TF-IDF pooling recommended!")
            elif avg_words > 20:
                st.info("📝 These are **long sentences** or **short paragraphs**")
            else:
                st.warning("⚠️ These are **sentences** - see Sentence tab instead")

        with col2:
            # Find most/least similar documents
            max_sim = -1
            max_pair = None
            for i in range(n):
                for j in range(i+1, n):
                    if sim_matrix[i, j] > max_sim:
                        max_sim = sim_matrix[i, j]
                        max_pair = (i, j)

            if max_pair:
                i, j = max_pair
                st.metric("Most Similar Pair", f"D{i+1} ↔ D{j+1}", f"{max_sim:.3f}")

                if max_sim > 0.8:
                    st.success("🟢 Very similar (likely same sentiment/topic)")
                elif max_sim > 0.5:
                    st.info("🟡 Moderately similar")
                else:
                    st.warning("🔴 Different content")

st.divider()

# Cross-granularity comparison
st.markdown("## 🔬 Cross-Granularity Comparison")
st.markdown("*Compare how the same text is embedded at different granularities*")

with st.expander("💡 **Why This Matters**"):
    st.markdown("""
    The same text can be analyzed at different levels:

    **Example**: "The movie was great"

    - **Word-level**: "movie" (single token)
    - **Phrase-level**: "the movie" (2 words)
    - **Sentence-level**: "The movie was great" (4 words)
    - **Document-level**: Multiple sentences forming a review

    **Key Insight**: Shorter granularities lose context, longer granularities may average out details.

    **Best practice**: Match granularity to your task:
    - Word similarity → Word embeddings
    - Phrase detection → Phrase embeddings
    - Sentence search → Sentence embeddings
    - Document classification → Document embeddings (your IMDB task!)
    """)

# Footer navigation
st.divider()

st.markdown("### 📚 Related Pages")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🔤 Word Analysis", use_container_width=True):
        st.switch_page("pages/3_🔤_Word_Analysis.py")

with col2:
    if st.button("🔬 Model Comparison", use_container_width=True):
        st.switch_page("pages/2_🔬_Model_Comparison.py")

with col3:
    if st.button("🧪 Failure Analysis", use_container_width=True):
        st.switch_page("pages/6_🧪_Failure_Analysis.py")

with col4:
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("Home.py")
