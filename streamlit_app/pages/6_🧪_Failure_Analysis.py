"""
Failure Analysis Page - Understanding why models fail
PhD-level insights for researchers
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PAGE_CONFIG, RESULTS_DIR, MODELS, MODEL_DISPLAY_NAMES
from utils.data_loader import (
    get_all_available_models,
    load_intrinsic_results,
    load_model_vectors
)

# Page configuration
st.set_page_config(**PAGE_CONFIG)

# Custom CSS
st.markdown("""
<style>
    .failure-case {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .success-case {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .hypothesis {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🧪 Failure Analysis")
st.markdown("**Understanding model limitations** through systematic error analysis")

st.divider()

# Introduction
with st.expander("📖 **Why Failure Analysis Matters**", expanded=True):
    st.markdown("""
    > "Success is a lousy teacher. It seduces smart people into thinking they can't lose."
    > — Bill Gates

    In machine learning, **understanding failures** is more valuable than celebrating successes.

    ### Goals of This Analysis:

    1. **Identify systematic failure patterns**
       - Which types of analogies fail consistently?
       - Why do certain linguistic patterns confuse models?

    2. **Understand root causes**
       - Data limitations (Text8 size, domain coverage)
       - Architectural constraints (context window, negative sampling)
       - Optimization issues (epochs, learning rate)

    3. **Improve future models**
       - What data would help? (More epochs, larger corpus, domain-specific text)
       - What architectures? (Sub-word models like FastText, contextual embeddings like BERT)

    4. **Set realistic expectations**
       - What can/cannot be achieved with static embeddings?
       - When to use modern alternatives (transformers, LLMs)?
    """)

st.divider()

# Load available models
available_models = get_all_available_models(RESULTS_DIR)

if not available_models:
    st.error("❌ No trained models found. Please train models first!")
    st.stop()

# Model selection for analysis
selected_model = st.selectbox(
    "Select Model to Analyze",
    available_models,
    format_func=lambda x: MODEL_DISPLAY_NAMES.get(x, x.upper()),
    key="failure_model_select"
)

st.divider()

# Load intrinsic results
intrinsic_results = load_intrinsic_results(selected_model, RESULTS_DIR)

if not intrinsic_results or "samples" not in intrinsic_results:
    st.warning(f"⚠️ No intrinsic evaluation results found for {selected_model}")
    st.info("Run intrinsic evaluation: `python intrinsic/evaluate.py`")
    st.stop()

# Extract analogy samples
analogy_samples = intrinsic_results["samples"].get("analogy_add", [])

if not analogy_samples:
    st.warning("No analogy samples found in evaluation results")
    st.stop()

# Categorize failures
failures = [s for s in analogy_samples if not s.get("ok", False)]
successes = [s for s in analogy_samples if s.get("ok", False)]

# Overview metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Analogies",
        len(analogy_samples),
        help="Number of analogy questions tested"
    )

with col2:
    st.metric(
        "Failures",
        len(failures),
        delta=f"-{len(failures)/len(analogy_samples)*100:.0f}%",
        delta_color="inverse"
    )

with col3:
    st.metric(
        "Successes",
        len(successes),
        delta=f"+{len(successes)/len(analogy_samples)*100:.0f}%"
    )

st.divider()

# Success cases (brief)
st.markdown("## ✅ Success Cases")
st.markdown("*What the model got right*")

for i, case in enumerate(successes):
    q = case["q"]
    pred = case["pred"]

    with st.container(border=True):
        st.markdown(f"""
        <div class="success-case">
            <strong>Analogy {i+1}:</strong> <code>{q[0]} - {q[1]} + {q[2]} = ?</code><br>
            <strong>Expected:</strong> <code>{q[3]}</code> | <strong>Predicted:</strong> <code>{pred}</code> ✅
        </div>
        """, unsafe_allow_html=True)

if not successes:
    st.warning("⚠️ No successful analogies found")

st.divider()

# Failure cases (detailed)
st.markdown("## ❌ Failure Cases")
st.markdown("*Deep dive into what went wrong*")

if not failures:
    st.success("🎉 No failures! (Unlikely with Text8 training)")
else:
    # Categorize failures by type
    failure_categories = {
        "Geographical": [],
        "Morphological (Comparatives)": [],
        "Morphological (Plurals)": [],
        "Morphological (Verb Forms)": [],
        "Relational (Family)": [],
        "Other": []
    }

    for case in failures:
        q = case["q"]

        # Categorize based on keywords
        if any(word in ["paris", "london", "france", "england", "city", "country"] for word in q):
            failure_categories["Geographical"].append(case)
        elif any(word in ["good", "better", "bad", "worse", "big", "bigger", "small", "smaller"] for word in q):
            failure_categories["Morphological (Comparatives)"].append(case)
        elif any(word in ["cat", "cats", "dog", "dogs"] for word in q):
            failure_categories["Morphological (Plurals)"].append(case)
        elif any(word in ["walk", "walking", "swim", "swimming"] for word in q):
            failure_categories["Morphological (Verb Forms)"].append(case)
        elif any(word in ["uncle", "aunt", "brother", "sister"] for word in q):
            failure_categories["Relational (Family)"].append(case)
        else:
            failure_categories["Other"].append(case)

    # Display by category
    for category, cases in failure_categories.items():
        if not cases:
            continue

        with st.expander(f"🔍 **{category}** ({len(cases)} failures)", expanded=(len(cases) > 0)):
            for i, case in enumerate(cases):
                q = case["q"]
                pred = case["pred"]
                expected = q[3]

                st.markdown(f"""
                <div class="failure-case">
                    <strong>Analogy {i+1}:</strong> <code>{q[0]} - {q[1]} + {q[2]} = ?</code><br>
                    <strong>Expected:</strong> <code>{expected}</code> | <strong>Predicted:</strong> <code>{pred}</code> ❌
                </div>
                """, unsafe_allow_html=True)

                # Hypothesis for why it failed
                st.markdown("**💡 Hypothesis:**")

                if category == "Geographical":
                    st.markdown("""
                    <div class="hypothesis">
                    <strong>Root Cause:</strong> Lack of geographical context in Text8<br><br>
                    Text8 is Wikipedia text, but may not have enough explicit "X is the capital of Y" patterns.
                    The model learns word associations, but <strong>geographical relationships require world knowledge</strong>
                    that's underrepresented in the training corpus.<br><br>
                    <strong>Fix:</strong> Train on larger corpus (Wikipedia full, news articles) or use knowledge-enhanced embeddings.
                    </div>
                    """, unsafe_allow_html=True)

                elif category == "Morphological (Comparatives)":
                    st.markdown("""
                    <div class="hypothesis">
                    <strong>Root Cause:</strong> Morphological similarity confuses skip-gram<br><br>
                    Words like "better", "bigger", "shorter", "smaller" all end in "-er" and appear in similar contexts
                    (comparative adjectives). The model captures this <strong>surface similarity</strong> but fails to distinguish
                    the <strong>semantic direction</strong> (good→better vs bad→worse).<br><br>
                    <strong>Fix:</strong> Use character-aware models (FastText) or train longer with more negative samples
                    to distinguish subtle semantic differences.
                    </div>
                    """, unsafe_allow_html=True)

                elif category == "Morphological (Plurals)":
                    st.markdown("""
                    <div class="hypothesis">
                    <strong>Root Cause:</strong> Competing word associations<br><br>
                    "Cat/cats" and "dog/dogs" are strongly associated with many other words in the semantic space.
                    The predicted word ("<code>{pred}</code>") likely appears in similar contexts to "dog" but doesn't
                    capture the <strong>+plural</strong> transformation.<br><br>
                    <strong>Fix:</strong> Character-level models (FastText) excel at morphology by learning sub-word patterns.
                    </div>
                    """.format(pred=pred), unsafe_allow_html=True)

                elif category == "Morphological (Verb Forms)":
                    st.markdown("""
                    <div class="hypothesis">
                    <strong>Root Cause:</strong> Sparse training signal for verb morphology<br><br>
                    "-ing" forms (gerunds) are common, but the <strong>parallel structure</strong> (walk:walking :: swim:swimming)
                    requires the model to learn a general transformation rule. With only 5 epochs on Text8, the model
                    hasn't seen enough examples to generalize this pattern.<br><br>
                    <strong>Fix:</strong> Train longer (20+ epochs), use larger corpus, or try FastText for sub-word awareness.
                    </div>
                    """, unsafe_allow_html=True)

                elif category == "Relational (Family)":
                    st.markdown("""
                    <div class="hypothesis">
                    <strong>Root Cause:</strong> Symmetric vs asymmetric relationships<br><br>
                    Family relationships have complex semantics: "uncle" and "aunt" are gender opposites, but
                    "brother" and "sister" are also gender opposites with different generational meaning.
                    The model predicts "cousin" (a related family term) but fails to capture the <strong>exact semantic axis</strong>.<br><br>
                    <strong>Fix:</strong> More training data with explicit family relationship contexts, or use knowledge graphs.
                    </div>
                    """, unsafe_allow_html=True)

                st.divider()

st.divider()

# Systematic analysis
st.markdown("## 📊 Failure Pattern Summary")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Failure Categories")
    category_counts = {k: len(v) for k, v in failure_categories.items() if v}
    if category_counts:
        df_categories = pd.DataFrame([
            {"Category": k, "Count": v}
            for k, v in category_counts.items()
        ]).sort_values("Count", ascending=False)
        st.dataframe(df_categories, use_container_width=True, hide_index=True)

with col2:
    st.markdown("### Top Root Causes")
    st.markdown("""
    1. **Limited training data** (17M tokens, 5 epochs)
    2. **Domain coverage** (Wikipedia lacks certain contexts)
    3. **No sub-word awareness** (pure word-level embeddings)
    4. **Context window size** (5 words may miss long-range dependencies)
    5. **Optimization** (5 negative samples, 0.025 LR may be suboptimal)
    """)

st.divider()

# Recommendations
st.markdown("## 🎯 Recommendations")

tab1, tab2, tab3 = st.tabs(["📚 Data", "🏗️ Architecture", "⚙️ Training"])

with tab1:
    st.markdown("### Data Improvements")
    st.markdown("""
    #### 1. Larger Corpus
    - **Current**: Text8 (17M tokens)
    - **Recommended**: Full Wikipedia (2B+ tokens) or CommonCrawl

    #### 2. Domain-Specific Data
    - For geographical analogies: Add news articles, travel guides
    - For morphology: Add grammatically rich text (literature, academic papers)

    #### 3. Preprocessing
    - Keep more rare words (lower min_count from 5 to 3)
    - Add sub-word tokenization for morphology

    #### 4. Data Augmentation
    - Synthetic analogy generation
    - Back-translation for paraphrases
    """)

with tab2:
    st.markdown("### Architectural Improvements")
    st.markdown("""
    #### 1. Sub-word Models (FastText)
    - **Why**: Handles morphology (plurals, verb forms) better
    - **How**: Character n-grams (3-6 chars)
    - **Trade-off**: Slower training, larger memory

    #### 2. Larger Embeddings
    - **Current**: 100-300 dims
    - **Recommended**: 300-500 dims for richer representations

    #### 3. Contextual Embeddings (BERT, GPT)
    - **Why**: Captures word sense disambiguation
    - **How**: Pre-trained transformers
    - **Trade-off**: Requires GPU, slower inference

    #### 4. Hybrid Approaches
    - Combine static (GloVe) + contextual (BERT)
    - Use static for efficiency, contextual for accuracy
    """)

with tab3:
    st.markdown("### Training Improvements")
    st.markdown("""
    #### 1. More Epochs
    - **Current**: 5 epochs
    - **Recommended**: 20-50 epochs with early stopping

    #### 2. Better Negative Sampling
    - **Current**: 5 negative samples
    - **Recommended**: 10-15 for rare words

    #### 3. Learning Rate Schedule
    - **Current**: Fixed 0.025
    - **Recommended**: Decay from 0.025 → 0.0001

    #### 4. Larger Context Window
    - **Current**: 5 words
    - **Recommended**: 10 words for long-range dependencies

    #### 5. Better Evaluation
    - Use standard benchmarks (Google analogies, SimLex-999)
    - Cross-validation on downstream tasks
    """)

st.divider()

# Realistic expectations
st.markdown("## 🎓 Setting Realistic Expectations")

with st.container(border=True):
    st.markdown("""
    ### What Static Embeddings CAN Do:

    ✅ Capture word associations (king → queen, man → woman)

    ✅ Measure semantic similarity (cat ≈ dog, car ≈ vehicle)

    ✅ Transfer learning for downstream tasks (sentiment, NER)

    ✅ Efficient inference (no GPU needed, fast lookups)

    ### What Static Embeddings CANNOT Do:

    ❌ **Handle polysemy** (bank = financial institution vs river bank)
       - *Why*: One vector per word, no context awareness

    ❌ **Understand compositional semantics** (not good ≠ bad)
       - *Why*: Negation requires syntactic understanding

    ❌ **Capture world knowledge** (Paris is capital of France)
       - *Why*: Limited to distributional statistics, not factual knowledge

    ❌ **Handle out-of-vocabulary (OOV) words** (new slang, typos)
       - *Why*: No sub-word awareness (except FastText)

    ### When to Use Modern Alternatives:

    - **BERT/RoBERTa**: When context matters (polysemy, long documents)
    - **GPT-3/4**: When world knowledge is critical (QA, reasoning)
    - **Sentence-BERT**: For semantic search over sentences/paragraphs
    - **Domain-specific LLMs**: For specialized tasks (medical, legal)
    """)

st.divider()

# Interactive failure investigation
st.markdown("## 🔬 Interactive Failure Investigation")
st.markdown("Investigate specific analogies in detail")

# Load model vectors for detailed analysis
model_data = load_model_vectors(selected_model, RESULTS_DIR)

if model_data:
    embeddings, word2id = model_data

    # Select a failure case
    failure_options = [
        f"{c['q'][0]} - {c['q'][1]} + {c['q'][2]} = ? (expected: {c['q'][3]}, got: {c['pred']})"
        for c in failures
    ]

    if failure_options:
        selected_failure = st.selectbox(
            "Select a failure case to investigate:",
            range(len(failure_options)),
            format_func=lambda i: failure_options[i],
            key="failure_investigation_select"
        )

        case = failures[selected_failure]
        q = case["q"]
        a, b, c, expected = q
        predicted = case["pred"]

        st.markdown(f"### Analyzing: `{a} - {b} + {c} = ?`")

        # Compute the analogy
        if a in word2id and b in word2id and c in word2id:
            # Get vectors
            v_a = embeddings[word2id[a]]
            v_b = embeddings[word2id[b]]
            v_c = embeddings[word2id[c]]

            # Compute target vector
            target = v_a - v_b + v_c

            # Normalize
            target_norm = target / (np.linalg.norm(target) + 1e-10)
            E_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)

            # Compute similarities
            sims = E_norm @ target_norm

            # Get top 20 candidates
            top_k = 20
            top_idx = np.argsort(-sims)[:top_k]

            # Exclude input words
            id2word = {v: k for k, v in word2id.items()}
            exclude = {a, b, c}
            top_words = [(id2word[i], sims[i]) for i in top_idx if id2word[i] not in exclude][:15]

            # Display top predictions
            st.markdown("#### 🔝 Top 15 Predictions")

            predictions_data = []
            for rank, (word, score) in enumerate(top_words, 1):
                is_expected = "✅ **EXPECTED**" if word == expected else ""
                is_predicted = "⚠️ **MODEL CHOSE**" if word == predicted else ""
                marker = f"{is_expected} {is_predicted}".strip()

                predictions_data.append({
                    "Rank": rank,
                    "Word": word,
                    "Similarity": f"{score:.4f}",
                    "Note": marker
                })

            df_predictions = pd.DataFrame(predictions_data)
            st.dataframe(df_predictions, use_container_width=True, hide_index=True)

            # Analysis
            st.markdown("#### 🔍 Analysis")

            # Where does expected word rank?
            expected_rank = None
            for rank, (word, score) in enumerate(top_words, 1):
                if word == expected:
                    expected_rank = rank
                    break

            if expected_rank:
                st.success(f"✅ Expected word **'{expected}'** appears at rank **{expected_rank}**")
                st.markdown("*The model has the right answer in its top predictions, but chose a different word.*")
            else:
                st.error(f"❌ Expected word **'{expected}'** not in top 15 predictions")
                st.markdown("*The model completely missed the expected answer - this is a hard failure.*")

            # Similarity to input words
            st.markdown("#### 📏 Similarity of Top Prediction to Input Words")

            if predicted in word2id:
                v_pred = embeddings[word2id[predicted]]
                v_pred_norm = v_pred / (np.linalg.norm(v_pred) + 1e-10)

                v_a_norm = v_a / (np.linalg.norm(v_a) + 1e-10)
                v_b_norm = v_b / (np.linalg.norm(v_b) + 1e-10)
                v_c_norm = v_c / (np.linalg.norm(v_c) + 1e-10)

                sim_a = np.dot(v_pred_norm, v_a_norm)
                sim_b = np.dot(v_pred_norm, v_b_norm)
                sim_c = np.dot(v_pred_norm, v_c_norm)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"Similarity to '{a}'", f"{sim_a:.4f}")
                with col2:
                    st.metric(f"Similarity to '{b}'", f"{sim_b:.4f}")
                with col3:
                    st.metric(f"Similarity to '{c}'", f"{sim_c:.4f}")

                # Interpretation
                if max(sim_a, sim_b, sim_c) > 0.5:
                    st.warning(f"⚠️ Predicted word is too similar to input words - model is choosing a related word rather than solving the analogy.")
        else:
            st.error("❌ Some input words not in vocabulary")

st.divider()

# Footer navigation
st.markdown("### 📚 Related Pages")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔬 Model Comparison", use_container_width=True):
        st.switch_page("pages/2_🔬_Model_Comparison.py")

with col2:
    if st.button("⚙️ Technical Details", use_container_width=True):
        st.switch_page("pages/7_⚙️_Technical_Details.py")

with col3:
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("Home.py")
