"""
Model Comparison Page - Detailed benchmarks and performance metrics
For data scientists and analysts
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PAGE_CONFIG, RESULTS_DIR, MODELS, MODEL_DISPLAY_NAMES, COLORS
from utils.data_loader import (
    get_all_available_models,
    load_intrinsic_results,
    load_downstream_results
)

# Page configuration
st.set_page_config(**PAGE_CONFIG)

# Custom CSS
st.markdown("""
<style>
    .metric-box {
        padding: 15px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .winner-badge {
        background-color: #ffd700;
        color: #000;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🔬 Model Comparison")
st.markdown("**Comprehensive performance analysis** for data scientists and analysts")

st.divider()

# Load data
available_models = get_all_available_models(RESULTS_DIR)

if not available_models:
    st.error("❌ No trained models found. Please train models first!")
    st.stop()

downstream_results = load_downstream_results(RESULTS_DIR)

# Overview metrics
st.markdown("## 📊 Performance Overview")

# Collect all intrinsic results
intrinsic_data = []
for model in available_models:
    result = load_intrinsic_results(model, RESULTS_DIR)
    if result:
        intrinsic_data.append({
            "Model": MODEL_DISPLAY_NAMES.get(model, model.upper()),
            "Analogy Accuracy": f"{result.get('analogy_accuracy', 0) * 100:.1f}%",
            "Correct": result.get('analogy_correct', 0),
            "Total": result.get('analogy_total', 20),
        })

if intrinsic_data:
    st.markdown("### 🎯 Intrinsic Evaluation (Word Analogies)")
    st.markdown("*How well does the model understand word relationships?*")
    df_intrinsic = pd.DataFrame(intrinsic_data)
    st.dataframe(df_intrinsic, use_container_width=True, hide_index=True)

    st.info("""
    **Note**: Analogy accuracy around 25% is **normal** for models trained on Text8 (17M tokens, 5 epochs).
    Text8 is Wikipedia text, so it lacks domain-specific knowledge (e.g., geography, celebrities).
    """)

st.divider()

# Downstream results - comprehensive view
st.markdown("## 🎬 Downstream Evaluation (IMDB Sentiment Classification)")
st.markdown("*How well do embeddings transfer to real-world tasks?*")

if not downstream_results:
    st.warning("⚠️ No downstream evaluation results found.")
    st.info("Run downstream evaluation: `python downstream/evaluate_sentiment.py`")
else:
    # Explain the configurations
    with st.expander("ℹ️ **Understanding Evaluation Configurations**", expanded=False):
        st.markdown("""
        Each model is evaluated with **4 different configurations**:

        ### Pooling Strategies (2 options):
        1. **Average Pooling**: Simple mean of word vectors
        2. **TF-IDF Weighted Pooling**: Weight words by informativeness

        ### Classifiers (2 options):
        1. **Logistic Regression**: Linear classifier (baseline)
        2. **MLP**: Multi-layer perceptron (non-linear)

        **Total**: 2 pooling × 2 classifiers = **4 configurations per model**
        """)

    # Collect all configurations
    all_results = []
    for model_name in available_models:
        if model_name not in downstream_results:
            continue

        configs = downstream_results[model_name]
        for config_name, result in configs.items():
            all_results.append({
                "Model": MODEL_DISPLAY_NAMES.get(model_name, model_name.upper()),
                "Pooling": result.get("pooling", "").upper(),
                "Classifier": result.get("classifier", "").upper(),
                "Test Acc": f"{result.get('test_accuracy', 0) * 100:.2f}%",
                "Test Acc (raw)": result.get('test_accuracy', 0),  # For sorting
            })

    if all_results:
        df_all = pd.DataFrame(all_results)

        # Sort by test accuracy
        df_all_sorted = df_all.sort_values("Test Acc (raw)", ascending=False)
        df_display = df_all_sorted.drop(columns=["Test Acc (raw)"])

        # Highlight best
        best_config = df_all_sorted.iloc[0]

        st.markdown(f"""
        ### 🏆 Best Configuration
        **Model**: {best_config['Model']} | **Pooling**: {best_config['Pooling']} | **Classifier**: {best_config['Classifier']}

        **Test Accuracy**: {best_config['Test Acc']}
        """)

        st.markdown("### 📋 All Configurations (Ranked by Test Accuracy)")
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        st.divider()

        # Analysis by pooling strategy
        st.markdown("## 🔍 Analysis: Pooling Strategy Impact")

        pooling_comparison = []
        for model_name in available_models:
            if model_name not in downstream_results:
                continue

            configs = downstream_results[model_name]

            # Find avg vs tfidf for same classifier (use logistic as baseline)
            avg_result = configs.get("avg_logistic")
            tfidf_result = configs.get("tfidf_logistic")

            if avg_result and tfidf_result:
                avg_acc = avg_result.get("test_accuracy", 0) * 100
                tfidf_acc = tfidf_result.get("test_accuracy", 0) * 100
                diff = tfidf_acc - avg_acc

                pooling_comparison.append({
                    "Model": MODEL_DISPLAY_NAMES.get(model_name, model_name.upper()),
                    "AVG Pooling": f"{avg_acc:.2f}%",
                    "TF-IDF Pooling": f"{tfidf_acc:.2f}%",
                    "Difference": f"{diff:+.2f}%",
                    "Winner": "TF-IDF" if diff > 0 else ("AVG" if diff < 0 else "Tie"),
                    "diff_raw": diff
                })

        if pooling_comparison:
            df_pooling = pd.DataFrame(pooling_comparison).drop(columns=["diff_raw"])
            st.markdown("*Comparing AVG vs TF-IDF pooling (using Logistic Regression classifier)*")
            st.dataframe(df_pooling, use_container_width=True, hide_index=True)

            # Summary insights
            avg_wins = sum(1 for p in pooling_comparison if p["diff_raw"] < 0)
            tfidf_wins = sum(1 for p in pooling_comparison if p["diff_raw"] > 0)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("AVG Wins", avg_wins)
            with col2:
                st.metric("TF-IDF Wins", tfidf_wins)
            with col3:
                avg_improvement = np.mean([p["diff_raw"] for p in pooling_comparison])
                st.metric("Avg TF-IDF Improvement", f"{avg_improvement:+.2f}%")

        st.divider()

        # Analysis by classifier
        st.markdown("## 🔍 Analysis: Classifier Impact")

        classifier_comparison = []
        for model_name in available_models:
            if model_name not in downstream_results:
                continue

            configs = downstream_results[model_name]

            # Find logistic vs mlp for same pooling (use avg as baseline)
            logistic_result = configs.get("avg_logistic")
            mlp_result = configs.get("avg_mlp")

            if logistic_result and mlp_result:
                logistic_acc = logistic_result.get("test_accuracy", 0) * 100
                mlp_acc = mlp_result.get("test_accuracy", 0) * 100
                diff = mlp_acc - logistic_acc

                classifier_comparison.append({
                    "Model": MODEL_DISPLAY_NAMES.get(model_name, model_name.upper()),
                    "Logistic Regression": f"{logistic_acc:.2f}%",
                    "MLP": f"{mlp_acc:.2f}%",
                    "Difference": f"{diff:+.2f}%",
                    "Winner": "MLP" if diff > 0 else ("Logistic" if diff < 0 else "Tie"),
                    "diff_raw": diff
                })

        if classifier_comparison:
            df_classifier = pd.DataFrame(classifier_comparison).drop(columns=["diff_raw"])
            st.markdown("*Comparing Logistic Regression vs MLP (using AVG pooling)*")
            st.dataframe(df_classifier, use_container_width=True, hide_index=True)

            # Summary insights
            logistic_wins = sum(1 for c in classifier_comparison if c["diff_raw"] < 0)
            mlp_wins = sum(1 for c in classifier_comparison if c["diff_raw"] > 0)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Logistic Wins", logistic_wins)
            with col2:
                st.metric("MLP Wins", mlp_wins)
            with col3:
                avg_improvement = np.mean([c["diff_raw"] for c in classifier_comparison])
                st.metric("Avg MLP Improvement", f"{avg_improvement:+.2f}%")

        st.divider()

        # Model-by-model breakdown
        st.markdown("## 📦 Model-by-Model Breakdown")

        for model_name in available_models:
            if model_name not in downstream_results:
                continue

            with st.expander(f"🔍 **{MODEL_DISPLAY_NAMES.get(model_name, model_name.upper())}** Details"):
                configs = downstream_results[model_name]

                # Show all 4 configurations
                model_data = []
                for config_name, result in configs.items():
                    model_data.append({
                        "Configuration": config_name.replace("_", " + ").upper(),
                        "Train Acc": f"{result.get('train_accuracy', 0) * 100:.2f}%",
                        "Val Acc": f"{result.get('val_accuracy', 0) * 100:.2f}%",
                        "Test Acc": f"{result.get('test_accuracy', 0) * 100:.2f}%",
                    })

                if model_data:
                    df_model = pd.DataFrame(model_data)
                    st.dataframe(df_model, use_container_width=True, hide_index=True)

                    # Show best config for this model
                    best_config_idx = max(range(len(model_data)),
                                         key=lambda i: configs[list(configs.keys())[i]].get('test_accuracy', 0))
                    best_config_name = list(configs.keys())[best_config_idx]
                    best_result = configs[best_config_name]

                    st.success(f"✅ **Best**: {best_config_name.replace('_', ' + ').upper()} → {best_result.get('test_accuracy', 0) * 100:.2f}%")

st.divider()

# Training details
st.markdown("## ⚙️ Training Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📚 Dataset")
    st.markdown("""
    - **Training**: Text8
    - **Size**: 17M tokens (100MB)
    - **Vocab**: ~71K words (min_count=5)
    - **Domain**: Wikipedia text
    """)

with col2:
    st.markdown("### 🎯 Hyperparameters")
    st.markdown("""
    - **Embedding dim**: 100
    - **Window size**: 5
    - **Epochs**: 5
    - **Negative samples**: 5
    - **Learning rate**: 0.025 (Word2Vec)
    """)

with col3:
    st.markdown("### 🎬 Downstream Task")
    st.markdown("""
    - **Dataset**: IMDB reviews
    - **Train**: 15,000 samples
    - **Val**: 5,000 samples
    - **Test**: 5,000 samples
    - **Task**: Binary sentiment
    """)

st.divider()

# Key takeaways
st.markdown("## 💡 Key Takeaways")

with st.container(border=True):
    st.markdown("""
    ### For Data Scientists:

    1. **Best Overall Performance**:
       - Check the ranked table above to see which model + config achieves highest test accuracy
       - Typically GloVe with TF-IDF weighted pooling performs best

    2. **Pooling Strategy Matters**:
       - TF-IDF weighted pooling often outperforms simple averaging
       - Helps filter out stopwords and emphasize informative terms

    3. **Classifier Choice**:
       - MLP can provide non-linear decision boundaries
       - But logistic regression is faster and often competitive

    4. **Model Selection**:
       - **GloVe**: Best for downstream tasks (captures global co-occurrence)
       - **Skip-gram**: Good for rare words and analogies
       - **CBOW**: Faster training, smoother representations
       - **TF-IDF**: Surprisingly competitive despite being purely statistical

    5. **Trade-offs**:
       - Higher accuracy vs training time
       - Model complexity vs interpretability
    """)

st.divider()

# Footer navigation
st.markdown("### 📚 Related Pages")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📊 Executive Summary", use_container_width=True):
        st.switch_page("pages/1_📊_Executive_Summary.py")

with col2:
    if st.button("🧪 Failure Analysis", use_container_width=True):
        st.switch_page("pages/6_🧪_Failure_Analysis.py")

with col3:
    if st.button("⚙️ Technical Details", use_container_width=True):
        st.switch_page("pages/7_⚙️_Technical_Details.py")

with col4:
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("Home.py")
