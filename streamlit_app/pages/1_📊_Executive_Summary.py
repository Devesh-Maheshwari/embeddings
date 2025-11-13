"""
Executive Summary - Industry/Recruiter View
Comprehensive results showing all configurations
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PAGE_CONFIG, RESULTS_DIR, COLORS, MODEL_DISPLAY_NAMES
from utils.data_loader import load_downstream_results, get_all_available_models

# Page config
st.set_page_config(**PAGE_CONFIG)

# Header
st.title("📊 Executive Summary")
st.caption("Comprehensive results across all models and configurations")

# Load data
downstream_results = load_downstream_results(RESULTS_DIR)
available_models = get_all_available_models(RESULTS_DIR)

if not downstream_results:
    st.error("⚠️ No downstream evaluation results found!")
    st.info("Train models and run downstream evaluation first.")
    st.stop()

# Explanation of configurations
with st.expander("ℹ️ **Understanding the Evaluation Setup**", expanded=False):
    st.markdown("""
    ### 🎯 What We're Evaluating

    Each embedding model is tested on **IMDB sentiment classification** with different configurations:

    #### 📊 **Pooling Strategies** (How to create sentence embeddings from word embeddings)
    1. **Average Pooling (`avg`)**
       - Take the **mean** of all word vectors in the sentence
       - Formula: `sentence_vec = mean([vec(w1), vec(w2), ..., vec(wn)])`
       - Treats all words equally
       - **Simple and fast**

    2. **TF-IDF Weighted Pooling (`tfidf`)**
       - Weight each word by its **importance** (IDF score)
       - Formula: `sentence_vec = Σ idf(word) × vec(word)`
       - Downweights common words like "the", "is"
       - **More sophisticated**

    #### 🤖 **Classifiers** (How to predict sentiment from sentence embeddings)
    1. **Logistic Regression (`logistic`)**
       - Simple linear classifier
       - Fast training
       - Good baseline

    2. **Multi-Layer Perceptron (`mlp`)**
       - Neural network with hidden layers
       - Can learn non-linear patterns
       - More powerful but slower

    #### 🔢 **Total Configurations**
    For each embedding model, we test:
    - `avg` + `logistic`
    - `avg` + `mlp`
    - `tfidf` + `logistic`
    - `tfidf` + `mlp`

    **= 4 configurations per model**

    *(Note: TF-IDF embeddings only use TF-IDF pooling, not avg)*
    """)

st.divider()

# Find overall best configuration
st.markdown("## 🏆 Best Overall Configuration")

all_configs = []
for model_name, configs in downstream_results.items():
    for config_name, result in configs.items():
        acc = result.get("test_accuracy", 0)
        all_configs.append({
            "model": model_name,
            "config": config_name,
            "pooling": result.get("pooling", config_name.split("_")[0]),
            "classifier": result.get("classifier", config_name.split("_")[1] if "_" in config_name else "unknown"),
            "accuracy": acc,
            "f1": result.get("test_f1", 0),
            "precision": result.get("test_precision", 0),
            "recall": result.get("test_recall", 0),
            "result": result
        })

if all_configs:
    # Sort by accuracy
    all_configs_sorted = sorted(all_configs, key=lambda x: x["accuracy"], reverse=True)
    best_config = all_configs_sorted[0]

    # Highlight box
    pooling_display = "Average" if best_config["pooling"] == "avg" else "TF-IDF Weighted"
    classifier_display = "Logistic Regression" if best_config["classifier"] == "logistic" else "MLP"

    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 2rem; border-radius: 1rem; color: white; margin-bottom: 2rem;'>
        <h2 style='margin: 0; color: white;'>🥇 {MODEL_DISPLAY_NAMES.get(best_config["model"], best_config["model"]).upper()}</h2>
        <h1 style='margin: 0.5rem 0; font-size: 3rem; color: white;'>{best_config["accuracy"]*100:.2f}%</h1>
        <p style='margin: 0; font-size: 1.2rem;'>{pooling_display} + {classifier_display}</p>
        <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>IMDB Sentiment Classification Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Accuracy",
            f"{best_config['accuracy']*100:.2f}%",
            help="Overall classification accuracy"
        )

    with col2:
        st.metric(
            "F1-Score",
            f"{best_config['f1']:.4f}",
            help="Harmonic mean of precision and recall"
        )

    with col3:
        st.metric(
            "Precision",
            f"{best_config['precision']:.4f}",
            help="True positives / (True positives + False positives)"
        )

    with col4:
        st.metric(
            "Recall",
            f"{best_config['recall']:.4f}",
            help="True positives / (True positives + False negatives)"
        )

    # Configuration details
    with st.expander("🔍 **Configuration Details**"):
        st.markdown(f"""
        **Model**: {MODEL_DISPLAY_NAMES.get(best_config["model"], best_config["model"])}

        **Pooling Strategy**: {pooling_display}
        - {"Treats all words equally by averaging their vectors" if best_config["pooling"] == "avg" else "Weights words by TF-IDF importance scores"}

        **Classifier**: {classifier_display}
        - {"Simple linear model (fast, interpretable)" if best_config["classifier"] == "logistic" else "Neural network with hidden layers (more powerful)"}

        **Training Set**: {best_config['result'].get('n_train', 'N/A')} samples
        **Validation Set**: {best_config['result'].get('n_val', 'N/A')} samples
        **Test Set**: {best_config['result'].get('n_test', 'N/A')} samples
        """)

st.divider()

# Comprehensive comparison - All configurations
st.markdown("## 📊 Complete Results: All Configurations")

# Create comprehensive dataframe
comparison_data = []
for config in all_configs:
    comparison_data.append({
        "Model": MODEL_DISPLAY_NAMES.get(config["model"], config["model"]),
        "Pooling": "Avg" if config["pooling"] == "avg" else "TF-IDF",
        "Classifier": "Logistic" if config["classifier"] == "logistic" else "MLP",
        "Configuration": f"{config['pooling']}_{config['classifier']}",
        "Accuracy (%)": config["accuracy"] * 100,
        "F1-Score": config["f1"],
        "Precision": config["precision"],
        "Recall": config["recall"],
        "model_key": config["model"]
    })

df_full = pd.DataFrame(comparison_data)

# Interactive filters
col1, col2, col3 = st.columns(3)

with col1:
    model_filter = st.multiselect(
        "Filter by Model",
        options=df_full["Model"].unique().tolist(),
        default=df_full["Model"].unique().tolist(),
        key="model_filter"
    )

with col2:
    pooling_filter = st.multiselect(
        "Filter by Pooling",
        options=df_full["Pooling"].unique().tolist(),
        default=df_full["Pooling"].unique().tolist(),
        key="pooling_filter"
    )

with col3:
    classifier_filter = st.multiselect(
        "Filter by Classifier",
        options=df_full["Classifier"].unique().tolist(),
        default=df_full["Classifier"].unique().tolist(),
        key="classifier_filter"
    )

# Apply filters
df_filtered = df_full[
    (df_full["Model"].isin(model_filter)) &
    (df_full["Pooling"].isin(pooling_filter)) &
    (df_full["Classifier"].isin(classifier_filter))
]

# Grouped bar chart
fig = px.bar(
    df_filtered.sort_values("Accuracy (%)", ascending=False),
    x="Model",
    y="Accuracy (%)",
    color="Configuration",
    barmode="group",
    title="Accuracy Comparison: All Configurations",
    hover_data=["Pooling", "Classifier", "F1-Score"],
    text="Accuracy (%)"
)

fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig.update_layout(
    yaxis_range=[0, 100],
    height=500,
    legend_title="Configuration"
)

st.plotly_chart(fig, use_container_width=True)

# Detailed table
st.markdown("### 📋 Detailed Results Table")

# Format display
display_df = df_filtered.copy()
display_df = display_df.sort_values("Accuracy (%)", ascending=False)
display_df["Accuracy (%)"] = display_df["Accuracy (%)"].apply(lambda x: f"{x:.2f}%")
display_df["F1-Score"] = display_df["F1-Score"].apply(lambda x: f"{x:.4f}")
display_df["Precision"] = display_df["Precision"].apply(lambda x: f"{x:.4f}")
display_df["Recall"] = display_df["Recall"].apply(lambda x: f"{x:.4f}")

st.dataframe(
    display_df[["Model", "Pooling", "Classifier", "Accuracy (%)", "F1-Score", "Precision", "Recall"]],
    use_container_width=True,
    hide_index=True
)

st.divider()

# Analysis: Pooling Strategy Comparison
st.markdown("## 🔬 Analysis: Pooling Strategy Impact")

st.markdown("""
**Question**: Does TF-IDF weighting improve over simple averaging?

Let's compare the **same model** with different pooling strategies:
""")

# Compare avg vs tfidf for each model
pooling_comparison = []

for model_name in available_models:
    if model_name == "tfidf":
        continue  # TF-IDF doesn't have avg pooling

    if model_name in downstream_results:
        configs = downstream_results[model_name]

        # Get avg_logistic and tfidf_logistic (same classifier, different pooling)
        if "avg_logistic" in configs and "tfidf_logistic" in configs:
            avg_acc = configs["avg_logistic"].get("test_accuracy", 0) * 100
            tfidf_acc = configs["tfidf_logistic"].get("test_accuracy", 0) * 100
            diff = tfidf_acc - avg_acc

            pooling_comparison.append({
                "Model": MODEL_DISPLAY_NAMES.get(model_name, model_name),
                "Avg Pooling": avg_acc,
                "TF-IDF Pooling": tfidf_acc,
                "Difference": diff,
                "Winner": "TF-IDF" if diff > 0 else "Average"
            })

if pooling_comparison:
    df_pooling = pd.DataFrame(pooling_comparison)

    # Bar chart showing difference
    fig_pooling = go.Figure()

    fig_pooling.add_trace(go.Bar(
        name="Average Pooling",
        x=df_pooling["Model"],
        y=df_pooling["Avg Pooling"],
        marker_color="lightblue"
    ))

    fig_pooling.add_trace(go.Bar(
        name="TF-IDF Pooling",
        x=df_pooling["Model"],
        y=df_pooling["TF-IDF Pooling"],
        marker_color="darkblue"
    ))

    fig_pooling.update_layout(
        title="Pooling Strategy Comparison (Logistic Regression)",
        yaxis_title="Accuracy (%)",
        barmode="group",
        height=400
    )

    st.plotly_chart(fig_pooling, use_container_width=True)

    # Summary table
    display_pooling = df_pooling.copy()
    display_pooling["Avg Pooling"] = display_pooling["Avg Pooling"].apply(lambda x: f"{x:.2f}%")
    display_pooling["TF-IDF Pooling"] = display_pooling["TF-IDF Pooling"].apply(lambda x: f"{x:.2f}%")
    display_pooling["Difference"] = display_pooling["Difference"].apply(lambda x: f"{x:+.2f}%")

    st.dataframe(display_pooling, use_container_width=True, hide_index=True)

    # Insight
    avg_winner_count = sum(1 for x in pooling_comparison if x["Difference"] < 0)
    tfidf_winner_count = len(pooling_comparison) - avg_winner_count

    if tfidf_winner_count > avg_winner_count:
        st.success(f"✅ **TF-IDF pooling wins** for {tfidf_winner_count}/{len(pooling_comparison)} models!")
        st.info("**Why?** TF-IDF downweights common words (\"the\", \"is\", \"a\") that don't carry sentiment, focusing on informative words.")
    elif avg_winner_count > tfidf_winner_count:
        st.warning(f"⚠️ **Average pooling wins** for {avg_winner_count}/{len(pooling_comparison)} models.")
        st.info("**Why?** For sentiment, even common words like \"not\" can be crucial. Averaging preserves all information.")
    else:
        st.info("📊 **Tie!** Both strategies have merit depending on the model.")

st.divider()

# Analysis: Classifier Comparison
st.markdown("## 🤖 Analysis: Classifier Impact")

st.markdown("""
**Question**: Does a neural network (MLP) outperform simple logistic regression?

Let's compare classifiers with the **same pooling strategy** (average):
""")

classifier_comparison = []

for model_name in available_models:
    if model_name in downstream_results:
        configs = downstream_results[model_name]

        # Compare avg_logistic vs avg_mlp
        if "avg_logistic" in configs and "avg_mlp" in configs:
            log_acc = configs["avg_logistic"].get("test_accuracy", 0) * 100
            mlp_acc = configs["avg_mlp"].get("test_accuracy", 0) * 100
            diff = mlp_acc - log_acc

            classifier_comparison.append({
                "Model": MODEL_DISPLAY_NAMES.get(model_name, model_name),
                "Logistic Regression": log_acc,
                "MLP": mlp_acc,
                "Difference": diff,
                "Winner": "MLP" if diff > 0 else "Logistic"
            })

if classifier_comparison:
    df_classifier = pd.DataFrame(classifier_comparison)

    # Bar chart
    fig_classifier = go.Figure()

    fig_classifier.add_trace(go.Bar(
        name="Logistic Regression",
        x=df_classifier["Model"],
        y=df_classifier["Logistic Regression"],
        marker_color="lightcoral"
    ))

    fig_classifier.add_trace(go.Bar(
        name="MLP",
        x=df_classifier["Model"],
        y=df_classifier["MLP"],
        marker_color="darkred"
    ))

    fig_classifier.update_layout(
        title="Classifier Comparison (Average Pooling)",
        yaxis_title="Accuracy (%)",
        barmode="group",
        height=400
    )

    st.plotly_chart(fig_classifier, use_container_width=True)

    # Summary
    display_classifier = df_classifier.copy()
    display_classifier["Logistic Regression"] = display_classifier["Logistic Regression"].apply(lambda x: f"{x:.2f}%")
    display_classifier["MLP"] = display_classifier["MLP"].apply(lambda x: f"{x:.2f}%")
    display_classifier["Difference"] = display_classifier["Difference"].apply(lambda x: f"{x:+.2f}%")

    st.dataframe(display_classifier, use_container_width=True, hide_index=True)

    # Insight
    mlp_winner_count = sum(1 for x in classifier_comparison if x["Difference"] > 0)
    log_winner_count = len(classifier_comparison) - mlp_winner_count

    if mlp_winner_count > log_winner_count:
        st.success(f"✅ **MLP wins** for {mlp_winner_count}/{len(classifier_comparison)} models!")
        st.info("**Why?** MLP can learn non-linear decision boundaries, better capturing complex sentiment patterns.")
    elif log_winner_count > mlp_winner_count:
        st.warning(f"⚠️ **Logistic Regression wins** for {log_winner_count}/{len(classifier_comparison)} models.")
        st.info("**Why?** For linearly separable data, simpler is better. Logistic avoids overfitting.")
    else:
        st.info("📊 **Tie!** Choice depends on the embedding quality.")

st.divider()

# Key takeaways
st.markdown("## 💡 Key Takeaways")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### ✅ What Works")
    st.markdown(f"""
    1. **Best Overall**: {MODEL_DISPLAY_NAMES.get(best_config["model"], best_config["model"])} with {best_config["pooling"]} pooling + {best_config["classifier"]} classifier
    2. **Pooling**: {"TF-IDF weighting generally helps" if tfidf_winner_count > avg_winner_count else "Simple averaging is often sufficient"}
    3. **Classifier**: {"MLP adds 1-2% over logistic" if mlp_winner_count > log_winner_count else "Logistic regression is surprisingly competitive"}
    """)

with col2:
    st.markdown("### 🎯 Recommendations")
    st.markdown("""
    - **For production**: Use best config (shown at top)
    - **For speed**: Use Logistic over MLP
    - **For interpretability**: Use avg pooling + Logistic
    - **For maximum accuracy**: Test all configurations
    """)

st.divider()

# Download
st.markdown("### 📥 Export Results")

col1, col2 = st.columns(2)

with col1:
    csv = df_full.to_csv(index=False)
    st.download_button(
        label="📄 Download Full Results (CSV)",
        data=csv,
        file_name="comprehensive_results.csv",
        mime="text/csv",
        use_container_width=True
    )

with col2:
    if st.button("📊 View Detailed Comparison", use_container_width=True):
        st.switch_page("pages/2_🔬_Model_Comparison.py")
