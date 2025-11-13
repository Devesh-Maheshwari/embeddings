"""
Word Analysis - Interactive word embedding exploration
Enhanced version with analogies and cross-model comparison
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import plotly.express as px

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PAGE_CONFIG, RESULTS_DIR, MODEL_DISPLAY_NAMES
from utils.data_loader import load_model_vectors, load_tfidf_data, get_all_available_models
from utils.similarity_demo import topk_neighbors

# Page config
st.set_page_config(**PAGE_CONFIG)

# Header
st.title("🔤 Word Embeddings Analysis")
st.caption("Explore word similarity, analogies, and semantic relationships")

# Load available models
available_models = get_all_available_models(RESULTS_DIR)

if not available_models:
    st.error("⚠️ No trained models found!")
    st.info("Train models first using the pretraining scripts.")
    st.stop()

# Model selection
st.markdown("## ⚙️ Configuration")
model_choice = st.selectbox(
    "Select Model",
    available_models,
    format_func=lambda x: MODEL_DISPLAY_NAMES.get(x, x),
    key="word_model_select",
    help="Choose which embedding model to explore"
)

# Load model data
if model_choice == "tfidf":
    data = load_tfidf_data(RESULTS_DIR)
    if data is None:
        st.error("Failed to load TF-IDF data")
        st.stop()
    idf, vocab = data
    id2tok = {i: t for t, i in vocab.items()}
else:
    data = load_model_vectors(model_choice, RESULTS_DIR)
    if data is None:
        st.error(f"Failed to load {model_choice} embeddings")
        st.stop()
    E, V = data

st.divider()

# TF-IDF specific view
if model_choice == "tfidf":
    st.markdown("## 🔤 TF-IDF Term Informativeness")

    st.info("""
    **What is IDF?** Inverse Document Frequency measures how **rare** a term is across documents.
    - **High IDF** = Rare, informative words (e.g., "quantum", "thermodynamics")
    - **Low IDF** = Common, less informative words (e.g., "the", "is", "a")
    """)

    top_n = st.slider(
        "Number of terms to display",
        min_value=10,
        max_value=100,
        value=25,
        step=5,
        key="tfidf_topn"
    )

    # Get top and bottom IDF scores
    top_idx = np.argsort(-idf)[:top_n]
    low_idx = np.argsort(idf)[:top_n]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ✅ Most Informative (High IDF)")
        st.caption("Rare words that carry specific meaning")

        top_df = pd.DataFrame({
            "Rank": range(1, top_n + 1),
            "Word": [id2tok[i] for i in top_idx],
            "IDF Score": idf[top_idx].round(4)
        })

        st.dataframe(
            top_df.style.background_gradient(cmap="Greens", subset=["IDF Score"]),
            use_container_width=True,
            hide_index=True
        )

    with col2:
        st.markdown("### ⚠️ Least Informative (Low IDF)")
        st.caption("Common words that appear in most documents")

        low_df = pd.DataFrame({
            "Rank": range(1, top_n + 1),
            "Word": [id2tok[i] for i in low_idx],
            "IDF Score": idf[low_idx].round(4)
        })

        st.dataframe(
            low_df.style.background_gradient(cmap="Reds", subset=["IDF Score"]),
            use_container_width=True,
            hide_index=True
        )

    st.warning("""
    ⚠️ **Note**: TF-IDF doesn't capture semantic similarity. Words like "king" and "queen"
    won't be considered similar unless they have similar IDF scores.

    For semantic similarity, use **Skip-gram**, **CBOW**, or **GloVe**.
    """)

else:
    # Word2Vec/GloVe features
    st.markdown("## 🔍 Word Similarity Search")

    col1, col2 = st.columns([3, 1])

    with col1:
        query = st.text_input(
            "Enter a word to find similar words",
            value="king",
            key="word_query",
            help="Type any word from the vocabulary"
        )

    with col2:
        k = st.slider(
            "Top-K neighbors",
            min_value=3,
            max_value=30,
            value=10,
            key="word_topk",
            help="Number of similar words to display"
        )

    # Check if word exists in vocabulary
    if query.lower() not in V:
        st.error(f"❌ Word **'{query}'** not found in vocabulary!")

        # Suggest similar words
        st.info("💡 **Suggestions**: Try common words like 'king', 'computer', 'good', 'city', 'music'")

        # Show vocab stats
        with st.expander("📊 Vocabulary Statistics"):
            st.markdown(f"""
            - **Total vocabulary size**: {len(V):,} words
            - **Embedding dimension**: {E.shape[1]}
            - **Model**: {MODEL_DISPLAY_NAMES.get(model_choice, model_choice)}
            """)
    else:
        # Find nearest neighbors
        nn = topk_neighbors(query.lower(), E, V, k)

        st.markdown(f"### 👑 Top {k} words similar to **{query}**")

        # Display as enhanced table
        if nn:
            df_nn = pd.DataFrame(nn, columns=["Word", "Cosine Similarity"])
            df_nn.insert(0, "Rank", range(1, len(nn) + 1))

            # Add similarity bar
            df_nn["Similarity"] = df_nn["Cosine Similarity"]

            st.dataframe(
                df_nn[["Rank", "Word", "Cosine Similarity"]].style.background_gradient(
                    cmap="Blues",
                    subset=["Cosine Similarity"]
                ),
                use_container_width=True,
                hide_index=True
            )

            # Visualize similarities
            fig = px.bar(
                df_nn,
                x="Word",
                y="Cosine Similarity",
                title=f"Cosine Similarity to '{query}'",
                color="Cosine Similarity",
                color_continuous_scale="Blues"
            )
            fig.update_layout(
                xaxis_title="Similar Words",
                yaxis_title="Cosine Similarity",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("No similar words found.")

    st.divider()

    # Word Analogies
    st.markdown("## 🧮 Word Analogy Calculator")

    st.info("""
    **Word analogies** test if embeddings capture semantic relationships:

    Formula: `vec(king) - vec(man) + vec(woman) ≈ vec(queen)`

    This tests if the model understands that "king is to man as queen is to woman".
    """)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        w1 = st.text_input("Word 1", value="king", key="analogy_w1")
    with col2:
        w2 = st.text_input("Word 2 (subtract)", value="man", key="analogy_w2")
    with col3:
        w3 = st.text_input("Word 3", value="woman", key="analogy_w3")
    with col4:
        k_analogy = st.number_input("Top-K results", min_value=1, max_value=10, value=5, key="analogy_k")

    # Quick examples - REALISTIC for Text8 training
    st.markdown("**💡 Try these examples (tested to work on Text8):**")
    example_col1, example_col2, example_col3 = st.columns(3)

    examples = [
        ("king", "man", "woman", "queen?"),
        ("he", "his", "she", "her?"),
        ("big", "bigger", "small", "smaller?")
    ]

    for idx, (e1, e2, e3, e4) in enumerate(examples):
        with [example_col1, example_col2, example_col3][idx]:
            if st.button(f"`{e1} - {e2} + {e3}` = {e4}", key=f"example_{idx}"):
                # Auto-fill the inputs
                st.session_state.analogy_w1 = e1
                st.session_state.analogy_w2 = e2
                st.session_state.analogy_w3 = e3
                st.rerun()

    st.warning("""
    ⚠️ **Note**: Analogies are HARD for models trained on small datasets!
    - Text8 is only 17M tokens (100MB)
    - Your model trained for just 5 epochs
    - **Expect ~25% accuracy** (2 out of 8 correct is normal!)

    Analogies like "paris-france+london=england" FAIL because Text8 lacks geographical context.
    **Stick to simple examples above** that are heavily present in Wikipedia text.
    """)

    if st.button("🔮 Compute Analogy", type="primary", use_container_width=True):
        # Check all words exist
        missing = []
        for word in [w1, w2, w3]:
            if word.lower() not in V:
                missing.append(word)

        if missing:
            st.error(f"❌ Words not in vocabulary: {', '.join(missing)}")
            st.info(f"**Available**: All words must be in the {len(V):,}-word vocabulary. Try common words like: man, woman, king, queen, big, small, he, she, his, her")
        else:
            # Compute analogy: w1 - w2 + w3
            try:
                v1 = E[V[w1.lower()]]
                v2 = E[V[w2.lower()]]
                v3 = E[V[w3.lower()]]

                # Analogy vector using 3CosAdd method
                target_vec = v1 - v2 + v3
                target_vec = target_vec / (np.linalg.norm(target_vec) + 1e-10)  # Normalize

                # Find closest words (excluding input words)
                # Normalize embeddings for cosine similarity
                E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-10)
                sims = E_norm @ target_vec

                order = np.argsort(-sims)

                id2tok_local = {i: t for t, i in V.items()}
                results = []
                exclude = {w1.lower(), w2.lower(), w3.lower()}

                for i in order:
                    word = id2tok_local.get(i)
                    if word and word not in exclude:
                        results.append((word, float(sims[i])))
                        if len(results) >= k_analogy * 3:  # Get more for debugging
                            break

                # Display results
                st.markdown(f"### 🎯 Results for: `{w1} - {w2} + {w3}` = ?")

                if results:
                    top_word, top_sim = results[0]

                    # Check if it makes sense
                    expected_words = {
                        ("king", "man", "woman"): "queen",
                        ("he", "his", "she"): "her",
                        ("big", "bigger", "small"): "smaller",
                        ("good", "better", "bad"): "worse"
                    }

                    expected = expected_words.get((w1.lower(), w2.lower(), w3.lower()))

                    if expected and top_word == expected:
                        st.success(f"✅ **CORRECT!** Top prediction: **{top_word}** (similarity: {top_sim:.4f})")
                        st.balloons()
                    elif expected:
                        st.warning(f"❌ **Predicted**: **{top_word}** (similarity: {top_sim:.4f})")
                        st.info(f"💡 **Expected**: {expected}")

                        # Check if expected is in top 10
                        top_words = [w for w, _ in results[:10]]
                        if expected in top_words:
                            rank = top_words.index(expected) + 1
                            st.info(f"📊 '{expected}' is ranked #{rank} in the results")
                    else:
                        st.info(f"✨ **Top prediction**: **{top_word}** (similarity: {top_sim:.4f})")

                    # Show top results
                    df_analogy = pd.DataFrame(
                        results[:k_analogy],
                        columns=["Predicted Word", "Cosine Similarity"]
                    )
                    df_analogy.insert(0, "Rank", range(1, len(df_analogy) + 1))

                    st.dataframe(
                        df_analogy.style.background_gradient(cmap="Purples", subset=["Cosine Similarity"]),
                        use_container_width=True,
                        hide_index=True
                    )

                    # Debugging info
                    with st.expander("🔍 **Debugging Info**"):
                        st.markdown(f"""
                        **Vector Arithmetic:**
                        - `vec({w1})` shape: {v1.shape}
                        - `vec({w2})` shape: {v2.shape}
                        - `vec({w3})` shape: {v3.shape}
                        - Target vector norm: {np.linalg.norm(target_vec):.4f}

                        **Top 15 predictions** (to see if expected answer appears):
                        """)

                        debug_df = pd.DataFrame(
                            results[:15],
                            columns=["Word", "Similarity"]
                        )
                        st.dataframe(debug_df, use_container_width=True, hide_index=True)

                        # Show which input word is closest to result
                        st.markdown("**Sanity check** - distances from result to input words:")
                        result_vec = E[V[top_word]]
                        result_vec_norm = result_vec / (np.linalg.norm(result_vec) + 1e-10)

                        for input_word in [w1, w2, w3]:
                            input_vec = E[V[input_word.lower()]]
                            input_vec_norm = input_vec / (np.linalg.norm(input_vec) + 1e-10)
                            sim_to_input = float(result_vec_norm @ input_vec_norm)
                            st.text(f"  sim('{top_word}', '{input_word}'): {sim_to_input:.4f}")

                else:
                    st.warning("No results found.")

            except Exception as e:
                st.error(f"Error computing analogy: {e}")
                import traceback
                with st.expander("Full error trace"):
                    st.code(traceback.format_exc())

    st.divider()

    # Cross-model comparison
    st.markdown("## 🔬 Compare Across Models")

    st.markdown("""
    See how different models represent the **same word**.
    Do Skip-gram and GloVe agree on what's similar to your query word?
    """)

    if len(available_models) > 1:
        compare_word = st.text_input(
            "Word to compare across models",
            value="king",
            key="compare_word"
        )

        compare_k = st.slider(
            "Top-K neighbors for comparison",
            min_value=3,
            max_value=10,
            value=5,
            key="compare_k"
        )

        if st.button("🔍 Compare Models", type="secondary", use_container_width=True):
            comparison_results = {}

            for model_name in available_models:
                if model_name == "tfidf":
                    continue  # Skip TF-IDF for semantic similarity

                model_data = load_model_vectors(model_name, RESULTS_DIR)
                if model_data:
                    E_m, V_m = model_data
                    if compare_word.lower() in V_m:
                        nn_m = topk_neighbors(compare_word.lower(), E_m, V_m, compare_k)
                        comparison_results[model_name] = [w for w, _ in nn_m]

            if comparison_results:
                st.markdown(f"### 📊 Top {compare_k} neighbors for **{compare_word}** across models:")

                # Create comparison dataframe
                max_len = max(len(v) for v in comparison_results.values())
                comp_df = pd.DataFrame({
                    MODEL_DISPLAY_NAMES.get(model, model): words + ["—"] * (max_len - len(words))
                    for model, words in comparison_results.items()
                })

                st.dataframe(comp_df, use_container_width=True, hide_index=True)

                # Analyze agreement
                if len(comparison_results) >= 2:
                    models_list = list(comparison_results.keys())
                    words_model1 = set(comparison_results[models_list[0]])
                    words_model2 = set(comparison_results[models_list[1]])

                    overlap = words_model1 & words_model2
                    overlap_pct = len(overlap) / compare_k * 100

                    if overlap_pct >= 60:
                        st.success(f"✅ **High agreement**: {overlap_pct:.0f}% overlap between {MODEL_DISPLAY_NAMES.get(models_list[0], models_list[0])} and {MODEL_DISPLAY_NAMES.get(models_list[1], models_list[1])}")
                        st.markdown(f"**Common words**: {', '.join(sorted(overlap))}")
                    elif overlap_pct >= 30:
                        st.info(f"📊 **Moderate agreement**: {overlap_pct:.0f}% overlap")
                    else:
                        st.warning(f"⚠️ **Low agreement**: Only {overlap_pct:.0f}% overlap - models have different semantic representations!")

            else:
                st.warning(f"Word '{compare_word}' not found in any model vocabulary.")

    else:
        st.info("Train more models to enable cross-model comparison.")

# Footer
st.divider()
st.markdown("### 💡 Learn More")

col1, col2 = st.columns(2)

with col1:
    if st.button("📝 Try Sentence Similarity", use_container_width=True):
        st.switch_page("pages/4_📝_Sentence_Analysis.py")

with col2:
    if st.button("🧪 Why Models Fail", use_container_width=True):
        st.switch_page("pages/6_🧪_Failure_Analysis.py")
