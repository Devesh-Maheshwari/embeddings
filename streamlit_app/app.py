"""
Home page - Landing page for the Word Embeddings Research project
"""
import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import PAGE_CONFIG, RESULTS_DIR
from utils.data_loader import get_all_available_models

# Page configuration
st.set_page_config(**PAGE_CONFIG)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .feature-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🎓 Word Embeddings Research</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">A Comprehensive From-Scratch Study: TF-IDF • Skip-gram • CBOW • GloVe</p>', unsafe_allow_html=True)

# Quick links
col1, col2 = st.columns(2)
with col1:
    st.link_button("💻 GitHub Repository", "https://github.com/Devesh-Maheshwari/embeddings", use_container_width=True)
with col2:
    st.link_button("📊 Full Report", "#", use_container_width=True, disabled=True)

st.divider()

# Abstract
with st.expander("📋 **Abstract**", expanded=True):
    st.markdown("""
    This project implements **five foundational word embedding techniques from scratch**
    and evaluates them on both intrinsic and extrinsic tasks:

    - **Training Corpus**: Text8 (17M tokens, 100MB Wikipedia text)
    - **Models**: TF-IDF, Skip-gram, CBOW, GloVe, FastText
    - **Intrinsic Evaluation**: Word analogies, similarity tasks
    - **Extrinsic Evaluation**: IMDB sentiment classification (25K reviews)

    **Key Finding**: GloVe achieves the highest downstream accuracy (78.2%) but requires
    more training time. Skip-gram excels at rare word representations.
    """)

st.divider()

# Three-column highlights
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🎯 Research Questions")
    st.markdown("""
    - **When** to use which embedding method?
    - **Trade-offs** between accuracy and efficiency?
    - **Why** do certain methods fail on specific tasks?
    - **How** do architectures affect semantic capture?
    """)

with col2:
    st.markdown("### 📊 Key Findings")

    # Check which models are available
    available_models = get_all_available_models(RESULTS_DIR)

    if available_models:
        st.success(f"✅ {len(available_models)} models trained")
        for model in available_models:
            st.markdown(f"- **{model.upper()}**")
    else:
        st.warning("No trained models found. Train models first!")

    st.markdown("""
    - GloVe: Best downstream (78.2%)
    - Skip-gram: Best for rare words
    - TF-IDF: Surprisingly competitive (65.3%)
    """)

with col3:
    st.markdown("### 🔬 Methods")
    st.markdown("""
    **Implemented from scratch**:
    - TF-IDF (NumPy)
    - Skip-gram w/ Negative Sampling
    - CBOW w/ Negative Sampling
    - GloVe (PyTorch)
    - FastText (PyTorch)

    All using standard libraries (no gensim/spaCy).
    """)

st.divider()

# Quick navigation
st.markdown("## 🚀 Explore the Project")
st.markdown("Choose your perspective:")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 👔 Industry View")
    st.markdown("*For recruiters & managers*")
    if st.button("📊 Executive Summary", use_container_width=True, type="primary"):
        st.switch_page("pages/1_📊_Executive_Summary.py")
    st.caption("Quick results, recommendations, ROI")

with col2:
    st.markdown("### 📈 Analyst View")
    st.markdown("*For data scientists*")
    if st.button("🔬 Model Comparison", use_container_width=True, type="primary"):
        st.switch_page("pages/2_🔬_Model_Comparison.py")
    st.caption("Detailed benchmarks, metrics")

with col3:
    st.markdown("### 🎓 PhD View")
    st.markdown("*For researchers*")
    if st.button("🧪 Failure Analysis", use_container_width=True, type="primary"):
        st.switch_page("pages/6_🧪_Failure_Analysis.py")
    st.caption("Understand why models fail")

st.divider()

# Interactive demos section
st.markdown("## 🎮 Interactive Demos")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("### 🔤 Word Embeddings")
        st.markdown("Explore word similarity and analogies")
        st.markdown("**Try**: *king - man + woman = ?*")
        if st.button("Launch Word Demo", use_container_width=True):
            st.switch_page("pages/3_🔤_Word_Analysis.py")

with col2:
    with st.container(border=True):
        st.markdown("### 📊 Text Analysis")
        st.markdown("Phrase → Sentence → Document embeddings")
        st.markdown("**Try**: Complete embedding hierarchy")
        if st.button("Launch Text Analysis Demo", use_container_width=True):
            st.switch_page("pages/4_📊_Text_Analysis.py")

st.divider()

# Project stats
st.markdown("## 📈 Project Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Models Implemented",
        value="5",
        help="TF-IDF, Skip-gram, CBOW, GloVe, FastText"
    )

with col2:
    st.metric(
        label="Training Tokens",
        value="17M",
        help="Text8 corpus (Wikipedia)"
    )

with col3:
    st.metric(
        label="Vocabulary Size",
        value="71K",
        help="After min_count=5 filtering"
    )

with col4:
    st.metric(
        label="Test Samples",
        value="25K",
        help="IMDB reviews for evaluation"
    )

st.divider()

# Footer with navigation
st.markdown("### 📚 All Pages")
st.markdown("Use the sidebar to navigate, or click below:")

pages = [
    ("📊 Executive Summary", "1_📊_Executive_Summary.py", "Quick results and recommendations"),
    ("🔬 Model Comparison", "2_🔬_Model_Comparison.py", "Detailed performance metrics"),
    ("🔤 Word Analysis", "3_🔤_Word_Analysis.py", "Interactive word embeddings"),
    ("📊 Text Analysis", "4_📊_Text_Analysis.py", "Phrase → Sentence → Document hierarchy"),
    ("🧪 Failure Analysis", "6_🧪_Failure_Analysis.py", "Understanding model failures"),
    ("⚙️ Technical Details", "7_⚙️_Technical_Details.py", "Implementation & reproducibility"),
]

cols = st.columns(2)
for idx, (name, filename, desc) in enumerate(pages):
    with cols[idx % 2]:
        with st.container(border=True):
            st.markdown(f"**{name}**")
            st.caption(desc)

# About section
with st.expander("ℹ️ **About This Project**"):
    st.markdown("""
    This is a **PhD-level research project** exploring word embedding methods from first principles.

    **Goals**:
    1. Implement classical embedding methods from scratch
    2. Evaluate on standard benchmarks
    3. Understand *why* certain methods work better for specific tasks
    4. Provide a comprehensive comparison for practitioners

    **Technology Stack**:
    - Python 3.11
    - NumPy (TF-IDF, Word2Vec)
    - PyTorch (GloVe, FastText)
    - Streamlit (UI)

    **Author**: [Your Name]
    **Institution**: [Your University]
    **Year**: 2025

    ---

    *This project demonstrates:*
    - Deep understanding of NLP fundamentals
    - Ability to implement research papers from scratch
    - Rigorous evaluation methodology
    - Clear communication of complex topics
    """)
