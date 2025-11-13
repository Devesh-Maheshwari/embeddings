"""
Centralized configuration for the Streamlit app
"""
from pathlib import Path

# Paths
APP_DIR = Path(__file__).parent
ROOT_DIR = APP_DIR.parent
RESULTS_DIR = ROOT_DIR / "results"

# Model names
MODELS = ["skipgram", "cbow", "glove", "tfidf"]
MODEL_DISPLAY_NAMES = {
    "skipgram": "Skip-gram",
    "cbow": "CBOW",
    "glove": "GloVe",
    "tfidf": "TF-IDF"
}

# Color scheme for consistent visualization
COLORS = {
    "primary": "#1f77b4",
    "success": "#2ca02c",
    "warning": "#ff7f0e",
    "danger": "#d62728",
    "info": "#9467bd",
    # Model-specific colors
    "skipgram": "#1f77b4",  # Blue
    "cbow": "#ff7f0e",      # Orange
    "glove": "#2ca02c",     # Green
    "tfidf": "#d62728",     # Red
    "fasttext": "#9467bd",  # Purple
}

# Page configuration
PAGE_CONFIG = {
    "page_title": "Word Embeddings Research",
    "page_icon": "🎓",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}
