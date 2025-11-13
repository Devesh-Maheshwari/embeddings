# Simplified Dataset Loading for Embedding Pretraining Comparison
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import requests
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from typing import List, Tuple,Dict, Any
import zipfile




class EmbeddingBenchmarkDatasets:
    """
    Simplified dataset loader focusing on pretraining comparison.
    Uses WikiText-2 for pretraining and IMDB for evaluation.
    """
    
    def __init__(self, data_dir='datasets'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "processed").mkdir(parents=True, exist_ok=True)

    def load_text8(self, chunk_size: int = 256, cache_name: str = "text8_pretraining")->List[List[str]]:
        """
        Download + load text8, then split into fixed-length token chunks.
        Returns: List[List[str]] (each inner list ~chunk_size tokens).
        Caches to datasets/processed/text8_pretraining.pkl
        """
        cached = self.load_processed_dataset(cache_name)
        if cached is not None:
            return cached
        if requests is None:
            raise RuntimeError(
                "The 'requests' package is required to download text8. "
                "Install it via: pip install requests"
            )
        print("Loading Text8 for pretraining...")
        url = "http://mattmahoney.net/dc/text8.zip"
        raw_txt_path = self.data_dir / "text8"
        if not raw_txt_path.exists():
            print("Downloading Text8 dataset...")
            response = requests.get(url)
            zip_path = self.data_dir / "text8.zip"
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
        with open(raw_txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        tokens = text.split()
        print(f"Total tokens in Text8: {len(tokens):,}")
        #split into chunks
        sentences: List[List[str]] = []
        for i in range(0, len(tokens), chunk_size):
            seg = tokens[i:i + chunk_size]
            if len(seg) >= 4:
                sentences.append(seg)
        print(f"[text8] Pseudo-sentences (chunk={chunk_size}): {len(sentences):,}")
        self.save_processed_dataset(sentences, cache_name)
        return sentences

    def load_wikitext2(self, cache_name: str = "wikitext2_pretraining") -> List[str]:
        """
        Load WikiText-2 dataset for pretraining all embedding models.
        Perfect size for Mac M1: ~12MB, 2M tokens, 33K vocab.
        """
        print("Loading WikiText-2 for pretraining...")
        
        try:
            # Try using datasets library (Hugging Face)
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
            
            # Combine train and validation for pretraining
            train_texts = [text for text in dataset['train']['text'] if text.strip()]
            val_texts = [text for text in dataset['validation']['text'] if text.strip()]
            all_texts = train_texts + val_texts
            
            # Filter out very short texts
            all_texts = [text for text in all_texts if len(text.split()) > 10]
            
            print(f"✅ WikiText-2 loaded: {len(all_texts)} documents")
            print(f"   Total tokens: ~{sum(len(text.split()) for text in all_texts[:1000]):,} (estimated)")
            return all_texts
        except ImportError:
            print("Hugging Face datasets not available. Downloading manually...")
            return self._download_wikitext2_manual()
    
 
    def _download_wikitext2_manual(self):
        """Manual download if datasets library not available"""
        url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip"
        
        zip_path = self.data_dir / "wikitext-2.zip"
        extract_path = self.data_dir / "wikitext-2"
        
        if not extract_path.exists():
            print("Downloading WikiText-2...")
            response = requests.get(url)
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
        
        # Load text files
        train_file = extract_path / "wikitext-2-raw" / "wiki.train.raw"
        valid_file = extract_path / "wikitext-2-raw" / "wiki.valid.raw"
        
        texts = []
        for file_path in [train_file, valid_file]:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().split('\n')
                    texts.extend([line.strip() for line in content if line.strip()])
        
        return texts

    def load_imdb_toy_for_evaluation(self):
        """
        Load IMDB dataset for downstream evaluation.
        Binary sentiment classification task.
        """
        print("Loading IMDB for evaluation...")
        try:
            return self._download_imdb_simple()
        except Exception as e:
            print(f"Error loading IMDB dataset: {e}")
            return self._download_imdb_simple()
        
    def load_imdb_real(self, cache_name: str = "imdb_evaluation") -> Dict[str, Any]:
        """
        Load the real IMDB dataset via HuggingFace.
        Returns a dict with:
            train_texts, train_labels, test_texts, test_labels, label_names
        """
        cached = self.load_processed_dataset(cache_name)
        if cached is not None:
            return cached

        if load_dataset is None:
            raise RuntimeError(
                "The 'datasets' library is required for IMDB. "
                "Install it via: pip install datasets"
            )

        print("[IMDB] Loading via HuggingFace…")
        ds = load_dataset("imdb")
        # HF IMDB has splits: 'train' (25k), 'test' (25k), 'unsupervised' (50k)
        train_texts = [t for t in ds["train"]["text"]]
        train_labels = [int(l) for l in ds["train"]["label"]]
        test_texts = [t for t in ds["test"]["text"]]
        test_labels = [int(l) for l in ds["test"]["label"]]
        label_names = ["negative", "positive"]

        payload = {
            "train_texts": train_texts,
            "train_labels": train_labels,
            "test_texts": test_texts,
            "test_labels": test_labels,
            "label_names": label_names,
        }
        print(f"[IMDB] Train: {len(train_texts):,} | Test: {len(test_texts):,}")
        self.save_processed_dataset(payload, cache_name)
        return payload

    def _download_imdb_simple(self):
        """Download a subset of IMDB for evaluation"""
        # For demo purposes, create a simple sentiment dataset
        # In practice, you'd download the real IMDB dataset
        
        positive_samples = [
            "This movie was absolutely fantastic! Great acting and story.",
            "Amazing film with incredible performances. Highly recommended!",
            "One of the best movies I've ever seen. Perfect in every way.",
            "Brilliant cinematography and outstanding direction. Loved it!",
            "Exceptional storytelling with wonderful character development."
        ] * 500  # Repeat to create more samples
        
        negative_samples = [
            "Terrible movie with poor acting and boring plot.",
            "Complete waste of time. Nothing good about this film.",
            "Awful storyline and terrible direction. Very disappointing.",
            "One of the worst movies ever made. Avoid at all costs.",
            "Boring and predictable. Poor quality throughout."
        ] * 500
        
        # Create balanced dataset
        texts = positive_samples + negative_samples
        labels = [1] * len(positive_samples) + [0] * len(negative_samples)
        
        # Shuffle
        indices = np.random.permutation(len(texts))
        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]
        
        # Split train/test
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"✅ IMDB-style dataset created: {len(train_texts)} train, {len(test_texts)} test")
        
        return {
            'train_texts': train_texts,
            'train_labels': train_labels,
            'test_texts': test_texts,
            'test_labels': test_labels,
            'label_names': ['negative', 'positive']
        }
    
    def get_word_analogy_pairs(self):
        """Get standard word analogy pairs for evaluation"""
        analogies = [
            ['king', 'man', 'queen', 'woman'],
            ['paris', 'france', 'london', 'england'],
            ['good', 'better', 'bad', 'worse'],
            ['big', 'bigger', 'small', 'smaller'],
            ['he', 'his', 'she', 'her'],
            ['uncle', 'aunt', 'brother', 'sister'],
            ['walk', 'walking', 'swim', 'swimming'],
            ['cat', 'cats', 'dog', 'dogs']
        ]
        return analogies
    
    def get_word_similarity_pairs(self):
        """Get word similarity pairs for evaluation"""
        pairs = [
            ('king', 'queen', 0.8),
            ('man', 'woman', 0.7),  
            ('car', 'automobile', 0.9),
            ('big', 'large', 0.8),
            ('happy', 'joyful', 0.8),
            ('cat', 'dog', 0.6),
            ('computer', 'laptop', 0.7),
            ('ocean', 'sea', 0.9)
        ]
        return pairs
    
    def save_processed_dataset(self, dataset, name):
        """Save processed dataset"""
        processed_dir = self.data_dir / 'processed'
        processed_dir.mkdir(exist_ok=True)
        
        file_path = processed_dir / f'{name}.pkl'
        with open(file_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"💾 Saved: {file_path}")
    
    def load_processed_dataset(self, name):
        """Load processed dataset"""
        file_path = self.data_dir / 'processed' / f'{name}.pkl'
        if file_path.exists():
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        return None

# Quick test and setup
if __name__ == "__main__":
    print("🚀 Setting up simplified benchmark datasets...")
    
    loader = EmbeddingBenchmarkDatasets()

    try:
        sents = loader.load_text8(chunk_size=256)
        print(f"Text8 sentences sample (2): {sents[0][:10]} … | {sents[1][:10]} …")
    except Exception as e:
        print(f"[text8] Skipped: {e}")

    try:
        docs = loader.load_wikitext2()
        print(f"WikiText-2 sample docs (2): {docs[0][:80]} … | {docs[1][:80]} …")
    except Exception as e:
        print(f"[wikitext2] Skipped: {e}")

    # imdb real
    try:
        imdb = loader.load_imdb_real()
        print(f"IMDB sample: {imdb['label_names']}, train[0][:80] = {imdb['train_texts'][0][:80]} …")
    except Exception as e:
        print(f"[imdb] Skipped: {e}")
    


### Specific test code for Mac M1 Air performance (uncomment to run but make sure to implement the methods above) ###

    # # Initialize dataset loader
    # loader = EmbeddingBenchmarkDatasets(subset_size=5000)
    # print("🔄 Loading benchmark datasets for Mac M1 Air...")
    # # Load 20 Newsgroups (small subset)
    # newsgroups = loader.load_20newsgroups(subset_size=2000)
    # print(f"✅ 20 Newsgroups: {len(newsgroups['train_texts'])} train, {len(newsgroups['test_texts'])} test")
    # # Load IMDB sample
    # imdb = loader.load_imdb_sample(sample_size=3000)
    # if imdb:
    #     print(f"✅ IMDB: {len(imdb['train_texts'])} train, {len(imdb['test_texts'])} test")
    # # Create analogy dataset
    # analogies = loader.create_sample_analogy_dataset()
    # print(f"✅ Sample analogies: {len(analogies)} analogy pairs")
    # # Save datasets
    # loader.save_processed_dataset(newsgroups, '20newsgroups_small')
    # if imdb:
    #     loader.save_processed_dataset(imdb, 'imdb_small')
    # print("\n🎯 Datasets ready for embedding benchmarks!")
    # print("💡 These datasets are optimized for Mac M1 Air performance testing.")
