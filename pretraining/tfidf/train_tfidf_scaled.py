import os, json, time, yaml, argparse
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from .tfidf_vectorizer import TfidfVectorizer

from ..utils.benchmark_datasets import EmbeddingBenchmarkDatasets

def main(cfg_path: str):
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)
        
    out_dir = Path(config["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = EmbeddingBenchmarkDatasets(data_dir="datasets")
    imdb = loader.load_processed_dataset("imdb_evaluation") or loader.load_imdb_for_evaluation()
    vec_cfg = config['embeddings']['tfidf']
    vectorizer = TfidfVectorizer(
        max_features=vec_cfg.get('max_features', None),
        min_df=vec_cfg.get('min_df', 1),
        max_df=vec_cfg.get('max_df', 1.0),
        ngram_range=tuple(vec_cfg.get('ngram_range', (1,1))),
        smooth_idf=vec_cfg.get('smooth_idf', True),
        sublinear_tf=vec_cfg.get('sublinear_tf', False),
        norm=vec_cfg.get('norm', 'l2'),
        lowercase=vec_cfg.get('lowercase', True),
    )
    print("🚀 Fitting TF-IDF vectorizer...")
    start_time = time.time()
    X_train = vectorizer.fit_transform(imdb['train_texts'])
    y_train = np.array(imdb['train_labels'])
    print(f"✅ TF-IDF fitting completed in {time.time() - start_time:.2f} seconds.")
    X_test = vectorizer.transform(imdb['test_texts'])
    y_test = np.array(imdb['test_labels'])

    vectorizer.save(out_dir / "tfidf")
    clf = LogisticRegression(max_iter=2000, n_jobs=None)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    metrics = dict(
        accuracy=accuracy_score(y_test, preds),
        precision=precision_score(y_test, preds),
        recall=recall_score(y_test, preds),
        f1=f1_score(y_test, preds)
    )
    print("📊 Evaluation Metrics:",metrics)
    with open(out_dir / "tfidf_evaluation_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate TF-IDF embeddings on IMDB dataset.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    main(args.config)


