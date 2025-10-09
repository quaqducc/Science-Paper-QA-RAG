from typing import Dict, Any, List

import json
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

from .config import topic_cfg, paths


def _load_metadata_texts() -> Dict[str, str]:
    with open(paths.abs_metadata_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    m = {}
    for r in records:
        did = str(r["id"])
        text = (r.get("title", "") + "\n" + r.get("abstract", "")).strip()
        m[did] = text
    return m


def compute_nmf_topic_vectors(n_components: int = 17) -> Dict[str, np.ndarray]:
    texts = _load_metadata_texts()
    ids = list(texts.keys())
    corpus = [texts[i] for i in ids]
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    X = tfidf.fit_transform(corpus)
    nmf = NMF(n_components=n_components, init="nndsvda", random_state=0, max_iter=400)
    W = nmf.fit_transform(X)
    # L2 normalize topic vectors
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    return {ids[i]: W[i] for i in range(len(ids))}


def infer_query_topic_vec_nmf(query: str, n_components: int = 17) -> np.ndarray:
    # For simplicity reuse the same vectorizer/model as compute; in production cache them
    texts = _load_metadata_texts()
    corpus = list(texts.values())
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    X = tfidf.fit_transform(corpus + [query])
    nmf = NMF(n_components=n_components, init="nndsvda", random_state=0, max_iter=400)
    W = nmf.fit_transform(X)
    q_vec = W[-1]
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-9)
    return q_vec


