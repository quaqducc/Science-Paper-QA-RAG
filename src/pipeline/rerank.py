from typing import List, Dict, Any, Optional

import numpy as np

from .config import retrieval_cfg


def _safe_sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-max(min(x, 20.0), -20.0)))


def rerank_weighted_fusion(
    hits: List[Dict[str, Any]],
    query_topic_vec: Optional[np.ndarray] = None,
    doc_topic_vecs: Optional[Dict[str, np.ndarray]] = None,
    graph_node_to_idx: Optional[dict] = None,
    graph_embeddings: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    alpha = retrieval_cfg.alpha_dense
    beta = retrieval_cfg.beta_topic
    gamma = retrieval_cfg.gamma_graph

    # Normalize dense scores to [0,1]
    dense_scores = np.array([h.get("dense_score", 0.0) for h in hits], dtype=float)
    if len(dense_scores) > 0:
        dmin, dmax = dense_scores.min(), dense_scores.max()
        if dmax > dmin:
            dense_norm = (dense_scores - dmin) / (dmax - dmin)
        else:
            dense_norm = np.zeros_like(dense_scores)
    else:
        dense_norm = dense_scores

    # Topic similarity per doc (cosine)
    topic_scores = np.zeros_like(dense_norm)
    if query_topic_vec is not None and doc_topic_vecs is not None:
        q = query_topic_vec.astype(float)
        q = q / (np.linalg.norm(q) + 1e-9)
        for i, h in enumerate(hits):
            did = str(h["metadata"].get("doc_id") or h.get("doc_id"))
            dt = doc_topic_vecs.get(did)
            if dt is None:
                continue
            v = dt.astype(float)
            v = v / (np.linalg.norm(v) + 1e-9)
            topic_scores[i] = float(np.dot(q, v))  # in [-1,1]
        # map to [0,1]
        topic_scores = 0.5 * (topic_scores + 1.0)

    # Graph proximity via cosine sim to centroid of top dense seeds
    graph_scores = np.zeros_like(dense_norm)
    if graph_node_to_idx is not None and graph_embeddings is not None:
        # choose top-m by dense score as seeds
        m = min(5, len(hits))
        dense_order = np.argsort(-dense_norm).tolist()
        seed_vecs = []
        for j in dense_order[:m]:
            did = str(hits[j]["metadata"].get("doc_id") or hits[j].get("doc_id"))
            idx = graph_node_to_idx.get(did)
            if idx is None:
                continue
            seed_vecs.append(graph_embeddings[idx])
        if len(seed_vecs) > 0:
            centroid = np.mean(seed_vecs, axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
            for i, h in enumerate(hits):
                did = str(h["metadata"].get("doc_id") or h.get("doc_id"))
                idx = graph_node_to_idx.get(did)
                if idx is None:
                    continue
                v = graph_embeddings[idx]
                v = v / (np.linalg.norm(v) + 1e-9)
                cos = float(np.dot(centroid, v))  # [-1,1]
                graph_scores[i] = 0.5 * (cos + 1.0)  # map to [0,1]

    fused = alpha * dense_norm + beta * topic_scores + gamma * graph_scores
    order = np.argsort(-fused)
    ranked = []
    for i in order.tolist():
        item = dict(hits[i])
        item["rerank_score"] = float(fused[i])
        ranked.append(item)
    return ranked


