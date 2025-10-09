import argparse
from typing import List

import numpy as np

from .config import retrieval_cfg
from .retrieve import ChromaRetriever
from .rerank import rerank_weighted_fusion
from .llm import SimpleLLM
from .topics import compute_nmf_topic_vectors, infer_query_topic_vec_nmf
from .embeddings import load_graphsage_embeddings


def run_pipeline(question: str, final_top_n: int = None):
    if final_top_n is None:
        final_top_n = retrieval_cfg.final_top_n

    # 1) retrieve
    retriever = ChromaRetriever()
    retrieved = retriever.retrieve(question, top_k=retrieval_cfg.top_k)
    hits = retrieved["hits"]

    # 2) optional features for reranking
    try:
        doc_topic_vecs = compute_nmf_topic_vectors(n_components=17)
        query_topic_vec = infer_query_topic_vec_nmf(question, n_components=17)
    except Exception:
        doc_topic_vecs = None
        query_topic_vec = None
    try:
        node_to_idx, g_embeds = load_graphsage_embeddings()
    except Exception:
        node_to_idx, g_embeds = None, None

    ranked = rerank_weighted_fusion(
        hits,
        query_topic_vec=query_topic_vec,
        doc_topic_vecs=doc_topic_vecs,
        graph_node_to_idx=node_to_idx,
        graph_embeddings=g_embeds,
    )
    selected = ranked[:final_top_n]

    # 3) prepare contexts for LLM
    contexts: List[str] = []
    for h in selected:
        did = h.get("doc_id") or (h.get("metadata", {}).get("doc_id") if h.get("metadata") else None)
        title = (h.get("metadata", {}) or {}).get("title", "")
        contexts.append(f"[doc_id={did}] {title}\n{h['document']}")

    # 4) LLM answer
    llm = SimpleLLM()
    answer = llm.answer(question, contexts)
    return {
        "question": question,
        "contexts": contexts,
        "answer": answer,
        "selected_doc_ids": [ (h.get("doc_id") or h.get("metadata", {}).get("doc_id")) for h in selected ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--top_n", type=int, default=None)
    args = parser.parse_args()
    out = run_pipeline(args.question, final_top_n=args.top_n)
    print("Selected doc IDs:", out["selected_doc_ids"])
    print("\nAnswer:\n", out["answer"])


