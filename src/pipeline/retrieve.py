from typing import List, Dict, Any

import numpy as np
import chromadb
from chromadb.config import Settings

from .embeddings import FinetunedQuestionEmbedding
from .config import retrieval_cfg, paths


class ChromaRetriever:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=paths.chroma_persist_dir, settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(name="papers")
        self.encoder = FinetunedQuestionEmbedding()

    def retrieve(self, query: str, top_k: int = None) -> Dict[str, Any]:
        if top_k is None:
            top_k = retrieval_cfg.top_k
        qvec = self.encoder.encode([query])[0].astype(float).tolist()
        res = self.collection.query(query_embeddings=[qvec], n_results=top_k, include=["metadatas", "documents", "distances"])
        # Chroma returns lower distances for better matches with cosine; convert to similarity
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        sims = [1.0 - float(d) for d in dists]
        results: List[Dict[str, Any]] = []
        for did, doc, meta, score in zip(ids, docs, metas, sims):
            results.append({
                "doc_id": did,
                "document": doc,
                "metadata": meta,
                "dense_score": score,
            })
        return {"query": query, "hits": results}


