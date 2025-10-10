import os
import json
from typing import List, Dict, Any

import numpy as np
import chromadb
from chromadb.config import Settings

from .config import paths, indexing_cfg
from .embeddings import E5Encoder, normalize_seven_digit_id
from .topics import compute_nmf_topic_vectors, compute_top_topic_indices
import pandas as pd


def read_abs_metadata(abs_metadata_path: str) -> Dict[str, Dict[str, Any]]:
    with open(abs_metadata_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    # Expect a list of {id,title,authors,abstract}
    idx: Dict[str, Dict[str, Any]] = {}
    for r in records:
        idx[str(r["id"])]= {
            "title": r.get("title", ""),
            "authors": r.get("authors", ""),
            "abstract": r.get("abstract", ""),
        }
    return idx


def build_chroma_collection():
    os.makedirs(paths.chroma_persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=paths.chroma_persist_dir, settings=Settings(anonymized_telemetry=False))
    # Use a descriptive collection name
    collection = client.get_or_create_collection(name="papers", metadata={"hnsw:space": "cosine"})

    # Load metadata
    meta_index = read_abs_metadata(paths.abs_metadata_path)

    # Load the list of allowed IDs from the questions embeddings CSV (question-domain list)
    # but we will compute DOC embeddings from abstracts with E5
    try:
        q_df = pd.read_csv(paths.finetuned_questions_embeddings_csv, usecols=["id"])  # just need ids
        allowed_ids = set(normalize_seven_digit_id(i) for i in q_df["id"].astype(str).tolist())
    except Exception:
        allowed_ids = set(normalize_seven_digit_id(i) for i in meta_index.keys())

    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    for raw_id, md in meta_index.items():
        norm_id = normalize_seven_digit_id(raw_id)
        if norm_id not in allowed_ids:
            continue
        if not md:
            # Skip docs not present in metadata
            continue
        text = md.get("title", "") + "\n" + md.get("abstract", "")
        documents.append(text)
        metadatas.append({
            "title": md.get("title", ""),
            "authors": md.get("authors", ""),
            "doc_id": str(norm_id),
        })
        ids.append(str(norm_id))

    # Compute E5 embeddings for abstracts (or later extend with PDF fulltext chunks)
    encoder = E5Encoder("intfloat/e5-large-v2")
    embeds_np = encoder.encode(documents)
    embeds = embeds_np.astype(float).tolist()

    # Compute and attach top_topic to metadata for retrieval-time filtering
    try:
        doc_topic_vecs = compute_nmf_topic_vectors(n_components=17)
        top_topic_map = compute_top_topic_indices(doc_topic_vecs)
        for meta in metadatas:
            did = meta["doc_id"]
            meta["top_topic"] = int(top_topic_map.get(did, -1))
    except Exception:
        for meta in metadatas:
            meta["top_topic"] = -1

    if len(ids) == 0:
        print("No overlapping docs between embeddings CSV and abs_metadata.json. Nothing to index.")
        return

    # Upsert into Chroma in chunks
    B = 2048
    for i in range(0, len(ids), B):
        collection.upsert(
            ids=ids[i:i+B],
            documents=documents[i:i+B],
            metadatas=metadatas[i:i+B],
            embeddings=embeds[i:i+B],
        )
    print(f"Indexed {len(ids)} documents into Chroma at {paths.chroma_persist_dir}")


if __name__ == "__main__":
    build_chroma_collection()


