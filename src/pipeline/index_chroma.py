import os
import json
from typing import List, Dict, Any

import numpy as np
import chromadb
from chromadb.config import Settings

from .config import paths
from .embeddings import load_precomputed_doc_embeddings


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

    # Load doc embeddings and metadata
    doc_ids, doc_embeddings = load_precomputed_doc_embeddings()
    meta_index = read_abs_metadata(paths.abs_metadata_path)

    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    for doc_id in doc_ids:
        md = meta_index.get(str(doc_id))
        if not md:
            # Skip docs not present in metadata
            continue
        text = md.get("title", "") + "\n" + md.get("abstract", "")
        documents.append(text)
        metadatas.append({
            "title": md.get("title", ""),
            "authors": md.get("authors", ""),
            "doc_id": str(doc_id),
        })
        ids.append(str(doc_id))

    # Align embeddings to filtered ids order
    id_to_idx = {str(did): i for i, did in enumerate(doc_ids)}
    embeds = []
    for did in ids:
        embeds.append(doc_embeddings[id_to_idx[did]].astype(float).tolist())

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


