import os
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel

from .config import embed_cfg, paths


class QuestionEncoder(nn.Module):
    def __init__(self, pretrained_model: str, out_dim: int = 128):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model)
        self.projection = nn.Linear(self.encoder.config.hidden_size, out_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]
        projected = self.projection(cls_output)
        return projected


class FinetunedQuestionEmbedding:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(embed_cfg.pretrained_model)
        self.model = QuestionEncoder(embed_cfg.pretrained_model, embed_cfg.projection_out_dim)
        if os.path.exists(embed_cfg.finetuned_state_path):
            state = torch.load(embed_cfg.finetuned_state_path, map_location=self.device)
            self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: List[str]) -> np.ndarray:
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=embed_cfg.max_length,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            vec = self.model(enc["input_ids"], enc["attention_mask"]).cpu().numpy()
        # L2 normalize
        norms = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-9
        return vec / norms


# ===== E5 encoder (intfloat/e5-large-v2) with mean pooling =====

class E5Encoder:
    def __init__(self, model_name: str = "intfloat/e5-large-v2", max_length: int = 512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.max_length = max_length

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def encode(self, texts: List[str]) -> np.ndarray:
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = self.model(**enc)
            pooled = self._mean_pool(outputs.last_hidden_state, enc["attention_mask"]).cpu().numpy()
        norms = np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-9
        return pooled / norms


def normalize_seven_digit_id(arxiv_like_id: Any) -> str:
    s = str(arxiv_like_id)
    digits = ''.join(ch for ch in s if ch.isdigit())
    if len(digits) == 0:
        return s
    if len(digits) < 7:
        digits = digits.zfill(7)
    return digits


def load_precomputed_doc_embeddings() -> Tuple[List[str], np.ndarray]:
    """
    Load document embeddings from the finetuned CSV. We will treat the 'id' as document id (e.g., arXiv id)
    and columns 0..127 as embedding dimensions. The file also has a 'questions' text which we ignore here.
    """
    df = pd.read_csv(paths.finetuned_questions_embeddings_csv)
    doc_ids: List[str] = df["id"].astype(str).tolist()
    embed_cols = [str(i) for i in range(128)]
    if not set(embed_cols).issubset(df.columns):
        # fallback: some CSVs have integer columns already
        embed_cols = list(range(128))
    embeddings = df[embed_cols].values.astype(np.float32)
    # normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    embeddings = embeddings / norms
    return doc_ids, embeddings


def load_graphsage_embeddings() -> Tuple[Dict[str, int], np.ndarray]:
    """
    Load GraphSAGE node embeddings. Returns mapping from node_id (str) -> index, and the embedding matrix.
    """
    df = pd.read_csv(paths.graphsage_embeddings_csv)
    node_ids = df["node_id"].astype(str).tolist()
    embed_cols = [c for c in df.columns if c != "node_id"]
    mat = df[embed_cols].values.astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    mat = mat / norms
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    return node_to_idx, mat


