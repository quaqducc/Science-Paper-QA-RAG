from dataclasses import dataclass


@dataclass
class Paths:
    abs_metadata_path: str = "data/abs_metadata.json"
    finetuned_questions_embeddings_csv: str = (
        "src/citation_net/finetune_embedding_model/combined_doi_questions_embeddings.csv"
    )
    graphsage_embeddings_csv: str = "src/citation_net/graphSAGE/graphsage_embeddings.csv"
    chroma_persist_dir: str = "src/vector_db/chroma_db"


@dataclass
class EmbeddingModelCfg:
    pretrained_model: str = "intfloat/e5-large-v2"
    projection_out_dim: int = 128
    finetuned_state_path: str = "src/citation_net/finetune_embedding_model/e5-large-v2_question_encoder.pt"
    max_length: int = 128


@dataclass
class RetrievalCfg:
    top_k: int = 50
    final_top_n: int = 10
    alpha_dense: float = 0.6  # weight for dense (finetuned) similarity in rerank
    beta_topic: float = 0.2   # weight for topic model similarity
    gamma_graph: float = 0.2  # weight for graph/GraphSAGE proximity
    use_cross_encoder: bool = False
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_topic_prefilter: bool = True  # apply topic filter at retrieval-time (RAG stage)


@dataclass
class TopicCfg:
    lda_model_path: str = "src/topic_model/model_lda_100_6.model"
    nmf_model_path: str = "src/topic_model/model_nmf_100_17.model"
    # Optional: we can compute topic vectors for abstracts on the fly


@dataclass
class IndexingCfg:
    use_pdf_fulltext: bool = True
    pdf_timeout_sec: int = 20
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 128


@dataclass
class LLMConfig:
    model: str = "sentence-transformers/all-MiniLM-L6-v2"  # placeholder; replace with chat LLM
    max_new_tokens: int = 512


paths = Paths()
embed_cfg = EmbeddingModelCfg()
retrieval_cfg = RetrievalCfg()
topic_cfg = TopicCfg()
llm_cfg = LLMConfig()
indexing_cfg = IndexingCfg()


