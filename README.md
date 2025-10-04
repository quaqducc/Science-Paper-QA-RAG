# Dataset & Code

This repository contains a compact version of the codebase used in my project. It includes all major scripts required to reproduce the core components of the project.  
**Note**: Large datasets—such as scientific articles, text embeddings, and graph embeddings—are **not included**, but can be generated using the provided scripts. Some file paths in the code may not align exactly with the current structure due to consolidation.

## Contents

- **`Topic Model/`**  
  Scripts for training topic models (LDA, NMF) and saving the resulting models.

- **`Co-author/`**  
  Code for constructing co-author networks using topic model results, along with saved graph data.

- **`SciQAG/`**  
  Code for generating and filtering QA pairs (modified from [SciQAG](https://github.com/MasterAI-EAM/SciQAG)).

- **`GraphSAGE/`**  
  Implementation of citation network construction and GraphSAGE-based graph embeddings.

- **`Finetune Embedding/`**  
  Scripts for fine-tuning a model to map from text embeddings to graph embeddings.

- **`RAG.py`**  
  Main script for integrating retrieval-augmented generation (RAG), topic modeling, co-author and citation networks, and QA generation.

- **`eval.ipynb`**  
  RAG evaluation using the [ragas](https://github.com/explodinggradients/ragas) library.

- **`requirements.txt`**  
  List of Python dependencies.

- **`README.md`**  
  This documentation file.

---

## Setup

### 1. Install Dependencies

It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

### 2. **Topic Models Training**
   Navigate to the `Topic Model/` directory and run the following notebooks:

```bash
lda-topic.ipynb
nmf-topic.ipynb
```

### 3. **Co-author Network Construction**
   Use the trained topic models to build the co-author network:

```bash
co-author.ipynb
```

### 4. **Citation Network Construction and GraphSAGE**
   Run graphsage_embedding.ipynb for constructing citation network and learning graph embeddings using GraphSAGE algorithm.

```bash
graphsage_embedding.ipynb
```

### 5. **Finetune Embedding**
   Given the citaion network, I finetune embedding model to map from text embedding space to graph embedding space. Run finetune-transformer-model-for-qa-embedding.ipynb for finetuning and Infer_Finetune_Model.py for inference.

```bash
# Train
finetune-transformer-model-for-qa-embedding.ipynb
```

```bash
# Infer
python Infer_Finetune_Model.py
```

### 6. **Main RAG**
   RAG.py contains all variants of RAG in this project.

```bash
python RAG.py
```

### 6.1 Minimal RAG (Windows-friendly)
A lightweight script that indexes abstracts from `Paper QA RAG/abs_metadata.json` into a local Chroma DB and supports quick retrieval.

Install extra dependency:

```bash
pip install -r requirements.txt
```

Build index (first run):

```bash
python rag_minimal.py index --metadata "../abs_metadata.json" --persist_dir "./chroma_db" --reset
```

Query:

```bash
python rag_minimal.py query --persist_dir "./chroma_db" --q "What is CP violation in phi decays?" --top_k 5
```

Notes:
- Defaults to `sentence-transformers/all-MiniLM-L6-v2` for speed.
- You can change the collection name via `--collection` and model via `--model`.

### 7. **Evaluation**
   Run eval.ipynb for RAG evaluation. You should provide your openai key.

```bash
eval.ipynb
```

   Before running, make sure to set your OpenAI API key:

```bash
import os
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```
