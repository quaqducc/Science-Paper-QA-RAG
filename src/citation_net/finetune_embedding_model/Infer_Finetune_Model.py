import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch.nn as nn
import argparse

# ====== Device setup ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Model Class ===
class QuestionEncoder(nn.Module):
    def __init__(self, pretrained_model, out_dim=128):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model)
        self.projection = nn.Linear(self.encoder.config.hidden_size, out_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        projected = self.projection(cls_output)
        return projected
    
def infer_query(model, tokenizer, query, doi_embeddings, doi_ids, max_len=64, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    model.eval()

    # Tokenize query
    encoded = tokenizer(query, return_tensors='pt', truncation=True, padding='max_length', max_length=max_len)
    encoded = {k: v.to(device) for k, v in encoded.items()}

    # Compute query embedding
    with torch.no_grad():
        query_vector = model(encoded['input_ids'], encoded['attention_mask']).cpu().numpy()

    # Compute cosine similarity
    similarities = cosine_similarity(query_vector, doi_embeddings)[0]

    # Get top 5 closest DOIs
    top_indices = np.argsort(similarities)[::-1][:5]
    top_dois = [doi_ids[i] for i in top_indices]
    top_scores = [similarities[i] for i in top_indices]

    return top_dois, top_scores

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define query
query = ("How do the studies of neutrino oscillations in the Sudbury Neutrino Observatory (SNO), "
         "the strong coupling dynamics of the standard Higgs sector, and the evolution of color exchange "
         "in QCD hard scattering collectively contribute to advancing our understanding of fundamental "
         "particle interactions and their implications for experimental observations at high-energy facilities like the LHC?")

# Load DOI embeddings
data_path = 'combined_doi_questions_embeddings.csv'
df = pd.read_csv(data_path)
doi_embeddings = df[[str(i) for i in range(128)]].values
doi_ids = df['id'].tolist()

# List of models and their fine-tuned weights
# Download model at https://www.kaggle.com/models/quangduc3122004/bge-large-en-v1.5-finetuned/pyTorch/default/1
model = ('intfloat/e5-large-v2', 'e5-large-v2_question_encoder.pt')

# Run inference for each model
model_name, model_path = model
print(f"\n=== Running inference for {model_name} ===")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = QuestionEncoder(pretrained_model=model_name, out_dim=128)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

# Perform inference
top_dois, top_scores = infer_query(model, tokenizer, query, doi_embeddings, doi_ids, device=device)

# Print results
print(f"🔍 Top 5 closest papers for {model_name}:")
for rank, (doi, score) in enumerate(zip(top_dois, top_scores), 1):
    print(f"{rank}. DOI: {doi} — Similarity: {score:.4f}")
