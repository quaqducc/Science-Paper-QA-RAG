from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import torch
print(torch.cuda.is_available())  # Should be True

import json
# Initialize model (you can specify another transformer model if needed)
kw_model = KeyBERT(model='all-MiniLM-L6-v2')

file_path = "/root/SciQAG/samples.json"

# Example: Load your full text (this could be from a .txt, .pdf, or other source)
with open(file_path, 'r') as file:
    datas = json.load(file) 
count = 0
for data in datas:
    full_text = data["txt"]

# Extract keywords (you can tweak the parameters)
    keywords = kw_model.extract_keywords(
        full_text,
        keyphrase_ngram_range=(1, 3),
        stop_words='english',
        top_n=15,
        use_maxsum=True,
        nr_candidates=15
    )
    kw = [i[0] for i in keywords]

    data["keywords"] = kw
    count+=1
    print(count)
with open("final_samples.json", "w") as json_file:
    json.dump(datas, json_file, indent=4)
