#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Topic Modeling Pipeline (No author_papers version)
--------------------------------------------------
Xử lý văn bản, xây dựng mô hình NMF để trích xuất chủ đề từ danh sách bài báo.
Không cần ánh xạ tác giả ↔ bài viết.
"""

import os
import re
import gensim
import gensim.corpora as corpora
from gensim.models.nmf import Nmf
from gensim.models.phrases import Phrases, Phraser
from pprint import pprint
from tqdm import tqdm

# =====================
# 1. Setup
# =====================
BASE_DIR = "."
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# =====================
# 2. Sample Data (thay bằng dữ liệu của bạn)
# =====================
documents = [
    "Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to learn.",
    "Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
    "Natural language processing enables computers to understand human language.",
    "Reinforcement learning is an area of machine learning concerned with how software agents should take actions in an environment.",
    "Computer vision is the field that deals with how computers can gain high-level understanding from digital images or videos."
]

print(f"Loaded {len(documents)} documents.\n")

# =====================
# 3. Preprocessing
# =====================
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [word for word in text.split() if word not in stop_words and len(word) > 2]
    return tokens

data_words = [preprocess(doc) for doc in documents]

# =====================
# 4. Detect bigrams/trigrams
# =====================
bigram = Phrases(data_words, min_count=2, threshold=5)
bigram_mod = Phraser(bigram)
data_words_bigrams = [bigram_mod[doc] for doc in data_words]

# =====================
# 5. Create Dictionary & Corpus
# =====================
id2word = corpora.Dictionary(data_words_bigrams)
texts = data_words_bigrams
corpus = [id2word.doc2bow(text) for text in texts]

print("Dictionary and Corpus created successfully.")
print(f"Sample vocab size: {len(id2word)}\n")

# =====================
# 6. Train NMF Model
# =====================
num_topics = 3
nmf_model = Nmf(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=42)

# =====================
# 7. Display Topics
# =====================
print("\n=== Topics extracted ===")
for idx, topic in nmf_model.show_topics(num_topics=num_topics, formatted=False):
    print(f"\nTopic #{idx}:")
    pprint([word for word, _ in topic])

# =====================
# 8. Save Outputs
# =====================
model_path = os.path.join(DATA_DIR, "nmf_model.gensim")
dict_path = os.path.join(DATA_DIR, "id2word.dict")

nmf_model.save(model_path)
id2word.save(dict_path)

print(f"\n✅ Model and dictionary saved in {DATA_DIR}/")

# =====================
# 9. Example: Infer topic for new text
# =====================
new_text = "Neural networks are essential for deep learning applications."
new_bow = id2word.doc2bow(preprocess(new_text))
topic_distribution = nmf_model[new_bow]
print("\nTopic distribution for sample text:")
print(topic_distribution)
