# =========================
# 1. Imports & Configuration
# =========================

import os
import sys
import json
import pickle
import random
import subprocess
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

# NLP & ML
import gensim
import spacy
import nltk
from nltk.corpus import stopwords
from gensim.models.word2vec import Word2Vec
from gensim.models import Nmf
from gensim.utils import simple_preprocess
import gensim.corpora as corpora

# Transformers & Torch
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

# Langchain & ChromaDB
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import chromadb
from chromadb.config import Settings
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch.nn as nn
import argparse

# =========================
# 2. Utility Functions
# =========================

def extract_base_id(full_id):
    return full_id.split('/')[-1].split('v')[0]

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts, stop_words):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts, bigram_mod, trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def download_documents(documents):
    for document_id in documents:
        if '/' in document_id:
            subdir, filename = document_id.split('/')
            dir_path = f"/root/thesis/download_papers/{subdir}"
            file_path = f"{dir_path}/{filename}.pdf"
            os.makedirs(dir_path, exist_ok=True)
        else:
            file_path = f"/root/thesis/download_papers/{document_id}.pdf"
        if os.path.isfile(file_path):
            print(f"File {file_path} already exists. Skipping download.")
        else:
            print(f"Downloading {file_path}...")
            subprocess.run(["wget", f"https://arxiv.org/pdf/hepph/{document_id}.pdf", "-O", file_path])

def load_documents(documents):
    final_documents = []
    for id in documents:
        file_path = f"/root/thesis/download_papers/{id}.pdf"
        if not os.path.isfile(file_path):
            print(f"❌ File {file_path} not found!")
            continue
        if os.path.getsize(file_path) == 0:
            print(f"⚠️ File {file_path} is empty! Skipping...")
            continue
        try:
            loader = PyPDFLoader(file_path)
            final_documents += loader.load()
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            continue
    return final_documents

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def infer_query(model, tokenizer, query, doi_embeddings, doi_ids, max_len=512, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    model.eval()

    encoded = tokenizer(query, return_tensors='pt', truncation=True, padding='max_length', max_length=max_len)
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        query_vector = model(encoded['input_ids'], encoded['attention_mask']).cpu().numpy()

    similarities = cosine_similarity(query_vector, doi_embeddings)[0]

    top_indices = np.argsort(similarities)[::-1][:4]
    top_dois = [doi_ids[i] for i in top_indices]
    top_scores = [similarities[i] for i in top_indices]

    return top_dois, top_scores

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

query = "What are the implications of a non-zero spectral index for inﬂationary models?"

data_path = '/root/thesis/graphsage_embeddings.csv'
df = pd.read_csv(data_path)
doi_embeddings = df[[str(i) for i in range(128)]].values
doi_ids = df['node_id'].tolist()
doi_ids = [str(item).zfill(7) for item in doi_ids]
id_to_embedding = dict(zip(doi_ids, doi_embeddings))

model = ('intfloat/e5-large-v2', '/root/thesis/encoder_model/e5-large-v2_question_encoder.pt'),

model_name, model_path = model
g_tokenizer = AutoTokenizer.from_pretrained(model_name)
g_model = QuestionEncoder(pretrained_model=model_name, out_dim=128)
g_model.load_state_dict(torch.load(model_path, map_location=device))
g_model.to(device)

# =========================
# 3. Data Loading
# =========================

# Load metadata
datas_meta = read_json_file("/root/thesis/arxiv_papers.json")

# Load author-paper mapping
author_papers = read_json_file("/root/thesis/author_papers.json")

# Load NMF graph
with open('/root/thesis/nmf_coauthor_17.pkl', 'rb') as file:
    G = pickle.load(file)

# =========================
# 4. Graph Embedding (DeepWalk)
# =========================

def getRandomWalk(graph, node, length_of_random_walk):
    start_node = node
    current_node = start_node
    random_walk = [node]
    for _ in range(length_of_random_walk):
        current_node_neighbours = list(graph.neighbors(current_node))
        if not current_node_neighbours:
            break
        current_node = random.choice(current_node_neighbours)
        random_walk.append(current_node)
    return random_walk

num_sampling = 10
length_of_random_walk = 10
random_walks = []
for node in tqdm(G.nodes(), desc="Iterating Nodes"):
    for _ in range(num_sampling):
        random_walks.append(getRandomWalk(G, node, length_of_random_walk))

deepwalk_model = Word2Vec(
    sentences=random_walks, window=5, sg=1, negative=5,
    vector_size=128, epochs=20, compute_loss=True
)
deepwalk_model.save("deepwalk.model")

def getSimilarNodes(model, nodes):
    res = []
    for node in nodes:
        try:
            similarity = model.wv.most_similar(node)
            res += [author[0] for author in similarity if author[1] >= 0.9]
        except:
            pass
    return res

def get_paper_from_authors(authors):
    res = []
    for author in authors:
        res += author_papers.get(author, [])
    return [extract_base_id(full_id) for full_id in res]

# =========================
# 5. Text Preprocessing
# =========================

nltk.download('stopwords')
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# Extract abstracts
abstracts = [item.get("summary", "") for item in datas_meta]
data_words = list(sent_to_words(abstracts))

# Build bigram/trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# Remove stopwords, make trigrams, lemmatize
data_words_nostops = remove_stopwords(data_words, stop_words)
data_words_bigrams = make_trigrams(data_words_nostops, bigram_mod, trigram_mod)
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
data_lemmatized = lemmatization(data_words_bigrams, nlp)

# =========================
# 6. Topic Modeling
# =========================

id2word = corpora.Dictionary(data_lemmatized)
texts = data_lemmatized
corpus = [id2word.doc2bow(text) for text in texts]
nmf = Nmf.load(f"/root/thesis/model_nmf_100_17.model")

# Compute topic distributions for all papers
filtered_meta = [item for item in datas_meta if extract_base_id(item.get('id', '')) != '']
ids = [extract_base_id(item.get('id', '')) for item in filtered_meta]
texts = [item.get('summary', '') for item in filtered_meta]

topic_dis = []
for text in texts:
    new_doc_tokens = text.lower().split()
    new_doc_bow = id2word.doc2bow(new_doc_tokens)
    num_topics = nmf.num_topics
    topics_sparse = dict(nmf.get_document_topics(new_doc_bow, minimum_probability=0.0))
    dense_vector = [topics_sparse.get(i, 0.0) for i in range(num_topics)]
    topic_dis.append(dense_vector)
topic_dis = np.array(topic_dis, dtype=np.float32)

# =========================
# 7. Embedding Model & ChromaDB Setup
# =========================

embed_model = HuggingFaceEmbeddings(
    model_name="intfloat/e5-large-v2",
    model_kwargs={"device": "cuda"}
)

chroma_client = chromadb.Client(Settings())
topic_collection = chroma_client.get_or_create_collection(name="topic_papers")
collection = chroma_client.get_or_create_collection(name="papers_abstract")

# Add topic embeddings to ChromaDB
batch_size = 5000
total = len(ids)
for i in range(0, total, batch_size):
    end = i + batch_size
    topic_collection.add(
        ids=ids[i:end],
        documents=texts[i:end],
        embeddings=topic_dis[i:end]
    )
    print(f"Added batch {i} to {min(end, total)}")

# Add abstract embeddings to ChromaDB
embeddings = np.load("/root/thesis/embeddings.npy", allow_pickle=True)
for i in range(0, total, batch_size):
    end = i + batch_size
    collection.add(
        ids=ids[i:end],
        documents=texts[i:end],
        embeddings=embeddings[i:end],
        metadatas=[{"paper_id": _id} for _id in ids[i:end]]
    )
    print(f"Added batch {i} to {min(end, total)}")

# =========================
# 8. Retriever Classes
# =========================

class ChunkedChromaRetriever:
    def __init__(self, collection, embed_model, top_k_docs=5, chunk_size=500, chunk_overlap=50, top_k_chunks=5):
        self.collection = collection
        self.embed_model = embed_model
        self.top_k_docs = top_k_docs
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k_chunks = top_k_chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def retrieve(self, query):
        query_embedding = self.embed_model.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k_docs,
            include=["documents"]
        )
        documents_ids = results["ids"][0]
        download_documents(documents_ids)
        raw_documents = load_documents(documents_ids)
        chunks = self.text_splitter.split_documents(raw_documents)
        chunk_vectorstore = Chroma.from_documents(chunks, embedding=embed_model)
        chunk_retriever = chunk_vectorstore.as_retriever(search_kwargs={"k": self.top_k_chunks})
        chunk_documents = chunk_retriever.invoke(query)
        return chunk_documents
    
class TopicChunkedChromaRetriever:
    def __init__(self, collection, topic_collection, embed_model, top_k_docs=5, chunk_size=500, chunk_overlap=50, top_k_chunks=5):
        self.collection = collection
        self.topic_collection = topic_collection
        self.embed_model = embed_model
        self.top_k_docs = top_k_docs
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k_chunks = top_k_chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def retrieve(self, query):
        new_doc = text
        new_doc_tokens = new_doc.lower().split()
        new_doc_bow = id2word.doc2bow(new_doc_tokens)
        num_topics = nmf.num_topics
        topics_sparse = dict(nmf.get_document_topics(new_doc_bow, minimum_probability=0.0))
        dense_vector = [topics_sparse.get(i, 0.0) for i in range(num_topics)]

        results = self.topic_collection.query(
            query_embeddings=[dense_vector],
            n_results=1000,
            include=["documents"]
        )

        ids = results["ids"][0]

        first_authors = []
        for pid in ids:
            for entry in datas_meta:
                if extract_base_id(entry.get('id',"")) == pid:
                    first_author = entry['authors'][0]['name'] if entry.get('authors') else 'N/A'
                    first_authors.append(first_author)

        author_papers_ids = []
        for author in first_authors:
            author_papers_ids+=getSimilarNodes(deepwalk_model,author)

        final_ids = ids + get_paper_from_authors(first_authors + author_papers_ids)

        query_embedding = self.embed_model.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k_docs,
            where={"paper_id": {"$in": final_ids}},
            include=["documents"]
        )

        retrieve_papers = results["ids"][0]
        download_documents(retrieve_papers)
        raw_documents = load_documents(retrieve_papers)
        chunks = self.text_splitter.split_documents(raw_documents)
        chunk_vectorstore = Chroma.from_documents(chunks, embedding=embed_model)
        chunk_retriever = chunk_vectorstore.as_retriever(search_kwargs={"k" : self.top_k_chunks})
        chunk_documents = chunk_retriever.invoke(query)

        return chunk_documents
    
class GraphTopicChunkedChromaRetriever:
    def __init__(self, collection, topic_collection, embed_model, top_k_docs=5, chunk_size=500, chunk_overlap=50, top_k_chunks=5):
        self.collection = collection
        self.topic_collection = topic_collection
        self.embed_model = embed_model
        self.top_k_docs = top_k_docs
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k_chunks = top_k_chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def retrieve(self, query):

        new_doc = text
        new_doc_tokens = new_doc.lower().split()  
        new_doc_bow = id2word.doc2bow(new_doc_tokens)
        num_topics = nmf.num_topics
        topics_sparse = dict(nmf.get_document_topics(new_doc_bow, minimum_probability=0.0))
        dense_vector = [topics_sparse.get(i, 0.0) for i in range(num_topics)]

        results = self.topic_collection.query(
            query_embeddings=[dense_vector],
            n_results=1000,
            include=["documents"]
        )

        ids = results["ids"][0]

        first_authors = []
        for pid in ids:
            for entry in datas_meta:
                if extract_base_id(entry.get('id',"")) == pid:
                    first_author = entry['authors'][0]['name'] if entry.get('authors') else 'N/A'
                    first_authors.append(first_author)

        author_papers_ids = []
        for author in first_authors:
            author_papers_ids+=getSimilarNodes(deepwalk_model,author)


        final_ids = ids + get_paper_from_authors(author_papers_ids)

        encoded = g_tokenizer(query, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            query_vector = g_model(encoded['input_ids'], encoded['attention_mask']).cpu().detach().numpy().flatten().tolist()

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=self.top_k_docs,
            where={"paper_id": {"$in": final_ids}},
            include=["documents"]
        )

        retrieve_papers = results["ids"][0]
        download_documents(retrieve_papers)
        raw_documents = load_documents(retrieve_papers)
        chunks = self.text_splitter.split_documents(raw_documents)
        chunk_vectorstore = Chroma.from_documents(chunks, embedding=embed_model)
        chunk_retriever = chunk_vectorstore.as_retriever(search_kwargs={"k" : self.top_k_chunks})
        chunk_documents = chunk_retriever.invoke(query)

        return chunk_documents
    
class CitationChunkedChromaRetriever:
    def __init__(self, collection, embed_model, top_k_docs=5, chunk_size=500, chunk_overlap=50, top_k_chunks=5):
        self.collection = collection
        self.embed_model = embed_model
        self.top_k_docs = top_k_docs
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k_chunks = top_k_chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def retrieve(self, query):

        top_dois, top_scores = infer_query(g_model, g_tokenizer, query, doi_embeddings, doi_ids, device=device)
        query_embedding = self.embed_model.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["documents"]
        )
        raw_documents = results["documents"][0]

        documents_ids = results["ids"][0] + top_dois

        download_documents(documents_ids)
        raw_documents = load_documents(documents_ids)
        chunks = self.text_splitter.split_documents(raw_documents)
        chunk_vectorstore = Chroma.from_documents(chunks, embedding=embed_model)
        chunk_retriever = chunk_vectorstore.as_retriever(search_kwargs={"k" : self.top_k_chunks})
        chunk_documents = chunk_retriever.invoke(query)

        return chunk_documents

# =========================
# 9. LLM Setup
# =========================

model_id = "Qwen/Qwen3-14B-FP8"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    max_new_tokens=100,
    pad_token_id=tokenizer.eos_token_id,
    return_full_text=False
)
llm = HuggingFacePipeline(pipeline=pipe)

# =========================
# 10. Prompt & Chain Setup
# =========================

prompt_template = PromptTemplate.from_template(
    "Use only the context below to answer the question.\n"
    "- If the answer is not in the context, respond exactly: I don't know\n"
    "- Do not write anything else.\n"
    "- Answer in exactly one short sentence, no more than 50 words.\n\n"
    "Question: {question}\n\nContext:\n{context}\n\nFinal Answer:"
)

class CleanSingleLineAnswerParser(StrOutputParser):
    def parse(self, text: str) -> str:
        for line in text.strip().split("\n"):
            if line.strip():
                return line.strip()
        return ""

retrieved_contexts = []
def save_context(inputs):
    retrieved_contexts.append(inputs["context"])
    return inputs

retriever = ChunkedChromaRetriever(
    collection=collection,
    embed_model=embed_model,
    top_k_docs=5,
    chunk_size=500,
    chunk_overlap=100,
    top_k_chunks=5
)

semantic_rag_chain = (
    {
        "context": RunnableLambda(lambda q: retriever.retrieve(q)) | format_docs,
        "question": RunnablePassthrough()
    }
    | RunnableLambda(save_context)
    | prompt_template
    | llm
    | CleanSingleLineAnswerParser()
)

# =========================
# 11. Main Execution
# =========================

if __name__ == "__main__":
    import torch
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    # Example query
    query = "How does the existence of a full scalar doublet at low energies benefit the Topcolor model?"
    top_chunks = retriever.retrieve(query)
    for idx, chunk in enumerate(top_chunks):
        print(f"--- Top Chunk {idx+1} ---")
        print(chunk.page_content)

    # Batch QA
    with open("/root/thesis/filter_qa.json", 'r') as file:
        qas = json.load(file)
    answers = []
    for qa in qas:
        q = qa[0]
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            answer = semantic_rag_chain.invoke(q)
            print(answer)
        except:
            answer = ""
            retrieved_contexts.append("")
        finally:
            answers.append(answer)
    with open("answers_rag.json", 'w') as file:
        json.dump(answers, file, indent=4)
    with open("retrieved_contexts_rag.json", 'w') as file:
        json.dump(retrieved_contexts, file, indent=4)
