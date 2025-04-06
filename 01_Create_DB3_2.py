import os
import time
import numpy as np
import faiss
import pickle
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# --- SETTINGS ---
pdf_folder = r"C:\Users\AsA\RAG\Liquefaction"
vectorstore_folder = r"C:\Users\AsA\RAG\vectorstore"
index_file = os.path.join(vectorstore_folder, "faiss.index")
docstore_file = os.path.join(vectorstore_folder, "documents.pkl")

# --- HELPERS ---
def normalize(v):
    v = np.array(v)
    norm = np.linalg.norm(v)
    return v if norm == 0 else (v / norm)

def save_documents(docs, path):
    with open(path, "wb") as f:
        pickle.dump(docs, f)
    print(f"✅ Saved {len(docs)} documents to {path}")

# --- LOAD AND SPLIT ---
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")

all_documents = []
for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        file_path = os.path.join(pdf_folder, filename)
        print(f"📄 Processing: {filename}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        valid_docs = [doc for doc in docs if doc.page_content.strip()]
        for i, doc in enumerate(valid_docs):
            doc.metadata["source"] = filename
            doc.metadata["page"] = i + 1
        all_documents.extend(valid_docs)

if not all_documents:
    raise ValueError("❌ No valid documents found.")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(all_documents)

# Prepend "passage: " for e5
for chunk in chunks:
    chunk.page_content = f"passage: {chunk.page_content.strip()}"

# --- EMBED IN BATCHES ---
texts = [doc.page_content for doc in chunks]
metadatas = [doc.metadata for doc in chunks]

vectors = []
batch_size = 50
print(f"🧠 Embedding {len(texts)} chunks...")
for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]
    batch_vectors = embedding_model.embed_documents(batch)
    vectors.extend([normalize(v) for v in batch_vectors])
    print(f"✅ Embedded batch {i // batch_size + 1}")
    time.sleep(5)

# --- BUILD FAISS INDEX ---
dim = len(vectors[0])
index = faiss.IndexFlatIP(dim)
index.add(np.array(vectors).astype("float32"))

# --- SAVE FAISS + DOCSTORE ---
os.makedirs(vectorstore_folder, exist_ok=True)
faiss.write_index(index, index_file)
save_documents(chunks, docstore_file)
print(f"✅ FAISS index saved at: {index_file}")
