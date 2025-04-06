import faiss
import numpy as np
import pickle
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from config import VECTORSTORE_PATH, SIMILARITY_THRESHOLD, MAX_CHUNKS

def normalize(v):
    v = np.array(v)
    norm = np.linalg.norm(v)
    return v if norm == 0 else (v / norm)

def retrieve_relevant_chunks(query, vectorstore_path, similarity_threshold, max_chunks):
    """Retrieve top-k relevant chunks using cosine similarity and E5 embeddings."""

    # Paths to stored index and documents
    index_path = f"{vectorstore_path}/faiss.index"
    docstore_path = f"{vectorstore_path}/documents.pkl"

    # Load FAISS index
    index = faiss.read_index(index_path)

    # Load stored chunked documents
    with open(docstore_path, "rb") as f:
        documents: list[Document] = pickle.load(f)

    # Load embedding model (E5)
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")

    # Embed and normalize the query
    formatted_query = f"query: {query.strip()}"
    query_vector = embedding_model.embed_query(formatted_query)
    query_vector = normalize(query_vector).astype("float32").reshape(1, -1)

    # Perform FAISS similarity search
    top_k = max_chunks * 2  # Retrieve more, then filter
    scores, indices = index.search(query_vector, top_k)

    selected_chunks = []
    references = []
    seen_metadata = set()

    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx == -1 or idx >= len(documents):
            continue  # invalid match
        doc = documents[idx]
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        # Avoid duplicate page references
        if (source, page) in seen_metadata:
            continue
        seen_metadata.add((source, page))

        if score >= similarity_threshold:
            reference = f"[Chunk {len(selected_chunks)+1}] Source: {source}, Page: {page}, Similarity: {score:.2f}"
            chunk_text = doc.page_content
            selected_chunks.append(f"{chunk_text}\n{reference}")
            references.append(reference)

        if len(selected_chunks) >= max_chunks:
            break

    return selected_chunks, references
