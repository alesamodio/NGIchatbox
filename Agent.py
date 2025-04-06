from Retrieve import retrieve_relevant_chunks
from Ask_llm import query_llm, ask_user_for_query_refinement
import hashlib
import re

from config import VECTORSTORE_PATH, SIMILARITY_THRESHOLD, MAX_CHUNKS, EXTRA_CHUNKS

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def hash_chunk(text):
    return hashlib.md5(clean_text(text).encode()).hexdigest()

def iterative_rag_agent(user_query):
    """Single-pass RAG agent for UI integration."""
    gathered_chunks = []
    gathered_references = []
    seen_chunk_hashes = set()

    # Step 1: Retrieve more chunks than needed
    new_chunks, new_references = retrieve_relevant_chunks(
        user_query,
        VECTORSTORE_PATH,
        SIMILARITY_THRESHOLD,
        EXTRA_CHUNKS
    )

    # Step 2: Filter duplicates, keep top unique chunks
    for chunk, ref in zip(new_chunks, new_references):
        chunk_hash = hash_chunk(chunk)
        if chunk_hash not in seen_chunk_hashes:
            seen_chunk_hashes.add(chunk_hash)
            gathered_chunks.append(chunk)
            gathered_references.append(ref)
        if len(gathered_chunks) >= MAX_CHUNKS:
            break

    # Step 3: Build context and query LLM
    context = "\n\n".join(gathered_chunks)
    response = query_llm(context, user_query)

    # Step 4: Check for refinement request
    if "NEED MORE INFO: YES" in response:
        new_query_start = response.find("Suggested Query:") + len("Suggested Query:")
        refined_query = response[new_query_start:].strip()

        # Package for UI to handle refinement
        refinement_info = ask_user_for_query_refinement(refined_query)
        refinement_info.update({
            "answer": response,
            "references": gathered_references,
            "chunks": gathered_chunks
        })
        return refinement_info

    # Step 5: Return final result
    return {
        "needs_refinement": False,
        "answer": response,
        "references": gathered_references,
        "chunks": gathered_chunks
    }
