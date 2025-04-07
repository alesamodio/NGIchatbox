import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings

st.title("Embedding Model Test – MiniLM")

try:
    st.write("Attempting to load lightweight model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model:\n\n{e}")
