import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings

st.title("Embedding Model Test – E5 Base")

try:
    st.write("Attempting to load intfloat/e5-base-v2 model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2"
    )
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model:\n\n{e}")
