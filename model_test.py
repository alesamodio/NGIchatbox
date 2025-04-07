import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings

st.title("Embedding Model Test")

try:
    st.write("Attempting to load model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2",
        model_kwargs={"use_auth_token": st.secrets["HF_TOKEN"]}
    )
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model:\n\n{e}")
