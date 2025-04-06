import os
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Paths
pdf_folder = r"C:\Users\AsA\RAG\New_PDFs"  # Update this path with your new PDFs folder
vectorstore_path = r"C:\Users\AsA\RAG\vectorstore"

# Load existing FAISS vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(vectorstore_path, embedding_model, allow_dangerous_deserialization=True)

def load_new_pdfs(pdf_folder):
    """Loads and extracts text from all PDFs in the folder."""
    all_documents = []
    
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_folder, filename)
            print(f"Processing: {filename}")
            
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            valid_docs = [doc for doc in documents if doc.page_content.strip()]
            
            if valid_docs:
                for doc in valid_docs:
                    doc.metadata["source"] = filename
                all_documents.extend(valid_docs)
            else:
                print(f"Skipping {filename}: No valid content found.")
    
    return all_documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Splits documents into chunks for better retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

# Load and process new documents
new_documents = load_new_pdfs(pdf_folder)
if new_documents:
    new_texts = split_documents(new_documents)
    
    # Add new documents to existing FAISS index
    vectorstore.add_documents(new_texts)

    # Save updated FAISS vectorstore
    vectorstore.save_local(vectorstore_path)
    print("Updated FAISS vectorstore saved successfully.")
else:
    print("No valid new documents found to add.")
