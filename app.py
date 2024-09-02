import streamlit as st
from pdf2image import convert_from_bytes
from pytesseract import image_to_string
from PIL import Image
import chromadb
from sentence_transformers import SentenceTransformer
import os
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ChromaDB client
client = chromadb.Client()

# SentenceTransformer model for embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to convert PDF to images
def convert_pdf_to_img(pdf_bytes):
    return convert_from_bytes(pdf_bytes)

# Function to convert image to text using OCR
def convert_image_to_text(image):
    text = image_to_string(image)
    return text

# Function to extract text from any PDF
def get_text_from_any_pdf(pdf_bytes):
    images = convert_pdf_to_img(pdf_bytes)
    final_text = ""
    for img in images:
        final_text += convert_image_to_text(img)
    return final_text

# Function to chunk text
def chunk_text(text, chunk_size=500, chunk_overlap=15):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    return text_splitter.split_text(text)

# Main Streamlit app
def main():
    st.title("PDF Text Extraction and Query System")

    # Multiple file upload
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        documents = []
        metadatas = []
        ids = []
        total_pdf_text = ""

        # Process each uploaded PDF
        for uploaded_file in uploaded_files:
            with st.spinner(f'Extracting text from {uploaded_file.name}...'):
                pdf_bytes = uploaded_file.read()  # Read the file contents as bytes
                pdf_text = get_text_from_any_pdf(pdf_bytes)  # Extract text
                total_pdf_text += pdf_text
                st.success(f"Text extraction from {uploaded_file.name} completed.")

                # Chunk the extracted text
                chunks = chunk_text(pdf_text)
                for chunk_idx, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadatas.append({"source": uploaded_file.name, "chunk_idx": chunk_idx})
                    ids.append(f"{uploaded_file.name}_{chunk_idx+1}")

        # Check if the collection already exists
        collection_name = "docu"
        try:
            # Create or get existing collection
            docu = client.get_or_create_collection(collection_name)
        except chromadb.db.base.UniqueConstraintError:
            # If already exists, get the collection
            docu = client.get_collection(collection_name)

        # Embed documents and add to ChromaDB
        document_embeddings = model.encode(documents)
        docu.add(
            documents=documents,
            embeddings=[embedding.tolist() for embedding in document_embeddings],
            metadatas=metadatas,
            ids=ids
        )
        
        st.success("All PDFs successfully processed and stored in vector database.")

        # User query input
        query = st.text_input("Enter your query:")

        if query:
            # Query the database
            results = docu.query(query_texts=[query], n_results=5)
            st.write("Query Results:")
            for result in results['documents']:
                st.write(result)

# Run the app
if __name__ == "__main__":
    main()

