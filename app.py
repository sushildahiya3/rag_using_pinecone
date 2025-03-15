import streamlit as st
from vipas import model
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import pdfplumber
from docx import Document
import pandas as pd
import numpy as np
import os
import asyncio

# Initialize Vipas SDK model client
client = model.ModelClient()
LLAMA_MODEL_ID = "mdl-b1mxve8nrq9cj"  

# Initialize SentenceTransformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY","pcsk_3Yg7jK_My6W7E4qLhbx1LYQu2P862chHfntFFkYftCtkJxPASXUdHsbYTV1BDmjHncmTSx")  # Set via environment variables
PINECONE_ENVIRONMENT = "aws-us-east-1"  

pc = Pinecone(api_key=PINECONE_API_KEY)

# Create or connect to an index
INDEX_NAME = "vipas-rag-index"
existing_indexes = pc.list_indexes().names()

if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# 🔹 Step 1: Optimize Document Processing (Streaming Processing)
def preprocess_document(file):
    """Processes PDF, DOCX, and Excel files in a memory-efficient way."""
    text = []
    try:
        if file.type == "application/pdf":
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    if page_text:
                        text.append(page_text)

        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(file)
            for para in doc.paragraphs:
                text.append(para.text)

        elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            for chunk in pd.read_excel(file, chunksize=500):
                text.append(chunk.to_string(index=False))

        else:
            st.error("Unsupported file type. Please upload a PDF, DOCX, or Excel file.")
            return ""

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        return ""

    return " ".join(text)  # Efficiently join text

# 🔹 Step 2: Optimize Embeddings Storage
async def store_embeddings(text, batch_size=4):  # Reduced batch size
    """Creates embeddings and stores them in Pinecone efficiently."""
    chunks = [text[i:i + 250] for i in range(0, len(text), 250)]  # Reduced chunk size
    chunks = [chunk for chunk in chunks if chunk.strip()]  # Remove empty chunks

    if not chunks:
        st.error("⚠️ No valid text found in the document.")
        return [], None

    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        try:
            batch_embeddings = embedding_model.encode(batch, batch_size=batch_size)  # Reduced memory load
        except Exception as e:
            st.error(f"❌ Error generating embeddings: {e}")
            return [], None

        async def upsert_data():
            vectors = [(f"doc_chunk_{i+j}", emb.tolist(), {"text": batch[j]}) for j, emb in enumerate(batch_embeddings)]
            index.upsert(vectors=vectors)

        await asyncio.get_event_loop().run_in_executor(None, upsert_data)
        embeddings.extend(batch_embeddings)

    return chunks, embeddings

# 🔹 Step 3: Optimize Context Retrieval
async def retrieve_context(query, top_k=3):
    """Retrieves relevant context from Pinecone."""
    query_embedding = embedding_model.encode([query]).tolist()[0]
    try:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, lambda: index.query(vector=query_embedding, top_k=top_k, include_metadata=True))

        if not results.matches:
            st.warning("⚠️ No relevant context found.")
            return ""

        retrieved_chunks = [match.metadata["text"] for match in results.matches]
        return " ".join(retrieved_chunks)[:170]

    except Exception as e:
        st.error(f"❌ Error retrieving context from Pinecone: {e}")
        return ""

# 🔹 Step 4: Query the Deployed LLM Model
def query_llm(query, context):
    """Queries the LLM model."""
    prompt = (
        "You are an expert. Answer the question using the provided context:\n\n"
        f"Context: {context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    payload = {
        "inputs": [
            {
                "name": "prompt",
                "shape": [1],
                "datatype": "BYTES",
                "data": [prompt]
            }
        ]
    }
    try:
        st.write("🤖 Querying the LLM model...")
        response = client.predict(model_id=LLAMA_MODEL_ID, input_data=payload)
        return response['outputs'][0]['data'][0]
    except Exception as e:
        st.error(f"❌ Error querying the LLM: {e}")
        return ""

# 🔹 Step 5: Streamlit UI
st.title("📄 RAG-based Q&A with Vipas LLM (Using Pinecone)")
st.write("Upload a document and ask questions using the LLM.")

# Limit file upload size to prevent OOM
uploaded_file = st.file_uploader("📂 Upload a file (Max 5MB)", type=["pdf", "docx", "xlsx"], accept_multiple_files=False)

if uploaded_file:
    if uploaded_file.size > 5 * 1024 * 1024:  # 5MB limit
        st.error("🚨 File too large! Please upload a file smaller than 5MB.")
        st.stop()

    # Step 1: Preprocess the file
    st.write("📖 Processing the file...")
    text = preprocess_document(uploaded_file)

    if text:
        # Step 2: Generate embeddings
        st.write("⚡ Generating embeddings and indexing...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        chunks, embeddings = loop.run_until_complete(store_embeddings(text))

        if embeddings is not None:
            st.success("✅ Document processed and indexed successfully!")

            # Step 3: Accept user query
            query = st.text_input("🔍 Enter your query:")

            if query:
                # Step 4: Retrieve relevant context
                context = loop.run_until_complete(retrieve_context(query))
                # Step 5: Query the LLM model
                st.write("🤖 Generating response from LLM...")
                response = query_llm(query, context)
                st.write("### ✨ Response")
                st.write(response)
