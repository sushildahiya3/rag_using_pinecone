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
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_3Yg7jK_My6W7E4qLhbx1LYQu2P862chHfntFFkYftCtkJxPASXUdHsbYTV1BDmjHncmTSx")
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

### 🔹 Step 1: Stream & Chunk Documents Efficiently
def stream_text_chunks(file, chunk_size=250):
    """Streams text from a document in chunks to avoid loading into memory."""
    buffer = ""
    
    try:
        if file.type == "application/pdf":
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    for line in (page.extract_text() or "").split("\n"):
                        buffer += line + " "
                        if len(buffer) >= chunk_size:
                            yield buffer.strip()
                            buffer = ""

        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(file)
            for para in doc.paragraphs:
                buffer += para.text + " "
                if len(buffer) >= chunk_size:
                    yield buffer.strip()
                    buffer = ""

        elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            for chunk in pd.read_excel(file, chunksize=500):
                for row in chunk.itertuples(index=False):
                    row_text = " ".join(str(cell) for cell in row)
                    buffer += row_text + " "
                    if len(buffer) >= chunk_size:
                        yield buffer.strip()
                        buffer = ""

        else:
            st.error("Unsupported file type. Please upload a PDF, DOCX, or Excel file.")
            return

        if buffer:
            yield buffer.strip()  # Yield remaining text

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

### 🔹 Step 2: Store Embeddings Without Holding in Memory
async def store_embeddings(file, batch_size=4):
    """Streams text chunks, generates embeddings, and stores them in Pinecone."""
    chunks = []
    vectors = []
    batch_id = 0

    async def upsert_data():
        """Uploads a batch of embeddings to Pinecone asynchronously."""
        nonlocal vectors
        if vectors:
            index.upsert(vectors=vectors)
            vectors = []  # Clear memory

    for chunk in stream_text_chunks(file):
        chunks.append(chunk)
        
        # Generate embeddings in small batches
        if len(chunks) >= batch_size:
            batch_embeddings = embedding_model.encode(chunks, batch_size=batch_size)
            vectors.extend(
                [(f"doc_chunk_{batch_id+j}", emb.tolist(), {"text": chunks[j]}) for j, emb in enumerate(batch_embeddings)]
            )
            chunks = []
            batch_id += batch_size

            # Store in Pinecone without keeping embeddings in memory
            await asyncio.get_event_loop().run_in_executor(None, upsert_data)

    # Store remaining embeddings
    if chunks:
        batch_embeddings = embedding_model.encode(chunks, batch_size=batch_size)
        vectors.extend(
            [(f"doc_chunk_{batch_id+j}", emb.tolist(), {"text": chunks[j]}) for j, emb in enumerate(batch_embeddings)]
        )
        await asyncio.get_event_loop().run_in_executor(None, upsert_data)

### 🔹 Step 3: Retrieve Relevant Context Efficiently
async def retrieve_context(query, top_k=3):
    """Retrieves relevant chunks from Pinecone."""
    query_embedding = embedding_model.encode([query]).tolist()[0]
    
    try:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, lambda: index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        )

        if not results.matches:
            st.warning("⚠️ No relevant context found.")
            return ""

        return " ".join([match.metadata["text"] for match in results.matches])[:170]

    except Exception as e:
        st.error(f"❌ Error retrieving context from Pinecone: {e}")
        return ""

### 🔹 Step 4: Query the Deployed LLM Model
def query_llm(query, context):
    """Queries the LLM model using provided context."""
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

### 🔹 Step 5: Streamlit UI
st.title("📄 RAG-based Q&A with Vipas LLM (Using Pinecone)")
st.write("Upload a document and ask questions using the LLM.")

# Limit file upload size to prevent OOM
uploaded_file = st.file_uploader("📂 Upload a file (Max 5MB)", type=["pdf", "docx", "xlsx"], accept_multiple_files=False)

if uploaded_file:
    if uploaded_file.size > 5 * 1024 * 1024:  # 5MB limit
        st.error("🚨 File too large! Please upload a file smaller than 5MB.")
        st.stop()

    # Step 1: Preprocess the file
    st.write("📖 Processing and indexing the file in chunks...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(store_embeddings(uploaded_file))

    st.success("✅ Document processed and indexed successfully!")

    # Step 2: Accept user query
    query = st.text_input("🔍 Enter your query:")

    if query:
        # Step 3: Retrieve relevant context
        context = loop.run_until_complete(retrieve_context(query))
        
        # Step 4: Query the LLM model
        st.write("🤖 Generating response from LLM...")
        response = query_llm(query, context)
        
        st.write("### ✨ Response")
        st.write(response)
