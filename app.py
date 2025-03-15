import streamlit as st
from vipas import model
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import pdfplumber
from docx import Document
import pandas as pd
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

### 🔹 Step 1: Efficient Document Streaming
def stream_text_chunks(file, chunk_size=250):
    """Streams text from a document in small chunks to reduce memory usage."""
    
    def process_text(text):
        """Yield text in small chunks instead of storing in memory."""
        buffer = []
        for word in text.split():
            buffer.append(word)
            if len(" ".join(buffer)) >= chunk_size:
                yield " ".join(buffer)
                buffer = []
        if buffer:
            yield " ".join(buffer)

    try:
        if file.type == "application/pdf":
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    yield from process_text(text)  # Stream chunks directly

        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(file)
            for para in doc.paragraphs:
                yield from process_text(para.text)  # Stream paragraphs chunk-wise

        elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            for chunk in pd.read_excel(file, chunksize=1):  # Process row-wise
                row_text = " ".join(str(cell) for cell in chunk.iloc[0].values)
                yield from process_text(row_text)  # Stream row-wise

        else:
            yield "Unsupported file type."

    except Exception as e:
        yield f"Error processing file: {e}"

### 🔹 Step 2: Upsert Embeddings Without Holding in Memory
async def store_embeddings(file):
    """Streams text chunks, generates embeddings, and upserts them immediately to Pinecone."""
    
    async def upsert_embedding(text_chunk):
        """Generate embedding and upsert it immediately."""
        embedding = embedding_model.encode([text_chunk])[0].tolist()
        index.upsert(vectors=[(f"doc_chunk_{hash(text_chunk)}", embedding, {"text": text_chunk})])

    tasks = [upsert_embedding(chunk) for chunk in stream_text_chunks(file)]
    await asyncio.gather(*tasks)  # Run all tasks asynchronously

### 🔹 Step 3: Retrieve Context Without Holding Memory
async def retrieve_context(query, top_k=3):
    """Retrieves relevant chunks from Pinecone without storing extra variables."""
    try:
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: index.query(
                vector=embedding_model.encode([query])[0].tolist(),  # Encode directly
                top_k=top_k, 
                include_metadata=True
            )
        )
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
        context_result = loop.run_until_complete(retrieve_context(query))

        if context_result and context_result.matches:
            context = " ".join([match.metadata["text"] for match in context_result.matches])[:170]
        
            # Step 4: Query the LLM model
            st.write("🤖 Generating response from LLM...")
            response = query_llm(query, context)
            
            st.write("### ✨ Response")
            st.write(response)
        else:
            st.warning("⚠️ No relevant context found.")
