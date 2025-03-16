import streamlit as st
from vipas import model
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import pdfplumber
import os
import asyncio
import io

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

### 🔹 **Step 1: Read PDF and Upsert While Reading**
async def process_and_upsert_pdf(file):
    """Extract text from a PDF page-by-page and upsert it into Pinecone."""
    
    async def upsert_embedding(text, page_num):
        """Generate embedding and upsert to Pinecone."""
        if text.strip():  # Avoid empty pages
            embedding = embedding_model.encode([text])[0].tolist()
            vector_id = f"{file.name}_page_{page_num}"
            index.upsert([
                {
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {"text": text, "page": page_num, "source": file.name}
                }
            ])
            print(f"✅ Upserted: {vector_id}")

    with pdfplumber.open(io.BytesIO(file.read())) as pdf:
        tasks = []
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            tasks.append(upsert_embedding(text, page_num))
        
        # Execute all upserts asynchronously
        await asyncio.gather(*tasks)

### 🔹 **Step 2: Retrieve Context from Pinecone**
async def retrieve_context(query, top_k=3):
    """Retrieves relevant pages from Pinecone."""
    try:
        query_embedding = embedding_model.encode([query])[0].tolist()
        response = await asyncio.get_event_loop().run_in_executor(
            None, lambda: index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
        )
        return response if response and response.get("matches") else None
    except Exception as e:
        st.error(f"❌ Error retrieving context: {e}")
        return None

### 🔹 **Step 3: Query the LLM Model**
def query_llm(query, context):
    """Queries the LLM model using retrieved context."""
    prompt = (
        "You are an AI assistant. Answer using the provided context:\n\n"
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
        response = client.predict(model_id=LLAMA_MODEL_ID, input_data=payload)
        return response['outputs'][0]['data'][0]
    except Exception as e:
        st.error(f"❌ Error querying the LLM: {e}")
        return ""

### 🔹 **Step 4: Streamlit UI**
st.title("📄 RAG-based Q&A with Vipas LLM (Using Pinecone)")
st.write("Upload a document and ask questions.")

uploaded_file = st.file_uploader(
    "📂 Upload a document (Max 5MB)", 
    type=["pdf"], 
    accept_multiple_files=False
)

if uploaded_file:
    if uploaded_file.size > 5 * 1024 * 1024:  # 5MB limit
        st.error("🚨 File too large! Please upload a smaller file.")
        st.stop()

    # Step 1: Process the file while upserting
    st.write("📖 Processing and indexing the document...")
    asyncio.run(process_and_upsert_pdf(uploaded_file))
    st.success("✅ Document processed and indexed!")

    # Step 2: Accept user query
    query = st.text_input("🔍 Enter your query:")

    # **Submit Button**
    if st.button("Submit Query") and query:
        # Step 3: Retrieve context
        context_result = asyncio.run(retrieve_context(query))
        
        if context_result:
            context = " ".join([match["metadata"]["text"] for match in context_result["matches"]])[:170]
            
            # Step 4: Query LLM
            st.write("🤖 Generating response...")
            response = query_llm(query, context)
            
            st.write("### ✨ AI Response")
            st.write(response)
        else:
            st.warning("⚠️ No relevant context found.")
