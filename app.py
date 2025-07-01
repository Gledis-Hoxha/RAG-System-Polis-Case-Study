import streamlit as st
import PyPDF2
import json
import os
import re
import gc
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading
import httpx
import asyncio

# Load environment variables
load_dotenv()

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 50
TOP_K = 5
BATCH_SIZE = 3
MAX_CHUNKS = 1500
DEFAULT_PROMPT = """Context: {context}\n\nQuestion: {question}\n\nPlease answer the question based on the context provided above. If the answer cannot be found in the context, say so."""

# FastAPI backend app
app = FastAPI()

# CORS config for FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages[:10]:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return clean_text(text)

def create_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length and len(chunks) < MAX_CHUNKS:
        end = start + chunk_size
        if end > text_length:
            end = text_length
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def save_to_json(chunks, embeddings, filename="data.json"):
    data = {
        "chunks": chunks,
        "embeddings": embeddings
    }
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_from_json(filename="data.json"):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return {"chunks": [], "embeddings": []}

# --- OpenAI client and functions ---

# Here using the official OpenAI python client:
import openai

def get_openai_api_key(user_api_key=None):
    return user_api_key or os.getenv("OPENAI_API_KEY")

def get_embedding(text, api_key):
    openai.api_key = api_key
    try:
        response = openai.Embedding.create(
            model="text-embedding-3-small",
            input=text
        )
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"Error getting embedding: {str(e)}")
        return None

def process_chunks_in_batches(chunks, api_key):
    embeddings = []
    total_chunks = len(chunks)
    for i in range(0, total_chunks, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        batch_embeddings = []
        for chunk in batch:
            embedding = get_embedding(chunk, api_key)
            if embedding is not None:
                batch_embeddings.append(embedding)
        embeddings.extend(batch_embeddings)
        gc.collect()
    return embeddings

def get_most_similar_chunks(query, chunks, embeddings, api_key, top_k=TOP_K):
    query_embedding = get_embedding(query, api_key)
    if query_embedding is None:
        return []
    similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]
    return [chunks[i] for i in top_indices]

def generate_response(query, context_chunks, prompt_template, extra_context, api_key):
    if not context_chunks and not extra_context:
        return "I couldn't find any relevant information to answer your question."
    context = "\n".join(context_chunks)
    if extra_context:
        context = extra_context + "\n" + context
    prompt = prompt_template.format(context=context, question=query)

    openai.api_key = api_key
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "Sorry, I encountered an error while generating the response."

# --- FastAPI endpoints ---

@app.post("/process_pdf")
async def process_pdf_endpoint(
    file: UploadFile = File(...),
    api_key: str = Form(...),
    chunk_size: int = Form(CHUNK_SIZE)
):
    try:
        text = extract_text_from_pdf(await file.read())
        chunks = create_chunks(text, chunk_size=chunk_size)
        embeddings = process_chunks_in_batches(chunks, api_key)

        if embeddings:
            save_to_json(chunks, embeddings)
            return {"status": "success", "message": f"Processed {len(chunks)} chunks"}
        else:
            return {"status": "error", "message": "Failed to generate embeddings"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/ask")
async def ask_endpoint(
    question: str = Form(...),
    api_key: str = Form(...),
    extra_context: str = Form("")
):
    try:
        data = load_from_json()
        chunks = data["chunks"]
        embeddings = data["embeddings"]

        if not chunks:
            return {"status": "error", "message": "No processed document found"}

        similar_chunks = get_most_similar_chunks(question, chunks, embeddings, api_key)
        answer = generate_response(question, similar_chunks, DEFAULT_PROMPT, extra_context, api_key)

        return {
            "status": "success",
            "answer": answer,
            "context_chunks": similar_chunks
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- Streamlit frontend UI ---

def streamlit_ui():
    st.title("RAG-based Document Q&A System")

    user_api_key = st.text_input("Enter your OpenAI API Key", type="password", help="Your key is only used in this session and never stored.")
    extra_context = st.text_area("Add extra context (optional)", value="", height=100, help="This context will be appended to the retrieved chunks.")
    uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])
    chunk_size = st.slider("Chunk Size", min_value=100, max_value=500, value=CHUNK_SIZE, step=50)
    prompt_template = st.text_area("Prompt Template", value=DEFAULT_PROMPT, height=150)

    if 'chunks' not in st.session_state:
        st.session_state['chunks'] = None
    if 'embeddings' not in st.session_state:
        st.session_state['embeddings'] = None
    if 'last_pdf' not in st.session_state:
        st.session_state['last_pdf'] = None
    if 'last_chunk_size' not in st.session_state:
        st.session_state['last_chunk_size'] = None
    if 'last_api_key' not in st.session_state:
        st.session_state['last_api_key'] = None

    # Helper async function to call FastAPI /process_pdf
    async def call_process_pdf(file_bytes, api_key, chunk_size):
        async with httpx.AsyncClient() as client:
            files = {'file': ('uploaded.pdf', file_bytes, 'application/pdf')}
            data = {'api_key': api_key, 'chunk_size': str(chunk_size)}
            resp = await client.post("http://localhost:8000/process_pdf", files=files, data=data)
            return resp.json()

    if uploaded_file is not None and user_api_key:
        # Process only if changed file or params
        if (
            st.session_state['last_pdf'] != uploaded_file.name or
            st.session_state['last_chunk_size'] != chunk_size or
            st.session_state['last_api_key'] != user_api_key
        ):
            file_bytes = uploaded_file.read()
            st.session_state['last_pdf'] = uploaded_file.name
            st.session_state['last_chunk_size'] = chunk_size
            st.session_state['last_api_key'] = user_api_key

            with st.spinner("Processing PDF and generating embeddings..."):
                result = asyncio.run(call_process_pdf(file_bytes, user_api_key, chunk_size))
            if result.get("status") == "success":
                st.success(result.get("message"))
                # Reload chunks and embeddings from file
                data = load_from_json()
                st.session_state['chunks'] = data["chunks"]
                st.session_state['embeddings'] = data["embeddings"]
            else:
                st.error(result.get("message"))

        else:
            st.success(f"Using cached data with {len(st.session_state['chunks']) if st.session_state['chunks'] else 0} chunks.")

    question = st.text_area("Ask a question about the uploaded document", height=100)

    # Async helper for /ask endpoint
    async def call_ask_api(question, api_key, extra_context):
        async with httpx.AsyncClient() as client:
            data = {
                "question": question,
                "api_key": api_key,
                "extra_context": extra_context
            }
            resp = await client.post("http://localhost:8000/ask", data=data)
            return resp.json()

    if st.button("Get Answer") and question.strip():
        if not user_api_key:
            st.warning("Please enter your OpenAI API Key.")
        elif not st.session_state.get('chunks'):
            st.warning("Please upload and process a PDF first.")
        else:
            with st.spinner("Generating answer..."):
                response = asyncio.run(call_ask_api(question, user_api_key, extra_context))
            if response.get("status") == "success":
                st.markdown("**Answer:**")
                st.write(response.get("answer"))
                st.markdown("---")
                st.markdown("**Retrieved Context Chunks:**")
                for chunk in response.get("context_chunks", []):
                    st.write(f"- {chunk[:300]}{'...' if len(chunk) > 300 else ''}")
            else:
                st.error(response.get("message"))

# Run FastAPI backend in a separate thread
def run_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=8000)

fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
fastapi_thread.start()

# Run Streamlit frontend
if __name__ == "__main__":
    streamlit_ui()
