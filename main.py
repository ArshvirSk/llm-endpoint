import uvicorn
from fastapi import FastAPI, Request, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
import requests
import tempfile
import fitz
import faiss
import os
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

from threading import Lock

# Auth Token (Hardcoded for challenge purposes)
AUTH_TOKEN = "14a917b2d58d35c9741f65d698be1f787283b1d8b606f4b121aeaacc51844167"

app = FastAPI()

# Lazy globals
model = None
gemini_model = None
index = None
chunks = []
lock = Lock()  # Ensure thread-safe lazy init

# Environment-safe port binding


@app.on_event("startup")
def configure_genai():
    genai.configure(api_key="AIzaSyBiIO4RIrlSA_jhEcI99YPCBRWkZhNxkyA")


class QueryRequest(BaseModel):
    documents: str
    questions: List[str]


class AnswerResponse(BaseModel):
    answers: List[str]


@app.post("/api/v1/hackrx/run", response_model=AnswerResponse)
async def run_submission(payload: QueryRequest, Authorization: Optional[str] = Header(None)):
    if Authorization != f"Bearer {AUTH_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    doc_url = payload.documents
    questions = payload.questions

    global index, chunks

    # Step 1: Download PDF to temp
    try:
        response = requests.get(doc_url)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to download document: {str(e)}")

    # Step 2: Extract and index chunks
    try:
        chunks = extract_chunks_from_pdf(tmp_path)
        index = build_faiss_index(chunks)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to process PDF: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Step 3: Answer questions using Gemini
    answers = []
    for q in questions:
        rel_chunks = search_chunks(q)
        ans = ask_llm_gemini(q, rel_chunks)
        answers.append(ans)

    return {"answers": answers}


# ============ Utility Functions ==============

def get_model():
    global model
    if model is None:
        with lock:
            if model is None:
                model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


def get_gemini_model():
    global gemini_model
    if gemini_model is None:
        with lock:
            if gemini_model is None:
                gemini_model = genai.GenerativeModel("gemini-2.5-pro")
    return gemini_model


def extract_chunks_from_pdf(pdf_path, max_len=400):
    doc = fitz.open(pdf_path)
    output = []
    for page in doc:
        for para in page.get_text().split("\n\n"):
            if len(para.strip()) > 50:
                output.append(para.strip())
    return output


def build_faiss_index(text_chunks):
    embedder = get_model()
    embeddings = embedder.encode(text_chunks)
    dim = embeddings[0].shape[0]
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(embeddings)
    return faiss_index


def search_chunks(query, top_k=5):
    global index, chunks
    embedder = get_model()
    query_vec = embedder.encode([query])
    _, I = index.search(query_vec, top_k)
    return [chunks[i] for i in I[0]]


def ask_llm_gemini(question, chunk_context):
    context = "\n".join(
        [f"Clause {i+1}: {c}" for i, c in enumerate(chunk_context)])
    prompt = f"""
You are an insurance policy assistant.
Based on the following question:
\"{question}\"
And using these relevant policy clauses:
{context}

Answer clearly and completely in 1-2 sentences. If no relevant info, say "Not mentioned in the document."
"""
    try:
        gemini = get_gemini_model()
        response = gemini.generate_content(prompt)
        return response.text.strip() if hasattr(response, "text") else "Error generating response."
    except Exception as e:
        return f"Error from Gemini: {str(e)}"


# Optional: Run for local testing
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
