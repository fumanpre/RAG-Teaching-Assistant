import faiss
import numpy as np
import joblib
from openai import OpenAI
import os

# Paths
EMBED_FILE = "../data/embeddings/chunks_with_embeddings.pkl"
FAISS_FILE = "../data/embeddings/faiss.index"

# 1️⃣ Load embeddings and FAISS index
df = joblib.load(EMBED_FILE)
index = faiss.read_index(FAISS_FILE)

# 2️⃣ Initialize OpenAI client
client = OpenAI(api_key="your_api_key_here")

# 3️⃣ Embedding function
def get_embedding(text):
    """Generate embeddings for query text."""
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(resp.data[0].embedding).astype("float32")

# 4️⃣ Retrieve top-k chunks with PDF info
def retrieve_chunks(query, top_k=5):
    """Retrieve most relevant text chunks from your PDF embeddings."""
    q_emb = get_embedding(query)
    distances, indices = index.search(np.array([q_emb]), top_k)
    chunks = [
        f"[PDF: {df.iloc[i]['pdf']}] {df.iloc[i]['text']}"
        for i in indices[0]
    ]
    return "\n\n".join(chunks)

# 5️⃣ Ask GPT-5 (supports MCQ / Fill-in / Normal Q&A)
def answer_question(query, options=None, mode="normal"):
    """
    Answer a question using retrieved PDF context.
    
    Parameters:
      - query: str — the question to ask.
      - options: dict — optional multiple-choice options, e.g. {"A": "Paris", "B": "Rome"}.
      - mode: str — "normal", "mcq", or "fill".
    """
    context = retrieve_chunks(query)

    if mode == "mcq" and options:
        # Dynamic handling of N options
        formatted_options = "\n".join([f"{k}) {v}" for k, v in options.items()])
        prompt = f"""
You are an expert assistant. Use the context below to select the correct multiple-choice answer.
Cite relevant PDF sources if possible.

Context:
{context}

Question:
{query}

Options:
{formatted_options}

Instructions:
- Select ONLY the best possible answer.
- Respond strictly in this format:
"<Letter>) <Answer> — [PDF: filename]"
"""
    elif mode == "fill":
        prompt = f"""
Use the following context to fill in the blank correctly.
Cite relevant PDF names if possible.

Context:
{context}

Question:
{query}

Instructions:
- Respond only with the correct word or phrase and include PDF source.
Example:
"Paris [PDF: Geography101.pdf]"
"""
    else:
        prompt = f"""
Use the following context to answer the question concisely.
Cite PDF names in your answer.

Context:
{context}

Question:
{query}

Answer:
"""

    # GPT-5 API call
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": prompt}],
        # 'temperature' defaults to 1.0 — required by gpt-5
    )

    return response.choices[0].message.content.strip()
