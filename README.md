# RAG-Teaching-Assistant

This project is a **Retrieval-Augmented Generation (RAG) based AI Teaching Assistant** that allows users to ask questions from their own PDF documents. The system ingests PDFs, creates embeddings, builds a FAISS index, and serves an interactive interface using **Streamlit**.

---

## Project Structure and Run Steps

```text
rag_gpt5/
├── data/
│   └── pdfs/              # Place your PDF files here
├── src/
│   ├── ingest_pdfs.py
│   ├── create_embeddings.py
│   ├── build_faiss.py
│   ├── create_embeddings.py
│   ├── query_rag.py
│   └── app.py
├── requirements.txt
├── Makefile
└── README.md



## Manual Steps

## 🔑 Step 0: Add Your API Key

Add your OpenAI API key in the following files:
rag_gpt5/src/query_rag.py
rag_gpt5/src/create_embeddings.py

## Step 1:Installing Dependencies (Without Makefile)

If you prefer not to use make:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


## Step 2: Add PDF Files

Place all your PDF documents into the following folder:
rag_gpt5/data/pdfs/

## Step 3: Run the Pipeline (Manual Method)

Navigate to the src directory:
cd rag_gpt5/src

Then run the following commands in order:
python3 ingest_pdfs.py
python3 create_embeddings.py
python3 build_faiss.py

Once indexing is complete, launch the application:
streamlit run app.py



## How to use Makefile

From the project root (rag_gpt5/):
make venv
make install
make ingest
make embed
make index
make run
