import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd

PDF_FOLDER = "../data/pdfs"
OUTPUT_CSV = "../data/embeddings/pdf_chunks.csv"

all_chunks = []

for filename in os.listdir(PDF_FOLDER):
    if filename.endswith(".pdf"):
        path = os.path.join(PDF_FOLDER, filename)
        reader = PyPDF2.PdfReader(path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        # Split text into chunks (~500 tokens, overlap 50)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)
        for chunk in chunks:
            all_chunks.append({"pdf": filename, "text": chunk})

df = pd.DataFrame(all_chunks)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(df)} chunks to {OUTPUT_CSV}")
