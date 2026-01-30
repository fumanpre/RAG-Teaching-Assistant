# create_embeddings.py
from openai import OpenAI
import os
import pandas as pd
import joblib

# 1️⃣ Initialize OpenAI client
# Make sure your API key is set:
# export OPENAI_API_KEY="your_api_key_here"
client = OpenAI(api_key="your_api_key_here")

# 2️⃣ Load text chunks (from your processed PDF or notes)
# Example: CSV file with one text chunk per row
# Or you can modify this section to read from JSON / TXT
input_file = "../data/embeddings/pdf_chunks.csv"   # change path as needed
df = pd.read_csv(input_file)

if "text" not in df.columns:
    raise ValueError("Your input file must have a 'text' column with text chunks.")

# 3️⃣ Generate embeddings using GPT-5 compatible model
embeddings = []
print(f"Generating embeddings for {len(df)} chunks...")

for i, text in enumerate(df["text"]):
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",  # Use text-embedding-3-large for higher accuracy
            input=text
        )
        emb = response.data[0].embedding
        embeddings.append(emb)
        if i % 10 == 0:
            print(f"✅ Processed chunk {i+1}/{len(df)}")
    except Exception as e:
        print(f"❌ Error on chunk {i+1}: {e}")
        embeddings.append([])

# 4️⃣ Add embeddings to DataFrame
df["embedding"] = embeddings

# 5️⃣ Save to disk for retrieval use later
output_file = "../data/embeddings/chunks_with_embeddings.pkl"
joblib.dump(df, output_file)
print(f"\n💾 Embeddings saved to {output_file}")
print(f"✅ Total chunks embedded: {len(df)}")
