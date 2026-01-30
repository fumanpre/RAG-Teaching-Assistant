import joblib
import faiss
import numpy as np

EMBED_FILE = "../data/embeddings/chunks_with_embeddings.pkl"
FAISS_FILE = "../data/embeddings/faiss.index"

# Load embeddings using joblib
df = joblib.load(EMBED_FILE)

embedding_dim = len(df['embedding'][0])
index = faiss.IndexFlatL2(embedding_dim)

emb_matrix = np.array(df['embedding'].tolist()).astype("float32")
index.add(emb_matrix)

# Save FAISS index
faiss.write_index(index, FAISS_FILE)
print(f"FAISS index saved with {index.ntotal} vectors")
