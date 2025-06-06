import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Paramètres
INDEX_PATH = "vectorstore/index.faiss"
CHUNKS_PATH = "data/chunks.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3  # Nombre de chunks à retourner

# Charger l'index FAISS
def load_faiss_index(index_path=INDEX_PATH):
    index = faiss.read_index(index_path)
    return index

# Charger les chunks (pour afficher le texte correspondant)
def load_chunks(path=CHUNKS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Charger le modèle d'embedding
def load_model(model_name=MODEL_NAME):
    return SentenceTransformer(model_name)

# Embed la question
def embed_query(query, model):
    return model.encode([query])

# Rechercher les k chunks les plus proches
def search(query, index, chunks, model, k=TOP_K):
    query_vector = embed_query(query, model)
    distances, indices = index.search(query_vector, k)
    results = [chunks[i] for i in indices[0]]
    return results

# Test rapide
if __name__ == "__main__":
    model = load_model()
    index = load_faiss_index()
    chunks = load_chunks()

    question = input("Pose ta question : ")
    top_chunks = search(question, index, chunks, model)

    print("\n🔎 Chunks les plus pertinents :\n")
    for i, chunk in enumerate(top_chunks, 1):
        print(f"Chunk {i}:\n{chunk}\n{'-'*50}")
