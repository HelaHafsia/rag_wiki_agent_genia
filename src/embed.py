import json
import os
import logging
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')

CHUNKS_PATH = "data/chunks.json"
INDEX_PATH = "vectorstore/index.faiss"
EMBEDDINGS_PATH = "vectorstore/embeddings.npy"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_chunks(path=CHUNKS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    logging.info(f"{len(chunks)} chunks chargés depuis {path}")
    return chunks

def embed_chunks(chunks, model_name=MODEL_NAME):
    model = SentenceTransformer(model_name)
    logging.info(f"Modèle chargé : {model_name}")
    embeddings = model.encode(chunks, show_progress_bar=True)
    return np.array(embeddings)

def save_embeddings(embeddings, index_path=INDEX_PATH, npy_path=EMBEDDINGS_PATH):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    logging.info(f"Index FAISS sauvegardé dans {index_path}")

    np.save(npy_path, embeddings)
    logging.info(f"Embeddings numpy sauvegardés dans {npy_path}")

if __name__ == "__main__":
    logging.info("▶ Démarrage de l'embedding des chunks")

    chunks = load_chunks()
    embeddings = embed_chunks(chunks)
    save_embeddings(embeddings)

    logging.info(" Embedding terminé")
