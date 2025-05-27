## RAG Wiki Agent – Projet IA Générative

Ce projet vise à construire un agent intelligent capable de répondre automatiquement à des questions en langage naturel à partir d’un corpus d’articles Wikipedia, en utilisant une approche RAG (Retrieval Augmented Generation).

## Structure du projet


rag-wiki-agent/
│
├── data/               # Données brutes/décompressées : Contient les fichiers bruts d'articles Wikipedia.
├── models/             # Modèles ou checkpoints locaux
├── notebooks/          # Explorations et tests Jupyter
├── src/                # Scripts Python (pipeline RAG) :Contient les scripts pour ingestion, embeddings, recherche et génération.
│   ├── ingest.py       # Chargement + découpage du corpus
│   ├── embed.py        # Vectorisation des chunks
│   ├── retriever.py    # Recherche sémantique (FAISS ou autre)
│   ├── generator.py    # Appel LLM avec contexte
│   └── app.py          # CLI ou serveur (API) du chatbot
├── vectorstore/        # Index vectoriel (FAISS, Chroma...)
├── README.md
├── requirements.txt
├── Dockerfile
└── .gitignore
 


##  Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
