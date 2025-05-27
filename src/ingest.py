import os
import json
import logging

# Configurer le logger
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

DATA_DIR = "data/text_data"
OUTPUT_FILE = "data/chunks.json"
CHUNK_SIZE = 500  # nombre de mots par chunk

def load_articles_from_folder(folder_path=DATA_DIR):
    articles = []
    filenames = [f for f in os.listdir(folder_path) if f.endswith(".txt.clean")]
    logging.info(f"{len(filenames)} fichiers trouvés dans {folder_path}")
    for filename in filenames:
        filepath = os.path.join(folder_path, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
                articles.append(text)
            logging.info(f"Chargé fichier : {filename} (taille {len(text)} caractères)")
        except Exception as e:
            logging.error(f"Erreur lors de la lecture du fichier {filename} : {e}")
    return articles

def split_articles_into_chunks(articles, chunk_size=CHUNK_SIZE):
    chunks = []
    for idx, article in enumerate(articles):
        words = article.split()
        logging.info(f"Découpage article {idx+1} avec {len(words)} mots en chunks de {chunk_size} mots")
        #Si l'article est plus court que la taille de chunk, on le garde tel quel
        if len(words) <= CHUNK_SIZE:
            chunks.append(article.strip())
        else:
            for i in range(0, len(words), CHUNK_SIZE):
                chunk = " ".join(words[i:i + CHUNK_SIZE])
                chunks.append(chunk.strip())
    logging.info(f"Total chunks créés : {len(chunks)}")
    return chunks

def save_chunks(chunks, output_file=OUTPUT_FILE):
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        logging.info(f"Chunks sauvegardés dans {output_file}")
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde des chunks : {e}")

if __name__ == "__main__":
    logging.info("Démarrage de l'ingestion des articles")
    articles = load_articles_from_folder()
    logging.info(f"{len(articles)} articles chargés")

    chunks = split_articles_into_chunks(articles)

    save_chunks(chunks)
    logging.info("Fin du script d'ingestion")
