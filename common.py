import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

COLLECTION = "playlist_chunks"
EMBEDDING_SIZE = 384
EMBED_MODEL = "all-MiniLM-L6-v2"


def get_client():
    url, key = os.getenv("QDRANT_URL"), os.getenv("QDRANT_API_KEY")
    if not url or not key:
        raise RuntimeError("Set QDRANT_URL and QDRANT_API_KEY in your .env file")
    return QdrantClient(url=url, api_key=key, timeout=60)


def get_embedder():
    return SentenceTransformer(EMBED_MODEL)