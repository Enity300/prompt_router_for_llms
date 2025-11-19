import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from chromadb.config import Settings
from config import config

# Connect to your ChromaDB
client = chromadb.PersistentClient(path=config.CHROMADB_PATH, settings=Settings())
collection = client.get_collection(name=config.COLLECTION_NAME)

# Get a sample embedding to check dimension
result = collection.get(limit=1, include=["embeddings"])

if result['embeddings'] is not None and len(result['embeddings']) > 0:
    embedding_dim = len(result['embeddings'][0])
    print(f"ChromaDB Embedding Dimension: {embedding_dim}")
    print(f"Collection Name: {config.COLLECTION_NAME}")
    print(f"Total Embeddings: {collection.count()}")
else:
    print("No embeddings found in the collection")