"""
Vector Database Module for RAG
Handles document storage, embedding, and retrieval
"""
import chromadb
from typing import List, Dict
import hashlib
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

class VectorStore:
    def __init__(self, persist_directory="/tmp/chroma_db"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # NO embedding model loaded here anymore!
        # ChromaDB will use its default when needed
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"description": "RAG document store"}
        )

    def add_documents(self, texts: List[str], metadatas: List[Dict] = None):
        if not texts:
            return
        
        print(f"Generating embeddings for {len(texts)} chunks...")
        
        # Generate unique IDs
        ids = [hashlib.md5(text.encode()).hexdigest() for text in texts]
        
        # Prepare metadata — ChromaDB needs all values as str
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in texts]
        else:
            metadatas = [{k: str(v) for k, v in m.items()} for m in metadatas]
        
        # Add to ChromaDB — embedding happens automatically
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✅ Added {len(texts)} chunks to vector store")

    def search(self, query: str, n_results: int = 3) -> List[Dict]:
        # ChromaDB handles embedding automatically
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i, doc in enumerate(results['documents'][0]):
                formatted_results.append({
                    'text': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else 0
                })
        return formatted_results

    def get_stats(self) -> Dict:
        count = self.collection.count()
        return {
            'total_chunks': count,
            'collection_name': self.collection.name
        }

    def clear(self):
        # ✅ FIXED: removed self.embedding_model (no longer exists)
        self.client.delete_collection(name="documents")
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"description": "RAG document store"}
        )
        print("✅ Vector store cleared")


# Global instance
vector_store = None

def get_vector_store():
    global vector_store
    if vector_store is None:
        vector_store = VectorStore(persist_directory="/tmp/chroma_db")
    return vector_store