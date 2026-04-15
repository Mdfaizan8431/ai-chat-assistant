"""
Vector Database Module for RAG
Handles document storage, embedding, and retrieval
"""

import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import hashlib
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

class VectorStore:
    def __init__(self, persist_directory="./chroma_db"):
        """Initialize ChromaDB and embedding model"""
        self.persist_directory = persist_directory
        
        # PersistentClient is the correct way in chromadb >= 0.4.x
        # This also fixes the telemetry crash automatically
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"description": "RAG document store"}
        )
        
        # Load embedding model (lightweight and fast)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded!")
        
    def add_documents(self, texts: List[str], metadatas: List[Dict] = None):
        """
        Add documents to the vector store
        
        Args:
            texts: List of text chunks
            metadatas: List of metadata dicts (filename, page, etc.)
        """
        if not texts:
            return
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Generate unique IDs
        ids = [hashlib.md5(text.encode()).hexdigest() for text in texts]
        
        # Prepare metadata — ChromaDB needs all values as str
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in texts]
        else:
            metadatas = [{k: str(v) for k, v in m.items()} for m in metadatas]
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Added {len(texts)} chunks to vector store")
        
    def search(self, query: str, n_results: int = 3) -> List[Dict]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of relevant documents with metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
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
        """Get statistics about the vector store"""
        count = self.collection.count()
        return {
            'total_chunks': count,
            'collection_name': self.collection.name
        }
    
    def clear(self):
        """Clear all documents from the collection"""
        # Delete and recreate collection
        self.client.delete_collection(name="documents")
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"description": "RAG document store"}
        )
        print("✅ Vector store cleared")

# Global instance
vector_store = None

def get_vector_store():
    """Get or create vector store instance"""
    global vector_store
    if vector_store is None:
        vector_store = VectorStore()
    return vector_store