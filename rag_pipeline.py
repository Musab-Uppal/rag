import os
import tempfile
import numpy as np
from typing import List, Dict, Any
import requests
import PyPDF2
import docx
import warnings
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import hashlib
warnings.filterwarnings('ignore')

class SimpleRAGPipeline:
    def __init__(self, vector_db_path: str = "./faiss_db"):
        """
        Initialize a simple RAG pipeline without complex dependencies
        """
        # Simple embedding function (no sentence-transformers)
        self.vector_db_path = vector_db_path
        self.documents = []
        self.embeddings = []
        self.metadata = []
        
        # Load existing data if exists
        self._load_from_disk()
        
        # Groq API
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
    
    def _simple_embedding(self, text: str) -> np.ndarray:
        """Simple TF-IDF like embedding (fallback when no sentence-transformers)"""
        words = text.lower().split()
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Simple hash-based embedding (32 dimensions)
        embedding = np.zeros(32)
        for word, freq in word_freq.items():
            # Use hash to get consistent position
            pos = abs(hash(word)) % 32
            embedding[pos] += freq
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _load_from_disk(self):
        """Load saved embeddings from disk"""
        try:
            if os.path.exists(f"{self.vector_db_path}.pkl"):
                with open(f"{self.vector_db_path}.pkl", 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data.get('documents', [])
                    self.embeddings = data.get('embeddings', [])
                    self.metadata = data.get('metadata', [])
        except:
            pass
    
    def _save_to_disk(self):
        """Save embeddings to disk"""
        os.makedirs(os.path.dirname(self.vector_db_path), exist_ok=True)
        with open(f"{self.vector_db_path}.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embeddings': self.embeddings,
                'metadata': self.metadata
            }, f)
    
    def load_document(self, file_path: str) -> List[Dict]:
        """
        Load document based on file type
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == ".pdf":
            return self._load_pdf(file_path)
        elif file_extension == ".txt":
            return self._load_txt(file_path)
        elif file_extension in [".docx", ".doc"]:
            return self._load_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    def _load_pdf(self, file_path: str) -> List[Dict]:
        """Load PDF file"""
        documents = []
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    documents.append({
                        'text': text,
                        'metadata': {
                            'source': os.path.basename(file_path),
                            'page': page_num + 1,
                            'type': 'pdf'
                        }
                    })
        return documents
    
    def _load_txt(self, file_path: str) -> List[Dict]:
        """Load text file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        return [{
            'text': text,
            'metadata': {
                'source': os.path.basename(file_path),
                'page': 1,
                'type': 'txt'
            }
        }]
    
    def _load_docx(self, file_path: str) -> List[Dict]:
        """Load DOCX file"""
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)
        
        text = '\n'.join(full_text)
        return [{
            'text': text,
            'metadata': {
                'source': os.path.basename(file_path),
                'page': 1,
                'type': 'docx'
            }
        }]
    
    def split_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def create_vector_store(self, documents: List[Dict]):
        """
        Create vector store from documents
        """
        for doc in documents:
            chunks = self.split_text(doc['text'])
            
            for chunk in chunks:
                # Create embedding
                embedding = self._simple_embedding(chunk)
                
                # Store
                self.documents.append(chunk)
                self.embeddings.append(embedding)
                self.metadata.append(doc['metadata'])
        
        print(f"Created {len(self.documents)} chunks from documents")
        
        # Save to disk
        self._save_to_disk()
    
    def search_similar(self, query: str, k: int = 4) -> List[Dict]:
        """Search for similar documents"""
        if not self.documents:
            return []
        
        # Create query embedding
        query_embedding = self._simple_embedding(query)
        
        # Calculate similarities
        similarities = []
        for doc_embedding in self.embeddings:
            # Convert to 2D arrays for cosine_similarity
            sim = cosine_similarity(
                query_embedding.reshape(1, -1),
                doc_embedding.reshape(1, -1)
            )[0][0]
            similarities.append(sim)
        
        # Get top k indices
        indices = np.argsort(similarities)[-k:][::-1]
        
        # Return results
        results = []
        for idx in indices:
            results.append({
                'content': self.documents[idx],
                'similarity': similarities[idx],
                'metadata': self.metadata[idx]
            })
        
        return results
    
    def query_documents(self, query: str, k: int = 4) -> Dict:
        """
        Query the documents and generate response using Groq API
        """
        # Search for similar documents
        similar_docs = self.search_similar(query, k=k)
        
        if not similar_docs:
            return {
                "answer": "No relevant documents found.",
                "sources": []
            }
        
        # Combine context from similar documents
        context_parts = []
        for i, doc in enumerate(similar_docs[:k]):
            context_parts.append(f"[Document {i+1}]: {doc['content']}")
        
        context = "\n\n".join(context_parts)
        
        # Call Groq API
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""Based on the following document excerpts, answer the question. 
If the answer cannot be found in the documents, say "I cannot find this information in the provided documents."

DOCUMENTS:
{context}

QUESTION: {query}

ANSWER:"""
        
        payload = {
            "model": "mixtral-8x7b-32768",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                answer = response.json()['choices'][0]['message']['content']
            else:
                answer = f"Error from Groq API (Status: {response.status_code})"
        except Exception as e:
            answer = f"Error calling Groq API: {str(e)}"
        
        # Prepare sources
        sources = []
        for doc in similar_docs:
            sources.append({
                'content': doc['content'][:300] + "..." if len(doc['content']) > 300 else doc['content'],
                'metadata': doc['metadata'],
                'similarity': f"{doc['similarity']:.3f}"
            })
        
        return {
            "answer": answer,
            "sources": sources
        }