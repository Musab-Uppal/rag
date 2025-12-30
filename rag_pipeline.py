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
import json
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
        
        # API endpoint
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
    
    def _simple_embedding(self, text: str) -> np.ndarray:
        """Simple TF-IDF like embedding"""
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
        
        # Convert embeddings to numpy array for efficient computation
        embeddings_array = np.array(self.embeddings)
        
        # Calculate similarities
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            embeddings_array
        )[0]
        
        # Get top k indices
        indices = np.argsort(similarities)[-k:][::-1]
        
        # Return results
        results = []
        for idx in indices:
            results.append({
                'content': self.documents[idx],
                'similarity': float(similarities[idx]),
                'metadata': self.metadata[idx]
            })
        
        return results
    
    def _call_groq_api(self, prompt: str) -> str:
        """Make API call to Groq"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on provided document excerpts. If you cannot find the answer in the provided context, say so."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 500,
            "top_p": 0.9,
            "stream": False
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            print(f"API Response Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
                else:
                    return f"Unexpected API response: {json.dumps(result, indent=2)}"
            else:
                error_detail = ""
                try:
                    error_json = response.json()
                    error_detail = f" | Error: {error_json.get('error', {}).get('message', 'Unknown error')}"
                except:
                    error_detail = f" | Response: {response.text[:200]}"
                
                return f"API Error (Status: {response.status_code}{error_detail})"
                
        except requests.exceptions.Timeout:
            return "API request timed out. Please try again."
        except requests.exceptions.ConnectionError:
            return "Connection error. Please check your internet connection."
        except Exception as e:
            return f"Error calling API: {str(e)}"
    
    def query_documents(self, query: str, k: int = 4) -> Dict:
        """
        Query the documents and generate response using Groq API
        """
        # Search for similar documents
        similar_docs = self.search_similar(query, k=k)
        
        if not similar_docs:
            return {
                "answer": "No relevant documents found. Please upload documents first or try a different query.",
                "sources": []
            }
        
        # Combine context from similar documents
        context_parts = []
        for i, doc in enumerate(similar_docs[:k]):
            context_parts.append(f"--- Excerpt {i+1} ---\n{doc['content']}")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Please answer the following question based ONLY on the provided document excerpts.

DOCUMENT EXCERPTS:
{context}

QUESTION: {query}

INSTRUCTIONS:
1. Answer based only on the provided excerpts
2. If the information is not in the excerpts, say "I cannot find this information in the provided documents"
3. Be concise and factual
4. Reference specific excerpts when possible

ANSWER:"""
        
        # Call API
        answer = self._call_groq_api(prompt)
        
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