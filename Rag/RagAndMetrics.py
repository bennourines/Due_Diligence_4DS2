"""
Optimized Retrieval-Augmented Generation (RAG) System with Evaluation Metrics
This file integrates all suggested enhancements while maintaining a clean, single-file structure.
uses all-mpnet-base-v2 for embeddings and cross-encoder/ms-marco-MiniLM-L-6-v2 for reranking.
different from prev rags that use qa
"""

import os
import numpy as np
import warnings
import requests
import json
import faiss
import pickle
import time
import functools
import hashlib
from typing import List, Dict, Any, Tuple, Generator, Optional, Union
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from cachetools import TTLCache, cached
import diskcache
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate import meteor_score
from rouge import Rouge
import re
from dotenv import load_dotenv


# Download NLTK data (first time only)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)

# Suppress warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', module='tensorflow')

# Load environment variables from .env file if present
load_dotenv(override=True)

# Configuration from environment variables with defaults
MODEL_NAME = os.getenv("RAG_MODEL_NAME", "sentence-transformers/multi-qa-mpnet-base-dot-v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
LLM_MODEL = "google/gemini-2.5-pro-exp-03-25"
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "../faiss_index_download/index.faiss")
METADATA_PATH = os.getenv("METADATA_PATH", "../faiss_index_download/merged_metadata.json")
DATA_DIR = os.getenv("DATA_DIR", "../DATA_COMPLETE/cleaned_texts")
USE_CACHE = os.getenv("USE_CACHE", "true").lower() == "true"

# Create cache directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)
disk_cache = diskcache.Cache(CACHE_DIR)

# Memory cache for fast lookups
memory_cache = TTLCache(maxsize=1000, ttl=3600)  # Cache with 1000 entries that expire after 1 hour

@dataclass
class EvaluationResult:
    """Container for RAG evaluation metrics"""
    retrieval_latency: float = 0.0
    generation_latency: float = 0.0
    total_latency: float = 0.0
    bleu_score: Optional[float] = None
    rouge_scores: Optional[Dict[str, Dict[str, float]]] = None
    meteor_score: Optional[float] = None
    retrieved_chunks: Optional[List[Dict]] = None
    answer: Optional[str] = None

class RAGSystem:
    """
    Enhanced Retrieval-Augmented Generation System
    Integrates vector search, keyword search, and reranking for improved retrieval accuracy.
    """
    
    def __init__(
        self, 
        model_name: str = MODEL_NAME,
        index_path: str = FAISS_INDEX_PATH,
        metadata_path: str = METADATA_PATH,
        api_key: str = OPENROUTER_API_KEY,
        llm_model: str = LLM_MODEL,
        use_cache: bool = USE_CACHE
    ):
        # Initialize parameters
        self.model_name = model_name
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.api_key = api_key
        self.llm_model = llm_model
        self.use_cache = use_cache
        
        # Initialize resources
        self.embedding_model = None
        self.cross_encoder = None
        self.index = None
        self.metadata = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # Load resources
        self._load_resources()
    
    def _load_resources(self):
        """Load embedding model, cross-encoder, and FAISS index"""
        print(f"Loading resources...")
        # Load embedding model
        self.embedding_model = self._load_embedding_model(self.model_name)
        
        # Load cross-encoder for reranking
        try:
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            print("Cross-encoder model loaded successfully")
        except Exception as e:
            print(f"Warning: Cross-encoder model could not be loaded: {str(e)}")
            self.cross_encoder = None
        
        # Load FAISS index
        self.index, self.metadata = self._load_faiss_index(self.index_path, self.metadata_path)
        if self.index is None or self.metadata is None:
            raise ValueError("Failed to load index and metadata")
        
        # Initialize TF-IDF vectorizer if we have metadata
        if self.metadata:
            self._initialize_tfidf()
    
    def _initialize_tfidf(self):
        """Initialize TF-IDF vectorizer with the corpus"""
        try:
            # Create corpus from metadata
            corpus = [item["text"] for item in self.metadata]
            
            # Create TF-IDF vectorizer and transform corpus
            self.tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
            print(f"TF-IDF vectorizer initialized with {len(corpus)} documents")
        except Exception as e:
            print(f"Warning: Failed to initialize TF-IDF vectorizer: {str(e)}")
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
    
    def _load_embedding_model(self, model_name: str):
        """Load SentenceTransformer model for generating embeddings"""
        cache_key = f"embedding_model_{model_name}"
        
        # Try to get from memory cache first
        if self.use_cache and cache_key in memory_cache:
            print(f"Loading embedding model {model_name} from memory cache")
            return memory_cache[cache_key]
        
        # Try to get from disk cache
        if self.use_cache:
            cached_model = disk_cache.get(cache_key)
            if cached_model is not None:
                print(f"Loading embedding model {model_name} from disk cache")
                memory_cache[cache_key] = cached_model
                return cached_model
        
        # Load model from scratch
        print(f"Loading SentenceTransformer model: {model_name}")
        model = SentenceTransformer(model_name)
        
        # Cache the model
        if self.use_cache:
            memory_cache[cache_key] = model
            disk_cache.set(cache_key, model)
        
        return model
    
    def _load_faiss_index(self, index_path: str, metadata_path: str):
        """Load existing FAISS index and metadata"""
        print(f"Loading FAISS index from {index_path}...")
        print(f"Loading metadata from {metadata_path}...") # Added for clarity

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            print("Index or metadata file not found.")
            return None, None
        
        try:
            # Load the FAISS index (binary)
            index = faiss.read_index(index_path)

            # Load the metadata (JSON)
            # Use 'r' for text mode and json.load
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            print(f"Loaded index with {index.ntotal} vectors and {len(metadata)} metadata entries")
            return index, metadata
        except json.JSONDecodeError as json_err: # Catch JSON specific errors
            print(f"Error loading metadata JSON: {str(json_err)}")
            return None, None
        except Exception as e:
            # Catch other potential errors (e.g., FAISS loading)
            print(f"Error loading index or metadata: {str(e)}")
            return None, None
    
    def semantic_search(self, question: str, top_k: int = 5) -> List[Dict]:
        """Perform semantic search using FAISS"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([question], show_progress_bar=False)
            
            # Search the index
            distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)
            
            # Get the relevant chunks
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.metadata):
                    # Get the full item {"text": ..., "metadata": ...}
                    item = self.metadata[idx].copy()
                    # Add distance score (convert to similarity score)
                    item["score"] = float(1.0 / (1.0 + distances[0][i]))  # Convert distance to similarity
                    item["retrieval_method"] = "semantic"
                    results.append(item) # Append the whole item

            return results
        except Exception as e:
            print(f"Error in semantic search: {str(e)}")
            return []
    
    def keyword_search(self, question: str, top_k: int = 5) -> List[Dict]:
        """Perform keyword-based search using TF-IDF"""
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            print("TF-IDF vectorizer not initialized. Skipping keyword search.")
            return []
        
        try:
            # Transform query and get scores
            query_tfidf = self.tfidf_vectorizer.transform([question])
            scores = (query_tfidf @ self.tfidf_matrix.T).toarray()[0]
            
            # Get top matches
            top_indices = np.argsort(-scores)[:top_k]
            
            # Create results
            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include results with matching terms
                    doc = self.metadata[idx].copy()
                    doc["score"] = float(scores[idx])
                    doc["retrieval_method"] = "keyword"
                    results.append(doc)
            
            return results
        except Exception as e:
            print(f"Error in keyword search: {str(e)}")
            return []
    
    def hybrid_search(self, question: str, top_k: int = 5) -> List[Dict]:
        """Combine semantic and keyword search results"""
        # Get semantic search results
        semantic_results = self.semantic_search(question, top_k=top_k*2)
        
        # Get keyword search results
        keyword_results = self.keyword_search(question, top_k=top_k*2)
        
        # Create a dictionary to store unique results by chunk_id
        combined_results = {}
        
         # Process semantic results
        for doc in semantic_results:
            # Access nested metadata for key creation
            source = doc.get('metadata', {}).get('source', 'unknown')
            chunk_id = doc.get('metadata', {}).get('chunk_id', 'unknown')
            key = f"{source}_{chunk_id}"
            doc["semantic_score"] = doc["score"]
            combined_results[key] = doc

        # Process keyword results and merge with semantic results
        for doc in keyword_results:
            # Access nested metadata for key creation
            source = doc.get('metadata', {}).get('source', 'unknown')
            chunk_id = doc.get('metadata', {}).get('chunk_id', 'unknown')
            key = f"{source}_{chunk_id}"
            keyword_score = doc["score"]

            if key in combined_results:
                # Update existing document
                combined_results[key]["keyword_score"] = keyword_score
                # Calculate combined score (adjust weights as needed)
                semantic_score = combined_results[key].get("semantic_score", 0)
                combined_results[key]["score"] = 0.7 * semantic_score + 0.3 * keyword_score # Example weighting
                combined_results[key]["retrieval_method"] = "hybrid"
            else:
                # Add new document from keyword search
                doc["keyword_score"] = keyword_score
                doc["semantic_score"] = 0 # No prior semantic match
                doc["score"] = 0.3 * keyword_score # Score based only on keyword match
                doc["retrieval_method"] = "keyword_only"
                combined_results[key] = doc

        # Convert to list and sort by score
        results = list(combined_results.values())
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]  # Limit to top_k results
    
    def rerank_chunks(self, question: str, chunks: List[Dict]) -> List[Dict]:
        """Rerank retrieved chunks using cross-encoder"""
        if not chunks:
            return []
            
        if self.cross_encoder is None:
            print("Cross-encoder not available. Skipping reranking.")
            return chunks
        
        try:
            # Create pairs of (question, chunk)
            pairs = [(question, chunk["text"]) for chunk in chunks]
            
            # Compute relevance scores
            scores = self.cross_encoder.predict(pairs)
            
            # Add scores to chunks
            for i, chunk in enumerate(chunks):
                chunk["cross_encoder_score"] = float(scores[i])
                # Update the final score to include cross-encoder score
                original_score = chunk["score"]
                chunk["score"] = 0.4 * original_score + 0.6 * float(scores[i])
            
            # Sort by updated score
            reranked_chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)
            
            return reranked_chunks
        except Exception as e:
            print(f"Error in reranking: {str(e)}")
            return chunks
    
    @cached(cache=memory_cache)
    def expand_query(self, query: str) -> List[str]:
        """Generate multiple search queries from the original query"""
        if not self.api_key:
            return [query]
            
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        prompt = (
            "You are a search query expansion tool. Given an original query, generate 2-3 alternative "
            "queries that could help retrieve more relevant information. Provide only the queries, "
            "separated by newlines, without any additional explanations.\n\n"
            f"Original query: {query}\n\n"
            "Alternative queries:"
        )
        data = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0.3
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                expanded_queries = response.json()["choices"][0]["message"]["content"].split("\n")
                # Remove any numbering or bullet points
                expanded_queries = [q.strip().lstrip("123456789.-*• ") for q in expanded_queries if q.strip()]
                return [query] + expanded_queries  # Include original query
            else:
                print(f"Error expanding query: {response.text}")
                return [query]
        except Exception as e:
            print(f"Exception in query expansion: {str(e)}")
            return [query]
    
    def compress_context(self, question: str, context: str) -> str:
        """Compress the context to the most relevant parts"""
        if not self.api_key or len(context) < 2000:  # Only compress if context is large
            return context
            
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        prompt = (
            "Your task is to extract only the most relevant information from the following context that helps "
            f"answer this question: \"{question}\"\n\n"
            f"Context:\n{context}\n\n"
            "Extract only the relevant facts and information needed to answer the question. "
            "Maintain critical details but remove redundant or irrelevant content."
        )
        data = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.3
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                compressed = response.json()["choices"][0]["message"]["content"]
                # Only use compressed context if it's not too short
                if len(compressed) > len(context) * 0.2:
                    print(f"Compressed context from {len(context)} to {len(compressed)} characters")
                    return compressed
        except Exception as e:
            print(f"Exception in context compression: {str(e)}")
        
        # Fall back to original context if compression fails
        print("Using original uncompressed context")
        return context
    
    def generate_answer(self, question: str, context: str, streaming: bool = False) -> Union[str, Generator[str, None, None]]:
        """Generate answer using LLM"""
        if not self.api_key:
            return "API key required for answer generation"
            
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        prompt = (
            "You are a highly knowledgeable expert in cryptocurrency, blockchain technology, "
            "and decentralized finance (DeFi), with extensive experience "
            "in both technical and practical aspects of the field. "
            "Your expertise includes understanding of consensus mechanisms, smart contracts, "
            "tokenomics, mining/staking processes, cryptocurrency exchanges, and regulatory "
            "frameworks across different jurisdictions. You will analyze the provided context thoroughly "
            "and respond to questions with precise, factual information supported by the given context. "
            "If the information cannot be found in the context, you will clearly state this limitation rather than making assumptions or providing speculative answers."
            " When answering, use clear, technical language while maintaining accessibility for both beginners and advanced users. "
            "Break down complex concepts when necessary, and provide relevant examples from the context to support your explanations."

            "Use the following context to answer the question. If you cannot find the answer in the context, "
            "explicitly state 'I cannot find this information in the provided context' and do not make up"
            "information that isn't directly supported by the context."

            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            "Answer: "
        )
        data = {
            "model": self.llm_model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 5000,
            "temperature": 0.6
        }
        
        if streaming:
            data["stream"] = True
            return self._stream_response(url, headers, data)
        else:
            return self._generate_full_response(url, headers, data)
    
    def _generate_full_response(self, url: str, headers: Dict, data: Dict) -> str:
        """Generate full response from LLM"""
        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                error_msg = f"Error calling LLM: {response.text}"
                print(error_msg)
                return error_msg
        except Exception as e:
            error_msg = f"Exception in answer generation: {str(e)}"
            print(error_msg)
            return error_msg
    
    def _stream_response(self, url: str, headers: Dict, data: Dict) -> Generator[str, None, None]:
        """Stream response from LLM"""
        try:
            response = requests.post(url, headers=headers, json=data, stream=True)
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        if line_text.startswith('data: ') and not line_text.startswith('data: [DONE]'):
                            try:
                                chunk_data = json.loads(line_text[6:])
                                content = chunk_data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                continue
            else:
                yield f"Error: {response.text}"
        except Exception as e:
            yield f"Exception in streaming: {str(e)}"
    
    def parallel_retrieve_and_rerank(self, questions: List[str], top_k: int = 5) -> List[List[Dict]]:
        """Process multiple questions in parallel"""
        with ProcessPoolExecutor(max_workers=min(os.cpu_count(), len(questions))) as executor:
            # Create a partial function with fixed arguments
            retrieve_func = functools.partial(
                self._retrieve_and_rerank_for_parallel,
                top_k=top_k
            )
            
            # Process in parallel
            results = list(executor.map(retrieve_func, questions))
            
        return results
    
    def _retrieve_and_rerank_for_parallel(self, question: str, top_k: int = 5) -> List[Dict]:
        """Helper function for parallel processing"""
        chunks = self.hybrid_search(question, top_k=top_k)
        return self.rerank_chunks(question, chunks)
    
    def query(self, 
              question: str, 
              top_k: int = 3, 
              use_hybrid: bool = True, 
              use_query_expansion: bool = True,
              use_reranking: bool = True,
              use_context_compression: bool = True,
              streaming: bool = False
             ) -> Tuple[Union[str, Generator[str, None, None]], EvaluationResult]:
        """
        Full RAG pipeline with evaluation metrics
        
        Parameters:
        - question: The question to answer
        - top_k: Number of chunks to retrieve
        - use_hybrid: Whether to use hybrid search (semantic + keyword)
        - use_query_expansion: Whether to expand the query for better retrieval
        - use_reranking: Whether to rerank retrieved chunks
        - use_context_compression: Whether to compress context before LLM generation
        - streaming: Whether to stream the response
        
        Returns:
        - Tuple of (answer, evaluation_result)
        """
        eval_result = EvaluationResult()
        
        # Start timing retrieval
        retrieval_start = time.time()
        
        # Expand query if enabled
        if use_query_expansion:
            expanded_queries = self.expand_query(question)
            print(f"Expanded queries: {expanded_queries}")
        else:
            expanded_queries = [question]
        
        # Retrieve chunks for all expanded queries
        all_chunks = []
        for query in expanded_queries:
            # Retrieve relevant chunks
            if use_hybrid:
                chunks = self.hybrid_search(query, top_k=top_k)
            else:
                chunks = self.semantic_search(query, top_k=top_k)
                
            if chunks:
                all_chunks.extend(chunks)
        
         # Remove duplicates based on chunk_id and source from nested metadata
        unique_chunks = {}
        for chunk in all_chunks:
            # Access nested metadata for key creation
            source = chunk.get('metadata', {}).get('source', 'unknown')
            chunk_id = chunk.get('metadata', {}).get('chunk_id', 'unknown')
            key = f"{source}_{chunk_id}"
            if key not in unique_chunks or chunk['score'] > unique_chunks[key]['score']:
                unique_chunks[key] = chunk

        relevant_chunks = list(unique_chunks.values())

        # Sort by score and limit to top_k
        relevant_chunks.sort(key=lambda x: x['score'], reverse=True)
        relevant_chunks = relevant_chunks[:top_k]
        
        # Rerank chunks if enabled and we have chunks
        if use_reranking and relevant_chunks:
            relevant_chunks = self.rerank_chunks(question, relevant_chunks)
        
        # Record retrieval latency
        retrieval_end = time.time()
        eval_result.retrieval_latency = retrieval_end - retrieval_start
        eval_result.retrieved_chunks = relevant_chunks
        
        if not relevant_chunks:
            eval_result.total_latency = eval_result.retrieval_latency
            return "No relevant information found for your question.", eval_result
        
        # Combine retrieved chunks into context
        # Access 'source' from nested metadata, 'text' from top level
        context = "\n\n".join([f"[Source: {chunk.get('metadata', {}).get('source', 'N/A')}, Score: {chunk['score']:.3f}]\n{chunk.get('text', '')}"
                            for chunk in relevant_chunks])

        
        # Compress context if enabled
        if use_context_compression:
            context = self.compress_context(question, context)
        
        # Generate answer and measure latency
        generation_start = time.time()
        answer = self.generate_answer(question, context, streaming)
        
        # If not streaming, we can measure generation latency
        if not streaming:
            generation_end = time.time()
            eval_result.generation_latency = generation_end - generation_start
            eval_result.total_latency = eval_result.retrieval_latency + eval_result.generation_latency
            eval_result.answer = answer
        else:
            # For streaming we can't measure until complete
            eval_result.total_latency = eval_result.retrieval_latency
        
        return answer, eval_result
    
    def evaluate_answer(self, question: str, answer: str, reference_answer: str = None) -> Dict:
        """Evaluate answer quality using NLP metrics"""
        results = {}
        
        if not answer:
            return {
                "error": "No answer provided for evaluation"
            }
        
        # If we have a reference answer, calculate reference-based metrics
        if reference_answer:
            try:
                # Calculate BLEU score
                reference_tokens = nltk.word_tokenize(reference_answer.lower())
                candidate_tokens = nltk.word_tokenize(answer.lower())
                bleu = sentence_bleu([reference_tokens], candidate_tokens)
                results["bleu"] = bleu
                
                # Calculate METEOR score
                meteor = meteor_score.meteor_score([reference_tokens], candidate_tokens)
                results["meteor"] = meteor
                
                # Calculate ROUGE scores
                rouge = Rouge()
                rouge_scores = rouge.get_scores(answer, reference_answer)[0]
                results["rouge"] = rouge_scores
            except Exception as e:
                print(f"Error calculating reference-based metrics: {str(e)}")
                results["reference_metrics_error"] = str(e)
        
        # Calculate reference-free metrics
        try:
            # Answer length
            results["answer_length"] = len(answer)
            results["answer_word_count"] = len(answer.split())
            
            # Question terms present in answer
            question_terms = set(re.findall(r'\b\w+\b', question.lower()))
            answer_terms = set(re.findall(r'\b\w+\b', answer.lower()))
            term_overlap = len(question_terms.intersection(answer_terms))
            results["question_term_overlap"] = term_overlap
            results["question_term_overlap_ratio"] = term_overlap / len(question_terms) if question_terms else 0
            
            # Detect uncertainty markers
            uncertainty_phrases = ["I cannot find", "not available", "unclear", "unknown", 
                                  "not mentioned", "cannot determine", "not provided"]
            results["contains_uncertainty"] = any(phrase in answer.lower() for phrase in uncertainty_phrases)
            
        except Exception as e:
            print(f"Error calculating reference-free metrics: {str(e)}")
            results["content_metrics_error"] = str(e)
            
        return results



def test_rag_system(rag: RAGSystem): # Pass the initialized RAG instance
    """Test various functionalities of the RAG system"""
    print("\n===== Testing RAG System =====\n")

    # Test questions
    test_questions = [
        "How safe and reliable are online and virtual payment and wallet platforms for cryptocurrency transactions?",
        "What are the main challenges in blockchain scalability?",
        "How do regulatory frameworks impact cryptocurrency adoption?",
        "Explain the concept of Proof-of-Work.", # Add a question likely answerable
        "What is the price of Bitcoin today?" # Add a question likely *not* answerable from typical context
    ]

    # Use the first question for detailed single tests
    q = test_questions[0]
    print(f"Using primary test question: {q}")

    # Test 1: Basic retrieval (Semantic)
    print("\n----- Test 1: Semantic Retrieval -----")
    semantic_results = rag.semantic_search(q, top_k=3)
    print(f"\nSemantic Search Results ({len(semantic_results)} chunks found):")
    if semantic_results:
        for i, res in enumerate(semantic_results):
            # Access 'source' from nested metadata, 'text' from top level
            source_name = res.get('metadata', {}).get('source', 'N/A')
            text_preview = res.get('text', '')[:100]
            print(f"{i+1}. Score: {res['score']:.4f}, Source: {source_name}") # Use Source
            print(f"   {text_preview}...\n")
    else:
        print("No semantic results found.")

     # Test 2: Hybrid search
    print("\n----- Test 2: Hybrid Search -----")
    hybrid_results = rag.hybrid_search(q, top_k=3)
    print(f"\nHybrid Search Results ({len(hybrid_results)} chunks found):")
    if hybrid_results:
        for i, res in enumerate(hybrid_results):
            semantic_score = res.get('semantic_score', 0)
            keyword_score = res.get('keyword_score', 0)
            # Access 'source' from nested metadata, 'text' from top level
            source_name = res.get('metadata', {}).get('source', 'N/A')
            text_preview = res.get('text', '')[:100]
            print(f"{i+1}. Combined Score: {res['score']:.4f} (Semantic: {semantic_score:.4f}, Keyword: {keyword_score:.4f})")
            print(f"   Source: {source_name}") # Use Source
            print(f"   {text_preview}...\n")
    else:
        print("No hybrid results found.")

    # Test 3: Reranking
    print("\n----- Test 3: Reranking -----")
    if rag.cross_encoder and hybrid_results:
        # Pass a copy to avoid modifying the original list used later
        reranked_results = rag.rerank_chunks(q, hybrid_results[:])
        print(f"\nReranked Results ({len(reranked_results)} chunks):")
        for i, res in enumerate(reranked_results):
            # Access 'source' from nested metadata, 'text' from top level
            source_name = res.get('metadata', {}).get('source', 'N/A')
            text_preview = res.get('text', '')[:100]
            print(f"{i+1}. Final Score: {res['score']:.4f}, Cross-Encoder: {res.get('cross_encoder_score', 0):.4f}")
            print(f"   Source: {source_name}") # Use Source
            print(f"   {text_preview}...\n")
    elif not hybrid_results:
         print("No results from hybrid search to rerank.")
    else:
        print("Cross-encoder not available - skipping reranking test")

    # Test 4: Query expansion
    print("\n----- Test 4: Query Expansion -----")
    expanded = rag.expand_query(q)
    print(f"Original Query: {q}")
    print(f"Expanded Queries ({len(expanded)}):")
    for i, eq in enumerate(expanded):
        print(f"{i+1}. {eq}")

    # Test 5: Full RAG pipeline (Non-Streaming)
    print("\n----- Test 5: Full RAG Pipeline (Non-Streaming) -----")
    print(f"Question: {q}")
    print("\nGenerating answer (non-streaming)...")
    answer, eval_result = rag.query(
        q,
        top_k=3,
        use_hybrid=True,
        use_query_expansion=True, # Set flags as desired for testing
        use_reranking=True,
        use_context_compression=True,
        streaming=False
    )

    

    print(f"\nGenerated Answer:\n{answer}")
    print(f"\nEvaluation Result (Non-Streaming):")
    print(f"  Retrieval Latency: {eval_result.retrieval_latency:.4f}s")
    print(f"  Generation Latency: {eval_result.generation_latency:.4f}s")
    print(f"  Total Latency: {eval_result.total_latency:.4f}s")
    print(f"  Retrieved Chunks: {len(eval_result.retrieved_chunks) if eval_result.retrieved_chunks else 0}")
    if eval_result.retrieved_chunks:
        # Access 'source' from nested metadata in the first chunk
        source_name = eval_result.retrieved_chunks[0].get('metadata', {}).get('source', 'N/A')
        print(f"    Top Chunk Score: {eval_result.retrieved_chunks[0]['score']:.4f}")
        print(f"    Top Chunk Source: {source_name}") # Use Source

    


    # Test 6: Full RAG pipeline (Streaming)
    print("\n----- Test 6: Full RAG Pipeline (Streaming) -----")
    print(f"Question: {test_questions[1]}") # Use a different question
    print("\nGenerating answer (streaming)...")
    stream_answer_gen, stream_eval_result = rag.query(
        test_questions[1],
        top_k=3,
        use_hybrid=True,
        use_query_expansion=False, # Try different flags
        use_reranking=True,
        use_context_compression=False,
        streaming=True
    )

    print("Streaming Answer Chunks:")
    full_streamed_answer = ""
    try:
        for chunk in stream_answer_gen:
            print(chunk, end="", flush=True)
            full_streamed_answer += chunk
    except Exception as e:
        print(f"\nError during streaming: {e}")
    print("\n--- End of Stream ---")
    print(f"\nStreaming Eval Result (Retrieval Only):")
    print(f"  Retrieval Latency: {stream_eval_result.retrieval_latency:.4f}s")
    print(f"  Retrieved Chunks: {len(stream_eval_result.retrieved_chunks) if stream_eval_result.retrieved_chunks else 0}")


    # Test 7: Evaluation Metrics
    print("\n----- Test 7: Evaluation Metrics -----")
    # Use the answer from the non-streaming test
    print(f"Evaluating answer for question: {q}")
    # Create a hypothetical reference answer for testing metrics
    reference_answer = "Online cryptocurrency platforms vary in safety. Reliability depends on the platform's security measures like encryption and multi-factor authentication, and user vigilance against phishing. Virtual wallets can be secure if private keys are managed properly."

    eval_metrics_with_ref = rag.evaluate_answer(q, answer, reference_answer)
    print(f"\nEvaluation Metrics (vs. Reference):")
    for key, value in eval_metrics_with_ref.items():
        if isinstance(value, dict): # Handle ROUGE scores
            print(f"  {key}:")
            for r_key, r_value in value.items():
                 print(f"    {r_key}: {r_value}")
        else:
            print(f"  {key}: {value}")


    eval_metrics_no_ref = rag.evaluate_answer(q, answer)
    print(f"\nEvaluation Metrics (No Reference):")
    for key, value in eval_metrics_no_ref.items():
         print(f"  {key}: {value}")

    # Test evaluation with an uncertain answer
    uncertain_answer_test, _ = rag.query(test_questions[4], top_k=1, streaming=False)
    print(f"\nEvaluating potentially uncertain answer for: {test_questions[4]}")
    print(f"Generated Answer: {uncertain_answer_test}")
    uncertain_eval = rag.evaluate_answer(test_questions[4], uncertain_answer_test)
    print(f"Evaluation Metrics (Uncertainty Check):")
    for key, value in uncertain_eval.items():
         print(f"  {key}: {value}")


    # Test 8: Parallel Retrieval
    print("\n----- Test 8: Parallel Retrieval -----")
    print(f"Processing {len(test_questions)} questions in parallel...")
    start_parallel = time.time()
    parallel_results = rag.parallel_retrieve_and_rerank(test_questions, top_k=2)
    end_parallel = time.time()
    print(f"Parallel processing took {end_parallel - start_parallel:.4f} seconds.")
    print(f"\nParallel Retrieval Results ({len(parallel_results)} questions processed):")
    for i, p_res in enumerate(parallel_results):
        print(f"  Question {i+1} ('{test_questions[i][:30]}...'): Found {len(p_res)} chunks")
        if p_res:
             # Access 'source' from nested metadata in the first chunk
             source_name = p_res[0].get('metadata', {}).get('source', 'N/A')
             print(f"    Top chunk score: {p_res[0]['score']:.4f}, Source: {source_name}") # Use Source
        else:
            print("    No chunks found.")

    

    print("\n===== RAG System Testing Complete =====\n")



def main():
    """
    Main function to initialize and test the RAGSystem.
    """
    print("--- Initializing RAG System ---")
    start_time = time.time()
    try:
        # Initialize the RAG system
        # Ensure your .env file is correctly set up with API keys and paths,
        # or that the default paths point to your index files.
        rag_system = RAGSystem()
        init_time = time.time()
        print(f"--- RAG System Initialized Successfully ({init_time - start_time:.2f}s) ---")

        # Run the comprehensive test function
        test_rag_system(rag_system)

    except ValueError as ve:
        print(f"\nError initializing RAG System: {ve}")
        print("Please ensure FAISS index and the merged metadata file exist at the specified paths:") # Updated message
        print(f"  Index: {os.getenv('FAISS_INDEX_PATH', '../faiss_index_download/index.faiss')}")
        print(f"  Metadata: {os.getenv('METADATA_PATH', '../faiss_index_download/merged_metadata.json')}") # Updated path
    
    except ImportError as ie:
         print(f"\nImport Error: {ie}")
         print("Please ensure all required dependencies are installed:")
         print("pip install faiss-cpu sentence-transformers requests nltk rouge-score numpy cachetools diskcache python-dotenv scikit-learn")
    except Exception as e:
        # Catch any other unexpected errors during initialization or testing
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging

    end_time = time.time()
    print(f"--- Total Execution Time: {end_time - start_time:.2f}s ---")


if __name__ == "__main__":
    # This ensures the main function runs only when the script is executed directly
    main()