"""
Optimized RAG System with Graph Expansion, Trimming, and Metrics
"""

import os
import json
import time
import faiss
import logging
import numpy as np
import diskcache
import pickle
from dataclasses import dataclass
from cachetools import TTLCache
from typing import List, Dict, Tuple 


import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Get absolute path to the vector stores directory
VECTOR_STORES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vector_stores')

# Configuration
CFG = {
    'MODEL_NAME': 'all-MiniLM-L6-v2',
    'FAISS_INDEX': os.path.join(VECTOR_STORES_DIR, 'index.faiss'),
    'METADATA_JSON': os.path.join(VECTOR_STORES_DIR, 'index_metadata.json'),
    'GRAPH_PATH': os.path.join(VECTOR_STORES_DIR, 'index.pkl'),
    'HF_MODEL': 'meta-llama/Llama-2-7b-chat-hf',
    'DEVICE': 'cpu',
    'USE_CACHE': True,
    'TOP_K_FAISS': 5,
    'TOP_K_FINAL': 3,
    'MAX_CTX_TOKENS': 2048,
    'MIN_CHUNK_WORDS': 20
}

# Create vector_stores directory if it doesn't exist
os.makedirs(VECTOR_STORES_DIR, exist_ok=True)

# Caches
memory_cache = TTLCache(maxsize=100, ttl=3600)
disk_cache = diskcache.Cache('./cache')

# Logging
logging.basicConfig(level=logging.INFO)

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

# Caching setup
disk_cache = diskcache.Cache('.cache')
memory_cache = TTLCache(maxsize=512, ttl=3600)

@dataclass
class EvaluationResult:
    retrieval_latency: float = 0.0
    generation_latency: float = 0.0
    total_latency:      float = 0.0
    bleu_score:        float = 0.0
    meteor_score:      float = 0.0
    rouge_scores:      dict  = None
    retrieved_chunks:  list  = None
    answer:            str   = ''

class RAGSystem:
    def __init__(self):
        self.index = None
        self.metadata = None
        self.graph = None
        self._load_resources()

    def _load_resources(self):
        # Check each required file
        missing_files = []
        if not os.path.exists(CFG['FAISS_INDEX']):
            missing_files.append(f"FAISS index file: {CFG['FAISS_INDEX']}")
        if not os.path.exists(CFG['METADATA_JSON']):
            missing_files.append(f"Metadata JSON file: {CFG['METADATA_JSON']}")
        
        if missing_files:
            error_msg = "Missing required files:\n" + "\n".join(missing_files)
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            # Load FAISS index
            logging.info(f"Loading FAISS index from {CFG['FAISS_INDEX']}")
            self.index = faiss.read_index(
                CFG['FAISS_INDEX'], faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
            )
            logging.info(f'Loaded FAISS index: {self.index.ntotal} vectors')
            
            # Load metadata
            logging.info(f"Loading metadata from {CFG['METADATA_JSON']}")
            with open(CFG['METADATA_JSON'], 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
                # Ensure each metadata entry has a text field
                for entry in self.metadata:
                    if 'text' not in entry:
                        entry['text'] = f"Document {entry['chunk_id']} from {entry['source']}"
            logging.info(f'Loaded metadata with {len(self.metadata)} entries')

            # Load knowledge graph if available
            if os.path.exists(CFG['GRAPH_PATH']):
                logging.info(f"Loading knowledge graph from {CFG['GRAPH_PATH']}")
                with open(CFG['GRAPH_PATH'], 'rb') as f:
                    self.graph = pickle.load(f)
                logging.info('Knowledge graph loaded')
            else:
                logging.warning('No graph file found; skipping graph expansion')
                self.graph = None

            # Embedding model
            emb_key = f"emb|{CFG['MODEL_NAME']}"
            if CFG['USE_CACHE'] and emb_key in memory_cache:
                self.embed_model = memory_cache[emb_key]
            else:
                self.embed_model = SentenceTransformer(CFG['MODEL_NAME'])
                if CFG['USE_CACHE']:
                    memory_cache[emb_key] = self.embed_model
                    disk_cache.set(emb_key, self.embed_model)
            logging.info('Sentence Transformer loaded')

            # TF-IDF vectorizer
            texts = [d['text'] for d in self.metadata]
            self.tfidf = TfidfVectorizer(max_features=50000, stop_words='english')
            self.tfidf_mat = self.tfidf.fit_transform(texts)
            logging.info('TF-IDF vectorizer initialized')

            # Cross-encoder for reranking
            try:
                self.cross = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=CFG['DEVICE'])
                logging.info('Cross-encoder loaded')
            except Exception as e:
                self.cross = None
                logging.warning(f'Cross-encoder unavailable: {str(e)}')

            # Local LLM
            try:
                token = os.getenv('HUGGINGFACE_HUB_TOKEN', None)
                self.tokenizer = AutoTokenizer.from_pretrained(CFG['HF_MODEL'], use_auth_token=token)
                self.llm = AutoModelForCausalLM.from_pretrained(
                    CFG['HF_MODEL'], use_auth_token=token
                ).to(CFG['DEVICE'])
                self.llm.eval()
                logging.info('Local LLM loaded')
            except Exception as e:
                self.llm = None
                self.tokenizer = None
                logging.error(f'LLM load failed: {str(e)}')

        except Exception as e:
            error_msg = f"Error loading resources: {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)

    def semantic_search(self, query:str) -> List[Dict]:
        emb = np.array(self.embed_model.encode([query], show_progress_bar=False), dtype='float32')
        faiss.normalize_L2(emb)
        D,I = self.index.search(emb, CFG['TOP_K_FAISS'])
        docs=[]
        for dist, idx in zip(D[0], I[0]):
            if idx<0: continue
            doc = {
                'metadata': {
                    'chunk_id': self.metadata[idx]['chunk_id'],
                    'source': self.metadata[idx]['source']
                },
                'text': self.metadata[idx].get('text', ''),
                'score': 1.0/(1.0+max(0,dist))
            }
            docs.append(doc)
        return docs

    def keyword_search(self, query:str) -> List[Dict]:
        # Initialize TF-IDF if not done yet
        if not hasattr(self, 'tfidf') or not hasattr(self, 'tfidf_mat'):
            texts = [d.get('text', '') for d in self.metadata]
            self.tfidf = TfidfVectorizer(max_features=50000, stop_words='english')
            self.tfidf_mat = self.tfidf.fit_transform(texts)
            
        vec = self.tfidf.transform([query])
        scores = (vec @ self.tfidf_mat.T).toarray()[0]         
        idxs = np.argsort(-scores)[:CFG['TOP_K_FAISS']]
        return [{'metadata': {'chunk_id': self.metadata[i]['chunk_id'], 
                            'source': self.metadata[i]['source']},
                'text': self.metadata[i].get('text', ''),
                'score': float(scores[i])}
                for i in idxs if scores[i]>0]

    def hybrid_search(self, query:str) -> List[Dict]:
        sem = self.semantic_search(query)
        kw  = self.keyword_search(query)
        comb={}
        for d in sem+kw:
            key=(d['metadata']['source'], d['metadata']['chunk_id'])
            if key not in comb or d['score']>comb[key]['score']:
                comb[key]=d
        # small fallback weight
        for d in comb.values():
            if 'text' in d and 'keyword' not in d:
                d['score']*=1.0
        return sorted(comb.values(), key=lambda x:-x['score'])[:CFG['TOP_K_FINAL']]

    def graph_expand(self, chunk_ids:List[str]) -> List[str]:
        if not self.graph: return chunk_ids
        expanded=set(chunk_ids)
        for cid in chunk_ids:
            if cid in self.graph:
                expanded.update(self.graph.neighbors(cid))
        return list(expanded)

    def rerank(self, query:str, docs:List[Dict]) -> List[Dict]:
        if not self.cross or not docs: return docs
        pairs=[(query,d['text']) for d in docs]
        scr=self.cross.predict(pairs)
        mn,mx = scr.min(), scr.max()
        norm = [(s-mn)/(mx-mn+1e-12) for s in scr]
        for d,s in zip(docs,norm): d['score']=0.3*d['score']+0.7*s
        return sorted(docs, key=lambda x:-x['score'])

    def trim_context(self, docs:List[Dict]) -> str:
        ctx, tokens = '', 0
        for d in docs:
            wcount=len(d['text'].split())
            if tokens + wcount > CFG['MAX_CTX_TOKENS']: break
            if wcount < CFG['MIN_CHUNK_WORDS']: continue
            ctx += d['text'] + '\n\n'
            tokens += wcount
        return ctx.strip()

    def generate(self, query:str, context:str) -> str:
        if not self.llm: return 'LLM unavailable'
        maxlen=self.tokenizer.model_max_length-256
        ctx = context[-maxlen:] if len(context.split())>maxlen else context
        prompt=f"Context: {ctx}\nQuestion: {query}\nAnswer:"
        inputs=self.tokenizer(prompt, return_tensors='pt', truncation=True,
                               max_length=self.tokenizer.model_max_length).to(CFG['DEVICE'])
        out=self.llm.generate(**inputs, max_new_tokens=256, temperature=0.3, top_p=0.9)
        gen=out[0, inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    def evaluate(self, question:str, answer:str, reference:str=None) -> EvaluationResult:
        er=EvaluationResult()
        er.answer = answer
        # retrieval and generation latencies must be set externally
        if reference:
            ch=SmoothingFunction()
            er.bleu_score = sentence_bleu([reference.split()], answer.split(), smoothing_function=ch.method1)
            er.meteor_score = nltk.translate.meteor_score.meteor_score([reference.split()], answer.split())
            er.rouge_scores  = Rouge().get_scores(answer, reference)[0]
        return er

    def query(self, question:str, reference:str=None) -> Tuple[str, EvaluationResult]:
        er=EvaluationResult()
        t0=time.time()
        # initial hybrid retrieval
        docs = self.hybrid_search(question)
        # graph expansion on their chunk_ids
        ids = [d['metadata']['chunk_id'] for d in docs]
        exp_ids = self.graph_expand(ids)
        # filter docs to expanded set
        docs = [d for d in docs if d['metadata']['chunk_id'] in exp_ids]
        er.retrieval_latency = time.time() - t0
        er.retrieved_chunks = docs
        if not docs:
            er.answer='No info found.'
            return er.answer, er
        # trim context
        ctx = self.trim_context(docs)
        t1=time.time()
        ans = self.generate(question, ctx)
        er.generation_latency = time.time() - t1
        er.total_latency = er.retrieval_latency + er.generation_latency
        er.answer = ans
        # evaluate if reference
        if reference:
            met = self.evaluate(question, ans, reference)
            er.bleu_score   = met.bleu_score
            er.meteor_score = met.meteor_score
            er.rouge_scores = met.rouge_scores
        # Log metrics
        logging.info(f"Metrics: retrieval={er.retrieval_latency:.3f}s, generation={er.generation_latency:.3f}s, "
                     f"BLEU={er.bleu_score:.3f}, METEOR={er.meteor_score:.3f}, ROUGE-1_f={er.rouge_scores.get('rouge-1',{}).get('f',0):.3f}")
        return ans, er

if __name__=='__main__':
    rag = RAGSystem()
    q   = "What are the main challenges in blockchain scalability?"
    ref = "High fees, slow processing, consensus issues, security risks, and interoperability limits."
    ans, metrics = rag.query(q, ref)
    print("Answer:", ans)
    print(metrics)
    print("Retrieved chunks:", len(metrics.retrieved_chunks))