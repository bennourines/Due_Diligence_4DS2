"""
Optimized Retrieval-Augmented Generation (RAG) System
With integrated evaluation, performance optimizations, and clear metrics output.
"""

import os
import logging
import json
import time
from typing import List, Dict, Tuple # Add this import
import faiss
import numpy as np
import diskcache
import nltk
from dataclasses import dataclass
from cachetools import TTLCache
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

# Setup logging
logging.basicConfig(level=logging.INFO)

# Download NLTK data
nltk.download('punkt', quiet=True)

# Environment variables
MODEL_NAME     = os.getenv('RAG_MODEL_NAME', 'sentence-transformers/all-mpnet-base-v2')
FAISS_INDEX    = os.getenv('FAISS_INDEX_PATH', 'index.faiss')
METADATA_JSON  = os.getenv('METADATA_PATH', 'merged_metadata.json')
HF_MODEL       = os.getenv('HF_MODEL_NAME', 'meta-llama/Llama-3.2-1B')
DEVICE         = os.getenv('DEVICE', 'cpu')
USE_CACHE      = os.getenv('USE_CACHE', 'true').lower() == 'true'
CACHE_DIR      = os.getenv('CACHE_DIR', '.cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# Caches
disk_cache   = diskcache.Cache(CACHE_DIR)
memory_cache = TTLCache(maxsize=512, ttl=3600)

@dataclass
class EvaluationResult:
    retrieval_latency: float = 0.0
    generation_latency: float = 0.0
    total_latency:      float = 0.0
    bleu_score:    float = 0.0
    meteor_score:  float = 0.0
    rouge_scores:  dict  = None
    retrieved_chunks: list = None
    answer:          str  = ''

class RAGSystem:
    def __init__(self):
        self._load_resources()

    def _load_resources(self):
        # Load FAISS + metadata
        if not os.path.exists(FAISS_INDEX) or not os.path.exists(METADATA_JSON):
            raise FileNotFoundError('Missing FAISS index or metadata')
        self.index = faiss.read_index(FAISS_INDEX, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
        with open(METADATA_JSON, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        logging.info(f'FAISS loaded: {self.index.ntotal} vectors')

        # Sentence embedding model
        key = f'emb|{MODEL_NAME}'
        if USE_CACHE and key in memory_cache:
            self.embed_model = memory_cache[key]
        else:
            self.embed_model = SentenceTransformer(MODEL_NAME)
            if USE_CACHE:
                memory_cache[key] = self.embed_model
                disk_cache.set(key, self.embed_model)
        logging.info('Embedding model ready')

        # TF-IDF vectorizer
        texts = [d['text'] for d in self.metadata]
        self.tfidf = TfidfVectorizer(max_features=50000, stop_words='english')
        self.tfidf_mat = self.tfidf.fit_transform(texts)
        logging.info('TF-IDF vectorizer ready')

        # Cross-encoder
        try:
            self.cross = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=DEVICE)
            logging.info('Cross-encoder ready')
        except:
            self.cross = None
            logging.warning('Cross-encoder unavailable')

        # LLM tokenizer + model
        try:
            token = os.getenv('HUGGINGFACE_HUB_TOKEN', None)
            self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, use_auth_token=token)
            self.llm = AutoModelForCausalLM.from_pretrained(
                HF_MODEL, use_auth_token=token,
                torch_dtype=(None if DEVICE=='cpu' else 'auto')
            ).to(DEVICE)
            self.llm.eval()
            logging.info('LLM loaded')
        except Exception as e:
            self.llm = None
            self.tokenizer = None
            logging.error(f'LLM load failed: {e}')

    def semantic_search(self, query: str, top_k: int =5) -> List[Dict]:
        emb = np.array(self.embed_model.encode([query], show_progress_bar=False), dtype='float32')
        D,I = self.index.search(emb, top_k)
        results=[]
        for dist,idx in zip(D[0],I[0]):
            if idx<0: continue
            doc = self.metadata[idx].copy()
            doc['score']=1/(1+max(0,dist))
            results.append(doc)
        return results

    def keyword_search(self, query: str, top_k: int=5) -> List[Dict]:
        vec = self.tfidf.transform([query])
        # Convert sparse matrix to dense array and flatten
        sc = (vec @ self.tfidf_mat.T).toarray()[0] 
        idxs = np.argsort(-sc)[:top_k]
        return [dict(self.metadata[i], score=float(sc[i])) for i in idxs if sc[i]>0]

    def hybrid_search(self, query: str, top_k:int=5) -> List[Dict]:
        sem = self.semantic_search(query, top_k*2)
        kw  = self.keyword_search(query, top_k*2)
        comb={}
        for d in sem+kw:
            key = (d['metadata']['source'],d['metadata']['chunk_id'])
            if key not in comb or d['score']>comb[key]['score']:
                comb[key]=d
        # light keyword fallback
        for d in comb.values():
            if 'keyword' in d: d['score']*=0.1
        return sorted(comb.values(), key=lambda x:-x['score'])[:top_k]

    def rerank(self, query:str, docs:List[Dict]) -> List[Dict]:
        if not self.cross or not docs: return docs
        pairs = [(query,d['text']) for d in docs]
        scores = self.cross.predict(pairs)
        mn,mx = float(min(scores)), float(max(scores))
        norm=[(s-mn)/(mx-mn+1e-12) for s in scores]
        for d,s in zip(docs,norm): d['score']=0.3*d['score']+0.7*s
        return sorted(docs, key=lambda x:-x['score'])

    def generate(self, query:str, context:str) -> str:
        if not self.llm: return 'LLM unavailable'
        max_len=self.tokenizer.model_max_length-256
        ctx = context[-max_len:] if len(context)>max_len else context
        prompt=f"Context: {ctx}\nQuestion: {query}\nAnswer:"
        inputs=self.tokenizer(prompt, return_tensors='pt', truncation=True,
                               max_length=self.tokenizer.model_max_length).to(DEVICE)
        out=self.llm.generate(**inputs, max_new_tokens=256, temperature=0.3, top_p=0.9)
        tok_len=inputs.input_ids.shape[1]
        gen=out[0,tok_len:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    def evaluate_metrics(self, question:str, answer:str, reference:str) -> Tuple[float, float, dict]:
        # BLEU with smoothing
        ch= SmoothingFunction()
        bleu=sentence_bleu([reference.split()], answer.split(), smoothing_function=ch.method1)
        meteor=nltk.translate.meteor_score.meteor_score([reference.split()], answer.split())
        rouge=Rouge().get_scores(answer,reference)[0]
        return bleu, meteor, rouge

    def query(self, question:str, reference:str=None) -> Tuple[str, EvaluationResult]:
        res=EvaluationResult()
        t0=time.time()
        docs=self.rerank(question, self.hybrid_search(question))
        res.retrieval_latency=time.time()-t0
        res.retrieved_chunks=docs
        if not docs:
            res.answer='No info found.'; return res.answer, res
        ctx='\n'.join(d['text'] for d in docs)
        t1=time.time()
        ans=self.generate(question,ctx)
        res.generation_latency=time.time()-t1
        res.total_latency=res.retrieval_latency+res.generation_latency
        res.answer=ans
        # Evaluate if reference provided
        if reference:
            b,m,r=self.evaluate_metrics(question,ans,reference)
            res.bleu_score=b; res.meteor_score=m; res.rouge_scores=r
        # Log metrics
        logging.info(f"Metrics: BLEU={res.bleu_score:.3f}, METEOR={res.meteor_score:.3f}, "
                     f"ROUGE-1_f={res.rouge_scores.get('rouge-1',{}).get('f',0):.3f}")
        return ans, res

if __name__=='__main__':
    rag=RAGSystem()
    # Example usage with hypothetical reference
    q="What are the main challenges in blockchain scalability?"
    ref="High fees, slow processing, consensus issues, security risks, and interoperability limits."
    ans,metrics = rag.query(q, ref)
    print("Answer:",ans)
    print(metrics)
