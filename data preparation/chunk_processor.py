import os
from pathlib import Path
import logging
from typing import List, Dict, Any, Tuple, Set
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import faiss
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.matcher import PhraseMatcher
import re
from collections import defaultdict
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chunk_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TextSplitter:
    """Simple text splitter that splits text into chunks of specified size with overlap."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        if not text:
            return []
            
        # Split text into sentences using spaCy
        doc = spacy.load("en_core_web_lg")(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size, save current chunk and start new one
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                
                # Keep last few sentences for overlap
                overlap_size = 0
                overlap_chunk = []
                for s in reversed(current_chunk):
                    if overlap_size + len(s) > self.chunk_overlap:
                        break
                    overlap_chunk.insert(0, s)
                    overlap_size += len(s)
                
                current_chunk = overlap_chunk
                current_length = sum(len(s) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

class EntityExtractor:
    """Extract named entities and other features from financial texts"""

    # Load NLP model
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("sentencizer")

    # Define entity configurations
    KNOWN_ENTITIES = {
        "crypto_projects": [
            "Bitcoin", "Ethereum", "Solana", "Ripple", "Cardano", "Binance",
            "Uniswap", "Compound", "Aave", "MakerDAO", "Chainlink", "Polygon",
            "Avalanche", "Tezos", "Polkadot", "Algorand", "Stellar", "Cosmos"
        ],
        "crypto_terms": [
            "blockchain", "cryptocurrency", "token", "protocol", "mining", "staking",
            "defi", "dao", "consensus", "wallet", "exchange", "yield", "liquidity"
        ],
        "organizations": [
            "Circle", "Coinbase", "Kraken", "FTX", "Binance", "Chainalysis", "Fireblocks",
            "BlockFi", "BitGo", "Gemini", "ConsenSys", "Tether", "CoinMarketCap"
        ],
        "locations": [
            "United States", "China", "Russia", "European Union", "Singapore", "Switzerland",
            "Malta", "Dubai", "Hong Kong", "Japan", "South Korea", "Cayman Islands"
        ],
        "risk_terms": {
            "regulatory": [
                "compliance violation", "unregistered", "non-compliant", "illegal", "banned",
                "unlicensed", "regulatory", "oversight", "fine", "penalty", "investigation"
            ],
            "technical": [
                "hack", "breach", "exploit", "vulnerability", "backdoor", "exposure",
                "bug", "malware", "phishing", "stolen keys", "51% attack"
            ],
            "financial": [
                "bankruptcy", "insolvency", "liquidity crisis", "bank run", "rugpull",
                "collapse", "fraud", "ponzi", "scam", "unsustainable", "hyperinflation"
            ],
            "operational": [
                "downtime", "outage", "suspended", "locked", "frozen assets",
                "withdrawal issues", "technical difficulties", "service disruption"
            ]
        }
    }

    BLACKLISTS = {
        "person_blacklist": ["administrator", "user", "customer", "client", "founder", "ceo"],
        "person_suffixes": ["corp", "inc", "llc", "ltd", "foundation", "group", "team"],
        "org_blacklist": ["the", "and", "that", "this", "these", "those"],
        "org_suffixes": ["ing", "ed", "ly", "day", "time", "year"],
        "org_prefixes": ["mr", "mrs", "ms", "dr", "prof"],
        "crypto_blacklist": ["the", "blockchain", "a", "an", "crypto", "mining"],
        "crypto_suffixes": ["ing", "ed", "s", "ly"]
    }

    # Patterns
    EMAIL_PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    CRYPTO_ADDRESS_PATTERN = r"\b(0x)?[0-9a-fA-F]{40}\b"
    PHONE_PATTERN = r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"

    def __init__(self):
        """Initialize entity extraction with phrase matchers"""
        self.matchers = self._setup_matchers()

    def _setup_matchers(self) -> Dict[str, PhraseMatcher]:
        """Create phrase matchers for known entities"""
        matchers = {}

        # Crypto project matcher
        crypto_matcher = PhraseMatcher(self.nlp.vocab)
        patterns = [self.nlp(text) for text in self.KNOWN_ENTITIES["crypto_projects"]]
        crypto_matcher.add("CRYPTO_PROJECT", patterns)
        matchers["crypto"] = crypto_matcher

        # Organization matcher
        org_matcher = PhraseMatcher(self.nlp.vocab)
        patterns = [self.nlp(text) for text in self.KNOWN_ENTITIES["organizations"]]
        org_matcher.add("ORG", patterns)
        matchers["org"] = org_matcher

        return matchers

    def is_valid_person(self, text: str) -> bool:
        """Validate person names with strict rules"""
        text_lower = text.lower()

        # Must have at least 2 words, proper capitalization, no numbers
        conditions = [
            len(text.split()) >= 2,
            text.istitle(),
            not any(c.isdigit() for c in text),
            not any(term in text_lower for term in self.BLACKLISTS["person_blacklist"]),
            not any(text_lower.endswith(suffix) for suffix in self.BLACKLISTS["person_suffixes"])
        ]

        return all(conditions)

    def is_valid_org(self, text: str) -> bool:
        """Validate organization names"""
        text_lower = text.lower()

        conditions = [
            3 <= len(text) <= 50,
            text[0].isupper(),
            not any(b in text_lower for b in self.BLACKLISTS["org_blacklist"]),
            not any(text_lower.endswith(suffix) for suffix in self.BLACKLISTS["org_suffixes"]),
            not any(text_lower.startswith(prefix) for prefix in self.BLACKLISTS["org_prefixes"])
        ]

        return all(conditions)

    def is_valid_crypto_project(self, text: str) -> bool:
        """Validate cryptocurrency projects"""
        text_lower = text.lower()

        # Check against known projects first
        if any(proj.lower() == text_lower for proj in self.KNOWN_ENTITIES["crypto_projects"]):
            return True

        # Generic project validation
        conditions = [
            2 <= len(text.split()) <= 4,
            not any(b in text_lower for b in self.BLACKLISTS["crypto_blacklist"]),
            not any(text_lower.endswith(suffix) for suffix in self.BLACKLISTS["crypto_suffixes"])
        ]

        # Must contain at least one known crypto term
        conditions.append(
            any(term in text_lower for term in self.KNOWN_ENTITIES["crypto_terms"])
        )

        return all(conditions)

    def is_valid_location(self, text: str) -> bool:
        """Validate locations against known list"""
        return text.lower() in {loc.lower() for loc in self.KNOWN_ENTITIES["locations"]}

    def normalize_entity(self, entity_type: str, text: str) -> str:
        """Standardize entity formatting"""
        # Apply type-specific normalization
        if entity_type == "person":
            # Standardize name formatting (Title Case)
            return " ".join(word.capitalize() for word in text.split())

        elif entity_type in ["crypto_project", "organization"]:
            # Remove common suffixes and standardize casing
            text = re.sub(r'\b(LLC|Inc|Ltd|Foundation|Labs|DAO|DeFi|Network|Protocol)\b', '', text, flags=re.IGNORECASE)
            return text.strip()

        return text

    def post_process_entities(self, entities: Dict[str, Set[str]]) -> Dict[str, List[str]]:
        """Clean and deduplicate extracted entities"""
        processed = {}

        for entity_type, entity_set in entities.items():
            # Normalize each entity
            normalized = {self.normalize_entity(entity_type, e) for e in entity_set}

            # Remove subsumed entities (shorter versions of longer entities)
            final_entities = set()
            for entity in sorted(normalized, key=len, reverse=True):
                if not any(e != entity and entity.lower() in e.lower() for e in final_entities):
                    if entity:  # Skip empty strings
                        final_entities.add(entity)

            processed[entity_type] = sorted(final_entities)

        return processed

    def extract_named_entities(self, text: str) -> Dict[str, List[str]]:
        """Comprehensive entity extraction pipeline"""
        doc = self.nlp(text)
        entities = defaultdict(set)

        # Stage 1: spaCy NER extraction with strict validation
        for ent in doc.ents:
            clean_text = ' '.join(ent.text.strip().split())

            if ent.label_ == "PERSON" and self.is_valid_person(clean_text):
                entities["person"].add(clean_text)

            elif ent.label_ == "ORG":
                if self.is_valid_crypto_project(clean_text):
                    entities["crypto_project"].add(clean_text)
                elif self.is_valid_org(clean_text):
                    entities["organization"].add(clean_text)

            elif ent.label_ == "GPE" and self.is_valid_location(clean_text):
                entities["location"].add(clean_text)

        # Stage 2: Phrase matching for known entities
        for match_id, start, end in self.matchers["crypto"](doc):
            entities["crypto_project"].add(doc[start:end].text)

        for match_id, start, end in self.matchers["org"](doc):
            entities["organization"].add(doc[start:end].text)

        # Stage 3: Pattern-based extraction
        entities["email"] = set(re.findall(self.EMAIL_PATTERN, text))
        entities["crypto_address"] = set(re.findall(self.CRYPTO_ADDRESS_PATTERN, text))

        # Stage 4: Post-processing
        processed_entities = self.post_process_entities(entities)

        return processed_entities

    def extract_risk_features(self, text: str) -> Tuple[Dict[str, List[str]], float]:
        """Enhanced risk term extraction"""
        risk_categories = {
            "regulatory": self.KNOWN_ENTITIES["risk_terms"]["regulatory"],
            "technical": self.KNOWN_ENTITIES["risk_terms"]["technical"],
            "financial": self.KNOWN_ENTITIES["risk_terms"]["financial"],
            "operational": self.KNOWN_ENTITIES["risk_terms"]["operational"]
        }

        found = {k: set() for k in risk_categories}
        doc = self.nlp(text.lower())

        for category, terms in risk_categories.items():
            for term in terms:
                if term.lower() in doc.text:
                    found[category].add(term)

        # Calculate weighted risk score
        weights = {"regulatory": 1.2, "technical": 1.1, "financial": 1.0, "operational": 0.9}
        score = min(sum(len(v) * 10 * weights[k] for k, v in found.items()), 100)

        return {k: sorted(v) for k, v in found.items()}, round(score, 2)

class FinanceUtility:
    """Utility functions for financial and cryptocurrency text processing"""

    FINANCE_TERMS = {
        "cryptocurrency": [
            "crypto", "bitcoin", "ethereum", "blockchain", "token", "coin", "altcoin",
            "defi", "mining", "wallet", "exchange", "nft", "smart contract", "node", "hash"
        ],
        "compliance": [
            "kyc", "aml", "know your customer", "anti-money laundering", "cft",
            "regulations", "regtech", "audit", "sanctions", "pep", "fincen", "ofac"
        ],
        "due_diligence": [
            "dd", "edd", "cdd", "risk assessment", "screening", "onboarding", "identity verification"
        ],
        "risk_analysis": [
            "risk", "fraud", "vulnerability", "score", "exposure", "mitigation", "sar", "suspicious activity"
        ],
        "transactions": [
            "payment", "transfer", "transaction", "wire", "p2p", "volume", "liquidity", "custody"
        ],
        "financial_crime": [
            "money laundering", "terrorism financing", "phishing", "darknet", "tumbler", "illicit", "sanction evasion"
        ]
    }

    @classmethod
    def get_domain_specific_terms(cls):
        """Flatten terms across all categories"""
        terms = []
        for group in cls.FINANCE_TERMS.values():
            terms.extend(group)
        return terms

    @classmethod
    def is_relevant_chunk(cls, chunk_text, min_terms=2):
        """Determine if a text chunk contains at least `min_terms` finance-domain words"""
        terms = cls.get_domain_specific_terms()
        return sum(1 for term in terms if term.lower() in chunk_text.lower()) >= min_terms

    @classmethod
    def enhance_query(cls, query):
        """Append additional related terms based on original query"""
        query = query.lower()
        extras = []

        for category, terms in cls.FINANCE_TERMS.items():
            for term in terms:
                if term in query:
                    related = [t for t in terms if t != term][:3]
                    extras.extend(related)

        if extras:
            return f"{query} {' '.join(extras[:5])}"
        return query

class ChunkProcessor:
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Create necessary directories
        self.create_directories()
        
        # Initialize components
        logger.info("Initializing SentenceTransformer model...")
        self.embedder = SentenceTransformer(embedding_model)
        
        logger.info("Initializing text splitter...")
        self.text_splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        logger.info("Initializing entity extractor...")
        self.entity_extractor = EntityExtractor()
        
        logger.info("Initializing finance utility...")
        self.utility = FinanceUtility()
        
        self.chunks = []
        self.embeddings = None
        self.faiss_index = None
        logger.info("Initialization complete.")

    def create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = ['chunks', 'vector_store']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            logger.info(f"Created directory: {directory}")

    def smart_chunking(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Apply smart chunking using domain-specific knowledge."""
        logger.info("Applying smart chunking...")
        try:
            # First split into basic chunks
            basic_chunks = self.text_splitter.split_text(text)
            smart_chunks = []
            
            for i, chunk in enumerate(basic_chunks):
                if self.utility.is_relevant_chunk(chunk):
                    # Extract entities and risk features
                    entities = self.entity_extractor.extract_named_entities(chunk)
                    risk_features, risk_score = self.entity_extractor.extract_risk_features(chunk)
                    
                    # Create enhanced chunk with metadata
                    smart_chunks.append({
                        "text": chunk,
                        "metadata": {
                            **(metadata or {}),
                            "entities": entities,
                            "risk_features": risk_features,
                            "risk_score": risk_score,
                            "chunk_type": "smart",
                            "chunk_index": i
                        }
                    })
            
            logger.info(f"Created {len(smart_chunks)} smart chunks")
            return smart_chunks
        except Exception as e:
            logger.error(f"Error in smart chunking: {str(e)}")
            return []

    def semantic_chunking(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Apply semantic chunking based on content similarity."""
        logger.info("Applying semantic chunking...")
        try:
            # First split into basic chunks
            basic_chunks = self.text_splitter.split_text(text)
            semantic_chunks = []
            
            for i, chunk in enumerate(basic_chunks):
                # Get embedding for the chunk
                embedding = self.embedder.encode([chunk])[0]
                
                # Extract entities and risk features
                entities = self.entity_extractor.extract_named_entities(chunk)
                risk_features, risk_score = self.entity_extractor.extract_risk_features(chunk)
                
                # Create semantic chunk with metadata
                semantic_chunks.append({
                    "text": chunk,
                    "embedding": embedding.tolist(),  # Convert to list for JSON serialization
                    "metadata": {
                        **(metadata or {}),
                        "entities": entities,
                        "risk_features": risk_features,
                        "risk_score": risk_score,
                        "chunk_type": "semantic",
                        "chunk_index": i
                    }
                })
            
            logger.info(f"Created {len(semantic_chunks)} semantic chunks")
            return semantic_chunks
        except Exception as e:
            logger.error(f"Error in semantic chunking: {str(e)}")
            return []

    def process_text(self, text: str, source: str) -> None:
        """Process text using all chunking methods and combine results."""
        logger.info(f"Processing text from {source}")
        metadata = {"source": source}
        
        try:
            # Apply chunking methods
            smart_chunks = self.smart_chunking(text, metadata)
            semantic_chunks = self.semantic_chunking(text, metadata)
            
            # Combine all chunks
            combined_chunks = smart_chunks + semantic_chunks
            
            if not combined_chunks:
                logger.warning(f"No chunks were generated for {source}")
                return
                
            # Add to main chunks list
            self.chunks.extend(combined_chunks)
            
            # Save chunks to file
            self.save_chunks_to_file(combined_chunks, source)
            logger.info(f"Successfully processed {source}")
        except Exception as e:
            logger.error(f"Error processing {source}: {str(e)}")

    def save_chunks_to_file(self, chunks: List[Dict[str, Any]], source: str) -> None:
        """Save chunks to a JSON file."""
        try:
            filename = Path(source).stem
            output_path = Path("chunks") / f"{filename}_chunks.json"
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved chunks to {output_path}")
        except Exception as e:
            logger.error(f"Error saving chunks to file: {str(e)}")

    def create_embeddings(self) -> None:
        """Create embeddings for all chunks and build FAISS index."""
        if not self.chunks:
            logger.warning("No chunks available to create embeddings")
            return
        
        logger.info("Creating embeddings...")
        try:
            # Extract texts and create embeddings
            texts = [chunk["text"] for chunk in self.chunks]
            self.embeddings = self.embedder.encode(texts)
            
            # Save embeddings
            embeddings_path = Path("vector_store/embeddings.pt")
            torch.save(self.embeddings, str(embeddings_path))
            logger.info(f"Saved embeddings to {embeddings_path}")
            
            # Create and save FAISS index
            dimension = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dimension)
            self.faiss_index.add(self.embeddings)
            
            faiss_path = Path("vector_store/faiss_index")
            faiss.write_index(self.faiss_index, str(faiss_path))
            logger.info(f"Created and saved FAISS index for {len(texts)} chunks")
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise

def main():
    # Initialize chunk processor
    processor = ChunkProcessor()
    
    # Process all cleaned text files
    cleaned_dir = Path("cleaned")
    total_files = len(list(cleaned_dir.glob("*.txt")))
    processed_files = 0
    
    logger.info(f"Found {total_files} files to process")
    
    for file_path in cleaned_dir.glob("*.txt"):
        try:
            logger.info(f"Processing file {processed_files + 1}/{total_files}: {file_path}")
            
            # Try different encodings
            encodings = ['utf-8', 'latin1', 'cp1252']
            text = None
            
            for encoding in encodings:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        text = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if text is None:
                logger.error(f"Failed to read {file_path} with any encoding")
                continue
                
            processor.process_text(text, str(file_path))
            processed_files += 1
            logger.info(f"Progress: {processed_files}/{total_files} files processed")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
    
    # Create embeddings and FAISS index
    processor.create_embeddings()
    logger.info(f"Chunk processing completed. Successfully processed {processed_files} out of {total_files} files.")

if __name__ == "__main__":
    main() 