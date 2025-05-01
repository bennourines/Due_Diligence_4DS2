import spacy
from spacy.matcher import PhraseMatcher
import re
from typing import List, Dict, Tuple, Set
from collections import defaultdict

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
