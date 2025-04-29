# utility.py
"""
Utility functions and data for financial and cryptocurrency analysis
"""

# Common financial and risk analysis terms for improved chunking and retrieval
FINANCE_TERMS = {
    # Cryptocurrency terms
    "cryptocurrency": ["crypto", "bitcoin", "ethereum", "blockchain", "token", "coin", "altcoin", 
                      "defi", "mining", "wallet", "exchange", "nft", "smart contract", "node", "hash"],
    
    # Compliance terms
    "compliance": ["kyc", "aml", "know your customer", "anti-money laundering", "cft", "combating financing of terrorism",
                  "regulations", "regtech", "compliance officer", "regulatory", "audit", "sanctions", "pep", 
                  "politically exposed person", "fatf", "fincen", "ofac"],
    
    # Due diligence terms
    "due_diligence": ["dd", "edd", "enhanced due diligence", "cdd", "customer due diligence", 
                     "onboarding", "verification", "identity verification", "risk assessment", 
                     "background check", "screening", "monitoring", "ongoing monitoring"],
    
    # Risk analysis terms
    "risk_analysis": ["risk", "fraud", "threat", "vulnerability", "assessment", "matrix", "score", 
                     "rating", "profile", "exposure", "mitigation", "control", "measure", 
                     "detection", "prevention", "suspicious activity", "sar", "suspicious activity report"],
    
    # Transaction terms
    "transactions": ["payment", "transfer", "transaction", "wire", "remittance", "p2p", "peer-to-peer", 
                    "volume", "liquidity", "trading", "settlement", "clearing", "custodian", "custody"],
    
    # Financial crime terms
    "financial_crime": ["money laundering", "terrorism financing", "fraud", "scam", "phishing", 
                       "ransomware", "darknet", "mixer", "tumbler", "illicit", "illegal", 
                       "sanction", "evasion", "criminal", "predicate offense"]
}

def get_domain_specific_terms():
    """Return a flattened list of all domain-specific terms"""
    all_terms = []
    for category, terms in FINANCE_TERMS.items():
        all_terms.extend(terms)
    return all_terms

def is_relevant_chunk(chunk_text, min_terms=2):
    """Check if a chunk contains enough domain-specific terms to be relevant"""
    terms = get_domain_specific_terms()
    count = sum(1 for term in terms if term.lower() in chunk_text.lower())
    return count >= min_terms

def enhance_query(query):
    """Enhance a query with related domain-specific terms"""
    enhanced_terms = []
    query_lower = query.lower()
    
    for category, terms in FINANCE_TERMS.items():
        for term in terms:
            if term in query_lower:
                # Add some related terms from the same category
                category_terms = [t for t in FINANCE_TERMS[category] if t != term]
                enhanced_terms.extend(category_terms[:3])  # Add up to 3 related terms
    
    if enhanced_terms:
        enhanced_query = f"{query} {' '.join(enhanced_terms[:5])}"  # Limit to 5 additional terms
        return enhanced_query
    return query