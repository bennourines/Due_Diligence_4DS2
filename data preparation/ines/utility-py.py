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
