from typing import List, Dict, Any, Optional
import json

class RiskAnalyzer:
    """Analyze crypto projects and generate risk reports"""
    
    def __init__(self):
        # Load question bank for comprehensive risk analysis
        self.question_bank = self._load_question_bank()
    
    def _load_question_bank(self) -> Dict:
        """Load question bank from JSON file or define it here"""
        return {
            "tokenomics": [
                {"question": "Is the token distribution centralized?", "weight": 20},
                {"question": "What percentage of tokens are allocated to the team and insiders?", "weight": 15},
                {"question": "Are there token vesting periods for team and early investors?", "weight": 10}
            ],
            "technical": [
                {"question": "Have the smart contracts been audited by reputable firms?", "weight": 25},
                {"question": "Are there any critical vulnerabilities in the codebase?", "weight": 25},
                {"question": "Is the source code open and verified on blockchain explorers?", "weight": 15}
            ],
            "team": [
                {"question": "Does the team have relevant experience in blockchain or crypto?", "weight": 20},
                {"question": "Are team members publicly identified (non-anonymous)?", "weight": 15},
                {"question": "Do team members have a history of successful projects?", "weight": 15}
            ],
            "regulatory": [
                {"question": "Does the project comply with relevant regulations?", "weight": 25},
                {"question": "Are there geographic restrictions for token holders?", "weight": 20},
                {"question": "Is the project at risk of being classified as a security?", "weight": 20}
            ],
            "market": [
                {"question": "Does the project solve a real problem?", "weight": 20},
                {"question": "Is there significant competition in the space?", "weight": 15},
                {"question": "Does the project have actual users or adoption?", "weight": 25}
            ]
        }
    
    def generate_report(self, project_id: str, search_engine, rag_pipeline) -> Dict:
        """Generate comprehensive risk report"""
        report = {
            "summary": "",
            "overall_score": 0,
            "categories": {},
            "recommendation": "",
            "key_findings": []
        }
        
        total_weight = 0
        total_score = 0
        
        # Process each category of questions
        for category, questions in self.question_bank.items():
            category_score = 0
            category_weight = sum(q["weight"] for q in questions)
            category_answers = []
            
            # Process each question in the category
            for question_data in questions:
                question = question_data["question"]
                weight = question_data["weight"]
                
                # Get context for this question
                contexts = search_engine.hybrid_search(question, project_id)
                
                # Generate answer
                answer = rag_pipeline.generate_response(question)
                
                # Calculate risk score (0-100, higher is better)
                # This is a simplified scoring, you'd implement more sophisticated logic
                risk_score = self._analyze_answer_risk(answer)
                
                # Add to category score
                weighted_score = risk_score * weight / 100
                category_score += weighted_score
                
                # Save answer
                category_answers.append({
                    "question": question,
                    "answer": answer,
                    "score": risk_score,
                    "weight": weight
                })
            
            # Calculate normalized category score (0-100)
            normalized_category_score = category_score * 100 / category_weight if category_weight > 0 else 0
            
            # Add to total score
            total_weight += category_weight
            total_score += category_score
            
            # Add category to report
            report["categories"][category] = {
                "score": normalized_category_score,
                "answers": category_answers
            }
            
            # Identify risk level
            risk_level = "Low"
            if normalized_category_score < 40:
                risk_level = "High"
            elif normalized_category_score < 70:
                risk_level = "Medium"
                
            # Add to report
            report["categories"][category]["risk_level"] = risk_level
        
        # Calculate overall score
        report["overall_score"] = total_score * 100 / total_weight if total_weight > 0 else 0
        
        # Generate recommendation
        if report["overall_score"] >= 80:
            report["recommendation"] = "Green - Favorable for investment with manageable risks"
        elif report["overall_score"] >= 60:
            report["recommendation"] = "Yellow - Proceed with caution, moderate risk identified"
        else:
            report["recommendation"] = "Red - High risk, significant concerns identified"
        
        # Generate key findings
        report["key_findings"] = self._generate_key_findings(report)
        
        # Generate summary
        report["summary"] = self._generate_summary(report)
        
        return report
    
    def _analyze_answer_risk(self, answer: str) -> float:
        """
        Analyze risk from answer text and return a score from 0-100
        Higher score means lower risk
        
        This is a simplified implementation - in production, you would implement
        a more sophisticated NLP analysis or use the LLM itself to score the answer
        """
        
        # List of negative signal words (indicating risk)
        negative_signals = [
            "risk", "unsafe", "vulnerable", "centralized", "scam", "fraud", 
            "hack", "exploit", "failure", "concern", "warning", "issue", 
            "problem", "critical", "danger", "inadequate", "insufficient"
        ]
        
        # List of positive signal words
        positive_signals = [
            "safe", "secure", "decentralized", "audited", "transparent",
            "compliant", "verified", "reliable", "proven", "successful",
            "strong", "rigorous", "distributed", "protected", "robust"
        ]
        
        # Count signal occurrences (case insensitive)
        answer_lower = answer.lower()
        negative_count = sum(answer_lower.count(signal) for signal in negative_signals)
        positive_count = sum(answer_lower.count(signal) for signal in positive_signals)
        
        # Calculate base score
        total_signals = negative_count + positive_count
        if total_signals == 0:
            return 50  # Neutral if no signals
        
        # Calculate score (higher is better)
        score = (positive_count / total_signals) * 100 if total_signals > 0 else 50
        
        # Adjust score based on certainty markers
        uncertainty_markers = ["unclear", "unknown", "insufficient information", "cannot determine"]
        if any(marker in answer_lower for marker in uncertainty_markers):
            # Reduce score if there's uncertainty
            score = max(score * 0.8, 30)  # Minimum score of 30
        
        return score
    
    def _generate_key_findings(self, report: Dict) -> List[str]:
        """Extract key findings from the report"""
        findings = []
        
        # Add high risk findings
        for category, data in report["categories"].items():
            if data["risk_level"] == "High":
                for answer in data["answers"]:
                    if answer["score"] < 40:  # High risk answers
                        findings.append(f"High Risk in {category}: {answer['question']}")
        
        # Add low risk findings (strengths)
        strengths = []
        for category, data in report["categories"].items():
            if data["risk_level"] == "Low":
                for answer in data["answers"]:
                    if answer["score"] > 80:  # Low risk answers
                        strengths.append(f"Strength in {category}: {answer['question']}")
        
        # Combine high risks and top strengths (if any)
        findings.extend(strengths[:3])  # Add up to 3 strengths
        
        return findings
    
    def _generate_summary(self, report: Dict) -> str:
        """Generate executive summary"""
        overall_score = report["overall_score"]
        
        if overall_score >= 80:
            sentiment = "favorable"
        elif overall_score >= 60:
            sentiment = "cautiously optimistic"
        else:
            sentiment = "concerning"
        
        # Count risk levels
        risk_counts = {"High": 0, "Medium": 0, "Low": 0}
        for category, data in report["categories"].items():
            risk_counts[data["risk_level"]] += 1
        
        summary = f"This crypto asset received an overall score of {overall_score:.1f}/100, indicating a {sentiment} risk profile. "
        summary += f"Analysis identified {risk_counts['High']} high-risk areas, {risk_counts['Medium']} medium-risk areas, and {risk_counts['Low']} low-risk areas. "
        
        # Add recommendation
        summary += f"Recommendation: {report['recommendation']}"
        
        return summary

