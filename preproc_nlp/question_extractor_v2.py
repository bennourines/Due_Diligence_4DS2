"""
Version 2 of question extraction with improved filtering and categorization.
"""

import fitz
import spacy
import re
import json
import os
import logging
from typing import List, Dict, Any, Set, Tuple
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('question_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TextBlock:
    """Data class for text block information."""
    text: str
    size: float
    font: str
    x0: float
    page: int
    line: int
    is_bold: bool = False
    is_italic: bool = False

@dataclass
class QuestionItem:
    """Data class for extracted questions."""
    text: str
    category: str
    subcategory: str
    page: int
    line: int
    confidence: float = 1.0

class ImprovedQuestionExtractor:
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        self.nlp = spacy.load("en_core_web_sm")
        self.questions: List[QuestionItem] = []
        self.seen_questions: Set[str] = set()
        
        # Enhanced patterns
        self.category_indicators = {
            'analysis', 'assessment', 'review', 'evaluation', 'structure', 
            'compliance', 'risk', 'management', 'strategy', 'overview'
        }
        
        self.question_starters = {
            'what', 'where', 'when', 'why', 'how', 'which', 'who',
            'is', 'are', 'can', 'could', 'should', 'would', 'will',
            'does', 'do', 'has', 'have', 'had', 'describe', 'explain'
        }
        
        # Track hierarchy
        self.current_category = ""
        self.current_subcategory = ""
        self.category_confidence = 0.0

    def is_likely_heading(self, block: TextBlock) -> bool:
        """Determine if a block is likely a heading."""
        text = block.text.lower()
        
        # Check format characteristics
        format_score = 0
        if block.is_bold:
            format_score += 0.4
        if block.size > 11:  # Larger text
            format_score += 0.3
        if block.x0 < 100:  # Left alignment
            format_score += 0.2
        
        # Check content characteristics
        content_score = 0
        words = text.split()
        if len(words) <= 6:  # Short phrases are more likely headings
            content_score += 0.3
        if any(indicator in text for indicator in self.category_indicators):
            content_score += 0.4
        if text.endswith(':'):
            content_score += 0.2
            
        # Roman numeral pattern
        if re.match(r'^[IVX]+\.\s+', block.text):
            return True
            
        return (format_score + content_score) > 0.7

    def validate_question(self, text: str, doc) -> Tuple[bool, float]:
        """Validate if text is a question with confidence score."""
        if not text or len(text) < 10:
            return False, 0.0
            
        confidence = 0.0
        
        # Definite indicators
        if text.strip().endswith('?'):
            return True, 1.0
            
        # Check first word
        first_word = doc[0].text.lower()
        if first_word in self.question_starters:
            confidence += 0.6
            
        # Structure analysis
        has_subject = any(token.dep_ == "nsubj" for token in doc)
        has_verb = any(token.pos_ == "VERB" for token in doc)
        has_aux = any(token.dep_ == "aux" for token in doc)
        
        if has_subject and has_verb:
            confidence += 0.2
        if has_aux and has_subject:
            confidence += 0.2
            
        # Length and complexity checks
        word_count = len(doc)
        if 5 <= word_count <= 30:  # Reasonable question length
            confidence += 0.1
            
        # Additional patterns
        if re.search(r'(please\s+)?(describe|explain|provide|list|identify)', text.lower()):
            confidence += 0.2
            
        return confidence >= 0.7, min(confidence, 1.0)

    def extract_text_blocks(self) -> List[TextBlock]:
        """Extract text blocks from PDF with formatting information."""
        blocks = []
        doc = fitz.open(self.pdf_path)
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            block_dict = page.get_text("dict")
            
            for block in block_dict["blocks"]:
                if block["type"] == 0:  # text block
                    for line_num, line in enumerate(block["lines"]):
                        for span in line["spans"]:
                            # Extract font properties
                            font_flags = span.get("flags", 0)
                            is_bold = bool(font_flags & 2**4)
                            is_italic = bool(font_flags & 2**1)
                            
                            blocks.append(TextBlock(
                                text=span["text"].strip(),
                                size=span["size"],
                                font=span["font"],
                                x0=span["bbox"][0],
                                page=page_num + 1,
                                line=line_num + 1,
                                is_bold=is_bold,
                                is_italic=is_italic
                            ))
        
        doc.close()
        return blocks

    def consolidate_categories(self) -> None:
        """Consolidate similar categories and subcategories."""
        # Group questions by category
        category_groups = defaultdict(list)
        for question in self.questions:
            category_groups[question.category.lower()].append(question)
            
        # Merge similar categories
        merged_questions = []
        processed_categories = set()
        
        for category, questions in category_groups.items():
            if category in processed_categories:
                continue
                
            # Find similar categories
            similar_categories = {
                cat for cat in category_groups.keys()
                if cat not in processed_categories and
                (cat == category or self.are_categories_similar(cat, category))
            }
            
            # Merge categories
            if similar_categories:
                main_category = max(similar_categories, key=len)
                processed_categories.update(similar_categories)
                
                # Update questions with consolidated category
                for q in questions:
                    q.category = main_category.title()
                merged_questions.extend(questions)
                
        self.questions = merged_questions

    @staticmethod
    def are_categories_similar(cat1: str, cat2: str) -> bool:
        """Check if two categories are similar enough to merge."""
        def normalize(text):
            return re.sub(r'[^a-z\s]', '', text.lower())
            
        norm1 = normalize(cat1)
        norm2 = normalize(cat2)
        
        # Check for substring relationship
        if norm1 in norm2 or norm2 in norm1:
            return True
            
        # Check word overlap
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        overlap = len(words1 & words2)
        total = len(words1 | words2)
        
        return overlap / total > 0.5 if total > 0 else False

    def process_blocks(self, blocks: List[TextBlock]) -> None:
        """Process text blocks to extract questions and structure."""
        for block in blocks:
            if not block.text.strip():
                continue
                
            # Parse text
            doc = self.nlp(block.text)
            
            # Check for headings/categories
            if self.is_likely_heading(block):
                if block.is_bold or block.size >= 12:
                    self.current_category = block.text.strip(' :')
                    self.category_confidence = 0.9
                else:
                    self.current_subcategory = block.text.strip(' :')
                continue
                
            # Check for questions
            is_question, confidence = self.validate_question(block.text, doc)
            if is_question:
                clean_text = re.sub(r'^[●○•\s-]*', '', block.text).strip()
                
                # Avoid duplicates
                if clean_text not in self.seen_questions:
                    self.seen_questions.add(clean_text)
                    self.questions.append(QuestionItem(
                        text=clean_text,
                        category=self.current_category or "General",
                        subcategory=self.current_subcategory or "Uncategorized",
                        page=block.page,
                        line=block.line,
                        confidence=confidence
                    ))

    def save_output(self, output_path: str) -> None:
        """Save extracted questions to JSON with metadata."""
        # First consolidate categories
        self.consolidate_categories()
        
        # Organize by category and subcategory
        organized = defaultdict(lambda: defaultdict(list))
        for q in self.questions:
            organized[q.category][q.subcategory].append({
                "question": q.text,
                "page": q.page,
                "line": q.line,
                "confidence": round(q.confidence, 2)
            })
            
        # Create output structure
        output = {
            "metadata": {
                "source": str(self.pdf_path),
                "total_questions": len(self.questions),
                "categories": len(organized),
                "extraction_date": "",  # TODO: Add timestamp
            },
            "questions": {
                category: dict(subcategories)
                for category, subcategories in organized.items()
            }
        }
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved {len(self.questions)} questions to {output_path}")

    def process(self, output_path: str) -> bool:
        """Main processing method."""
        try:
            if not self.pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
                
            logger.info(f"Processing PDF: {self.pdf_path}")
            
            # Extract text blocks
            blocks = self.extract_text_blocks()
            logger.info(f"Extracted {len(blocks)} text blocks")
            
            # Process blocks
            self.process_blocks(blocks)
            logger.info(f"Found {len(self.questions)} potential questions")
            
            # Save results
            self.save_output(output_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise

def main():
    # Example usage
    pdf_path = r"C:\Users\ferie\Documents\ferielWork\4ds2PIDS\Due_Diligence_4DS2\given_data\Questions Bank Example Due Diligence on Crypto Assets.pdf"
    output_path = "questions_extracted_v2.json"
    
    extractor = ImprovedQuestionExtractor(pdf_path)
    success = extractor.process(output_path)
    
    if success:
        logger.info("Processing completed successfully")
        # Print summary
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print("\nExtraction Summary:")
            print(f"Total Questions: {data['metadata']['total_questions']}")
            print(f"Categories: {data['metadata']['categories']}")
            
            # Show sample questions
            print("\nSample Questions by Category:")
            for category, subcats in list(data['questions'].items())[:3]:
                print(f"\nCategory: {category}")
                for subcat, questions in list(subcats.items())[:2]:
                    print(f"  Subcategory: {subcat}")
                    for q in questions[:2]:
                        print(f"    - [{q['confidence']}] {q['question']}")
    else:
        logger.error("Processing failed")

if __name__ == "__main__":
    main()