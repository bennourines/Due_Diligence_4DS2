"""
Version 3 of question extraction with text block merging and improved filtering.
"""

import fitz
import spacy
import re
import json
import os
import logging
from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, deque

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
    merged: bool = False

@dataclass
class QuestionItem:
    """Data class for extracted questions."""
    text: str
    category: str
    subcategory: str
    page: int
    line: int
    confidence: float = 1.0

class TextProcessor:
    """Helper class for text processing operations."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text by removing extra spaces and special characters."""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip(' -•●○*')
        return text

    @staticmethod
    def merge_line_fragments(fragments: List[str]) -> str:
        """Merge text fragments into a single coherent line."""
        result = []
        for fragment in fragments:
            fragment = fragment.strip()
            if not fragment:
                continue
            if result and not result[-1][-1] in '.?!,:;':
                result[-1] = result[-1] + ' ' + fragment
            else:
                result.append(fragment)
        return ' '.join(result)

class QuestionExtractorV3:
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        self.nlp = spacy.load("en_core_web_sm")
        self.questions: List[QuestionItem] = []
        self.seen_questions: Set[str] = set()
        self.text_buffer: deque = deque(maxlen=3)  # Buffer for multi-line text
        
        # Patterns and indicators
        self.category_indicators = {
            'analysis', 'assessment', 'review', 'evaluation', 'structure', 
            'compliance', 'risk', 'management', 'strategy', 'overview'
        }
        
        self.question_starters = {
            'what', 'where', 'when', 'why', 'how', 'which', 'who',
            'is', 'are', 'can', 'could', 'should', 'would', 'will',
            'does', 'do', 'has', 'have', 'had', 'describe', 'explain'
        }
        
        # Track document structure
        self.current_category = ""
        self.current_subcategory = ""
        self.last_font_size = 0
        self.pending_text = []

    def extract_text_blocks(self) -> List[TextBlock]:
        """Extract text blocks with improved multi-line handling."""
        blocks = []
        doc = fitz.open(self.pdf_path)
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            block_dict = page.get_text("dict")
            
            line_buffer = []
            current_y = None
            
            for block in block_dict["blocks"]:
                if block["type"] == 0:  # text block
                    for line in block["lines"]:
                        y_pos = line["bbox"][1]  # y-coordinate of line
                        
                        # Check if this is a continuation of previous line
                        if current_y is not None and abs(y_pos - current_y) < 2:
                            line_buffer.extend([span["text"] for span in line["spans"]])
                        else:
                            # Process previous line buffer
                            if line_buffer:
                                merged_text = TextProcessor.merge_line_fragments(line_buffer)
                                if merged_text:
                                    blocks.append(TextBlock(
                                        text=merged_text,
                                        size=line_buffer[0]["size"] if isinstance(line_buffer[0], dict) else 0,
                                        font=line_buffer[0]["font"] if isinstance(line_buffer[0], dict) else "",
                                        x0=line_buffer[0]["bbox"][0] if isinstance(line_buffer[0], dict) else 0,
                                        page=page_num + 1,
                                        line=len(blocks) + 1,
                                        merged=len(line_buffer) > 1
                                    ))
                            
                            # Start new line buffer
                            line_buffer = [span["text"] for span in line["spans"]]
                            current_y = y_pos
            
            # Process final line buffer
            if line_buffer:
                merged_text = TextProcessor.merge_line_fragments(line_buffer)
                if merged_text:
                    blocks.append(TextBlock(
                        text=merged_text,
                        size=line_buffer[0]["size"] if isinstance(line_buffer[0], dict) else 0,
                        font=line_buffer[0]["font"] if isinstance(line_buffer[0], dict) else "",
                        x0=line_buffer[0]["bbox"][0] if isinstance(line_buffer[0], dict) else 0,
                        page=page_num + 1,
                        line=len(blocks) + 1,
                        merged=len(line_buffer) > 1
                    ))
        
        doc.close()
        return blocks

    def validate_question(self, text: str, doc) -> Tuple[bool, float]:
        """Validate if text is a question with enhanced confidence scoring."""
        if not text or len(text.split()) < 3:
            return False, 0.0
        
        confidence = 0.0
        
        # Definite indicators
        if text.strip().endswith('?'):
            return True, 1.0
        
        # First word check
        first_word = doc[0].text.lower()
        if first_word in self.question_starters:
            confidence += 0.6
        
        # Structure analysis
        root_token = None
        subject_found = False
        for token in doc:
            if token.dep_ == "ROOT":
                root_token = token
            elif token.dep_ in {"nsubj", "nsubjpass"}:
                subject_found = True
        
        if subject_found and root_token:
            confidence += 0.2
        
        # Check for question-like structure
        has_wh_word = any(token.tag_.startswith('W') for token in doc)
        if has_wh_word:
            confidence += 0.2
        
        # Check for imperative verbs at start
        if doc[0].pos_ == "VERB":
            confidence += 0.1
        
        # Length and complexity
        word_count = len(doc)
        if 5 <= word_count <= 40:  # Reasonable question length
            confidence += 0.1
        
        # Penalize incomplete sentences
        if text.strip().endswith(('and', 'or', 'the', ',', ';')):
            confidence = 0.0
            return False, confidence
        
        # Additional checks for question-like patterns
        if re.search(r'(please\s+)?(describe|explain|provide|list|identify|specify|detail)', text.lower()):
            confidence += 0.2
        
        return confidence >= 0.7, min(confidence, 1.0)

    def process_blocks(self, blocks: List[TextBlock]) -> None:
        """Process text blocks with improved continuity handling."""
        self.pending_text.clear()
        
        for i, block in enumerate(blocks):
            if not block.text.strip():
                continue
            
            # Check for text continuation
            if self.pending_text and not block.text[0].isupper() and not block.is_bold:
                self.pending_text.append(block.text)
                continue
            
            # Process any pending text
            if self.pending_text:
                merged_text = ' '.join(self.pending_text)
                self._process_text_block(merged_text, blocks[i-1])
                self.pending_text.clear()
            
            # Start new text or process current block
            if block.text.endswith(('and', 'or', ',', ';')):
                self.pending_text.append(block.text)
            else:
                self._process_text_block(block.text, block)
        
        # Process any remaining pending text
        if self.pending_text:
            merged_text = ' '.join(self.pending_text)
            self._process_text_block(merged_text, blocks[-1])

    def _process_text_block(self, text: str, block: TextBlock) -> None:
        """Process a single text block or merged text."""
        doc = self.nlp(text)
        
        # Update category/subcategory
        if self._is_heading(text, block):
            if block.is_bold or block.size > self.last_font_size:
                self.current_category = text.strip(' :')
                self.current_subcategory = ""
            else:
                self.current_subcategory = text.strip(' :')
            self.last_font_size = block.size
            return
        
        # Check for questions
        is_question, confidence = self.validate_question(text, doc)
        if is_question:
            clean_text = TextProcessor.clean_text(text)
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

    def _is_heading(self, text: str, block: TextBlock) -> bool:
        """Enhanced heading detection."""
        text_lower = text.lower()
        
        # Skip obvious non-headers
        if text_lower.startswith(('and', 'or', 'the', 'to', 'in', 'of', 'for')):
            return False
        
        # Strong indicators
        if (re.match(r'^[IVX]+\.\s+', text) or  # Roman numerals
            re.match(r'^\d+[.)]\s+', text) or    # Numbered sections
            re.match(r'^[A-Z][.)]\s+', text)):   # Lettered sections
            return True
        
        # Format-based scoring
        score = 0
        if block.is_bold:
            score += 0.4
        if block.size > self.last_font_size:
            score += 0.3
        if block.x0 < 100:  # Left alignment
            score += 0.2
        if text.endswith(':'):
            score += 0.2
        if text.isupper():
            score += 0.3
        
        # Content-based scoring
        words = text_lower.split()
        if len(words) <= 6:
            score += 0.2
        if any(indicator in text_lower for indicator in self.category_indicators):
            score += 0.3
        
        return score >= 0.7

    def save_output(self, output_path: str) -> None:
        """Save extracted questions with enhanced organization."""
        # Organize by category and subcategory
        organized = defaultdict(lambda: defaultdict(list))
        for q in self.questions:
            organized[q.category][q.subcategory].append({
                "question": q.text,
                "page": q.page,
                "line": q.line,
                "confidence": round(q.confidence, 2)
            })
        
        # Sort questions by confidence
        for cat in organized.values():
            for subcat in cat.values():
                subcat.sort(key=lambda x: x["confidence"], reverse=True)
        
        output = {
            "metadata": {
                "source": str(self.pdf_path),
                "total_questions": len(self.questions),
                "categories": len(organized),
                "average_confidence": round(
                    sum(q.confidence for q in self.questions) / len(self.questions)
                    if self.questions else 0,
                    3
                )
            },
            "questions": {
                category: dict(subcategories)
                for category, subcategories in organized.items()
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.questions)} questions to {output_path}")

    def process(self, output_path: str) -> bool:
        """Main processing method with enhanced error handling."""
        try:
            if not self.pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
            
            logger.info(f"Processing PDF: {self.pdf_path}")
            
            # Extract and process text blocks
            blocks = self.extract_text_blocks()
            logger.info(f"Extracted {len(blocks)} text blocks")
            
            self.process_blocks(blocks)
            logger.info(f"Found {len(self.questions)} potential questions")
            
            self.save_output(output_path)
            return True
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise

def main():
    pdf_path = r"C:\Users\ferie\Documents\ferielWork\4ds2PIDS\Due_Diligence_4DS2\given_data\Questions Bank Example Due Diligence on Crypto Assets.pdf"
    output_path = "questions_extracted_v3.json"
    
    extractor = QuestionExtractorV3(pdf_path)
    success = extractor.process(output_path)
    
    if success:
        logger.info("Processing completed successfully")
        # Print summary
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print("\nExtraction Summary:")
            print(f"Total Questions: {data['metadata']['total_questions']}")
            print(f"Categories: {data['metadata']['categories']}")
            print(f"Average Confidence: {data['metadata']['average_confidence']}")
            
            print("\nSample Questions by Category:")
            for category, subcats in list(data['questions'].items())[:3]:
                print(f"\nCategory: {category}")
                for subcat, questions in list(subcats.items())[:2]:
                    print(f"  Subcategory: {subcat}")
                    for q in questions[:2]:
                        print(f"    - [{q['confidence']}] {q['question']}")

if __name__ == "__main__":
    main()