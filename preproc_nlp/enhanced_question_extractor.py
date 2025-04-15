"""
Enhanced version of question extraction from PDF with improved detection and structure.
"""

import fitz
import spacy
import re
import json
import os
import logging
from typing import List, Dict, Any, Set, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from tqdm import tqdm

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
    """Data class for storing text block information."""
    text: str
    size: float
    font: str
    bbox: Tuple[float, float, float, float]
    x0: float
    line_number: int
    page_number: int

@dataclass
class Question:
    """Data class for storing question information."""
    text: str
    category: str
    subcategory: str
    page: int
    line: int

class EnhancedPDFQuestionExtractor:
    def __init__(self, pdf_path: str):
        """Initialize the extractor with improved configuration."""
        self.pdf_path = Path(pdf_path)
        self.nlp = spacy.load("en_core_web_sm")
        self.categories: List[Dict[str, Any]] = []
        self.seen_questions: Set[str] = set()
        self.questions: List[Question] = []
        
        # Enhanced patterns
        self.category_patterns = [
            re.compile(r'^(I{1,3}|IV|V|VI{1,3})\.\s+(.+)$'),  # Roman numerals
            re.compile(r'^\d+\.\s+(.+)$'),  # Numeric
            re.compile(r'^[A-Z][.]\s+(.+)$')  # Letter headings
        ]
        self.subcategory_patterns = [
            re.compile(r'^\s*([A-Za-z][.]|\d+[.])\s+(.+?):?\s*$'),  # Numbered/lettered with colon
            re.compile(r'^\s*([A-Za-z\s&]+):$'),  # Text with colon
            re.compile(r'^\s*([A-Za-z\s&]+)\s*$')  # All caps text
        ]
        self.question_indicators = {
            "what", "where", "when", "why", "how", "which", "who", "is", "are", "can", "could",
            "should", "would", "will", "does", "do", "has", "have", "had"
        }

    def extract_text_blocks(self) -> List[TextBlock]:
        """Extract text blocks with enhanced formatting detection."""
        try:
            doc = fitz.open(self.pdf_path)
            blocks = []
            for page_num, page in enumerate(doc, 1):
                dict_blocks = page.get_text("dict")["blocks"]
                line_num = 1
                for block in dict_blocks:
                    if block["type"] == 0:  # Text block
                        for line in block["lines"]:
                            for span in line["spans"]:
                                blocks.append(TextBlock(
                                    text=span["text"].strip(),
                                    size=span["size"],
                                    font=span["font"],
                                    bbox=span["bbox"],
                                    x0=span["bbox"][0],
                                    line_number=line_num,
                                    page_number=page_num
                                ))
                            line_num += 1
            doc.close()
            return blocks
        except Exception as e:
            logger.error(f"Error extracting text blocks: {e}")
            raise

    def is_question(self, text: str) -> bool:
        """Enhanced question detection using multiple methods."""
        if not text:
            return False

        # Quick checks
        if text.strip().endswith("?"):
            return True
        
        # Clean text for analysis
        clean_text = re.sub(r'^[●○•\s-]*', '', text).strip()
        if not clean_text:
            return False

        # Analyze with spaCy
        doc = self.nlp(clean_text)
        
        # Check for question indicators
        first_word = doc[0].text.lower()
        if first_word in self.question_indicators:
            return True

        # Check for auxiliary verbs at start
        if doc[0].dep_ == "aux" and doc[0].pos_ == "AUX":
            return True

        # Check for complex question patterns
        has_wh_word = any(token.tag_ == "WDT" or token.tag_ == "WP" or token.tag_ == "WRB" for token in doc)
        has_aux_start = any(token.dep_ == "aux" and token.i == 0 for token in doc)
        
        return has_wh_word or has_aux_start

    def identify_category(self, block: TextBlock, prev_font_sizes: List[float]) -> str:
        """Identify if a block is a category heading."""
        text = block.text.strip()
        
        # Check patterns
        for pattern in self.category_patterns:
            if match := pattern.match(text):
                return match.group(1) if len(match.groups()) > 1 else text

        # Check formatting
        is_large = len(prev_font_sizes) > 0 and block.size > max(prev_font_sizes)
        is_bold = "bold" in block.font.lower()
        
        if (is_large or is_bold) and len(text) < 100:  # Avoid long text
            return text
            
        return ""

    def process_blocks(self, blocks: List[TextBlock]) -> None:
        """Process text blocks with improved structure detection."""
        current_category = None
        current_subcategory = None
        font_sizes = []
        
        for i, block in enumerate(blocks):
            text = block.text.strip()
            if not text:
                continue

            # Update font size history
            if block.size > 0:
                font_sizes.append(block.size)
                if len(font_sizes) > 5:  # Keep last 5 sizes
                    font_sizes.pop(0)

            # Detect category
            if category := self.identify_category(block, font_sizes):
                current_category = category
                current_subcategory = None
                continue

            # Detect subcategory
            for pattern in self.subcategory_patterns:
                if match := pattern.match(text):
                    subcategory_text = match.group(1) if len(match.groups()) > 0 else text
                    if current_category:
                        current_subcategory = subcategory_text.strip(': ')
                    break

            # Detect question
            if self.is_question(text) and current_category:
                clean_text = re.sub(r'^[●○•\s-]*', '', text).strip()
                if clean_text not in self.seen_questions:
                    self.seen_questions.add(clean_text)
                    self.questions.append(Question(
                        text=clean_text,
                        category=current_category,
                        subcategory=current_subcategory or "General",
                        page=block.page_number,
                        line=block.line_number
                    ))

    def save_to_json(self, output_path: str) -> None:
        """Save extracted questions to JSON with enhanced structure."""
        # Organize questions by category and subcategory
        organized_data = {}
        for question in self.questions:
            if question.category not in organized_data:
                organized_data[question.category] = {}
            
            if question.subcategory not in organized_data[question.category]:
                organized_data[question.category][question.subcategory] = []
            
            organized_data[question.category][question.subcategory].append({
                "question": question.text,
                "page": question.page,
                "line": question.line
            })

        # Save to file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": {
                        "source": str(self.pdf_path),
                        "total_questions": len(self.questions),
                        "categories": len(organized_data)
                    },
                    "questions": organized_data
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.questions)} questions to {output_path}")
        except Exception as e:
            logger.error(f"Error saving to JSON: {e}")
            raise

    def process(self, output_path: str) -> bool:
        """Main processing method with progress tracking."""
        try:
            if not self.pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")

            logger.info(f"Processing PDF: {self.pdf_path}")
            blocks = self.extract_text_blocks()
            logger.info(f"Extracted {len(blocks)} text blocks")

            logger.info("Analyzing text structure and extracting questions...")
            self.process_blocks(blocks)
            logger.info(f"Found {len(self.questions)} questions")

            self.save_to_json(output_path)
            return True

        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return False

def main():
    # Example usage
    pdf_path = r"C:\Users\ferie\Documents\ferielWork\4ds2PIDS\Due_Diligence_4DS2\given_data\Questions Bank Example Due Diligence on Crypto Assets.pdf"
    output_path = "questions_extracted.json"
    
    extractor = EnhancedPDFQuestionExtractor(pdf_path)
    success = extractor.process(output_path)
    
    if success:
        logger.info("Processing completed successfully")
        # Print some statistics
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print("\nExtraction Statistics:")
            print(f"Total Questions: {data['metadata']['total_questions']}")
            print(f"Categories: {data['metadata']['categories']}")
            print("\nSample Questions:")
            for category, subcats in list(data['questions'].items())[:2]:
                print(f"\nCategory: {category}")
                for subcat, questions in list(subcats.items())[:2]:
                    print(f"  Subcategory: {subcat}")
                    for q in questions[:2]:
                        print(f"    - {q['question']}")
    else:
        logger.error("Processing failed")

if __name__ == "__main__":
    main()