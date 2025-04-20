"""
Final version of question extraction with improved text segmentation and hierarchy detection.
"""

import fitz
import spacy
import re
import json
import os
import logging
from typing import List, Dict, Any, Set, Optional
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
class Section:
    """Represents a document section with hierarchical structure."""
    level: int
    text: str
    parent: Optional[str] = None
    font_size: float = 0
    is_bold: bool = False

@dataclass
class Question:
    """Represents an extracted question with metadata."""
    text: str
    section: str
    subsection: str
    page: int
    line: int
    confidence: float
    original_text: str = ""

class DocumentStructureExtractor:
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        self.nlp = spacy.load("en_core_web_sm")
        self.sections: List[Section] = []
        self.questions: List[Question] = []
        self.current_section = ""
        self.current_subsection = ""
        self.section_patterns = [
            re.compile(r'^(?:I{1,3}|IV|V|VI{1,3})\.\s+(.+)$'),  # Roman numerals
            re.compile(r'^\d+\.\s+(.+)$'),  # Decimal numbers
            re.compile(r'^[A-Z]\.\s+(.+)$'),  # Capital letters
            re.compile(r'^(?:[A-Za-z][\).]|\d+[\).])?\s*([A-Z][A-Za-z\s]+):')  # Labeled sections
        ]
        
        # Improved question detection patterns
        self.question_starters = {
            'what', 'where', 'when', 'why', 'how', 'which', 'who',
            'is', 'are', 'can', 'could', 'should', 'would', 'will',
            'does', 'do', 'has', 'have', 'had', 'describe', 'explain',
            'analyze', 'evaluate', 'identify', 'list', 'specify'
        }

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove bullet points and special characters
        text = re.sub(r'^[●○•\s-]+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def is_section_header(self, text: str) -> bool:
        """Determine if text is a section header."""
        text = text.strip()
        if not text:
            return False

        # Check patterns
        for pattern in self.section_patterns:
            if pattern.match(text):
                return True

        # Check formatting
        words = text.split()
        if len(words) <= 7 and text.endswith(':'):
            return True

        return False

    def extract_questions(self, text: str) -> List[str]:
        """Extract individual questions from text."""
        # Split on question marks, but handle special cases
        questions = []
        current = []
        
        # Handle multi-line questions
        lines = text.split('\n')
        for line in lines:
            parts = line.split('?')
            for i, part in enumerate(parts):
                part = part.strip()
                if not part:
                    continue
                
                if i < len(parts) - 1:  # Has question mark
                    current.append(part + '?')
                    questions.append(' '.join(current))
                    current = []
                else:  # No question mark
                    current.append(part)
        
        # Add any remaining text
        if current:
            text = ' '.join(current)
            if text.endswith('?'):
                questions.append(text)
        
        return [self.clean_text(q) for q in questions if self.is_valid_question(q)]

    def is_valid_question(self, text: str) -> bool:
        """Validate if text is a proper question."""
        text = self.clean_text(text)
        if not text or len(text.split()) < 3:
            return False
            
        # Must end with question mark or start with question word
        if not (text.endswith('?') or any(text.lower().startswith(w + ' ') for w in self.question_starters)):
            return False
            
        # Check structure using spaCy
        doc = self.nlp(text)
        has_verb = any(token.pos_ == "VERB" for token in doc)
        has_subject = any(token.dep_ in {"nsubj", "nsubjpass"} for token in doc)
        
        return has_verb and (has_subject or text.endswith('?'))

    def process_pdf(self) -> None:
        """Process PDF and extract structure and questions."""
        doc = fitz.open(self.pdf_path)
        current_text = []
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if block["type"] == 0:  # text block
                    for line in block["lines"]:
                        text = " ".join(span["text"] for span in line["spans"]).strip()
                        if not text:
                            continue
                            
                        # Check if it's a section header
                        if self.is_section_header(text):
                            # Process any pending text
                            if current_text:
                                self.process_text_block(" ".join(current_text), page_num + 1)
                                current_text = []
                            
                            # Update section information
                            self.update_section(text)
                        else:
                            current_text.append(text)
                            
                            # Process if we have a complete question
                            if text.endswith('?'):
                                self.process_text_block(" ".join(current_text), page_num + 1)
                                current_text = []
            
            # Process any remaining text at page end
            if current_text:
                self.process_text_block(" ".join(current_text), page_num + 1)
                current_text = []
        
        doc.close()

    def update_section(self, text: str) -> None:
        """Update current section and subsection based on header text."""
        text = self.clean_text(text)
        
        # Determine section level
        level = 1
        if text.startswith(('●', '○', '•')):
            level = 2
        elif not any(p.match(text) for p in self.section_patterns):
            level = 2
        
        # Update section structure
        if level == 1:
            self.current_section = text
            self.current_subsection = ""
        else:
            self.current_subsection = text

    def process_text_block(self, text: str, page_num: int) -> None:
        """Process a block of text to extract questions."""
        questions = self.extract_questions(text)
        
        for q_text in questions:
            # Skip if appears truncated
            if q_text.endswith(('and', 'or', ',', ';')):
                continue
                
            # Calculate confidence score
            confidence = 1.0 if q_text.endswith('?') else 0.8
            
            self.questions.append(Question(
                text=q_text,
                section=self.current_section or "General",
                subsection=self.current_subsection or "Uncategorized",
                page=page_num,
                line=len(self.questions) + 1,
                confidence=confidence,
                original_text=text
            ))

    def save_output(self, output_path: str) -> None:
        """Save extracted questions to JSON with hierarchical structure."""
        # Organize by section and subsection
        organized = {}
        for question in self.questions:
            section = question.section.strip(' :')
            subsection = question.subsection.strip(' :')
            
            if section not in organized:
                organized[section] = {}
            if subsection not in organized[section]:
                organized[section][subsection] = []
            
            organized[section][subsection].append({
                "question": question.text,
                "page": question.page,
                "line": question.line,
                "confidence": question.confidence
            })

        # Create output structure
        output = {
            "metadata": {
                "source": str(self.pdf_path),
                "total_questions": len(self.questions),
                "sections": len(organized),
                "average_confidence": sum(q.confidence for q in self.questions) / len(self.questions) if self.questions else 0
            },
            "questions": organized
        }
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved {len(self.questions)} questions to {output_path}")

def main():
    pdf_path = r"C:\Users\ferie\Documents\ferielWork\4ds2PIDS\Due_Diligence_4DS2\given_data\Questions Bank Example Due Diligence on Crypto Assets.pdf"
    output_path = "questions_final.json"
    
    extractor = DocumentStructureExtractor(pdf_path)
    
    try:
        logger.info(f"Processing PDF: {pdf_path}")
        extractor.process_pdf()
        extractor.save_output(output_path)
        logger.info("Processing completed successfully")
        
        # Print summary
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print("\nExtraction Summary:")
            print(f"Total Questions: {data['metadata']['total_questions']}")
            print(f"Sections: {data['metadata']['sections']}")
            print(f"Average Confidence: {data['metadata']['average_confidence']:.2f}")
            
            print("\nSample Questions by Section:")
            for section, subsections in list(data['questions'].items())[:3]:
                print(f"\nSection: {section}")
                for subsection, questions in list(subsections.items())[:2]:
                    print(f"  Subsection: {subsection}")
                    for q in questions[:2]:
                        print(f"    - [{q['confidence']}] {q['question']}")
    
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise

if __name__ == "__main__":
    main()