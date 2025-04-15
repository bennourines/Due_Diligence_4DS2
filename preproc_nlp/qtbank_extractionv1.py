import fitz  # PyMuPDF
import spacy
import re
import json
import os
import logging
from typing import List, Dict, Any, Set
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

class PDFQuestionExtractor:
    def __init__(self, pdf_path: str):
        """Initialize the extractor with the path to the PDF file."""
        self.pdf_path = pdf_path
        self.categories: List[Dict[str, Any]] = []
        self.nlp = spacy.load("en_core_web_sm")
        self.seen_questions: Set[str] = set()  # Track questions to detect duplicates
    
    def validate_file(self) -> bool:
        """Check if the PDF file exists and is readable."""
        if not os.path.exists(self.pdf_path):
            logger.error(f"File '{self.pdf_path}' does not exist.")
            return False
        if not self.pdf_path.lower().endswith('.pdf'):
            logger.error(f"File '{self.pdf_path}' is not a PDF.")
            return False
        logger.info(f"Validated file: {self.pdf_path}")
        return True

    def extract_text_blocks(self) -> List[Dict[str, Any]]:
        """Extract text blocks with formatting details from the PDF."""
        try:
            doc = fitz.open(self.pdf_path)
            blocks = []
            for page in doc:
                page_blocks = page.get_text("dict")["blocks"]
                for block in page_blocks:
                    if block["type"] == 0:  # Text block
                        for line in block["lines"]:
                            for span in line["spans"]:
                                blocks.append({
                                    "text": span["text"].strip(),
                                    "size": span["size"],
                                    "font": span["font"],
                                    "bbox": span["bbox"],  # (x0, y0, x1, y1)
                                    "x0": span["bbox"][0]  # Indentation reference
                                })
            doc.close()
            logger.info(f"Extracted {len(blocks)} text blocks from PDF")
            return blocks
        except Exception as e:
            logger.error(f"Error extracting text blocks from PDF: {e}")
            return []

    def is_question(self, text: str) -> bool:
        """Use spaCy to determine if a text is a question."""
        if not text:
            return False
        doc = self.nlp(text)
        # Check for question-like patterns: question words, auxiliary verbs, or question mark
        question_words = {"what", "where", "when", "why", "how", "which", "who"}
        has_question_word = any(token.text.lower() in question_words for token in doc)
        has_aux_verb = any(token.dep_ == "aux" and token.tag_ in {"VBZ", "VBP", "VBD"} for token in doc)
        has_question_mark = text.strip().endswith("?")
        return has_question_mark or has_question_word or has_aux_verb

    def parse_questions(self, blocks: List[Dict[str, Any]]) -> None:
        """Parse text blocks to extract and classify questions."""
        current_category = None
        current_subcategory = None
        category_pattern = re.compile(r'^(I{1,3}|IV|V)\.\s+(.+)$')  # Matches "I.", "II.", etc.
        subcategory_pattern = re.compile(r'^\s*([A-Za-z\s&]+):$')  # Matches "Jurisdictional Analysis:", etc.

        # Estimate indentation levels and font sizes for hierarchy
        x0_values = sorted(set(block["x0"] for block in blocks if block["text"]))
        font_sizes = sorted(set(block["size"] for block in blocks if block["text"]))
        large_font = max(font_sizes) if font_sizes else 12  # Fallback font size
        indent_threshold = x0_values[1] if len(x0_values) > 1 else x0_values[0] + 10

        for block in blocks:
            text = block["text"]
            if not text:
                continue

            # Detect category (larger font or Roman numeral)
            if block["size"] >= large_font or category_pattern.match(text):
                match = category_pattern.match(text)
                category_name = match.group(2).strip() if match else text
                current_category = {
                    "category": category_name,
                    "subcategories": []
                }
                self.categories.append(current_category)
                current_subcategory = None
                logger.debug(f"Detected category: {category_name}")
                continue

            # Detect subcategory (colon-terminated or indented)
            if (subcategory_pattern.match(text) or 
                (block["x0"] <= indent_threshold and text.endswith(":"))):
                match = subcategory_pattern.match(text)
                subcategory_name = match.group(1).strip() if match else text.rstrip(":")
                current_subcategory = {
                    "subcategory": subcategory_name,
                    "questions": []
                }
                if current_category:
                    current_category["subcategories"].append(current_subcategory)
                    logger.debug(f"Detected subcategory: {subcategory_name}")
                continue

            # Detect question (spaCy-based or bullet points)
            if current_subcategory and self.is_question(text):
                # Clean text (remove bullet points if present)
                clean_text = re.sub(r'^[●○\s-]*', '', text).strip()
                if clean_text in self.seen_questions:
                    logger.warning(f"Duplicate question detected: {clean_text}")
                    continue
                self.seen_questions.add(clean_text)
                current_subcategory["questions"].append(clean_text)
                logger.debug(f"Detected question: {clean_text}")

        # Validate structure
        self.validate_structure()

    def validate_structure(self) -> None:
        """Check for incomplete categories/subcategories or duplicates."""
        for category in self.categories:
            if not category["subcategories"]:
                logger.warning(f"Category '{category['category']}' has no subcategories")
            for subcategory in category["subcategories"]:
                if not subcategory["questions"]:
                    logger.warning(f"Subcategory '{subcategory['subcategory']}' in '{category['category']}' has no questions")
                # Check for duplicates within subcategory
                questions = subcategory["questions"]
                if len(questions) != len(set(questions)):
                    logger.warning(f"Duplicate questions found in subcategory '{subcategory['subcategory']}'")

    def save_to_json(self, output_path: str) -> bool:
        """Save extracted questions to a JSON file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({"categories": self.categories}, f, indent=4, ensure_ascii=False)
            logger.info(f"Questions saved to '{output_path}'")
            return True
        except Exception as e:
            logger.error(f"Error saving JSON: {e}")
            return False

    def process(self, output_json_path: str) -> bool:
        """Main method to process the PDF and extract questions."""
        if not self.validate_file():
            return False

        blocks = self.extract_text_blocks()
        if not blocks:
            logger.error("No text blocks extracted from the PDF")
            return False

        self.parse_questions(blocks)
        if not self.categories:
            logger.warning("No categories extracted from the PDF")
            return False

        return self.save_to_json(output_json_path)


def main():
    # Example usage
    pdf_path = r"C:\Users\ferie\Documents\ferielWork\4ds2PIDS\Due_Diligence_4DS2\given_data\Questions Bank Example Due Diligence on Crypto Assets.pdf"  # Replace with your PDF file path
    output_json_path = "questions_output.json"
    
    extractor = PDFQuestionExtractor(pdf_path)
    success = extractor.process(output_json_path)
    
    if success:
        logger.info("Processing completed successfully")
    else:
        logger.error("Processing failed")


if __name__ == "__main__":
    main()