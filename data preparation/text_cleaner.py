import os
import re
import nltk
import spacy
import emoji
import logging
from pathlib import Path
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from datetime import datetime

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load('en_core_web_sm')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('text_cleaning.log'),
        logging.StreamHandler()
    ]
)

class TextCleaner:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.html_pattern = re.compile(r'<.*?>')
        self.emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        
    def remove_html(self, text):
        """Remove HTML tags from text"""
        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text()
    
    def remove_urls(self, text):
        """Remove URLs from text"""
        return self.url_pattern.sub('', text)
    
    def remove_emojis(self, text):
        """Remove emojis from text"""
        return self.emoji_pattern.sub('', text)
    
    def remove_punctuation(self, text):
        """Remove punctuation from text"""
        return re.sub(r'[^\w\s]', ' ', text)
    
    def handle_chat_conversation(self, text):
        """Clean chat conversation format"""
        # Remove timestamps
        text = re.sub(r'\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?', '', text)
        # Remove usernames
        text = re.sub(r'@\w+', '', text)
        # Remove message indicators
        text = re.sub(r'\[.*?\]', '', text)
        return text
    
    def handle_incorrect_text(self, text):
        """Handle common text errors"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove repeated characters (e.g., "helloooo" -> "hello")
        text = re.sub(r'(.)\1+', r'\1', text)
        return text.strip()
    
    def remove_stopwords(self, text):
        """Remove stopwords from text"""
        words = word_tokenize(text)
        return ' '.join([word for word in words if word.lower() not in self.stop_words])
    
    def lemmatize_text(self, text):
        """Lemmatize text"""
        words = word_tokenize(text)
        return ' '.join([self.lemmatizer.lemmatize(word) for word in words])
    
    def to_lowercase(self, text):
        """Convert text to lowercase"""
        return text.lower()
    
    def pos_tagging(self, text):
        """Perform POS tagging"""
        words = word_tokenize(text)
        return pos_tag(words)
    
    def coreference_resolution(self, text):
        """Perform basic coreference resolution"""
        doc = nlp(text)
        # This is a simplified version - spaCy's coreference resolution is limited
        # You might want to use a more specialized library like neuralcoref
        return doc
    
    def clean_text(self, text):
        """Apply all cleaning steps"""
        # Apply cleaning steps in order
        text = self.remove_html(text)
        text = self.remove_urls(text)
        text = self.remove_emojis(text)
        text = self.handle_chat_conversation(text)
        text = self.handle_incorrect_text(text)
        text = self.remove_punctuation(text)
        text = self.to_lowercase(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize_text(text)
        
        return text

def create_output_directory():
    """Create the cleaned directory if it doesn't exist"""
    output_dir = Path("cleaned")
    output_dir.mkdir(exist_ok=True)
    return output_dir

def process_files():
    """Process all text files in the text_extractions directory"""
    cleaner = TextCleaner()
    output_dir = create_output_directory()
    input_dir = Path("text_extractions")
    
    # Create a summary file
    summary_file = output_dir / "cleaning_summary.txt"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Text Cleaning Summary\n")
        f.write("===================\n\n")
        
        # Process each text file
        for file_path in input_dir.glob("*.txt"):
            try:
                # Read the file
                with open(file_path, 'r', encoding='utf-8') as input_file:
                    text = input_file.read()
                
                # Clean the text
                cleaned_text = cleaner.clean_text(text)
                
                # Save cleaned text
                output_filename = f"{file_path.stem}_cleaned.txt"
                output_path = output_dir / output_filename
                
                with open(output_path, 'w', encoding='utf-8') as output_file:
                    output_file.write(cleaned_text)
                
                # Write summary
                f.write(f"\nFile: {file_path.name}\n")
                f.write(f"Original length: {len(text)} characters\n")
                f.write(f"Cleaned length: {len(cleaned_text)} characters\n")
                f.write(f"Processing time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                logging.info(f"Successfully cleaned {file_path.name}")
                
            except Exception as e:
                logging.error(f"Error processing {file_path}: {str(e)}")
                continue

if __name__ == "__main__":
    logging.info("Starting text cleaning...")
    process_files()
    logging.info("Text cleaning completed.") 