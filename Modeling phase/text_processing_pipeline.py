import os
import re
import nltk
import spacy
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from bs4 import BeautifulSoup
import emoji
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import pickle
from tqdm import tqdm

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Initialize spaCy for advanced NLP tasks
nlp = spacy.load('en_core_web_sm')

class TextProcessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def clean_text(self, text):
        # Remove HTML tags
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove emojis
        text = emoji.replace_emoji(text, '')
        
        # Remove special characters and punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.lower()
    
    def tokenize_text(self, text, tokenize_by='word'):
        if tokenize_by == 'word':
            return word_tokenize(text)
        elif tokenize_by == 'sentence':
            return sent_tokenize(text)
        else:
            raise ValueError("tokenize_by must be either 'word' or 'sentence'")
    
    def remove_stopwords(self, tokens):
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def stem_tokens(self, tokens):
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize_tokens(self, tokens):
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def get_pos_tags(self, tokens):
        return pos_tag(tokens)
    
    def process_text(self, text, chunk_size=512):
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Tokenize by sentence for chunking
        sentences = self.tokenize_text(cleaned_text, 'sentence')
        
        # Create chunks
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            words = self.tokenize_text(sentence, 'word')
            words = self.remove_stopwords(words)
            words = self.lemmatize_tokens(words)
            
            if current_size + len(words) > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = len(words)
            else:
                current_chunk.append(sentence)
                current_size += len(words)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def create_embeddings(self, chunks):
        return self.model.encode(chunks)

class VectorDatabase:
    def __init__(self, n_neighbors=5):
        self.index = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        self.documents = []
        self.metadata = []
        self.is_fitted = False
    
    def store_embeddings(self, chunks, embeddings, metadata):
        if len(chunks) > 0:
            if not self.is_fitted:
                self.index.fit(embeddings)
                self.is_fitted = True
            else:
                # Create a new index with all data
                all_embeddings = np.vstack([self.index._fit_X, embeddings])
                self.index.fit(all_embeddings)
            
            self.documents.extend(chunks)
            self.metadata.extend(metadata)
    
    def save(self, directory="vector_db"):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "vector_db.pkl"), "wb") as f:
            pickle.dump({
                'index': self.index,
                'documents': self.documents,
                'metadata': self.metadata,
                'is_fitted': self.is_fitted
            }, f)
    
    def load(self, directory="vector_db"):
        with open(os.path.join(directory, "vector_db.pkl"), "rb") as f:
            data = pickle.load(f)
            self.index = data['index']
            self.documents = data['documents']
            self.metadata = data['metadata']
            self.is_fitted = data['is_fitted']

def process_whitepapers(directory_path):
    processor = TextProcessor()
    vector_db = VectorDatabase()
    
    for filename in tqdm(os.listdir(directory_path)):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            # Process the text
            chunks = processor.process_text(text)
            
            # Create embeddings
            embeddings = processor.create_embeddings(chunks)
            
            # Prepare metadata
            metadata = [{"source": filename} for _ in range(len(chunks))]
            
            # Store in vector database
            vector_db.store_embeddings(chunks, embeddings, metadata)
    
    # Save the database
    vector_db.save()

if __name__ == "__main__":
    whitepapers_dir = "Amine/whitepapers/txt_whitepapers_np"
    process_whitepapers(whitepapers_dir) 