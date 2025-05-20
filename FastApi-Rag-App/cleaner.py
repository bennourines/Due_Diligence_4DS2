import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure necessary NLTK data is downloaded
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


def clean_text_nlp(text: str, language: str = 'english') -> str:
    """
    Clean text for NLP by:
      - Removing form feeds and lowercasing
      - Stripping URLs, emails, non-letter characters
      - Tokenizing (without sentence-splitting to avoid punkt_tab errors)
      - Removing stopwords and short tokens
      - Lemmatizing
    Returns a cleaned string of space-separated tokens.
    """
    # Basic cleanup
    text = text.replace("\f", " ")
    text = text.lower()
    # Remove URLs and emails
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    # Retain only letters and whitespace
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Tokenize: use preserve_line=True to bypass sentence tokenizer and avoid punkt_tab lookup
    try:
        tokens = word_tokenize(text, preserve_line=True)
    except TypeError:
        # Older NLTK versions may not support preserve_line; fallback to simple split
        tokens = text.split()
    except LookupError:
        # Fallback: use basic tokenizer if punkt data missing
        from nltk.tokenize import TreebankWordTokenizer
        tokens = TreebankWordTokenizer().tokenize(text)

    # Load stopwords
    stop_words = set(stopwords.words(language))
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    cleaned_tokens = []
    for token in tokens:
        # Remove stopwords and short tokens
        if token in stop_words or len(token) <= 2:
            continue
        # Lemmatize
        lemma = lemmatizer.lemmatize(token)
        cleaned_tokens.append(lemma)

    return ' '.join(cleaned_tokens)


def clean_folder_nlp(input_folder: str, output_folder: str) -> None:
    """
    Apply NLP cleaning to all .txt files in input_folder and save
    the cleaned output into output_folder.
    """
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith('.txt'):
            continue

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        with open(input_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        cleaned = clean_text_nlp(raw_text)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned)

        print(f"✔ NLP-cleaned '{filename}' → '{output_folder}'")


if __name__ == '__main__':
    SOURCE_FOLDER = "Extracted_Text"
    CLEANED_FOLDER = "cleaned_text"
    clean_folder_nlp(SOURCE_FOLDER, CLEANED_FOLDER)