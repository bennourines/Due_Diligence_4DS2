"""
Advanced Text Cleaning Utility for Web-Extracted Content

This utility prepares text files for NLP applications by:
1. Converting text to lowercase.
2. Stripping out URLs, HTML tags, email addresses, and social media handles.
3. Eliminating page indicators and common navigation elements.
4. Expanding abbreviated forms (contractions).
5. Removing punctuation marks.
6. (Optionally) correcting typos using pyspellchecker.
7. Filtering out stopwords.
8. Applying lemmatization with POS tagging.
9. Standardizing whitespace.
"""

import re
import unicodedata
import sys
import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

# Fetch required NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

# Optional spell checker - install via `pip install pyspellchecker` if desired
try:
    from spellchecker import SpellChecker
    IS_SPELLCHECK_ENABLED = True
except ImportError:
    IS_SPELLCHECK_ENABLED = False

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Dictionary for expanding contractions
contraction_map = {
    "ain't": "am not", "aren't": "are not", "can't": "cannot", 
    "can't've": "cannot have", "'cause": "because", 
    "could've": "could have", "couldn't": "could not", 
    "couldn't've": "could not have", "didn't": "did not", 
    "doesn't": "does not", "don't": "do not", "hadn't": "had not", 
    "hadn't've": "had not have", "hasn't": "has not", 
    "haven't": "have not", "he'd": "he would", "he'd've": "he would have", 
    "he'll": "he will", "he'll've": "he will have", 
    "he's": "he is", "how'd": "how did", 
    "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
    "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 
    "I'll've": "I will have", "I'm": "I am", "I've": "I have", 
    "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 
    "it'll": "it will", "it'll've": "it will have", "it's": "it is", 
    "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
    "might've": "might have", "mightn't": "might not", 
    "mightn't've": "might not have", "must've": "must have", 
    "mustn't": "must not", "mustn't've": "must not have", 
    "needn't": "need not", "needn't've": "need not have",
    "o'clock": "of the clock", "oughtn't": "ought not", 
    "oughtn't've": "ought not have", "shan't": "shall not", 
    "sha'n't": "shall not", "shan't've": "shall not have", 
    "she'd": "she would", "she'd've": "she would have", 
    "she'll": "she will", "she'll've": "she will have",
    "she's": "she is", "should've": "should have", 
    "shouldn't": "should not", "shouldn't've": "should not have", 
    "so've": "so have", "so's": "so is", 
    "that'd": "that had", "that'd've": "that would have", 
    "that's": "that is", "there'd": "there would", 
    "there'd've": "there would have", "there's": "there is", 
    "they'd": "they would", "they'd've": "they would have", 
    "they'll": "they will", "they'll've": "they will have", 
    "they're": "they are", "they've": "they have", 
    "to've": "to have", "wasn't": "was not", "we'd": "we would", 
    "we'd've": "we would have", "we'll": "we will", 
    "we'll've": "we will have", "we're": "we are", 
    "we've": "we have", "weren't": "were not", 
    "what'll": "what will", "what'll've": "what will have", 
    "what're": "what are", "what's": "what is", 
    "what've": "what have", "when's": "when is", 
    "when've": "when have", "where'd": "where did", 
    "where's": "where is", "where've": "where have", 
    "who'll": "who will", "who'll've": "who will have", 
    "who's": "who is", "who've": "who have", 
    "why's": "why is", "why've": "why have", "will've": "will have", 
    "won't": "will not", "won't've": "will not have",
    "would've": "would have", "wouldn't": "would not", 
    "wouldn't've": "would not have", "y'all": "you all", 
    "y'all'd": "you all would", "y'all'd've": "you all would have", 
    "y'all're": "you all are", "y'all've": "you all have", 
    "you'd": "you would", "you'd've": "you would have",
    "you'll": "you will", "you'll've": "you will have", 
    "you're": "you are", "you've": "you have"
}

# --- Utility Functions ---

def expand_contractions(content):
    """
    Expands contractions in the content using the contraction_map dictionary.
    """
    tokens = content.split()
    expanded = [contraction_map[token.lower()] if token.lower() in contraction_map else token for token in tokens]
    return ' '.join(expanded)

def strip_urls_and_tags(data):
    """
    Strips URLs and HTML tags from the data.
    """
    data = re.sub(r'https?://\S+', '', data)
    data = re.sub(r'<.*?>', '', data)
    return data

def erase_email_addresses(text_input):
    """
    Erases email addresses from the text input.
    """
    return re.sub(r'\S+@\S+', '', text_input)

def eliminate_social_tags(raw_text):
    """
    Eliminates social media tags (e.g., Twitter usernames) from the raw text.
    """
    return re.sub(r'@\w+', '', raw_text)

def clear_page_indicators(source_text):
    """
    Clears page indicators and common navigation/header/footer text.
    """
    source_text = re.sub(r'---\s*page\s*\d+\s*---', '', source_text, flags=re.IGNORECASE)
    source_text = re.sub(r'\bpage\s*\d+\b', '', source_text, flags=re.IGNORECASE)
    nav_terms = r'\b(home|about|services|contact|legal|privacy policy|cookie policy|terms and conditions|menu)\b'
    source_text = re.sub(nav_terms, '', source_text, flags=re.IGNORECASE)
    return source_text

def purge_punctuation(text_data):
    """
    Purges punctuation using a translation table.
    """
    return text_data.translate(str.maketrans('', '', string.punctuation))

def filter_common_words(input_text):
    """
    Filters out common words using NLTK's English stopword list.
    """
    common_words = set(stopwords.words('english'))
    common_words.update(('cm', 'kg', 'mr', 'wa', 'nv', 'ore', 'da', 'pm', 'am', 'cx'))
    if 'not' in common_words:
        common_words.remove('not')
    word_list = input_text.split()
    return " ".join([word for word in word_list if word not in common_words])

def apply_lemmatization(text_content):
    """
    Applies lemmatization with POS tagging to enhance accuracy.
    """
    lemmatizer_instance = WordNetLemmatizer()
    tag_mapping = {'N': wn.NOUN, 'V': wn.VERB, 'J': wn.ADJ, 'R': wn.ADV}
    tagged_tokens = nltk.pos_tag(text_content.split())
    processed_words = [lemmatizer_instance.lemmatize(word, tag_mapping.get(tag[0], wn.NOUN))
                       for word, tag in tagged_tokens]
    return " ".join(processed_words)

def correct_typos(text_to_fix):
    """
    Corrects typos using pyspellchecker.
    (This step may be slow for very large texts.)
    """
    if not IS_SPELLCHECK_ENABLED:
        print("pyspellchecker not installed. Skipping typo correction.")
        return text_to_fix

    spell_checker = SpellChecker()
    word_list = text_to_fix.split()
    misspelled = spell_checker.unknown(word_list)
    fixed_words = [spell_checker.correction(word) if word in misspelled else word for word in word_list]
    return " ".join(fixed_words)

def render_wordcloud_visual(data, mask_shape=None, max_terms=500, max_text_size=40, 
                           plot_dimensions=(12, 6), chart_title=None, title_font_size=15):
    """
    Renders and displays a wordcloud from the processed data.
    """
    wordcloud_obj = WordCloud(background_color='white', max_words=max_terms,
                              random_state=42, width=350, height=150, 
                              mask=mask_shape, stopwords=set(stopwords.words('english')),
                              collocations=False).generate(str(data))
    plt.figure(figsize=plot_dimensions)
    plt.imshow(wordcloud_obj, interpolation='bilinear')
    plt.title(chart_title, fontdict={'size': title_font_size, 'color': 'black', 'verticalalignment': 'bottom'})
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# --- Core Processing Function ---

def process_text(source_file, destination_file):
    """
    Reads a source file, applies various cleaning steps to the text,
    and saves the processed text to the destination file.
    """
    with open(source_file, "r", encoding="utf-8") as f:
        raw_content = f.read()

    # Step 1: Convert to lowercase for uniformity
    raw_content = raw_content.lower()
    
    # Step 2: Strip URLs, HTML tags, email addresses, and social media handles
    raw_content = strip_urls_and_tags(raw_content)
    raw_content = erase_email_addresses(raw_content)
    raw_content = eliminate_social_tags(raw_content)
    
    # Step 3: Clear page indicators and navigation/header/footer text
    raw_content = clear_page_indicators(raw_content)
    
    # Step 4: Expand contractions
    raw_content = expand_contractions(raw_content)
    
    # Step 5: Purge punctuation
    raw_content = purge_punctuation(raw_content)
    
    # Optional Step 6: Correct typos (uncomment if needed)
    # raw_content = correct_typos(raw_content)
    
    # Step 7: Filter out common words
    raw_content = filter_common_words(raw_content)
    
    # Step 8: Apply lemmatization
    raw_content = apply_lemmatization(raw_content)
    raw_content = unicodedata.normalize('NFKD', raw_content)
    
    # Step 9: Standardize whitespace
    raw_content = " ".join(raw_content.split())
    
    # Step 10: Optionally remove all whitespace (uncomment if needed)
    # raw_content = re.sub(r'\s+', '', raw_content)
    
    with open(destination_file, "w", encoding="utf-8") as f:
        f.write(raw_content)
    print(f"Processed text written to {destination_file}")

# --- Command Line Interface ---

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python text_cleaner.py source_file destination_file")
    else:
        process_text(sys.argv[1], sys.argv[2])