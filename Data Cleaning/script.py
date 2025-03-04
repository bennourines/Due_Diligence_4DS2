# -*- coding: utf-8 -*-
"""
Enhanced Text Preprocessing Script for Web-Scraped Data

This script cleans a text file for NLP tasks by:
1. Converting to lowercase.
2. Removing URLs, HTML tags, email addresses, and Twitter handles.
3. Removing page markers and typical navigation headers/footers.
4. Expanding contractions.
5. Removing punctuation.
6. (Optionally) correcting spelling using pyspellchecker.
7. Removing stopwords.
8. Lemmatizing with POS tagging.
9. Normalizing whitespace.
"""

import re
import unicodedata

import sys
import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

# Download required nltk resources
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

# Optional: Spell checker - install via pip install pyspellchecker if desired
try:
    from spellchecker import SpellChecker
    SPELLCHECK_AVAILABLE = True
except ImportError:
    SPELLCHECK_AVAILABLE = False

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Contraction mapping dictionary
appos = {
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

# --- Helper Functions ---

def replace_words(text):
    """
    Expand contractions in the text using the appos dictionary.
    """
    words = text.split()
    replaced = [appos[word.lower()] if word.lower() in appos else word for word in words]
    return ' '.join(replaced)

def remove_urls_and_html(text):
    """
    Remove URLs and HTML tags from text.
    """
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    return text

def remove_emails(text):
    """
    Remove email addresses from the text.
    """
    return re.sub(r'\S+@\S+', '', text)

def remove_social_handles(text):
    """
    Remove social media handles (e.g., Twitter usernames) from the text.
    """
    return re.sub(r'@\w+', '', text)

def remove_page_markers(text):
    """
    Remove page markers and common navigation/header/footer text.
    """
    # Remove page markers like '--- Page 1 ---' or 'page 1'
    text = re.sub(r'---\s*page\s*\d+\s*---', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bpage\s*\d+\b', '', text, flags=re.IGNORECASE)
    # Optionally remove common navigation keywords if isolated on a line
    nav_keywords = r'\b(home|about|services|contact|legal|privacy policy|cookie policy|terms and conditions|menu)\b'
    text = re.sub(nav_keywords, '', text, flags=re.IGNORECASE)
    return text

def remove_punctuation(text):
    """
    Remove punctuation using a translation table.
    """
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stopwords(text):
    """
    Remove stopwords using nltk's English stopword list.
    """
    stoplist = set(stopwords.words('english'))
    # Customize stoplist if needed
    stoplist.update(('cm', 'kg', 'mr', 'wa', 'nv', 'ore', 'da', 'pm', 'am', 'cx'))
    if 'not' in stoplist:
        stoplist.remove('not')
    words = text.split()
    return " ".join([word for word in words if word not in stoplist])

def lem(text):
    """
    Lemmatize text using POS tagging to improve results.
    """
    lemmatizer = WordNetLemmatizer()
    pos_dict = {'N': wn.NOUN, 'V': wn.VERB, 'J': wn.ADJ, 'R': wn.ADV}
    tagged_words = nltk.pos_tag(text.split())
    lemmatized_words = [lemmatizer.lemmatize(word, pos_dict.get(tag[0], wn.NOUN))
                        for word, tag in tagged_words]
    return " ".join(lemmatized_words)

def spelling_checks(text):
    """
    Correct spelling errors using pyspellchecker.
    (This step may be time-consuming for very large texts.)
    """
    if not SPELLCHECK_AVAILABLE:
        print("pyspellchecker not installed. Skipping spell correction.")
        return text

    spell = SpellChecker()
    words = text.split()
    unknown_words = spell.unknown(words)
    corrected = [spell.correction(word) if word in unknown_words else word for word in words]
    return " ".join(corrected)

def plot_wordcloud(text, mask=None, max_words=500, max_font_size=40, 
                   figure_size=(12, 6), title=None, title_size=15):
    """
    Generate and display a wordcloud from the processed text.
    """
    wordcloud = WordCloud(background_color='white', max_words=max_words,
                          random_state=42, width=350, height=150, 
                          mask=mask, stopwords=set(stopwords.words('english')),
                          collocations=False).generate(str(text))
    plt.figure(figsize=figure_size)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontdict={'size': title_size, 'color': 'black', 'verticalalignment': 'bottom'})
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# --- Main Cleaning Function ---

def clean_text(input_file, output_file):
    """
    Read an input file, process the text through various cleaning steps,
    and write the cleaned text to the output file.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Step 1: Lowercase for consistency
    text = text.lower()
    
    # Step 2: Remove URLs, HTML tags, email addresses, and social media handles
    text = remove_urls_and_html(text)
    text = remove_emails(text)
    text = remove_social_handles(text)
    
    # Step 3: Remove page markers and navigation/header/footer text
    text = remove_page_markers(text)
    
    # Step 4: Expand contractions
    text = replace_words(text)
    
    # Step 5: Remove punctuation
    text = remove_punctuation(text)
    
    # Optional Step 6: Spell correction (uncomment if needed)
    # text = spelling_checks(text)
    
    # Step 7: Remove stopwords
    text = remove_stopwords(text)
    
    # Step 8: Lemmatize the text
    text = lem(text)
    text = unicodedata.normalize('NFKD', text)
    # Step 9: Normalize whitespace
    text = " ".join(text.split())
    # Step 10: Remove ALL whitespace (final output has no spaces)
    #text = re.sub(r'\s+', '', text)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Cleaned text saved to {output_file}")

# --- Command Line Interface ---

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python enhanced_scrpt.py input_file output_file")
    else:
        clean_text(sys.argv[1], sys.argv[2])
