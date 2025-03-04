# clean.py
from gettext import find
import script
import nltk
import os
# Function to check if an NLTK package is already downloaded
def download_nltk_package(package_name):
    try:
        find(f'taggers/{package_name}')
    except LookupError:
        nltk.download(package_name)

# Download required NLTK data
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('stopwords')



#input_path = "us-crypto-regulatory-whitepaper.txt"  # or any file you want to clean
#output_path = "us-crypto-regulatory-whitepaper_cleaned.txt"
input_dir = "C:\PI\Data Cleaning\converted"
output_dir = "C:\PI\Data Cleaning\conveted_cleaned"


#script.clean_text(input_path, output_path)
# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each text file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_cleaned.txt")
        script.clean_text(input_path, output_path)