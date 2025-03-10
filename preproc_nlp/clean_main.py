# process_files.py
from gettext import find
import script
import nltk
import os

# Function to ensure an NLTK resource is available
def fetch_nltk_resource(resource_name):
    try:
        find(f'taggers/{resource_name}')
    except LookupError:
        nltk.download(resource_name)

# Download required NLTK data
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('stopwords')

# Define directories for source and destination files
source_directory = "C:\Users\ferie\Documents\ferielWork\4ds2PIDS\Due_Diligence_4DS2\preproc_nlp"
destination_directory = "C:\Users\ferie\Documents\ferielWork\4ds2PIDS\Due_Diligence_4DS2\preproc_nlp"

# Ensure the destination directory exists
os.makedirs(destination_directory, exist_ok=True)

# Process each text file in the source directory
for filename in os.listdir(source_directory):
    if filename.endswith(".txt"):
        source_file_path = os.path.join(source_directory, filename)
        destination_file_path = os.path.join(destination_directory, f"{os.path.splitext(filename)[0]}_processed.txt")
        script.process_text(source_file_path, destination_file_path)