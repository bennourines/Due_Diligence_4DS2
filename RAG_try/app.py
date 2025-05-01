from flask import Flask, render_template, request, jsonify
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from pypdf import PdfReader
import os
from huggingface_hub import login

# Login to Hugging Face
login(token="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")

app = Flask(__name__)

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize the language model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=512)
local_llm = HuggingFacePipeline(pipeline=pipe)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

def process_pdf(file_path):
    # Read PDF
    pdf_reader = PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Create and return the vector store
    vectorstore = Chroma.from_texts(chunks, embeddings)
    return vectorstore

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        try:
            global vectorstore
            vectorstore = process_pdf(file_path)
            return jsonify({'message': 'File successfully uploaded and processed'}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'error': 'No question provided'}), 400
        
        question = data['question']
        
        if hasattr(app, 'vectorstore'):
            # If a document is uploaded, use RAG
            qa_chain = RetrievalQA.from_chain_type(
                llm=local_llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever()
            )
            answer = qa_chain.run(question)
        else:
            # If no document is uploaded, use the language model directly
            prompt = f"Question: {question}\nAnswer:"
            answer = local_llm(prompt)
        
        return jsonify({'answer': answer}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 