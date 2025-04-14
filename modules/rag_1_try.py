import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter

# Load environment variables
load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

# Load text file
with open("data/bitaml.com_cleaned.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()

# Split text into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(raw_text)

# Use HuggingFace embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build vector database with FAISS
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_texts(chunks, embeddings)
vectorstore.save_local("faiss_index")

# Import the necessary chat model
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

# Initialize the chat model with OpenRouter
chat_model = ChatOpenAI(
    model="mistralai/mistral-7b-instruct-v0.1",
    openai_api_key=openrouter_api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0
)

# Define a custom prompt template
template = """Utilise les informations suivantes extraites d'un document pour répondre à la question.
Si tu n'es pas certain de la réponse, indique simplement que tu ne sais pas.

Contexte: {context}

Question: {question}
Réponse:"""

# Test the prompt with fictional values
test_prompt = template.format(context="Contenu de test", question="Question de test")
print("Test du prompt:", test_prompt)

# Ask a question
query = "What is Enhanced Due Diligence (EDD) in the context of cryptocurrency compliance?"
print("Query:", query)

# Retrieve relevant documents
docs = vectorstore.similarity_search(query)
context = " ".join([doc.page_content for doc in docs])

# Format the prompt with actual context and query
prompt_text = template.format(context=context, question=query)

# Use the chat model directly
response = chat_model.invoke([HumanMessage(content=prompt_text)])
print("Response:", response.content)