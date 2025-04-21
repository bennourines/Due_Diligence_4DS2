import os
import glob
import re
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
import utility

# Load environment variables
load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

def preprocess_whitepaper(text):
    """Clean and preprocess whitepaper text"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page numbers (common in PDFs)
    text = re.sub(r'\b\d+\s*\|\s*[pP]age\b', '', text)
    
    # Replace special characters
    text = text.replace('•', '* ')
    
    # Clean up references and citations
    text = re.sub(r'\[\d+\]', '', text)
    
    return text.strip()

def load_whitepapers(directory="data"):
    """Load all text files from a directory"""
    all_text = ""
    
    # Get all txt files in the directory
    txt_files = glob.glob(os.path.join(directory, "*.txt"))
    
    if not txt_files:
        print(f"No text files found in {directory}")
        return all_text
    
    for file_path in txt_files:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                print(f"Loading: {file_path}")
                file_text = file.read()
                cleaned_text = preprocess_whitepaper(file_text)
                all_text += cleaned_text + "\n\n"
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return all_text

def create_domain_specific_chunks(text):
    """Split text into chunks with domain-specific handling"""
    # Use RecursiveCharacterTextSplitter for better semantic chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Larger chunks for better context
        chunk_overlap=200,  # Higher overlap to maintain context
        separators=["\n\n", "\n", ". ", " ", ""],  # Custom separators
        length_function=len
    )
    
    chunks = text_splitter.split_text(text)
    
    # Filter chunks to ensure they're relevant to the domain
    relevant_chunks = []
    for i, chunk in enumerate(chunks):
        if utility.is_relevant_chunk(chunk):
            # Add section numbering for traceability
            relevant_chunks.append(f"[Section {i+1}] {chunk}")
    
    print(f"Created {len(relevant_chunks)} relevant chunks from {len(chunks)} total chunks")
    return relevant_chunks

def build_vector_database(chunks):
    """Build and save a vector database from text chunks"""
    # Use a more powerful model for financial documents
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",  # Better model for financial text
        model_kwargs={'device': 'cpu'}  # Use GPU if available by changing to 'cuda'
    )
    
    # Convert chunks to documents with metadata
    documents = [
        Document(
            page_content=chunk,
            metadata={"chunk_id": i, "source": "whitepaper"}
        ) for i, chunk in enumerate(chunks)
    ]
    
    # Create vector store
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Save the vector store
    vectorstore.save_local("faiss_finance_index")
    print("Vector database saved to 'faiss_finance_index'")
    
    return vectorstore

def setup_qa_system(vectorstore):
    """Set up the question-answering system"""
    # Initialize the chat model
    chat_model = ChatOpenAI(
        model="mistralai/mistral-7b-instruct-v0.1",
        openai_api_key=openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.2  # Slightly higher temperature for more comprehensive answers
    )
    
    # Create a retriever with higher k value for more context
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Retrieve top 5 chunks for better context
    )
    
    # Define a specialized prompt template for financial context
    template = """You are a specialized financial and cryptocurrency compliance expert. 
Use the following information extracted from financial and cryptocurrency whitepapers to answer the question.
Be precise and technical in your response, citing specific terms and concepts where relevant.
If the information is insufficient to give a complete answer, explain what is known based on the context and indicate what additional information would be needed.

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

def query_system(qa_chain, query):
    """Send a query to the QA system with enhanced processing"""
    # Enhance query with domain-specific terms
    enhanced_query = utility.enhance_query(query)
    print(f"Enhanced query: {enhanced_query}")
    
    # Get response
    response = qa_chain({"query": enhanced_query})
    
    # Format the output
    answer = response["result"]
    
    # Format and add sources
    sources = []
    for doc in response["source_documents"]:
        source_text = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
        sources.append(f"- {source_text}")
    
    return {
        "answer": answer,
        "sources": sources
    }

def main():
    """Main function to run the enhanced system"""
    # Load whitepaper texts
    raw_text = load_whitepapers()
    if not raw_text:
        print("No text loaded. Please check your data directory.")
        return
    
    # Create chunks
    chunks = create_domain_specific_chunks(raw_text)
    
    # Build vector database
    vectorstore = build_vector_database(chunks)
    
    # Set up QA system
    qa_chain = setup_qa_system(vectorstore)
    
    # Interactive query loop
    print("\n=== Financial Whitepaper Q&A System ===")
    print("Enter your questions about cryptocurrency compliance, finance, or risk analysis.")
    print("Type 'exit' to quit.")
    
    while True:
        query = input("\nYour question: ")
        if query.lower() in ['exit', 'quit']:
            break
            
        try:
            result = query_system(qa_chain, query)
            print("\nAnswer:")
            print(result["answer"])
            print("\nSources:")
            for source in result["sources"]:
                print(source)
        except Exception as e:
            print(f"Error processing question: {e}")

if __name__ == "__main__":
    main()