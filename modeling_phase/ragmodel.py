"""
rag.py

Sets up the Retrieval Augmented Generation (RAG) system. It imports the
text_chunker and vector modules, prepares the data, and launches an interactive
Q&A session using an LLM.
"""

import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import text_chunker
import text_vector as vector
import utility  # For functions such as enhance_query

# Load environment variables (e.g., OPENROUTER_API_KEY)
load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

def setup_qa_system(vectorstore):
    """Set up the QA system with a specialized prompt and retriever."""
    chat_model = ChatOpenAI(
        model="meta-llama/llama-2-13b-chat",
        openai_api_key=openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.2
    )
    
    # Configure a retriever for the vectorstore: retrieve top 5 similar chunks
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # Define a custom prompt template that guides the LLM's responses
    template = """You are a specialized financial and cryptocurrency compliance expert.
Use the following information extracted from whitepapers to answer the question.
Be precise and technical in your response, citing specific terms and concepts when relevant.
If the context is insufficient, explain what is known and what further details are needed.

Context:
{context}

Question: {question}

Answer:"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

def query_system(qa_chain, query):
    """Enhance the query and send it to the QA system."""
    enhanced_query = utility.enhance_query(query)
    print(f"Enhanced query: {enhanced_query}")
    
    response = qa_chain({"query": enhanced_query})
    answer = response["result"]
    
    # Extract and format excerpts from the source documents
    sources = []
    for doc in response["source_documents"]:
        source_text = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
        sources.append(f"- {source_text}")
    
    return {"answer": answer, "sources": sources}

def main():
    """Main function to run the RAG system interactively."""
    # Load the whitepaper texts using text_chunker module
    raw_text = text_chunker.load_whitepapers(directory="data")
    if not raw_text:
        print("No text loaded. Please check your data directory.")
        return
    
    # Create domain-specific chunks from the loaded text
    chunks = text_chunker.create_domain_specific_chunks(raw_text)
    
    # Build the vector database using the vector module
    vectorstore = vector.build_vector_database(chunks)
    
    # Set up the QA system with the prepared vectorstore
    qa_chain = setup_qa_system(vectorstore)
    
    # Start an interactive query loop
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
