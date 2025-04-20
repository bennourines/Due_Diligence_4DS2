from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
import requests
import os
from typing import List, Dict, Optional
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(
        self,
        embedding_model_name: str = "text-embedding-3-small",
        llm_model: str = "meta-llama/llama-4-maverick:free",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        index_path: str = "./vector_store",
        temperature: float = 0.7,
        max_tokens: int = 512
    ):
        """
        Initialize the RAG system with LangChain components.
        
        Args:
            embedding_model: HuggingFace model for embeddings
            llm_model: HuggingFace model for text generation
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            index_path: Path to store the vector database
        """
        # Initialize embeddings model
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model_name,
            api_base="https://api.openai.com/v1"  # Using OpenAI's API directly
        )
        # Initialize language model with OpenRouter base URL
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            headers={
                "HTTP-Referer": "https://github.com/langchain-ai/langchain",
                "X-Title": "LangChain RAG System"
            }
        )
        
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        
        self.index_path = index_path
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.vectorstore = None
        
    def process_documents(self, input_dir: str, file_pattern: str = "*.txt") -> None:
        """
        Process documents from a directory and create/update the vector store.
        
        Args:
            input_dir: Directory containing the documents
            file_pattern: File pattern to match (default: "*.txt")
        """
        try:
            # Use DirectoryLoader to load all matching documents
            loader = DirectoryLoader(
                input_dir,
                glob=file_pattern,
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'}
            )
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents")
            
            # Split documents into chunks
            texts = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(texts)} chunks")
            
            # Create or update vector store
            if os.path.exists(self.index_path):
                # Load existing index and add new documents
                self.vectorstore = FAISS.load_local(
                    self.index_path,
                    self.embeddings
                )
                self.vectorstore.add_documents(texts)
                logger.info("Updated existing vector store")
            else:
                # Create new vector store
                self.vectorstore = FAISS.from_documents(
                    texts,
                    self.embeddings
                )
                logger.info("Created new vector store")
            
            # Save the updated vector store
            self.vectorstore.save_local(self.index_path)
            logger.info(f"Saved vector store to {self.index_path}")
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise

    def load_vectorstore(self) -> None:
        """Load the vector store from disk."""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Vector store not found at {self.index_path}")
        
        self.vectorstore = FAISS.load_local(
            self.index_path,
            self.embeddings
        )
        logger.info("Loaded vector store from disk")

    def query(
        self,
        question: str,
        top_k: int = 4,
        chat_history: Optional[List] = None
    ) -> Dict:
        """
        Query the RAG system with a question.
        
        Args:
            question: User's question
            top_k: Number of relevant chunks to retrieve
            chat_history: Optional chat history for context
            
        Returns:
            Dict containing answer and source documents
        """
        try:
            if not self.vectorstore:
                self.load_vectorstore()
            
            # Create retrieval chain
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": top_k}),
                memory=self.memory,
                return_source_documents=True,
                verbose=True
            )
            
            # Get response
            result = qa_chain({"question": question})
            
            return {
                "answer": result["answer"],
                "source_documents": [
                    {
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", "Unknown"),
                        "start_index": doc.metadata.get("start_index", 0)
                    }
                    for doc in result["source_documents"]
                ]
            }
            
        except Exception as e:
            logger.error(f"Error querying RAG system: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    rag = RAGSystem()
    
    # Process documents from the nlp_cleaned_data directory
    rag.process_documents("./nlp_cleaned_data")
    
    # Example query
    result = rag.query(
        "What are the main points about cryptocurrency exchanges?",
        top_k=4
    )
    print("\nAnswer:", result["answer"])
    print("\nSources:")
    for doc in result["source_documents"]:
        print(f"\n- {doc['source']}:")
        print(f"  {doc['content'][:200]}...")
