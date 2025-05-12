from typing import List, Optional
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import logging
import re

logger = logging.getLogger(__name__)

# Define the cryptocurrency expert prompt template
CRYPTO_EXPERT_TEMPLATE = """You are a highly knowledgeable expert in cryptocurrency, blockchain technology, \
decentralized finance (DeFi), with extensive experience \
in both technical and practical aspects of the field.

Instructions: Use ONLY the following context to answer the question. Begin your answer DIRECTLY with relevant information. \
DO NOT use phrases like "Based on the provided context" or "I can provide". \
If you cannot find the answer in the context, state 'Information not found in the provided documents'. \
Be concise but thorough.

{context}

Question: {question}

Comprehensive answer:"""

CRYPTO_EXPERT_PROMPT = PromptTemplate(
    template=CRYPTO_EXPERT_TEMPLATE,
    input_variables=["context", "question"]
)

# Define an enhanced reasoning template for phi4-mini
REASONING_TEMPLATE = """You are a highly knowledgeable expert in cryptocurrency, blockchain technology, \
decentralized finance (DeFi), with extensive experience \
in both technical and practical aspects of the field.

You are analyzing a question and an initial answer to enhance the reasoning and quality of the response.

CONTEXT INFORMATION:
{context}

QUESTION: {question}

INITIAL ANSWER:
{initial_answer}

Instructions:
1. Analyze the question and the initial answer
2. Apply step-by-step reasoning to enhance the answer
3. Make logical connections between different pieces of information
4. Ensure the answer directly addresses the question without starting with phrases like "Based on the provided context"
5. If information is not available in the context, maintain the statement "Information not found in the provided documents"
6. Provide a comprehensive yet concise answer

ENHANCED ANSWER:"""

REASONING_PROMPT = PromptTemplate(
    template=REASONING_TEMPLATE,
    input_variables=["context", "question", "initial_answer"]
)

# List of common greetings for detection
GREETING_PATTERNS = [
    r"^hi$", r"^hello$", r"^hey$", r"^greetings$", r"^good morning$", 
    r"^good afternoon$", r"^good evening$", r"^howdy$", r"^hola$",
    r"^hi there$", r"^hello there$", r"^hey there$", r"^yo$"
]

# List of identity questions
IDENTITY_PATTERNS = [
    r"who are you\??$", r"what are you\??$", r"what is your name\??$",
    r"who (am i|is this)\s?(talking|speaking|chatting) (to|with)\??$",
    r"introduce yourself$", r"tell me about yourself$",
    r"what do you do\??$", r"what can you do\??$", r"what is your purpose\??$",
    r"what are you capable of\??$", r"how can you help( me)?\??$"
]

class CryptoQASystem:
    """
    Enhanced QA system for cryptocurrency queries using a two-stage approach:
    1. Retrieval and initial answer generation with base LLM and custom crypto prompt
    2. Answer enhancement using phi4-mini reasoning capabilities
    """
    
    def __init__(self, vectordb, base_model="llama3", reasoning_model="phi4-mini"):
        """Initialize the enhanced QA system with two models."""
        self.vectordb = vectordb
        self.base_model_name = base_model
        self.reasoning_model_name = reasoning_model
        
        # Initialize both models
        try:
            self.base_llm = Ollama(model=base_model, temperature=0.1)
            self.reasoning_llm = Ollama(model=reasoning_model, temperature=0.2)
            logger.info(f"Initialized base LLM ({base_model}) and reasoning LLM ({reasoning_model})")
        except Exception as e:
            logger.error(f"Error initializing LLMs: {str(e)}")
            raise RuntimeError(f"Failed to initialize language models: {str(e)}")
    
    def is_greeting(self, text):
        """Check if the input text is a simple greeting."""
        text = text.lower().strip()
        for pattern in GREETING_PATTERNS:
            if re.match(pattern, text):
                return True
        return False

    def is_identity_question(self, text):
        """Check if the input text is asking about the AI's identity."""
        text = text.lower().strip()
        for pattern in IDENTITY_PATTERNS:
            if re.match(pattern, text):
                return True
        return False

    def get_greeting_response(self):
        """Return a friendly greeting response."""
        return {
            "answer": "Hello! I'm your cryptocurrency and blockchain assistant. How can I help you today?",
            "sources": []
        }
    
    def get_identity_response(self):
        """Return a response about the AI's identity and capabilities."""
        return {
            "answer": "I'm a specialized cryptocurrency and blockchain assistant designed to provide expert information on digital assets, blockchain technology, decentralized finance (DeFi), and related topics. I can answer questions about cryptocurrencies, explain blockchain concepts, analyze crypto trends, and provide information from my knowledge base. Feel free to ask me anything related to the crypto space!",
            "sources": []
        }
    
    def create_retriever(self, doc_id=None, k=10):
        """Create a retriever with optional document filtering."""
        from qdrant_client.http import models
        
        search_filter = None
        if doc_id and doc_id != "all" and doc_id != "":
            search_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.doc_id",
                        match=models.MatchValue(value=doc_id)
                    )
                ]
            )
        
        retriever = self.vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": k,
                "filter": search_filter
            }
        )
        
        return retriever
    
    def get_standard_answer(self, question, doc_id=None):
        """Get a standard answer using the base LLM with document retrieval and crypto expert prompt."""
        retriever = self.create_retriever(doc_id)
        
        # Create a custom QA chain with our crypto expert prompt
        # Fetch documents first
        retrieved_docs = retriever.get_relevant_documents(question)
        
        if not retrieved_docs:
            return {
                "answer": "Information not found in the provided documents.",
                "source_documents": []
            }
        
        # Prepare context by concatenating document content
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Format the prompt with context and question
        formatted_prompt = CRYPTO_EXPERT_PROMPT.format(
            context=context,
            question=question
        )
        
        # Get response from base model
        answer = self.base_llm.invoke(formatted_prompt)
        
        return {
            "answer": answer,
            "source_documents": retrieved_docs,
            "context": context
        }
    
    def enhance_answer(self, question, initial_answer, context):
        """
        Enhance an answer using phi4-mini's reasoning capabilities.
        
        Args:
            question: The original question
            initial_answer: The answer from the base model
            context: Retrieved document context
        
        Returns:
            Enhanced answer with better reasoning
        """
        # Create a reasoning prompt that leverages phi4's strengths
        formatted_prompt = REASONING_PROMPT.format(
            context=context,
            question=question,
            initial_answer=initial_answer
        )
        
        # Generate the enhanced answer
        try:
            enhanced_answer = self.reasoning_llm.invoke(formatted_prompt)
            return enhanced_answer.strip()
        except Exception as e:
            logger.error(f"Error during answer enhancement: {str(e)}")
            return initial_answer  # Fallback to the initial answer
    
    def answer_question(self, question, doc_id=None):
        """
        Answer a question using the enhanced two-stage process with crypto expertise.
        
        Args:
            question: The question to answer
            doc_id: Optional document ID to restrict the search
            
        Returns:
            dict with enhanced answer and sources
        """
        logger.info(f"Processing query: '{question}'")
        
        # Check if the question is a simple greeting and handle accordingly
        if self.is_greeting(question):
            logger.info("Detected greeting, responding with welcome message")
            return self.get_greeting_response()
        
        # Check if the question is asking about identity
        if self.is_identity_question(question):
            logger.info("Detected identity question, responding with identity info")
            return self.get_identity_response()
        
        try:
            # Stage 1: Get initial answer and sources using base model with crypto prompt
            standard_result = self.get_standard_answer(question, doc_id)
            initial_answer = standard_result["answer"]
            source_documents = standard_result["source_documents"]
            context = standard_result["context"]
            
            # If no relevant documents were found, return early
            if not source_documents:
                return {
                    "answer": initial_answer,
                    "sources": []
                }
            
            # Stage 2: Enhance the answer using phi4-mini reasoning
            enhanced_answer = self.enhance_answer(
                question, 
                initial_answer, 
                context
            )
            
            # Extract sources for transparency
            sources = []
            for doc in source_documents:
                source_info = {
                    "source": doc.metadata.get("source", "Unknown"),
                    "doc_id": doc.metadata.get("doc_id", "Unknown"),
                    "page": doc.metadata.get("page", 0),
                    "file_type": doc.metadata.get("file_type", "Unknown")
                }
                sources.append(source_info)
            
            logger.info(f"Enhanced crypto answer generated with {len(sources)} source documents.")
            
            return {
                "answer": enhanced_answer,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error in crypto enhanced QA process: {str(e)}")
            return {"answer": f"Error: {str(e)}", "sources": []}

