# rag_pipeline.py
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any, Optional
import os
import logging

# Set up logging
logger = logging.getLogger(__name__)

class RAGPipeline:
    """RAG pipeline using LangChain and OpenRouter"""
    
    def __init__(self):
        logger.info("Initializing RAG Pipeline")
        try:
            # Initialize LLM using OpenRouter
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "your_openrouter_api_key")
            logger.debug("Setting up ChatOpenAI with OpenRouter")
            
            self.llm = ChatOpenAI(
                model="meta-llama/llama-4-maverick:free",  # Can be adjusted to other models
                temperature=0.2,
                max_tokens=1000,
                openai_api_base="https://openrouter.ai/api/v1",
                openai_api_key=openrouter_api_key,
                
            )
            logger.info("LLM initialized successfully")
            
            # Define QA prompt template
            logger.debug("Setting up QA prompt template")
            qa_template = """You are a helpful assistant specialized in cryptocurrency due diligence, analyzing blockchain projects, and identifying potential risks.

CONTEXT:
{context}

USER QUERY:
{question}

Provide a detailed and well-reasoned answer based on the given context. If the context doesn't contain enough information to answer confidently, mention that clearly. Include references to specific documents or data points from the context to support your analysis.
"""
            
            qa_prompt = PromptTemplate(
                template=qa_template,
                input_variables=["context", "question"]
            )
            
            # Create QA chain
            logger.debug("Creating QA chain")
            self.qa_chain = load_qa_chain(
                llm=self.llm,
                chain_type="stuff",# # Can be adjusted to other chain types, but chose "stuff" for simplicity
                prompt=qa_prompt
            )
            logger.info("QA chain created successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG Pipeline: {str(e)}", exc_info=True)
            raise
    
    def generate_response(self, query: str, contexts: List) -> str:
        """Generate response using RAG pipeline"""
        try:
            logger.info("Generating response")
            logger.debug(f"Query: {query}")
            logger.debug(f"Number of context documents: {len(contexts)}")
            
            # Extract text from context documents
            context_text = "\n\n".join([doc.page_content for doc in contexts])
            logger.debug(f"Combined context length: {len(context_text)} characters")
            
            # Generate response
            logger.debug("Invoking QA chain")
            response = self.qa_chain.invoke({  # Pass input as a dictionary
                "input_documents": contexts,
                "question": query
            })
            logger.info("Response generated successfully")
             # Extract the actual answer string from the dictionary
            answer = response.get('output_text', '') # Use .get for safety
            logger.debug(f"Response length: {len(answer)} characters")
            
            return answer # Return only the string answer bc in main,it needs a string
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            raise