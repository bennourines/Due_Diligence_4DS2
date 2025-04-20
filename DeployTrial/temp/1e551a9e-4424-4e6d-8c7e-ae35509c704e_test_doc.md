# Crypto Due Diligence Assistant

This document outlines the complete system architecture and setup instructions for the Crypto Due Diligence Assistant, a Python-based chatbot that automates the analysis of cryptocurrency assets.

## System Overview

The Crypto Due Diligence Assistant is built using:
- **FastAPI**: Backend API for document processing and query handling
- **Streamlit**: Frontend user interface for document upload and interaction
- **LangChain**: Framework for document processing, RAG, and LLM integration
- **FAISS**: Vector database for semantic search
- **BM25**: Keyword-based search algorithm
- **MongoDB**: Database for storing conversations and report history
- **OpenRouter**: API gateway for accessing various LLMs

The system implements a hybrid retrieval-augmented generation (RAG) pipeline that combines vector search with keyword search for improved document retrieval and question answering.

## Installation

### Prerequisites
- Python 3.9+
- MongoDB
- OpenRouter API key (for LLM access)

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/crypto-due-diligence.git
cd crypto-due-diligence
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with the following variables:
```
OPENROUTER_API_KEY=your_openrouter_api_key
MONGODB_URL=mongodb://localhost:27017
API_URL=http://localhost:8000
```

## Project Structure

```
crypto-due-diligence/
├── app.py                    # Streamlit frontend
├── main.py                   # FastAPI backend
├── document_processor.py     # Document loading and chunking
├── search_engine.py          # Hybrid search implementation
├── rag_pipeline.py           # RAG implementation with OpenRouter
├── risk_analyzer.py          # Risk scoring and report generation
├── database.py               # MongoDB connection handlers
├── models.py                 # Pydantic models for API
└── requirements.txt          # Project dependencies
```

## Running the Application

1. Start the FastAPI backend:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

2. Start the Streamlit frontend:
```bash
streamlit run app.py
```

3. Access the application at http://localhost:8501

## Dependencies

Create a `requirements.txt` file with the following:

```
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
streamlit==1.28.0
langchain==0.0.335
faiss-cpu==1.7.4
sentence-transformers==2.2.2
pymongo==4.6.0
motor==3.3.1
openai==1.3.0
langchain-openai==0.0.2
pypdf==3.17.1
docx2txt==0.8
pandas==2.1.3
matplotlib==3.8.2
plotly==5.18.0
python-dotenv==1.0.0
rank-bm25==0.2.2
unstructured==0.10.30
```

## System Components

### Document Processor
- Handles PDF, DOCX, XLSX, and unstructured text files
- Chunks documents using RecursiveCharacterTextSplitter
- Preserves document metadata

### Hybrid Search Engine
- Combines FAISS vector store for semantic search
- Integrates BM25 for keyword-based search
- Creates an ensemble retriever with weighted combination

### RAG Pipeline
- Uses LangChain's RetrievalQA chain
- Integrates with OpenRouter for LLM access
- Customized prompt engineering for crypto analysis

### Risk Analyzer
- Question bank covering key risk categories:
  - Tokenomics
  - Technical security
  - Team background
  - Regulatory compliance
  - Market positioning
- Weighted scoring algorithm for risk assessment
- Summary report generation with actionable recommendations

### Database Integration
- MongoDB for storing conversations
- Document metadata storage
- Report persistence

## Key Features

### 1. Document Analysis
The system processes various document formats, extracting text and metadata to create searchable chunks. Documents are split into optimal segments for retrieval.

### 2. Hybrid Search
The hybrid search combines:
- **Vector Search**: FAISS indexes document embeddings for semantic similarity
- **Keyword Search**: BM25 algorithm for exact keyword matching
- **Ensemble Retrieval**: Weighted combination of both search methods

### 3. Risk Report Generation
The system generates comprehensive risk reports by:
1. Processing a predefined question bank
2. Retrieving relevant contexts for each question
3. Generating answers using the RAG pipeline
4. Scoring answers based on risk indicators
5. Compiling weighted scores into category and overall risk assessments
6. Generating actionable recommendations

### 4. User Interface
The Streamlit interface provides:
- Document upload functionality
- Interactive chat interface
- Visual risk reports with charts and heatmaps
- Detailed category breakdowns

## Customization

### Modifying the Question Bank
Edit the `_load_question_bank` method in `risk_analyzer.py` to add or modify questions, categories, and weights.

### Changing the LLM Provider
Update the `RAGPipeline` class in `rag_pipeline.py` to use different models from OpenRouter or other providers.

### Adjusting Risk Scoring
Modify the `_analyze_answer_risk` method in `risk_analyzer.py` to refine the risk scoring algorithm.

## Security Considerations

- Document storage is temporary and can be configured for secure deletion
- API endpoints include user authentication placeholders
- MongoDB connection should be secured in production
- OpenRouter API key should be kept secure

## Deployment

For production deployment:
1. Set up proper authentication
2. Use a production-ready MongoDB instance
3. Deploy FastAPI with Gunicorn or similar WSGI server
4. Deploy Streamlit with proper authentication
5. Use environment variables for all sensitive credentials

## Conclusion

The Crypto Due Diligence Assistant provides a powerful tool for analyzing cryptocurrency assets through document analysis, interactive question answering, and comprehensive risk assessment. The modular architecture allows for easy customization and extension.
