import os
import json
from pathlib import Path
from typing import List, Dict
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

def load_chunks_from_folder(folder: str) -> List[Document]:
    """Load all chunks from JSON files in a folder and convert to LangChain Documents"""
    documents = []
    folder_path = Path(folder)

    for json_file in folder_path.glob("*_chunks.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            for chunk in chunks:
                documents.append(
                    Document(
                        page_content=chunk["text"],
                        metadata=chunk["metadata"]
                    )
                )
        print(f"✅ Loaded {len(chunks)} chunks from {json_file.name}")
    return documents

# Extend SmartChunker class with vector store capabilities
def add_vector_store_methods(SmartChunkerClass):
    def _init_embeddings(self):
        """Initialize embeddings model if not already done"""
        if self.embeddings is None:
            print("🔍 Initializing embeddings model...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={'device': 'cpu'}  # Use GPU if available by changing to 'cuda'
            )

    def create_vectorstore(self, chunks_folder="chunks", save_path=None):
        """Create FAISS vectorstore from processed chunks"""
        save_path = save_path or self.data_dir
        documents = load_chunks_from_folder(chunks_folder)

        if not documents:
            raise ValueError(f"No documents found in {chunks_folder}. Please process texts first.")

        # Initialize embeddings
        self._init_embeddings()

        print(f"🔍 Creating vectorstore with {len(documents)} documents...")
        self.vectordb = FAISS.from_documents(documents, self.embeddings)

        # Save FAISS index
        os.makedirs(save_path, exist_ok=True)
        self.vectordb.save_local(save_path)
        print(f"✅ FAISS index saved to '{save_path}/'")

        # Save metadata as JSON
        metadata_path = os.path.join(save_path, "index_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump([doc.metadata for doc in documents], f, indent=2, ensure_ascii=False)
        print(f"✅ Metadata saved to '{metadata_path}' (JSON)")

        return len(documents)

    def load_index(self, path=None):
        """Load existing FAISS index"""
        path = path or self.data_dir

        if not os.path.exists(path):
            raise FileNotFoundError(f"Index directory '{path}' not found.")

        # Initialize embeddings
        self._init_embeddings()

        print(f"📂 Loading FAISS index from {path}...")
        self.vectordb = FAISS.load_local(path, self.embeddings)

        # Load metadata
        metadata_path = os.path.join(path, "index_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                print(f"✅ Loaded metadata for {len(metadata)} documents")
        else:
            print("⚠️ No metadata file found.")

        return self.vectordb is not None

    def search(self, query: str, k: int = 3, enhance_query: bool = True) -> List[Dict]:
        """Search the vectorstore with an enhanced query"""
        if not self.vectordb:
            raise ValueError("No vectorstore loaded. Call load_index() first.")

        # Enhance query with domain terms if enabled
        if enhance_query:
            enhanced = self.utility.enhance_query(query)
            print(f"Enhanced query: '{enhanced}'")
            query = enhanced

        # Perform search
        results = self.vectordb.similarity_search_with_score(query, k=k)

        # Format results
        formatted_results = []
        for doc, score in results:
            result = {
                "text": doc.page_content,
                "score": 1.0 - score,  # Convert distance to similarity score
                "file_name": doc.metadata.get("source", "Unknown"),
                "entities": doc.metadata.get("entities", {}),
                "risk_score": doc.metadata.get("risk_score", 0)
            }
            formatted_results.append(result)

        return formatted_results

    # Add methods to class
    SmartChunkerClass._init_embeddings = _init_embeddings
    SmartChunkerClass.create_vectorstore = create_vectorstore
    SmartChunkerClass.load_index = load_index
    SmartChunkerClass.search = search

    return SmartChunkerClass
