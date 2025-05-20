# 📊 Due_Diligence_4DS2 — AI-Powered Crypto Fund Due Diligence

**Due_Diligence_4DS2** is an intelligent platform that automates and enhances the due diligence process for digital assets and crypto investment funds. Developed as a capstone project by 4DS2 students at [ESPRIT University](https://esprit.tn), it leverages advanced NLP, data engineering, and machine learning to deliver a trustworthy, automated due diligence pipeline.

[![Watch the Technical Demo](https://img.icons8.com/clouds/2x/video-playlist.png)](https://drive.google.com/file/d/1D5GI3hBaL3rQXcfnjsXt4R3KMuIhq8pZ/view)

---

## 🚀 Project Overview

In the volatile and fast-moving crypto landscape, **due diligence is critical**. This platform enables investors, analysts, and researchers to assess crypto projects with greater **accuracy, consistency, and speed**.

> 🔍 From entity extraction and risk profiling to AI-generated Q&A and investor-ready reports — **Due_Diligence_4DS2** offers an end-to-end intelligent due diligence solution powered by **Retrieval-Augmented Generation (RAG)** and **Knowledge Graphs**.

---

## 🔧 Core Features

- 📡 **Automated Data Ingestion**  
  Collects structured and unstructured data from APIs, PDFs, websites, GitHub, and regulatory databases using crawlers and pipelines.

- 🧠 **Custom NLP & Risk Profiling**  
  Uses domain-specific Named Entity Recognition (NER), smart chunking, and multi-category risk tagging (legal, technical, financial, operational) with `spaCy`, regex, and blacklists.

- 🔎 **Hybrid Retrieval & Knowledge Graphs**  
  Combines semantic vector search (Qdrant) with keyword-based TF-IDF scoring. Results are enriched using a semantic knowledge graph for multi-hop reasoning.

- 🤖 **AI-Powered Question Generation**  
  Uses **Retrieval-Augmented Generation (RAG)** with **Qdrant indexing** to fetch context from legal, financial, and technical documents. Responses are generated using **locally deployed LLaMA 3.2 models**, ensuring privacy, auditability, and domain specificity.

- 📊 **Professional Report Generation**  
  Automatically produces clear, well-structured due diligence reports and PowerPoint slides across categories like Legal, Financial, Technical, Governance, ESG, Risk, and more.

---

## 🛠️ Tech Stack

**Languages & Frameworks:**  
Python · JavaScript (React) · FastAPI

**ML & NLP:**  
LLaMA 3.2 1B (local) · Sentence Transformers (MPNet) · Cross-Encoder · spaCy (custom NER)

**Data & Pipelines:**  
Apache Airflow · LangChain · Qdrant · TF-IDF · SQLite · Pandas · NumPy

**Document & Web Scraping:**  
PyMuPDF · PDFMiner · BeautifulSoup · Selenium · GitHub API · Reddit API · CoinGecko · CryptoPanic

**Retrieval & Indexing:**  
Qdrant · Hybrid Search (semantic + keyword) · Smart Chunking · Knowledge Graph

**Visualization & Reporting:**  
python-pptx · Matplotlib · Seaborn

**CI/CD & Hosting:**  
GitHub Actions · GitHub Pages

---

## 📈 Evaluation

Evaluation uses [RAGAS](https://github.com/explodinggradients/ragas), covering:

- 🔍 **Retrieval Quality** – Context precision, recall, and noise sensitivity  
- ✍️ **Answer Generation** – Factuality, completeness, and semantic alignment  
- 📊 **NLP Metrics** – BLEU, ROUGE, BERTScore, Exact Match

> Designed for auditability, explainability, and use in high-stakes financial domains.

---

## 🌐 Deployment & Access

- 🚀 Deployed using **FastAPI**  
- 🧠 **LLaMA 3.2** runs locally (GPU or CPU)  
- 🔍 Document vectors indexed via **Qdrant** only  
- 📄 Reports generated with **python-pptx**

---

## 📚 References

- [Meta LLaMA 3.2](https://openrouter.ai/meta-llama/llama-4-maverick)  
- [Qdrant](https://qdrant.tech/)  
- [LangChain](https://docs.langchain.com)  
- [RAGAS Evaluation Toolkit](https://github.com/explodinggradients/ragas)  
- [spaCy](https://spacy.io)

---

## 🙏 Acknowledgments

Developed by 4DS2 students at **ESPRIT University**, under faculty mentorship.  
---

## 💬 Contact

📧 [info@esprit.tn](mailto:info@esprit.tn)  
🌐 [Project Website](https://bennourines.github.io/Due_Diligence_4DS2)
