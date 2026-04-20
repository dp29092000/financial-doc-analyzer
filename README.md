# Financial Document Analyzer

An AI-powered RAG (Retrieval Augmented Generation) application that lets users 
upload financial documents (PDF) and ask natural language questions, 
getting accurate answers grounded in the document content.

🚀 **[Live Demo](URL_HERE)**

## Problem Statement

Financial documents like annual reports are dense, lengthy, and difficult to 
navigate manually. Analysts spend significant time extracting specific insights 
from hundreds of pages of document.

This app addresses that by enabling instant, accurate question-answering 
directly from uploaded financial PDFs thus making document analysis faster.


## Architecture

The app follows a standard RAG pipeline with targeted optimizations:

**Ingestion (runs once per document)**
PDF Upload → Text Extraction (PDFPlumber) → Chunking (RecursiveCharacterTextSplitter) → Embedding (FastEmbed) → Vector Store (Chroma)

**Query (runs per question)**
User Question → Retrieve Top 8 Chunks (Chroma) → Re-rank to Top 4 (FlashrankRerank) → LLM Answer Generation (Groq / LLaMA 3.3 70B) → Display Answer

## Key Design Decisions

**PDFPlumberLoader over PyPDFLoader**
PyPDFLoader extracts text character by character, causing spacing issues with 
certain PDFs. PDFPlumberLoader uses spatial analysis to reconstruct words and 
sentences correctly, producing clean text extraction.

**FastEmbedEmbeddings over HuggingFaceEmbeddings**
HuggingFace embeddings caused heavy dependency build failures on Streamlit Cloud. 
FastEmbed is lightweight, faster, and cloud-deployment friendly.

**Re-ranking (K=8 → Top 4)**
Retrieving 8 chunks and re-ranking to top 4 improves retrieval precision over 
naive top-K retrieval, reducing noise passed to the LLM.

**Session State for Vector Store**
Storing the vector store in st.session_state prevents reprocessing the PDF on 
every Streamlit rerun, significantly improving response time.

**Temporary File Handling**
Uploaded PDFs are saved as temporary files to avoid storage costs, naming 
conflicts, and privacy concerns in a cloud deployment.

## Implementation Details

**Clear Document Button**
Allows users to analyze a new document by clearing the vector store from 
session state and resetting the file uploader, without restarting the app.

**Empty Question Validation**
Users are warned if they attempt to submit an empty question, preventing 
unnecessary LLM API calls.

**Custom Financial Prompt Template**
Prompt engineered specifically for financial document analysis - instructs the 
LLM to answer confidently when context is clear, use exact figures with units 
for numerical questions, cite document sections, and explicitly avoid hedging 
language.

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq API (llama-3.3-70b-versatile) |
| Embeddings | FastEmbedEmbeddings |
| Vector Store | Chroma |
| Re-ranker | FlashrankRerank |
| PDF Extraction | PDFPlumberLoader |
| Orchestration | LangChain |
| UI | Streamlit |
| Deployment | Streamlit Cloud |

## Future Enhancements

- **Query Rewriting** - Expand user queries with domain-specific terminology 
before retrieval to improve chunk relevance and reduce lexical overlap failures
- **Table-Aware Chunking** - Leverage PDFPlumber's table detection capabilities 
combined with structure-aware chunking to correctly handle financial tables 
and structured data, rather than treating them as plain text
- **Conversation Memory** - Maintain chat history to support follow-up questions
- **Streaming Responses** - Stream LLM output token by token for better UX
- **CohereRerank** - Replace FlashrankRerank with Cohere's reranking API for 
production-grade retrieval precision

## Author

**Prasanna D**
B.Tech, IIT Gandhinagar (2022)
Incoming MS Applied Machine Learning — University of Maryland, Fall 2026
