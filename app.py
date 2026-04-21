from dotenv import load_dotenv
import os
import streamlit as st
import tempfile
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank

# Setup
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
st.set_page_config(page_title="Financial Document Analyzer", layout = "wide")
st.title("Financial Document Analyzer")
st.write("Upload a financial document and ask questions to get instant insights powered by AI.")

# Session state initializations
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], key=st.session_state.uploader_key)

if uploaded_file is not None:
    st.success("File uploaded successfully!")

    if st.button("Clear Document"):
        if "vector_store" in st.session_state:
            del st.session_state["vector_store"]
            st.session_state.uploader_key += 1
            st.rerun()

    if "vector_store" not in st.session_state:
        with st.spinner("Processing your document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            loader = PDFPlumberLoader(tmp_path)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap = 200)
            chunks = text_splitter.split_documents(documents)

            embeddings = FastEmbedEmbeddings()
            vector_store = Chroma.from_documents(chunks, embeddings)

            st.session_state.vector_store = vector_store
    
    prompt = PromptTemplate(
            template="""You are an expert financial document analyst. 
                Use the provided context to answer the question accurately.

                Follow these guidelines:
                - When the answer is clearly present in the context, state it directly and confidently without hedging phrases like "the document suggests" or "it can be inferred"
                - For numerical questions: use exact figures from the document, include units (millions, billions, percentage etc.)
                - For comparison questions: clearly state the change in absolute numbers and percentages
                - For explanation questions: connect multiple reasons logically and explain their relationship
                - Always cite which part of the document your answer is based on
                - If the answer is not found in the context, say so clearly instead of guessing
                - Answer directly without repeating or rephrasing the context

                Context: {context}

                Question: {question}

                Answer:""",
            input_variables=["context", "question"])

    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
   
    base_retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 8})
    compressor = FlashrankRerank(top_n=4)
    retriever = ContextualCompressionRetriever(
                        base_compressor = compressor,
                        base_retriever = base_retriever
                    )

    question = st.text_input("Ask a question about your document")
    if st.button("Get Answer"):
        if question:
            with st.spinner("Generating Answer..."):
                docs = retriever.invoke(question)
                context = "\n\n".join([doc.page_content for doc in docs])
                chain = prompt | llm
                response = chain.invoke({"context": context, "question": question})
                st.markdown(response.content)
                
        else:
            st.warning("Please enter your Question!")
