from dotenv import load_dotenv
import os
import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Financial Document Analyzer", layout = "wide")
st.title("Financial Document Analyzer")

st.write("Upload a financial document and ask questions to get instant insights powered by AI.")


if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

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
            
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap = 200)
            chunks = text_splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_store = Chroma.from_documents(chunks, embeddings)

            st.session_state.vector_store = vector_store
    
    prompt = PromptTemplate(
            template="""You are a financial document query resolver. 
                Answer the question precisely based on the provided context.
                If you don't know the answer, say so clearly.

                Context: {context}

                Question: {question}

                Answer:""",
            input_variables=["context", "question"])

    llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    api_key=groq_api_key
                )

    question = st.text_input("Ask a question about your document")
    if st.button("Get Answer"):
        if question:
            retriever = st.session_state.vector_store.as_retriever()
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt}
                )
            response = chain.invoke({"query": question})
            st.markdown(response["result"])
        else:
            st.warning("Please enter your Question!")




    
    