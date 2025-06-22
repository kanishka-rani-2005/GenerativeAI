import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.embeddings import OllamaEmbeddings
import os
import time
from dotenv import load_dotenv 

# Load API Key
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

prompt = ChatPromptTemplate.from_template(
    '''Answer the question based on the provided context only.
<context>
{context}
</context>
Question: {input}
'''
)

st.title("RAG Document Q&A with Groq")

def create_vector_embedding():
    if "vectors" not in st.session_state:
        with st.spinner("Loading and Preprocessing documents...."):
            st.session_state.embeddings = OllamaEmbeddings(model='llama3.2')
            loader = PyPDFDirectoryLoader("research_papers")
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            final_document = text_splitter.split_documents(docs)
            st.session_state.vectors = FAISS.from_documents(final_document, embedding=st.session_state.embeddings)
        st.success("Vector Database is ready!")

if st.button("Embed Documents"):
    create_vector_embedding()

user_prompt = st.text_input("Enter your query from research papers:")

if user_prompt:
    if "vectors" not in st.session_state:
        st.warning(" Please embed documents first using the 'Embed Documents' button.")
    else:
        with st.spinner("Thinking..."):
            doc_chain = create_stuff_documents_chain(llm=model, prompt=prompt)
            retriever = st.session_state.vectors.as_retriever()
            retriever_chain = create_retrieval_chain(retriever, doc_chain)

            starttime = time.process_time()
            response = retriever_chain.invoke({"input": user_prompt})
            elapsed = time.process_time() - starttime

            st.markdown(f"**Answer:** {response['answer']}")
            st.caption(f"Response time: {elapsed:.2f} seconds")

            with st.expander("View Similar Document Chunks"):
                for i, doc in enumerate(response.get("context", [])):
                    st.write(doc.page_content)
                    st.markdown("-"*100)
