import os 
from dotenv import load_dotenv

from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings import OllamaEmbeddings
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


#LANGSMITH TRACKING
os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2']="true"
os.environ['LANGCHAIN_PROJECT']=os.getenv('LANGCHAIN_PROJECT')


## Prompt Template
try:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Please respond to the question asked."),
        ("user", "Question: {question}")
    ])
except Exception as e:
    st.error(f"Prompt template error: {e}")


## Streamlit Framework
st.title('Langchain  with LLAMA3.2 Model')

input_text=st.text_input("What question you have in mind?")
if st.button('Ask'):
    llm=Ollama(model="llama3.2")
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser

    if input_text:
        question=input_text
        response = chain.invoke({"question": input_text})
        st.markdown(f"**Answer:** {response}")

    else:
        st.markdown(f"**Answer:** Please Enter something.")


