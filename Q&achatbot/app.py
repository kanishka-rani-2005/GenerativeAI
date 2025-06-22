import streamlit as st
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.llms.ollama import Ollama

from dotenv import load_dotenv
load_dotenv()


## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Q&A Chatbot "


## Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assitant . Please response to the user queries in language {language}"),
        ("user","Question : {question}")
    ]
)



def generate_response(question,temperature,max_tokens,language="English"):

    llm=Ollama(model="llama3.2")
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({'question':question,'language':language})
    return answer


st.title("Enhanced Q&A Chatbot")

    
st.sidebar.title("Settings")
language = st.sidebar.selectbox("Select Language", ["English", "Spanish", "French","Hindi","German","Marathi","Punjabi","Chinese","Bhojpuri"])

temperature=st.sidebar.slider("Temperature ",min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)

## Main Interface

st.write("Go ahead ask any question.")
user_input=st.text_input("You :")


if user_input:
    response=generate_response(question=user_input,temperature=temperature,max_tokens=max_tokens,language=language)
    st.write("Assistant : ", response)
else:
    st.write("Please enter a question to get a response.")