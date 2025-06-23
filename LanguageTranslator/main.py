import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

groq_api_key=st.secrets('GROQ_API_KEY')
model=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)


prompt=ChatPromptTemplate.from_messages(
    [
        ("system","Translate the following into {language}"),
        ("user","{text}")
    ]
)

# Chain: prompt -> model -> parser
parser=StrOutputParser()

chain=prompt|model|parser

st.title("Language Translator")

language=st.text_input('Enter Language.')
text=st.text_input('Enter any Sentence.')


if st.button("Generate") and text and language:
    result = chain.invoke({"language": language,"text": text})
    st.markdown(f"**Translated Text:**\n\n {result}")
elif language=='' and text:
    st.markdown(f"**Please Enter Language.**")
elif text=='' and language:
    st.markdown(f"**Please enter any sentence.**")
else :
    st.markdown(f"**Enter Required fields.**")

