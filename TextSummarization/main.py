import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader

## Streamlit APP

st.title("🔍 LangChain: Summarize Text from Website")
st.subheader("Summarize any webpage content")

# Input Groq api key
with st.sidebar:
    api_key=st.text_input("Groq Api Key :",value="",type="password")


# Enter url
generic_url=st.text_input("Enter any url .",label_visibility='collapsed')


# Gemma model
llm = ChatGroq(model='gemma2-9b-it', groq_api_key=api_key)

prompt_template=''' 
provide summary of the following content in 300 words
Content:{text}

'''

map_prompt=PromptTemplate(
    input_variables=["text"],
    template=prompt_template
)

final_prompt = ''' 
Provide the final summary of entire speech with these important points:
- Add a motivational title
- Start with an introduction
- Provide the summary as a numbered list of key points

Speech: {text}
'''

final_prompt_template = PromptTemplate(
    input_variables=['text'],
    template=final_prompt
)

if st.button("Summarize"):

    if not api_key.strip() or not generic_url.strip():
       st.error("Please provide the Groq API key and a URL.")

    elif not validators.url(generic_url):
        st.error('Please enter a valid URL. It can may be a website url.')

    else:
        try:
            with st.spinner("Loading and summarizing..."):

                loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={
                                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                                "Chrome/113.0.0.0 Safari/537.36"}
                                                 )
                    
                documents = loader.load()

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = splitter.split_documents(documents)
               
                chain = load_summarize_chain(
                    llm=llm,
                    chain_type='map_reduce',
                    map_prompt=map_prompt,
                    combine_prompt=final_prompt_template,
                    verbose=False
                )
                result=chain.invoke({'input_documents': docs})

                summary = result['output_text']
                st.success("Summary Complete!")
                st.write(summary) 
                
        except Exception as e:

            st.exception(f"Error !!! {e}")