import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler

import os
from dotenv import load_dotenv


st.title("Langchain - Chat with search")

st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your GROQ API KEY .",type='password')


if not api_key:
    st.warning("Please enter your GROQ API key to continue.")
    st.stop()

##Arxiv and Wikipedia tools

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)

arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)
wiki=WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state['messages']=[
        {"role":"assistant","content":"Hi, I am a chatbot. How can i help you ? "}
    ]
# Display previous messages
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])
# Get user input
if prompt:=st.chat_input("Ask your question here..."):
    st.session_state["messages"].append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    ## Load LLM from Groq
    llm=ChatGroq(groq_api_key=api_key,model='Gemma2-9b-It',streaming=True)
    # Make tool kit from various tools
    tools=[arxiv,wiki]
    # initialize agent
    search_agent=initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handling_parsing_errors=True
    )

    with st.chat_message("assistant"):
       st_cb=StreamlitCallbackHandler(
            st.container(),
            expand_new_thoughts=False,
        )
 
       
    response=search_agent.run(st.session_state.messages,callbacks=[st_cb])

    st.session_state.messages.append({"role":"assistant","content":response})
    st.write(response)









# st.title("Chatbot")