import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMMathChain,LLMChain
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import Tool,initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os 
from dotenv import load_dotenv

load_dotenv()

## Setup Streamlit APP

st.set_page_config(page_title='Text To Math Problem Solver')

st.title('Text to Math problem solver using google gemma 2')


groq_api_key=os.getenv('GROQ_API_KEY')

if groq_api_key:
    chat_groq = ChatGroq(groq_api_key=groq_api_key,model='Gemma2-9b-It')
else:
    chat_groq = None
    raise ValueError("GROQ_API_KEY not found. Please set it in your environment.")


# Initializing the tools
wiki_wrap=WikipediaAPIWrapper()
wiki_tool=Tool(
    name='Wikipedia',
     func=wiki_wrap.run,
    description='Wikipedia API',
)

# Initializing the Math Tool
math_chains=LLMMathChain.from_llm(llm=chat_groq)

calculator = Tool(
    name='Calculator',
    func=math_chains.run,
    description='A tool that solves pure mathematical expressions, like arithmetic or algebra.'
)

prompt = '''
You are an intelligent math assistant. Solve the user's question logically and explain each step clearly.

Display the solution in a step-by-step format:

Question: {question}

Answer:
'''


prompt_template=PromptTemplate(
    input_variables=['question'],
    template=prompt
)

#Combine tool to chain

llmchain=LLMChain(
    prompt=prompt_template,
    llm=chat_groq
)


reasoning_tool = Tool(
    name='Reasoning Tool',
    func=llmchain.run,
    description='A tool that solves word problems and explains the reasoning step by step.'
)

## Initialize the agents

assistant_agent=initialize_agent(
    tools=[wiki_tool, calculator, reasoning_tool],
    name='Assistant',
    llm=chat_groq,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_error=True,
    description='A tool for answering math related answer. Only input mathematical expression need to be provided .'
)


if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {
            'role':'assistant',
            'content':'Hello, I am an assistant. I can answer math related questions.',
        }
    ]


for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])



def generate_response(user_question):
    response=assistant_agent.invoke({"input":user_question})
    return response

question=st.text_area('Enter Your Question Here.')

if st.button('Solve'):
    if question:
        with st.spinner('Generating Response....'):
            st.session_state.messages.append({'role':'User','content':question})
            response=generate_response(question)
            st.success('Response')
            st.chat_message('user').write(question)
            st.chat_message('assistant').markdown(response['output'])
            st.session_state.messages.append({'role':'Assistant','content':response['output']})


    else:
        st.warning('Please enter a question')


