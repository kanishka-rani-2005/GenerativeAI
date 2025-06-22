import os
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
import streamlit as st
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
# groq_api_key=os.getenv("GROQ_API_KEY")

os.environ['HF_TOKEN']=os.getenv('HF_TOKEN')

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Set up app

st.title('Conversation RAG with PDF uploads and Chat History')
st.write('Uploads PDFS and chat with their content.')

# Get input of api key
api_key=st.text_input('Enter your Groq API Key :' ,type='password')

# if api key is there then initialize the model
if api_key is not None and api_key.strip() != "":
    model=ChatGroq(model='Gemma2-9b-It',groq_api_key=api_key)

    session_id=st.text_input("Session Id ",value="Default_Session_Id")

    if "store" not in st.session_state:
        st.session_state.store={}
    
    uploaded_files=st.file_uploader("Choose a Pdf file ",type="pdf",accept_multiple_files=True)

    if uploaded_files:
        documents=[]
        #  for each file
        for file in uploaded_files:
            # create temp file
            temppdf = f"./temp_{file.name}"


            # open in write byte mode 
            with open (temppdf,'wb') as f:
                # get pdf value
                f.write(file.getvalue())
                # get pdf name
                file_name=file.name

            # load pdf
            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            # store each in documents
            documents.extend(docs)

        text_splitter=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=100)
        splits=text_splitter.split_documents(documents=documents)

        vectorstore=Chroma.from_documents(documents=splits,embedding=embeddings)
        retriever=vectorstore.as_retriever()

        ## new prompt
        contextual_prompt=(
            'Given a chat history and the latest user question'
            'which might refer the context in chat history'
            'Donot answer question without refering the chat history'
            'Just reformulate it if needed and otherwise return it as it is '

        )

        contextual_q_prompt = ChatPromptTemplate.from_messages([
             ("system", contextual_prompt + "\n\n Please respond in {language}"),
             MessagesPlaceholder("chat_history"),
                ("user", "Question : {input}")  
        ])

        history_aware_retriever=create_history_aware_retriever(model,retriever,contextual_q_prompt)
        

        ## Answer question prompt

        system_prompt=(
            'You are an AI assistant for answering the various questions'
            'Use the following pieces of retrieved context to answer the question'
            "If you don't know the answer ,say that you don't know "
            '{context}'
        )

        qa_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt+"\n\n Please respond in {language}"),
                MessagesPlaceholder("chat_history"),
                ("user","{input}")
            ]
        )

        question_answer_chain=create_stuff_documents_chain(model,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session_id:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()

            return st.session_state.store[session_id]


        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input=st.text_input("Your question")
        # language='English'
        language=st.text_input("Your language")
        if user_input and language:
            session_hist=get_session_history(session_id=session_id)

            response=conversational_rag_chain.invoke(
                {"input":user_input,"language":language},
                config={
                    "configurable": {"session_id":session_id}
               },
            )
            st.success("Assistant Response:")
            st.write(response['answer'])

            st.markdown("### Chat History")
            for msg in session_hist.messages:
                st.markdown(f"**{msg.type.capitalize()}:** {msg.content}")


else:
    st.warning("Please Enter Your groq api key .")