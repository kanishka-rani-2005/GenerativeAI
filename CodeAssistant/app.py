import streamlit as st
import requests
import json

url = st.secrets["OLLAMA_URL"]

headers={
    'Content-Type': 'application/json',
}

def generate_response(prompt,history):
    history.append(prompt)

    final_prompt='\n'.join(history)

    data={
        'model':'CodeGuru',
        'prompt':final_prompt,
        'stream':False
    }


    try:
        response=requests.post(url,headers=headers,data=json.dumps(data))

        response.raise_for_status()

        response_data=response.json()
        actual_response=response_data.get('response','No response Field in JSON.')

        history.append(actual_response)

        return actual_response
    except Exception as e:
        print(f"An error occurred: {e}")
        return
    


st.title("Code Assistant ")

st.text("It is simple code writer. You need to provide prompt it will generate code for that prompt. ")


if 'history' not in st.session_state:
    st.session_state.history = []

st.text_area('Conversation History ','\n\n'.join(st.session_state.history),width=1500,height=300,disabled=True)

prompt=st.text_input('Enter your prompt')


if st.button('Generate Response'):
    if prompt:
        with st.spinner("Generating response..."):
            response = generate_response(prompt, st.session_state.history)
        if response:
            st.success("Here is the latest response:")
            st.write(response)

    else:
        st.warning("Please enter a prompt")
