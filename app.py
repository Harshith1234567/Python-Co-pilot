# import streamlit for the app 
import streamlit as st

# Import langchain stuff 

from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All,
from langchain.chains import SimpleSequentialChain
from langchain.memory import ConversationBufferMemory





PATH = '/Users/admin/Library/Application Support/nomic.ai/GPT4All/mistral-7b-instruct-v0.1.Q4_0.gguf'
PATH2= '/Users/admin/Documents/datascience/LLm/LLM-Models/codellama-7b-instruct.Q3_K_S.gguf'
#stremlit Environment
st.title('🐍 Python Co-Pilot 🤖')
prompt = st.text_input('what code do you need')
col1, col2 = st.columns([5,5]) 

with col1:
    model_name=st.selectbox('Model to be used',('Mistral Instruct','Code llama2'),index=0)


code_templete = PromptTemplate(input_variables=['question'], 
                                  template='write a python program to {question}')

memory= ConversationBufferMemory(input_key='question', memory_key='chat_history')

#LLM initialization
llm = GPT4All(model=PATH, verbose=True)
code_chain=LLMChain(prompt=code_templete, llm=llm, verbose=True, memory=memory)

llama = GPT4All(model=PATH2, verbose=True)
code2_chain=LLMChain(prompt=code_templete, llm=llama, verbose=True, memory=memory)



if prompt and model_name=='Mistral Instruct': 
    
    response=code_chain.run(question = prompt)
    st.write(response)
    
    with st.expander('Message History'):
        st.info(memory.buffer)
    
if prompt and model_name=='Code llama2': 
    
    response=code2_chain.run(question = prompt)
    st.write(response)
    
    with st.expander('Message History'):
        st.info(memory.buffer)
    


