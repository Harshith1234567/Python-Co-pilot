# import streamlit for the app 
import streamlit as st

# Import langchain stuff 

from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.chains import SimpleSequentialChain
from langchain.memory import ConversationBufferMemory





PATH = '/Users/admin/Library/Application Support/nomic.ai/GPT4All/mistral-7b-instruct-v0.1.Q4_0.gguf'

#stremlit Environment
st.title('🐍 Python Co-Pilot 🤖')
prompt = st.text_input('what code do you need')


code_templete = PromptTemplate(input_variables=['question'], 
                                  template='write a python program to {question}')

memory= ConversationBufferMemory(input_key='question', memory_key='chat_history')

#LLM initialization
llm = GPT4All(model=PATH, verbose=True)
code_chain=LLMChain(prompt=code_templete, llm=llm, verbose=True, memory=memory)



if prompt: 
    
    response=code_chain.run(question = prompt)
    st.write(response)
    
    with st.expander('Message History'):
        st.info(memory.buffer)
    






































