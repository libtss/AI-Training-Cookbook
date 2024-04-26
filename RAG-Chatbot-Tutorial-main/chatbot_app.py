import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import AzureChatOpenAI
import vector_store
from langchain.callbacks import get_openai_callback

def main():
    # Displaying a header in the web application
    st.header("Simple RAG Chatbot 💬")
    # Displaying a text input field in the web application
    query = st.text_input("Ask questions about your PDF file:")
    
    if query:
        # Performing a similarity search using a vector store module
        docs = vector_store.Vector_store.similarity_search(query=query, k=3) 
        os.environ["AZURE_OPENAI_API_KEY"] = os.getenv('AZURE_OPENAI_API_KEY_chat')
        os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv('AZURE_OPENAI_ENDPOINT_chat')
        # Creating an instance of the AzureChatOpenAI class
        #input your api details    
        llm = AzureChatOpenAI(
            openai_api_version="openai_api_version",
            azure_deployment="azure_deployment",
            temperature=0.5)

        # Loading a question-answering chain using the load_qa_chain function
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb:
                # Running the question-answering chain with the provided input documents and query
                response = chain.run(input_documents=docs, question=query)
                print(cb)
        # Displaying the response in the web application        
        st.write(response)
        
       
if __name__ == '__main__':
    main()
