from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
import os
from pathlib import Path


def main():
    load_dotenv(Path(".streamlit\.ENV"))
    openai_api_key = os.getenv("OPEN_API_KEY") 
    YOUR_ORGANIZATION_ID = os.getenv("YOUR_ORGANIZATION_ID")
    st.set_page_config("Chat with multiple PDF")
    st.header("Chat with PDF")

    pdf = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=False)

    try:
        if pdf is not None:
            reader = PdfReader(pdf)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        
            # Splitting into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            chunks = text_splitter.split_text(text)
        
            #Openai Embeddings
            embeddings = OpenAIEmbeddings(
                openai_api_key= openai_api_key
            )
            knowledge_base = FAISS.from_texts(chunks, embeddings)
        
            #user question
            user_question = st.text_input("Ask a question about your pdf")
            if user_question is not None:
                trigger = st.button("Submit")
                if trigger:
                    docs = knowledge_base.similarity_search(user_question)
                    llm = OpenAI(openai_api_key=openai_api_key, openai_organization=YOUR_ORGANIZATION_ID)
                    chain = load_qa_chain(llm, chain_type="stuff")
                    response = chain.run(input_documents=docs, question=user_question)
                    st.write(response)
    except Exception as e:
        st.write(e)

if __name__ == "__main__":
    main()
