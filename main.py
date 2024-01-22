from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
import google.generativeai as genai
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
import os


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# GET PDF TEXT
def get_pdf_text(pdf_docs):
    texts = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            texts += page.extract_text()
    return texts


def get_text_chunk(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=1000,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # vector_store = FAISS.from_texts(texts=[text_chunks], embedding=embeddings)
    db = FAISS.from_documents(text_chunks, embeddings)
    db.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to produce 
    provided context just say, "answer is not available in the context", don't provide the 
    Context:\n {context}?\n
    Question: \n{question}\n 

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model_name="gemini-pro",
        temperature = 0.3
    )

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(
    model, chain_type="stuff", prompt=prompt
    )
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {
            "input_documents": docs,
            "question": user_question   
        },return_only_outputs=True
    )
    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Chat with multiple PDF")
    st.header("Chat with multiple PDF using Gemini")

    user_question = st.text_input("Enter your question...")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing.."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunk(raw_text)
                get_vector_store(text_chunks)
                st.success("Done!")

if __name__ == "__main__":
    main()