import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain


OPENAI_API_KEY = ""

st.set_page_config(page_title="Chatbot")
st.header("My first chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("upload a pdf file and start asking questions", type="pdf")

if file is not None:
    pdf_reader = PdfReader(file)
    txt = ""
    for page in pdf_reader.pages:
        txt += page.extract_text()
        # st.write(txt)

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(txt)
    # st.write(chunks)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    vector_store = FAISS.from_texts(chunks, embeddings)

    user_ques = st.text_input("Type your question here")

    if user_ques:
        match = vector_store.similarity_search(user_ques)

        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
            max_tokens=1000,
            model_name="gpt-3.5-turbo"
        )

        chain = load_qa_chain(llm,chain_type="stuff")
        response = chain.run(input_documents=match,question=user_ques)
        st.write(response)