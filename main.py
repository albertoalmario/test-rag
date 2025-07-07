import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def setup_qa_system(file_path):
    loader = PyPDFLoader(file_path)
    
    # check different loading methods
    # docs_load = loader.load()
    # print(f"docs_load: {docs_load}")
    # docs_load_and_split = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100))
    # print(f"docs_load_and_split: {docs_load_and_split}")

    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0, api_key=OPENAI_API_KEY)

    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, return_source_documents=True)

    return qa_chain


if __name__ == "__main__":
    file_path = "data/The Sparkling Circuit Quest.pdf"
    # file_path = "data/file_example_for_load_and_load_split.pdf"
    qa_chain = setup_qa_system(file_path)

    while True:
        question = input("Enter your question (or 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        
        answer = qa_chain.invoke(question)

        print(f"Answer: {answer['result']}")