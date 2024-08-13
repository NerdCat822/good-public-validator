from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain_openai.chat_models import ChatOpenAI
import openai
import os
from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.environ.get('OPENAI_API_KEY')

def RAG_bad(bad_text):
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size = 600,
        chunk_overlap=100,
    )

    loader =PyPDFLoader("./민원 처리에 관한 법령 해설집(개정판).pdf") # pdf loader

    pages = loader.load_and_split()
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=600, chunk_overlap=100)
    texts = text_splitter.split_documents(pages)
    
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever()

    combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    text = retrieval_chain.invoke({"input": bad_text + "앞의 악성민원과 관련해 부적절한 단어들을 , 로 구분 지어서 문자열로 반환해줘. 글자를 수정하지말고."})

    return text['answer']