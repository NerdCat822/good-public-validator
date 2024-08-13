from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from operator import itemgetter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.document_loaders import PyPDFLoader, UnstructuredFileLoader, TextLoader
from langchain.chains import RetrievalQA
import openai
import os
from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.environ.get('OPENAI_API_KEY')

model = ChatOpenAI(openai_api_key=openai_api_key, 
                  model = "gpt-3.5-turbo",
                  temperature=0.1)

splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size = 600,
    chunk_overlap=100,
)

loader =PyPDFLoader("./민원 처리에 관한 법령 해설집(개정판).pdf") # pdf loader

docs = loader.load_and_split(text_splitter=splitter)

texts = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings() # 문서 embeding 수행
db = FAISS.from_documents(texts, embeddings) # db 수행

retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="refine", # 그 외에도 refine, map_reduce, map_rerank 존재
    retriever=db.as_retriever(),   
)

text = chain.invoke("assistant는 현재 악성민원을 방지하는 업무를 맡고있다. 악성민원과 관련해 지속적인 악성민원, 민원처리법의 몇조, 몇항과 관련해서 알려줘.")
print(text['result'])