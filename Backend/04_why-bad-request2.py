# 6.6 RetrievalQA

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.storage import LocalFileStore
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.callbacks import StreamingStdOutCallbackHandler

import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.environ.get('OPENAI_API_KEY')

model = ChatOpenAI(openai_api_key=openai_api_key, 
                  model = "gpt-3.5-turbo",
                  temperature=0.1, 
                  streaming=True, 
                  callbacks=[StreamingStdOutCallbackHandler()])


splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size = 600,
    chunk_overlap=100,
)

loader = loader =PyPDFLoader("./민원 처리에 관한 법령 해설집(개정판).pdf") # pdf loader

docs = loader.load_and_split(text_splitter=splitter)

embeddings = OpenAIEmbeddings()

#chache_dir = LocalFileStore("./.cache/")
#cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
#    embeddings, chache_dir)

vectorstore = Chroma.from_documents(docs, embeddings)
#vectorstore_cache = FAISS.from_documents(docs, cached_embeddings)

#vectorstore.similarity_search("where does winston live")

chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="refine", # 그 외에도 refine, map_reduce, map_rerank 존재
    retriever=vectorstore.as_retriever(),   
)
bad_text = """
왜 이런 일이 일어나고 있는 거야? 기초수급비용을 왜 이렇게 적게 주는 거야? 내가 이미 너희들에게 얼마나 돈을 받아야 하는데? 더 이상 날 괴롭히지 마! 내 돈으로 무엇을 하는지 말해봐!

그냥 가라. 너희들은 어디서 이 모든 돈을 사용하고 있는 거야? 내가 도대체 뭘 위해서 받는 건데? 나는 이런 불공평한 일을 더 이상 참지 않겠어!
"""
print(chain.invoke(bad_text + "이 내용이 왜 악성민원인지 민원처리법, 폭언과 관련해서 알려줘."))
