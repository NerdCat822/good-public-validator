from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from openai import OpenAI
import openai
from fastapi import FastAPI
from pydantic import BaseModel
import os
from inference_finetune import *
from Check_list_Few_shot import *
from Bad2good_request import *
from why_bad_request import *
from valid_flag import *
from why_bad import *
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputText(BaseModel):
    text: str

@app.post("/summarize")
def post_summarize(input_text: InputText):
    summary = finetuned_summarize(input_text.text)
    return {"summary": summary}

@app.post("/check_list")
def post_check_list(input_text: InputText):
    check_list = fewshot_checklist(input_text.text)
    return {"checkList": check_list}

@app.post("/bad2good")
def post_bad2good(input_text: InputText):
    response = bad2good(input_text.text)
    return {"bad2good": response}

@app.post("/why_bad_request")
def post_bad_request(input_text: InputText):
    bad_request = RAG_bad_request()
    return {"badRequest": bad_request}

@app.post("/valid_flag")
def post_valid_flag(input_text: InputText):
    flag_value = flag(input_text.text)
    return {"validFlag": flag_value}

@app.post("/why_bad")
def post_why_bad(input_text: InputText):
    bad_value = RAG_bad(input_text.text)
    return {"bad": bad_value}