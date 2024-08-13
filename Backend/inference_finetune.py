from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from openai import OpenAI
import openai
from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

def finetuned_summarize(text):
    client = OpenAI()

    model = "ft:gpt-3.5-turbo-1106:nerdcat822::9DZMWdZy"

    llm = ChatOpenAI(model=model)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "'{input}' 이 내용 bullet point 사용해서 3줄 요약해줘"),
            ("user", "{input}" )
        ]
    )

    chain = prompt_template | llm | StrOutputParser()

    return chain.invoke({"input": text})

app = FastAPI()

class InputText(BaseModel):
    text: str

@app.post("/summarize")
def post_summarize(input_text: InputText):
    summary = finetuned_summarize(input_text.text)
    return {"summary": summary}