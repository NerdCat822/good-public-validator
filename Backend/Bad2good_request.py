from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

def bad2good(text):
    system_instruction = "assistant는 민원인의 말을 공손한 방식으로 필터링해 요약해줘."

    messages = [{"role": "system", "content": system_instruction},
                {"role": "user", "content": text} 
                ]

    response = openai.chat.completions.create(model="gpt-4-1106-preview", messages=messages)
    result = response.choices[0].message.content
    
    return result
