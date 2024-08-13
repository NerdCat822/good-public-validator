from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
openai.azure_endpoint = os.getenv('AZURE_ENDPOINT')
openai.api_type = os.getenv('API_TYPE')
openai.api_version = os.getenv('API_VERSION')


def summarize(text):
    system_instruction = "assistant는 user의 입력을 bullet point로 3줄 요약해준다."

    messages = [{"role": "system", "content": system_instruction},
                {"role": "user", "content": text} 
                ]

    response = openai.chat.completions.create(model="dev-gpt-35-turbo", messages=messages)
    result = response.choices[0].message.content
    
    return result

bad_texts = """
9급 욕하는 새끼들 특징이 



이상하게 현실세계에서 공부 잘하는 새끼들이나 아니면 어설프게라도 대학물 먹은애들은 욕 잘 안한다.



꼭 고졸 생산직, 좆소 영업직, 차팔이, 타이어팔이... 이런 ㅎㅌㅊ 인생 사는새끼들이 기를쓰고 욕함.



왜냐, 이새끼들은 지들이 '20살부터 돈을 벌어왔다는' 자부심이 상상외로 엄청나게 강함.



공고졸업하고 몸쓰는 일하는 새끼들 공통적인 특징이 뭐냐, 



대부분 집안이 별볼일 없는집안, 가난한 집안이라는거야.



사람은 누구나 자기가 가진걸로 그 세계를 평가한다.



본인의 가난한, 앰생 집안을 대입함과 동시에



이제 9급이 되면? 이라고 머리를 굴리는거지.



그래서 이새끼들이 맨날 9급 까는 코스가 비슷해.



"박봉이다, 찌질이다, 평생 가난하다, 그돈 받을거면 왜하냐? " 



이런식으로 욕을하면서 본인의 학벌 열등감과 



사회적으로 천대받는 본인의 직업에 대한 방어의식이 작동하는거지.



내가 주변 친구들 보면서 느낀게 공무원 까는 새끼들 공통적인 특징



1. 못배운 새끼들. 기술하는 새끼들. 자영업 하는 새끼들 -> 사회적으로 천대받는 직업



2. 대학은 다녀도 '집안이 가난한' 새끼들. 이새끼들도 본인이 가난해서 세상을 돈으로만 바라봄.



아니 시발 돈으로만 따지면 창녀가 검사보다 위 아니냐? ㅋㅋㅋ



내가 장인이어도 딸자식 줄때 9급이랑 고졸 생산직새끼랑 데려오면 9급한테 주겠다. 



9급 붙으면 상위 15%안에 들어가고 7급붙으면 5%안에 들어간다.



그만큼 대다수의 사람은 생산직, 좆소, 영세 자영업 하는 사람이 대한민국 80%임.



7급 5%라하면 또 말도 안된다 할텐데, 



건동홍이 상위 5%, 인터넷에서 지잡이라 까이는 국숭세단이 7~8%임.



아무튼 공시생들 본인이 이거 하기로 마음먹었으면 근거없는 자부심은 지양하더라도



비하하는 새끼들 말에 너무 슬퍼하지마라.



세상은 손가락으로 두드리는 키보드 너머에 있다. 
"""
print(summarize(bad_texts))

'''
app = FastAPI()

class InputText(BaseModel):
    text: str

@app.post("/summarize")
def post_summarize(input_text: InputText):
    summary = summarize(input_text.text)
    return {"summary": summary}
'''