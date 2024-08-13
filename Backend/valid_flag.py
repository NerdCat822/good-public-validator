from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.schema import BaseOutputParser
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain.schema import StrOutputParser

import os
from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.environ.get('OPENAI_API_KEY')

def flag(text):
    chat = ChatOpenAI(openai_api_key=openai_api_key, 
                      model = "gpt-3.5-turbo",
                      temperature=0.1)

    examples = [
        {
            "question": "씨발, 이 병신 개새끼야! 미친놈처럼 지랄하네!",
            "answer": "false"
        },
        {
            "question": "안녕, 당신은 나의 모든 것입니다. 당신과 함께하는 순간은 내게 무엇보다 소중하며, 당신의 존재만으로도 나는 행복을 느낍니다. 당신과 함께 있는 모든 순간은 나에게 사랑의 따뜻함을 전달해줍니다. 당신은 나의 세상을 밝게 빛내는 별이고, 나의 삶에는 당신이 함께 있어야만 완벽합니다.",
            "answer": "true"
        },
        {
            "question": "너는 정말 특별해. 너와 함께 있는 시간은 내게 큰 의미가 있어. 네가 내 곁에 있을 때마다 행복한 기분이 들어. 너 없이는 내 삶이 완전하지 않아. 너는 내게 필요한 사람이야.",
            "answer": "true"
        }
    ]
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{question}"),
            ("ai", "{answer}"),
        ]
    )

    example_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )


    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "assistant는 나쁜말과 착한말을 구분 가능하다. '{input}' 이 문장이 나쁜 말이라면 false, 착한 말이면 true 를 출력해줘"),
            example_prompt,
            ("human", "{input}"),
        ]
    )

    chain = final_prompt | chat | StrOutputParser()

    return chain.invoke({"input": text})
