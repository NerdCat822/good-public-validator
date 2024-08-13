from datasets import load_dataset
import openai
import json
import os
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
#openai.azure_endpoint = os.getenv('AZURE_ENDPOINT')
#openai.api_type = os.getenv('API_TYPE')
#openai.api_version = os.getenv('API_VERSION')

dataset = load_dataset('maywell/korean_textbooks', 'claude_evol')
train_valid_ds = dataset['train'].train_test_split(test_size=30)
train_df = train_valid_ds['train'].to_pandas()
valid_df = train_valid_ds['test'].to_pandas()


# 실습을 위한 학습 시간을 줄이기 위해 샘플을 줄임, 일반적으로 많을 수록 좋음
train_df = train_df.sample(100)

# JSONL 파일 생성 함수
def create_jsonl(df, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for i, item in df.iterrows():
            doc = item['text']
            system_instruction = "assistant는 user의 입력을 bullet point로 3줄 요약해준다."

            messages = [{"role": "system", "content": system_instruction},
                        {"role": "user", "content": doc} 
                        ]
            response = openai.chat.completions.create(model="gpt-4-1106-preview", messages=messages)
            json_line = json.dumps({"messages": [{"role": "system", "content": system_instruction},
                                                 {"role": "user", "content": doc},
                                                 {"role": "assistant", "content": response.choices[0].message.content}
                                                ],
                                    }
                                  ,ensure_ascii=False)
            f.write(json_line + '\n')


# 훈련 및 검증 데이터셋을 JSONL 파일로 변환
create_jsonl(train_df, 'korean_textbooks_train.jsonl')
create_jsonl(valid_df, 'korean_textbooks_valid.jsonl')