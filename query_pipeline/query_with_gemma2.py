import torch
from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, FewShotChatMessagePromptTemplate
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
# from langchain_core.prompts import PromptTemplate
from subquerying_prompt import SYSTEM_FINAL_0129
from few_shot import examples_final
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextIteratorStreamer
from langchain_huggingface import HuggingFacePipeline
import os
from datetime import datetime

# 모델 로드


def load_prompt(system_prompt :str, fewshot_ex=None) -> PromptTemplate:
    '''
    프롬프트를 받아, LangChain의 프롬프트 클래스를 생성하여 반환하는 함수
    args:
        system_prompt: 시스템 프롬프트
        fewshot_ex: list, few-shot 학습 데이터
    return:
        chat_prompt: PromptTemplate
    '''

    today = datetime.today().strftime("%Y년 %m월 %d일")
    today_prompt = f"오늘은 {today}입니다. 올해는 {datetime.today().year}년입니다. 이번 달은 {datetime.today().month}월입니다."

    assistent_template = """다음 질문을 여러 개의 하위 질문으로 나누어주세요.
    <|im_start|>user {query}<|im_end|>
    <|im_start|>assistant
    """
    
    if fewshot_ex is not None:
        few_shot_prompt = """<|im_start|>example아래는 몇 가지 예시입니다. 아래 예시를 참고해서 답변을 작성하세요.""" \
        + '\n'.join([f"예시 입력: {ex['input']}\n예시 출력: {ex['output']}\n" for ex in fewshot_ex]) + "<|im_end|>"
        chat_prompt = PromptTemplate.from_template(today_prompt + system_prompt + few_shot_prompt + '\n\n' + assistent_template)

    else:
        chat_prompt = PromptTemplate.from_template(today_prompt + system_prompt + '\n\n' + assistent_template)

    return chat_prompt


def get_sub_queries(query: str, tokenizer, model) -> list[str]:
    '''
    사용자 입력을 받아, 하위 쿼리로 나누어 반환하는 함수
    args:
        query: 사용자 입력
    return:
        sub_queries: 하위 쿼리
    '''

    # 프롬프트 설정
    chat_prompt = load_prompt(SYSTEM_FINAL_0129, examples_final)

    # llm 호출
    # MODEL_NAME = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
    # MODEL_NAME = "recoilme/recoilme-gemma-2-9B-v0.4"
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", use_cache=True)
    model.eval()
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    # LangChain의 LLM으로 Wrapping
    llm = HuggingFacePipeline(pipeline=pipe)

    # chaining
    chain = chat_prompt | llm | StrOutputParser()

    # pipeline 실행
    sub_queries = chain.invoke({"query": query})
    sub_queries = sub_queries.split("<|im_start|>assistant")[1].split("<|im_end|>")[0].strip()

    # print(sub_queries)  # for debugging
    # 답변을 json으로 저장
    with open('sub_queries.json', 'w') as f:
        f.write(str(sub_queries))
    # json 파일을 읽어들여 list 형태의 subquery 저장
    with open('sub_queries.json', 'r') as f:
        sub_queries = json.load(f)

    return sub_queries['response']