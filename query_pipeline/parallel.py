import openai
from langchain_openai import ChatOpenAI
import yaml
import dotenv
import os
from langchain.chains import SequentialChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, FewShotChatMessagePromptTemplate
from routing_prompts import *
import json
import time
import datetime
import concurrent.futures
from typing import List, Dict

def load_prompt(system_prompt: str) -> ChatPromptTemplate:
    '''
    프롬프트를 받아, LangChain의 프롬프트 클래스를 생성하여 반환하는 함수
    system_prompt: str, 시스템 프롬프트
    '''
    system_prompt += ("\n 현재 시각: " + str(datetime.datetime.now())) 
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template("사용자 입력: {user_input}")
        ]
    )
    return chat_prompt

def process_single_query(query: str, chain) -> Dict:
    '''
    단일 쿼리를 처리하는 함수
    query: str, 처리할 단일 쿼리
    chain: LangChain chain, 사용할 체인
    '''
    try:
        result = chain.invoke({"user_input": query})
        return result
    except Exception as e:
        print(f"Error processing query '{query}': {str(e)}")
        return {
            "response": [{
                "subquery": query,
                "routing": "False",
                "reasoning": f"Error processing: {str(e)}"
            }]
        }

def prompt_routing(subqueries: List[str]):
    '''
    subquery를 받아, LLM prompting을 통해 routing을 병렬로 실행하는 함수
    subqueries: str[], 사용자 입력을 subquery로 분해한 list
    '''
    # openai key 받아오기
    # 모델 교체되면 삭제할 것!!!
    with open('../jo/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    dotenv.load_dotenv('~/agent_ai')
    # OpenAI API 키 설정
    openai.api_key = config.get('OPENAI_API_KEY')
    os.environ["OPENAI_API_KEY"] = openai.api_key

    # Setup
    chat_prompt = load_prompt(PARALLEL)
    dotenv.load_dotenv()
    llm = ChatOpenAI(model='gpt-4o-mini')
    chain = chat_prompt | llm | StrOutputParser()
    
    # 병렬 처리 코드
    all_responses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(subqueries), 10)) as executor:
        future_to_query = {
            executor.submit(process_single_query, query, chain): query 
            for query in subqueries
        }
        
        for future in concurrent.futures.as_completed(future_to_query):
            query = future_to_query[future]
            try:
                result = future.result()
                all_responses.append(json.loads(result))
                print(f"Processed query: {query}")
            except Exception as e:
                print(f"Query '{query}' generated an exception: {str(e)}")
                all_responses.append({
                    "subquery": query,
                    "routing": "False",
                    "reasoning": f"Error processing: {str(e)}"
                })
    
    combined_result = {
        "response": all_responses
    }
    
    with open('sub_queries.json', 'w', encoding='utf-8') as f:
        json.dump(combined_result, f, ensure_ascii=False, indent=2)
    
    # return combined_result
    return all_responses