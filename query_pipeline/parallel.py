import yaml
import dotenv
import torch
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from routing_prompts import *
import json
import datetime
import concurrent.futures
from typing import List, Dict
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from queries import SAMPLE_QUERIES

def load_prompt(system_prompt: str) -> PromptTemplate:
    '''
    프롬프트를 받아, LangChain의 PromptTemplate 클래스를 생성하여 반환하는 함수
    system_prompt: str, 시스템 프롬프트
    '''
    template = f"""{system_prompt}
    현재 시각: {{current_time}}

    쿼리: {{user_input}}

    답변:"""
    return PromptTemplate.from_template(template)

def process_single_query(query: str, chain) -> Dict:
    '''
    단일 쿼리를 처리하는 함수
    query: str, 처리할 단일 쿼리
    chain: LangChain chain, 사용할 체인
    '''
    try:
        result = chain.invoke({"current_time":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"user_input":query})
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
    chat_prompt = load_prompt(PARALLEL)
    
    # Model Selection
    model_name = "recoilme/recoilme-gemma-2-9B-v0.4"  # Change model name!


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=0.15, # Change it up
        max_new_tokens=512,
        top_p=0.9, # Change it up
        eos_token_id=terminators,
        repetition_penalty = 1.1, # Change it up
        model_kwargs={"torch_dtype": torch.bfloat16}
    )
    
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    chain = (
        chat_prompt
        | llm
        | StrOutputParser()
    )
    
    all_responses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(subqueries), 10)) as executor:
        futures = {executor.submit(process_single_query, query, chain): query for query in subqueries}
        
        for future in concurrent.futures.as_completed(futures):
            query = futures[future]
            try:
                result = future.result()
                print(result)
                json_start = result.find("답변: {") + 4
                json_end = json_start
                while result[json_end] != "}":
                    json_end += 1
                json_end += 1
                json_str = result[json_start:json_end]
                print("\n==============================================================\njson_str:\n\n\n")
                print(json_str)
                all_responses.append(json.loads(json_str))
                print(f"Processed query: {query}")
            except Exception as e:
                print(f"Query '{query}' generated exception: {str(e)}")
                all_responses.append({
                    "subquery": query,
                    "routing": "False",
                    "reasoning": f"Error processing: {str(e)}"
                })
    
    print("All queries processed!")
    with open('sub_queries.json', 'w', encoding='utf-8') as f:
        json.dump({"response": all_responses}, f, ensure_ascii=False, indent=2)
    
    return all_responses

if __name__ == "__main__":
    prompt_routing(SAMPLE_QUERIES[0:3])
