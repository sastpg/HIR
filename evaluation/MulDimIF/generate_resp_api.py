import json
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os

api_key = ""
base_url = ""
client = OpenAI(api_key=api_key, base_url=base_url)

CONCURRENT_REQUESTS = 40

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Strip out whitespace and check if the line is not empty
            if line.strip():
                data.append(json.loads(line))
    return data


dataset = json.load(open("./data/test.json", "r", encoding='utf-8'))

def generate_resp(task):
    model_name = "deepseek-v3.1"
    temperature = 0.6
    message = task["conversations"][0]["content"]
    messages = [{"role":"user", "content": message}]
    i = 0
    response = "N/A"
    answer_content = "N/A"
    maxtry = 3
    while i < maxtry:
        try:
            i += 1
            response = client.chat.completions.create(
                model = model_name,
                messages=messages,
                extra_body={"enable_thinking": True},
                # temperature=temperature,
                stream=True,
                stream_options={
                    "include_usage": True
                },
            )
            # response = response.choices[0].message.content
            # print(response)
            reasoning_content = ""  # 完整思考过程
            answer_content = ""  # 完整回复
            is_answering = False  # 是否进入回复阶段

            for chunk in response:
                if not chunk.choices:
                    print(chunk.usage)
                    continue

                delta = chunk.choices[0].delta

                # 只收集思考内容
                if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                    if not is_answering:
                        pass
                    reasoning_content += delta.reasoning_content

                # 收到content，开始进行回复
                if hasattr(delta, "content") and delta.content:
                    if not is_answering:
                        is_answering = True
                    answer_content += delta.content

            break
        except Exception as e:
            print(f"Try {i}/{maxtry}\t message:{message} \tError:{e}", flush=True)
            i += 1
            time.sleep(2)
            continue
    return answer_content
    # return {"prompt": task["prompt"], "response": answer_content}

# Moonshot-Kimi-K2-Instruct
def generate_resp2(message):
    model_name = "Moonshot-Kimi-K2-Instruct"
    temperature = 0.6
    messages = [{"role":"user", "content": message}]
    i = 0
    response = "N/A"
    maxtry = 3
    while i < maxtry:
        try:
            i += 1
            responses = client.chat.completions.create(
                model = model_name,
                messages=messages,
                temperature=temperature,
            )
            response = responses.choices[0].message.content
            break
        except Exception as e:
            print(f"Try {i}/{maxtry}\t message:{message} \tError:{e}", flush=True)
            i += 1
            time.sleep(2)
            continue
    return response


def process_item_idealab(message):
    messages = [
        {
            "role": "user",
            "content": message
        }
    ]
    maxtry = 3
    while maxtry > 0:
        try:
            model = 'claude_sonnet4'#'gemini-2.5-flash-06-17'
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.6,
            )
            print(f"usage: {completion.usage}")
            res = completion.choices[0].message.content
            break
        except Exception as e:
            print(f"request ideatalk error: {e}")
            res = ""
            time.sleep(2)
            maxtry -= 1
    
    return res
    # return {"prompt": task["prompt"], "response": res}

import copy
from tqdm import tqdm
# for task in dataset:
#     res = generate_resp2(task["conversations"][0]["content"])
#     output_new = copy.deepcopy(task)
#     output_new["id"] = str(output_new["id"])
#     output_new["conversations"].append({"role": "assistant", "content": res})
#     with open("data/input_response_kimi_k2_ins.jsonl", 'a+', encoding="utf-8") as f:
#         f.write(json.dumps(output_new, ensure_ascii=False) + "\n")


with ThreadPoolExecutor(max_workers=35) as executor:
    results = list(tqdm(
        executor.map(generate_resp, dataset), 
        total=len(dataset),  # 总任务数
        desc="Processing"  # 进度条描述
    ))
    assert len(results) == len(dataset)

    with open("data/deepseek_v31_infer.jsonl", 'a+', encoding="utf-8") as f:
        for task, res in zip(dataset, results):
            output_new = copy.deepcopy(task)
            output_new["id"] = str(output_new["id"])
            output_new["conversations"].append({"role": "assistant", "content": res})
            f.write(json.dumps(output_new, ensure_ascii=False) + "\n")