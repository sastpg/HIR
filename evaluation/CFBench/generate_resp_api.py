import json
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm

api_key = ""
base_url = ""
client = OpenAI(api_key=api_key, base_url=base_url)

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Strip out whitespace and check if the line is not empty
            if line.strip():
                data.append(json.loads(line))
    return data


dataset = json.load(open("./data/cfbench_data.json",'r',encoding='utf-8'))

def generate_resp(task):
    model_name = "deepseek-v3.1"
    temperature = 0.6
    message = task["prompt"]
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
                    # print(chunk.usage)
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
            model = 'gemini-2.5-flash-06-17'
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

# import copy
# for task in dataset:
#     res = process_item_idealab(task["prompt"])
#     task["response"] = res
#     with open("output/response/gemini25flash_infer.jsonl", 'a+', encoding="utf-8") as f:
#         f.write(json.dumps(task, ensure_ascii=False) + "\n")

with ThreadPoolExecutor(max_workers=30) as executor:
    results = list(tqdm(
        executor.map(generate_resp, dataset), 
        total=len(dataset),  # 总任务数
        desc="Processing"  # 进度条描述
    ))
    assert len(results) == len(dataset)

    with open("output/response/deepseek-v31_infer.jsonl", 'a+', encoding="utf-8") as f:
        for task, res in zip(dataset, results):
            task["response"] = res
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

