import os
import json
from argparse import ArgumentParser
import numpy as np

from utils import convert_to_api_input, add_model_args
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os
from tqdm import tqdm

api_key = "sk-c8e8edb9b99242bdae2264bbce13247d"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1" # 'https://idealab.alibaba-inc.com/api/openai/v1'
client = OpenAI(api_key=api_key, base_url=base_url)
# os.environ['OPENAI_API_KEY'] = '030cd8aa1c3691559c60a5336922900b'
# client = OpenAI(base_url=base_url)


def generate_resp(task):
    model_name = "deepseek-v3.1"
    temperature = 0.6
    message = task["prompt_new"]
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
    # 请用给定的ak，联系@黄苍
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
                max_tokens=2048
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



# def generate_responses(data, args, constraint_type):
#     for example in data:
#         responses = process_item_idealab(example['prompt_new'])
#         with open(f"{args.api_output_path}/{args.model_path}/{constraint_type}_constraint.jsonl", 'a+', encoding='utf-8') as output_file:
#             o = {'prompt': example['prompt_new'], "choices": [{"message": {"content": responses}}]}
#             output_file.write(json.dumps(o) + "\n")



def run_inference(args):
    args.max_tokens
    for constraint_type in args.constraint_types:
        # data = []
        # with open(os.path.join(args.api_input_path, f"{constraint_type}_constraint.jsonl"), 'r', encoding='utf-8') as data_file:
        #     for line in data_file:
        #         data.append(json.loads(line))

        # os.makedirs(f"{args.api_output_path}/{args.model_path}", exist_ok=True)
        # generate_responses(data, args, constraint_type)

        data = []
        with open(os.path.join(args.api_input_path, f"{constraint_type}_constraint.jsonl"), 'r', encoding='utf-8') as data_file:
            for line in data_file:
                data.append(json.loads(line))

        with ThreadPoolExecutor(max_workers=35) as executor:
            # 结果顺序与输入顺序一致，即使执行完成顺序不同
            output_list = list(tqdm(
                executor.map(generate_resp, data), 
                total=len(data),  # 总任务数
                desc="Processing"  # 进度条描述
            ))
            assert len(output_list) == len(data)


        os.makedirs(f"{args.api_output_path}/{args.model_path}", exist_ok=True)
        with open(f"{args.api_output_path}/{args.model_path}/{constraint_type}_constraint.jsonl", 'w', encoding='utf-8') as output_file:
            for idx, res in enumerate(output_list):
                o = {'prompt': data[idx]['prompt_new'], "choices": [{"message": {"content": res}}]}
                output_file.write(json.dumps(o) + "\n")




def main():
    parser = ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--constraint_types", nargs='+', type=str, default=['content', 'situation', 'style', 'format', 'example', 'mixed'])
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--api_input_path", type=str, default="api_input")
    parser.add_argument("--api_output_path", type=str, default="api_output_vllm")

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()

    os.makedirs(args.api_input_path, exist_ok=True)
    for constraint_type in args.constraint_types:
        convert_to_api_input(
                            data_path=args.data_path, 
                            api_input_path=args.api_input_path, 
                            constraint_type=constraint_type
                            )

    run_inference(args)

if __name__ == '__main__':
    main()