from vllm import LLM, SamplingParams
import json
from transformers import AutoTokenizer

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Strip out whitespace and check if the line is not empty
            if line.strip():
                data.append(json.loads(line))
    return data


model_path = "model_name"
dataset = json.load(open("./data/cfbench_data.json",'r',encoding='utf-8'))


sampling_params = SamplingParams(temperature=0.6, max_tokens=4096)

# 加载模型
llm = LLM(model=model_path) # , enforce_eager=True
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

prompts = [tokenizer.apply_chat_template([{"role": "user", "content": item["prompt"]}], tokenize=False, add_generation_prompt=True) for item in dataset]
all_prompts_token_ids = tokenizer(prompts, padding=False, add_special_tokens=False, max_length=4096, truncation=True)["input_ids"]
# outputs = llm.generate(prompts, sampling_params=sampling_params)
outputs = llm.generate(sampling_params=sampling_params, prompt_token_ids=all_prompts_token_ids)
responses = [item.outputs[0].text for item in outputs]

new = []
for content, resp in zip(dataset, responses):
    content["response"] = resp
    new.append(content)
json.dump(new, open("output/response/qwen25_7b_rl_infer.json",'w',encoding='utf-8'),ensure_ascii=False,indent=4)