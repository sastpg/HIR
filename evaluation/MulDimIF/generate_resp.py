from vllm import LLM, SamplingParams
import json
from transformers import AutoTokenizer
import copy
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Strip out whitespace and check if the line is not empty
            if line.strip():
                data.append(json.loads(line))
    return data


model_path = ""
dataset = json.load(open("./data/test.json", "r", encoding='utf-8'))


sampling_params = SamplingParams(temperature=0.6, max_tokens=3000)

# 加载模型
llm = LLM(model=model_path, max_model_len=4096) # , enforce_eager=True
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# prompts = [item["prompt"] for item in dataset]
prompts = [tokenizer.apply_chat_template(item["conversations"], tokenize=False, add_generation_prompt=True) for item in dataset]

all_prompts_token_ids = tokenizer(prompts, padding=False, add_special_tokens=False, max_length=4096, truncation=True)["input_ids"]
# outputs = llm.generate(prompts, sampling_params=sampling_params)
outputs = llm.generate(sampling_params=sampling_params, prompt_token_ids=all_prompts_token_ids)

responses = [item.outputs[0].text for item in outputs]

prompts = [item for item in dataset]
for prompt, resp in zip(prompts, responses):
    output_new = copy.deepcopy(prompt)
    output_new["id"] = str(output_new["id"])
    output_new["conversations"].append({"role": "assistant", "content": resp})
    with open("data/input_response_data_qwen3_4b_hir_dpo_ckpt246.jsonl", "a+") as f:
        f.write(json.dumps(output_new, ensure_ascii=False) + "\n")
