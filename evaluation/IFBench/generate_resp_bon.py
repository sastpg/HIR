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


model_path = "/shared-nas/zkc/checkpoints/llama_3b_rule_sparse/global_step_400/actor/huggingface/"#"/shared-nas/zkc/Qwen2.5-3B-Instruct/"
dataset = read_jsonl("/gruntdata/zkc/code/iclr2026/IFBench/data/IFBench_test.jsonl")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# prompts = [item["prompt"] for item in dataset]
prompts = sum([[tokenizer.apply_chat_template([{"role": "user", "content": item["prompt"]}], tokenize=False, add_generation_prompt=True)] * 1024 for item in dataset], [])

# 加载模型
llm = LLM(model=model_path) # , enforce_eager=True
sampling_params = SamplingParams(temperature=1.0, max_tokens=4096)


for index in range(0, len(prompts), 1024):
    all_prompts_token_ids = tokenizer(prompts[index:index+1024], padding=False, add_special_tokens=False, max_length=4096, truncation=True)["input_ids"]
    # outputs = llm.generate(prompts, sampling_params=sampling_params)
    outputs = llm.generate(sampling_params=sampling_params, prompt_token_ids=all_prompts_token_ids)

    responses = [item.outputs[0].text for item in outputs]
    for resp in responses:
        with open("data/input_response_data_llama_3b_rlvr_pass1024.jsonl", "a+") as f:
            f.write(json.dumps({"prompt": index // 1024, "response": resp}, ensure_ascii=False) + "\n")
