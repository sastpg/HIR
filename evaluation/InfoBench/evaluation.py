import json
import os
import time
import tiktoken
import argparse

from os.path import join,exists
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


encoder = tiktoken.get_encoding("cl100k_base")

SYS_MSG ="Based on the provided Input (if any) and Generated Text, answer the ensuing Questions with either a YES or NO choice. Your selection should be based on your judgment as well as the following rules:\n\n- YES: Select 'YES' if the generated text entirely fulfills the condition specified in the question. However, note that even minor inaccuracies exclude the text from receiving a 'YES' rating. As an illustration. consider a question that asks. \"Does each sentence in the generated text use a second person?” If even one sentence does not use the second person, the answer should NOT be 'YES'. To qualify for a 'YES' rating, the generated text must be entirely accurate and relevant to the question\n\n- NO: Opt for 'NO' if the generated text fails to meet the question's requirements or provides no information that could be utilized to answer the question. For instance, if the question asks. \"Is the second sentence in the generated text a compound sentence?\" and the generated text only has one sentence. it offers no relevant information to answer the question. Consequently, the answer should be 'NO'.'''"

def load_jsonl(file_path):
    "General function to load jsonl file"
    _data = []
    with open(file_path, 'r') as f:
        for data in f:
            jline = json.loads(data)
            _data.append(jline)
    return _data


def bool_ratio2(fpath):
    "Calculate true false ratio for eval results"
    _data = load_jsonl(fpath)
    count = {"true":0, "false":0}
    visited_ids = set()

    instruction_level_hard =  {"true":0, "false":0}
    instruction_level_easy =  {"true":0, "false":0}
    prompt_level_hard =  {"true":0, "false":0}
    prompt_level_easy =  {"true":0, "false":0}

    for entry in _data:
        if entry['id'] in visited_ids:
            continue

        if entry.get("eval", None) is None:
            print("Wrong output")
            print(entry['id'])

        if len(entry['decomposed_questions']) != len(entry['eval']):
            print("Wrong length")
            print(entry['id'])

        if None in entry['eval']:
            print("None in eval")
            print(entry['id'])
        
        num_instruction = len(entry['eval'])
        num_correct = 0
        for eva_value in entry['eval']:
            if eva_value:
                count["true"] += 1
                num_correct += 1
            else:
                count["false"] += 1
        if num_correct == num_instruction:
            prompt_correct = 1
        else:
            prompt_correct = 0
        
        if "easy" in (entry["subset"]).lower():
            instruction_level_easy["true"] += num_correct
            instruction_level_easy["false"] += (num_instruction-num_correct)
            prompt_level_easy["true"] += prompt_correct
            prompt_level_easy["false"] += (1-prompt_correct)

        else:
            instruction_level_hard["true"] += num_correct
            instruction_level_hard["false"] += (num_instruction-num_correct)
            prompt_level_hard["true"] += prompt_correct
            prompt_level_hard["false"] += (1-prompt_correct)

        visited_ids.add(entry['id'])
    
    print("-------- True False Table --------")
    print(count)
    print(f"Percentage of True: {count['true']/(sum(count.values())+1e-4)}")

    stats_jsonl_path = fpath + "_score.jsonl"
    res = {}
    res["overall"] = float(count['true']/(sum(count.values())+1e-4))
    res["count"] = count
    instruction_level_easy_true = instruction_level_easy["true"]/(sum(instruction_level_easy.values())+1e-4)
    prompt_level_easy_true = prompt_level_easy["true"]/(sum(prompt_level_easy.values())+1e-4)
    instruction_level_hard_true = instruction_level_hard["true"]/(sum(instruction_level_hard.values())+1e-4)
    prompt_level_hard_true = prompt_level_hard["true"]/(sum(prompt_level_hard.values())+1e-4)
    
    instruction_level_all_true = (instruction_level_easy["true"]+instruction_level_hard["true"])/(sum(instruction_level_easy.values())+sum(instruction_level_hard.values())+1e-4)
    prompt_level_all_true = (prompt_level_easy["true"]+prompt_level_hard["true"])/(sum(prompt_level_easy.values())+sum(prompt_level_hard.values())+1e-4)
    
    res["prompt_total_count"] = sum(prompt_level_hard.values()) + sum(prompt_level_easy.values())
    assert(res["prompt_total_count"]==500)
    res["instruction_level_easy_true"] = instruction_level_easy_true
    res["prompt_level_easy_true"] = prompt_level_easy_true
    res["instruction_level_hard_true"] = instruction_level_hard_true
    res["prompt_level_hard_true"] = prompt_level_hard_true
    res["instruction_level_all_true"] = instruction_level_all_true
    res["prompt_level_all_true"] = prompt_level_all_true

    with open(stats_jsonl_path, "w") as fw:
        fw.write(json.dumps(res, ensure_ascii=False)+"\n")
    
    return

def bool_ratio(fpath):
    "Calculate true false ratio for eval results"
    _data = load_jsonl(fpath)
    count = {"true":0, "false":0}
    for entry in _data:
        if entry.get("eval", None) is None:
            print("Wrong output")
            print(entry['id'])
        if len(entry['decomposed_questions']) != len(entry['eval']):
            print("Wrong length")
            print(entry['id'])
        if None in entry['eval']:
            print("None in eval")
            print(entry['id'])
        
        for eva_value in entry['eval']:
            if eva_value:
                count["true"] += 1
            else:
                count["false"] += 1
    
    print("-------- True False Table --------")
    print(count)
    print(f"Percentage of True: {count['true']/sum(count.values())}")
    return

def _judge_one(entry):
    if entry.get('eval', None) is not None:
        return entry
    
    input_task = entry['input']
    output = entry['output']
    if output is None: # skip if result hasn't been generated
        raise ValueError("Result hasn't been generated")
    
    message = []
    answer = ""
    # print(f"--------Instance {entry['id']}--------")
    for question in entry['decomposed_questions']:
        if len(message) == 0:
            if input_task:
                content =  f"{SYS_MSG}\n\nInput:\n\"{input_task}\"\n\nGenerated Text:\n\"{output}\"\n\nQuestion:\n{question}\n"
            else:
                content =  f"{SYS_MSG}\n\nGenerated Text:\n\"{output}\"\n\nQuestion:\n{question}\n"
        else:
            content = f"{question}\n"
        # content =  f"{SYS_MSG}\n\nInput:\n\"{input_task}\"\n\nGenerated Text:\n\"{output}\"\n\nQuestion:\n{question}\n"
        # message = [{"role": "user", "content": content}]
        message.append({"role": "user", "content": content})
        # create a chat completion
        early_stop = False
        max_retry = 3
        while max_retry > 0:
            try:
                completion = client.chat.completions.create(
                    model=eval_model,
                    messages=message,
                    temperature=temperature,
                )
                generation = completion.choices[0].message.content
                message.append(
                    {"role": "assistant", "content": generation})
                # check if generation is yes or no
                if generation.lower().startswith("yes") or generation.lower().startswith("no"):
                    if generation.lower().startswith("yes"):
                        answer += "Yes\n"
                    else:
                        answer += "No\n"
                else:
                    if "YES" in generation and "NO" not in generation:
                        answer += "Yes\n"
                    elif "YES" not in generation and "NO" in generation:
                        answer += "No\n"
                    else:
                        # for msg in message:
                        #     print(msg['content'])
                        # print("NO YES or NO answer!" + generation)
                        answer += "None\n"
                break
            except Exception as e:
                print("ERROR!")
                print(e)
                print("Retry!")
                time.sleep(2)
                max_retry -= 1

        # when no answer occurs, break the loop and continue to next instance
        # if early_stop:
        #     break

    answer = answer[:-1]
    # save eval results as List[bool]
    bool_results = []
    for i in answer.split('\n'):
        if i == "Yes":
            bool_results.append(True)
        elif i == "No":
            bool_results.append(False)
        else:
            bool_results.append(None)

    entry['eval'] = bool_results
    return entry

def run_evaluation(client, in_path, o_dir, eval_model="gpt-4-0314", temperature=0):
    """
    Main function to run decomposed questisons evaluation on models' outputs
        in_path: str, path to the model output file
        o_dir: str, path to the output folder
        eval_model: str, default "gpt-4-0314", model name to be used for evaluation
        temperature: float, default 0, temperature to be used for evaluation
    """
    _data = load_jsonl(in_path)
    _model_name = in_path.split('/')[1].split('_')[0]
    
    # ceate output folder if not exists
    _o_dir = join(o_dir, eval_model)
    if not exists(_o_dir):
        os.mkdir(_o_dir)

    _opath = join(_o_dir, f"{_model_name}_DecomposeEval.jsonl")
    
    # load_results if exists
    if os.path.exists(_opath):
        _exist = load_jsonl(_opath)
        _exist_ids = [i['id'] for i in _exist]
        for pos, instance in enumerate(_data):
            if instance['id'] in _exist_ids:
                _data[pos] = _exist[_exist_ids.index(instance['id'])]
    
    # result_writer = open(_opath, 'w')
    
    print(f"--------Evaluating output from {in_path}--------")
    print(f"--------Evaluation Using {eval_model}--------")


    results = []
    with ThreadPoolExecutor(20) as executor:
        for entry in tqdm(executor.map(_judge_one, _data), total=len(_data), desc=f'eval'):
            results.append(entry)
    if None in results:
        raise ValueError("Some entries are not annotated due to errors in judge_one, please inspect and retry.")
    
    
    scores = []
    with open(_opath, 'a+') as f:
        for item in results:
            scores.append(all(item["eval"]))
            json.dump(item, f)
            f.write('\n')
    print(f"Acc: {sum(scores)/len(scores)}")

    # run true false ratio calculation
    # bool_ratio(_opath)
    bool_ratio2(_opath)
    
    return _opath

def main_run(args):
    if not exists(results_file):
        print(f"results_dir {results_file} not exists")
        return
    
    # run evaluation for each model
    run_evaluation(client, results_file, output_dir, eval_model, temperature) 
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default="sk-77fcc652744f42fc9ef3448c757f5c49") #  sk-c8e8edb9b99242bdae2264bbce13247d
    parser.add_argument("--model", type=str, default="deepseek-v3", help="model name to be used for evaluation")
    
    parser.add_argument("--input", type=str, required=True, help="path to the results file")
    parser.add_argument("--output_dir", type=str, required=True, help="path to the output folder")
    
    parser.add_argument("--temperature", type=float, default=0.6, help="temperature to be used for evaluation")
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    results_file = args.input
    output_dir = args.output_dir
    eval_model = args.model
    temperature = args.temperature

    main_run(args)