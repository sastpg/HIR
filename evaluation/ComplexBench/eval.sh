data_path=/gruntdata/zkc/code/iclr2026/ComplexBench/data/data_final.json
llm_output_path=/gruntdata/zkc/code/iclr2026/ComplexBench/llm_generations/deepseek-v3.1.jsonl
output_dir=evaluation
api_key="sk-c8e8edb9b99242bdae2264bbce13247d"
api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
model_name=deepseek-v3.1
language=zh


python3 evaluation/llm_based_extraction.py \
    --data_path $data_path \
    --llm_output_path $llm_output_path \
    --output_path "${output_dir}/${model_name}_llm_extraction_results.jsonl" \
    --api_key $api_key \
    --api_base $api_base \
    --language $language


python3 evaluation/llm_based_evaluation.py \
    --data_path $data_path \
    --llm_output_path $llm_output_path \
    --output_path "${output_dir}/${model_name}_llm_evaluation_results.jsonl" \
    --api_key $api_key \
    --api_base $api_base \
    --language $language


python3 evaluation/rule_based_evaluation.py \
    --data_path $data_path \
    --extraction_path "${output_dir}/${model_name}_llm_extraction_results.jsonl" \
    --output_path "${output_dir}/${model_name}_rule_evaluation_results.jsonl"


python3 evaluation/aggregation.py \
    --data_path $data_path \
    --llm_evaluation_path "${output_dir}/${model_name}_llm_evaluation_results.jsonl" \
    --rule_evaluation_path "${output_dir}/${model_name}_rule_evaluation_results.jsonl" \
    --model $model_name \
    --output_path $output_dir