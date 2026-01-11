infer_model=deepseek-v31
infer_out_dir=output/response
# infer_out_dir=output/judge

eval_out_dir=output/judge
eval_score_path=output/scores.xlsx
eval_max_threads=30
python code/evalaute.py \
    --infer_model ${infer_model} \
    --in_dir ${infer_out_dir} \
    --out_dir ${eval_out_dir}  \
    --score_path ${eval_score_path} \
    --max_threads ${eval_max_threads} \
    --eval_model deepseek_v3