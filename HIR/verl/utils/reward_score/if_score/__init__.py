from verl.utils.reward_score.if_score.ifeval.evaluation_lib import test_ifeval_strict
from verl.utils.reward_score.if_score.ifbench.evaluation_lib import test_ifbench_strict
from verl.utils.reward_score.if_score.muldimif.evaluation import test_muldimif_strict

import json

def compute_score(model_output: str, ground_truth, type) -> bool:
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)

    try:
        if type == "type1":
            sparse_score, dense_score, is_following_list = test_ifeval_strict(model_output, ground_truth)
            return {"score": sparse_score, "sparse": sparse_score, "dense": dense_score, "judge_list": is_following_list}
        elif type == "type2":
            sparse_score, dense_score, is_following_list = test_ifbench_strict(model_output, ground_truth)
            return {"score": sparse_score, "sparse": sparse_score, "dense": dense_score, "judge_list": is_following_list}
        elif type == "type3":
            sparse_score, dense_score, is_following_list = test_muldimif_strict(model_output, ground_truth)
            return {"score": sparse_score, "sparse": sparse_score, "dense": dense_score, "judge_list": is_following_list}
        else:
            raise Exception("Invalid type")
    except:
        print("rule verify timeout!")
        return 0
