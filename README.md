<div align='center'>
<h2>Replay Failures as Successes:<br>Sample-Efficient Reinforcement Learning for Instruction Following</h2>

<!-- TODO:  Thread,Paper,Dataset,Weights-->
[![Paper](https://img.shields.io/badge/paper-5f16a8?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2512.23457)
[![Dataset](https://img.shields.io/badge/Datasets-4d8cd8?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/sastpg/HIR-16K)
</div>

## Overview
Reinforcement Learning (RL) has shown promise for aligning Large Language Models (LLMs) to follow instructions with various constraints. Despite the encouraging results, RL improvement inevitably relies on sampling successful, high-quality responses; however, the initial model often struggles to generate responses that satisfy all constraints due to its limited capabilities, yielding sparse or indistinguishable rewards that impede learning. In this work, we propose ***H***indsight ***i***nstruction ***R***eplay (HiR), a novel sample-efficient RL framework for complex instruction following tasks, which employs a *select*-then-*rewrite* strategy to *replay failed attempts as successes* based on the constraints that have been satisfied in hindsight. We perform RL on these replayed samples as well as the original ones, theoretically framing the objective as dual-preference learning at both the instruction- and response-level to enable efficient optimization using only a binary reward signal.

![](./images/framework.png)

## Requirements

We recommend using the official docker provided by verl. Besides, it needs to install the following package additionally:

- langdetect
- nltk
- immutabledict
- emoji
- syllapy

## Key Modification

The core strategies and modification of HiR are implemented in:

- `verl/trainer/ppo/ray_trainer.py`
- `verl/utils/reward_score/if_score`

We demonstrate the key selection and rewrite part modification in the following:

**SELECT**

```python
# NOTE: Alg. Selection of trajectory for replay
# NOTE: F_int = CLA, F_div = entropy
all_valid_replay_index = [i for i, value in enumerate(reward_extra_infos_dict["dense"]) if 0<value<1]
trajectory_score = []
_lambda = (1.05 ** self.global_steps) * 2.0
for idx in all_valid_replay_index:
    ent = agg_loss(loss_mat=entropys[idx], loss_mask=batch[idx].batch["response_mask"], loss_agg_mode=loss_agg_mode)
    trajectory_score.append(_lambda * reward_extra_infos_dict["dense"][idx] + ent)
indices = np.argpartition(np.array(trajectory_score), -num_replay)[-num_replay:]
replay_index = [all_valid_replay_index[i] for i in indices]
```

**REWRITE**

```python
for index in sorted(replay_index, reverse=True):
    assert index < len(batch)
    # replay
    raw_prompt = batch[index].non_tensor_batch["extra_info"]["question"]
    full_criteria = batch[index].non_tensor_batch["extra_info"]["criteria"]
    judge_list = reward_extra_infos_dict["judge_list"][index]
    statsified_criteria = [full_criteria[ii] for ii, judge in enumerate(judge_list) if judge]
    assert 0 < len(statsified_criteria) < len(full_criteria)
    new_prompt = self._concat_question(raw_prompt, statsified_criteria)
    new_msg = {"role": "user", "content": new_prompt}

    rewrite_prompt = self.tokenizer.apply_chat_template([new_msg], add_generation_prompt=True, tokenize=False)
    model_inputs = self.tokenizer(rewrite_prompt, return_tensors="pt", add_special_tokens=False)
    
    # ... omit

    new_experience = TensorDict(
        {
            "attention_mask": attention_mask,
            "input_ids": input_ids,
            # \pi_{old} (y | q)
            "old_log_probs": batch[index].batch["old_log_probs"].unsqueeze(0),
            "position_ids": position_ids,
            "prompts": prompt_ids,
            "ref_log_prob": batch[index].batch["ref_log_prob"].unsqueeze(0),
            "response_mask": batch[index].batch["response_mask"].unsqueeze(0),
            "responses": batch[index].batch["responses"].unsqueeze(0),
            "is_replay": torch.ones(1,1),
        },
        torch.Size([1])
    )
```



## Acknowledgement

We thank the [verl](https://github.com/volcengine/verl) for providing the awesome open-source RL infrastructure. We also thank the developers of [Qwen](https://github.com/QwenLM) and [Llama](https://github.com/meta-llama) for their awesome open-source models.
