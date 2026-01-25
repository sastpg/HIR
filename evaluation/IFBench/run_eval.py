# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Binary of evaluating instruction following. See README.md."""

import os
from typing import Sequence

from absl import app
from absl import flags
from absl import logging

import evaluation_lib


_INPUT_DATA = flags.DEFINE_string(
    "input_data", None, "path to input data", required=True
)

_INPUT_RESPONSE_DATA = flags.DEFINE_string(
    "input_response_data", None, "path to input response data", required=False
)

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "Output directory for inference and eval results.",
    required=True,
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  inputs = evaluation_lib.read_prompt_list(_INPUT_DATA.value)
  prompt_to_response = evaluation_lib.read_prompt_to_response_dict(
      _INPUT_RESPONSE_DATA.value)

  # get instruction following results
  for func, output_file_name in [
      (evaluation_lib.test_instruction_following_strict, "eval_results_strict"),
      (evaluation_lib.test_instruction_following_loose, "eval_results_loose"),
  ]:
    logging.info("Generating %s...", output_file_name)
    outputs = []
    for inp in inputs:
      outputs.append(func(inp, prompt_to_response))
    follow_all_instructions = [o.follow_all_instructions for o in outputs]
    accuracy = sum(follow_all_instructions) / len(outputs)
    logging.info("Accuracy: %f", accuracy)

    output_file_name = os.path.join(
        _OUTPUT_DIR.value, output_file_name + ".jsonl"
    )
    evaluation_lib.write_outputs(output_file_name, outputs)
    logging.info("Generated: %s", output_file_name)

    # Prints instruction following accuracy report.
    print("=" * 64)
    print(f"{output_file_name} Accuracy Scores:")
    evaluation_lib.print_report(outputs)


if __name__ == "__main__":
  app.run(main)


First Draft, Then Revise: Reinforcement Learning for LLM Reasoning with online experience guided exploration
First Solve, Then Polish: Online Experience drives effective exploration in Reinforcement Learning for LLMs
First Draft, Then Revise: Online Experience-guided exploration induces effective Reinforcement Learning for LLMs
Consistent Paths Lead to Truth: Self-Rewarding Reinforcement Learning for LLM Reasoning
Breaking the Exploration Bottleneck: Rubric-Scaffolded Reinforcement Learning for General LLM Reasoning
Replay Failures as Successes: Sample-Efficient Reinforcement Learning for Instruction Following
Consistent Paths Lead to Truth: Self-Rewarding Reinforcement Learning for LLM Reasoning
SeRL: Self-Play Reinforcement Learning for Large Language Models with Limited Data
MUSE: MCTS-Driven Red Teaming Framework for Enhanced Multi-Turn Dialogue Safety in Large Language Models
Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning
Satori: Reinforcement Learning with Chain-of-Action-Thought Enhances LLM Reasoning via Autoregressive Search
First Draft, Then Refine: Online Experience Drives Effective Exploration in Reinforcement Learning for LLMs

Experience is the Best Teacher: Motivating Effective Exploration in Reinforcement Learning for LLMs
Experience is the Best Teacher: Online Experience drives effective exploration in Reinforcement Learning for LLMs


# 论文 Story
## 第一段：从大模型->RLVR范式
1. LLM现在很厉害，在数学分析、代码编程、机器人控制等等都展现了很强的能力
2. Deepseek-R1及其后续工作表明仅仅通过RLVR就可以大幅提升模型的推理能力

## 第二段：RLVR->rubrics
1. 尽管RLVR取得了很好的进展，但是在open-ended的task上仍然困难，因为没有明确规则的可验证答案
2. 因此非常近期有研究使用rubrics作为评分标准，使用LLM-as-a-Judge范式打分进行RL训练

## 第三段：讲探索问题
1. 但是现有的范式仍然存在探索难的问题，一些工作比如FR3E，TreeRL依靠entropy或者树搜索，但是还是需要大量rollout不高效的探索
2. Intro开头的奖励假说，RL优化实际上可以被视为引导策略朝着由奖励空间定义的理想分布方向优化，表明有效的探索应该朝着奖励期望的行为努力，而不是entropy那样胡乱探索
3. 幸运的是，最近rubric-based reward范式使得奖励的语言描述成为可能，那么，关键问题就是如何利用好此类反馈来指导有效探索（让努力和目标一致），而不是在采样时依赖低效的多轮试错。


## 第四段：介绍我们的方法
1. 我们提出了HeRL，具体是咋做的，有什么好处（参考摘要）， LLMs能够从初始生成和奖励的事后经验中学习