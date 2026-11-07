#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线验证 GPT3Custom 插件:加载 vllm_model 并生成一段文本。"""

from vllm import LLM, SamplingParams


def main():
    llm = LLM(
        model="./vllm_model",
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
    )

    prompts = ["从前有座山，山里有座庙，"]
    sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=80, stop=None)
    outs = llm.generate(prompts, sp)
    for o in outs:
        print("=" * 60)
        print("PROMPT:", o.prompt)
        print("OUTPUT:", repr(o.outputs[0].text))
        print("finish_reason:", o.outputs[0].finish_reason,
              "tokens:", len(o.outputs[0].token_ids))


if __name__ == "__main__":
    main()
