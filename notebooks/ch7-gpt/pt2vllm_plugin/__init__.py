# -*- coding: utf-8 -*-
"""pt2vllm 插件:向 vLLM 注册自定义架构 GPT3Custom。"""


def register():
    """vLLM general plugin 入口:惰性注册模型架构,避免子进程 CUDA 初始化。"""
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model(
        "GPT3Custom",
        "pt2vllm_plugin.model:GPT3CustomForCausalLM",
    )
