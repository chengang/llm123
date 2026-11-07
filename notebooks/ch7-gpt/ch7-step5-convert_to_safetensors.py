#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 ch7-step3-gpt-train.py 训练得到的 ckpt_best.pt 转成 vLLM 可直接 serve 的目录:

  vllm_model/
    model.safetensors        # bf16,键名与 ckpt 一一对应(扁平命名,nn.Linear 布局)
    config.json              # 复用 HF GPT2Config,architectures=["GPT3Custom"]
    tokenizer.model          # 原始 SentencePiece BPE
    tokenizer_config.json    # LlamaTokenizer 包装,add_bos_token=false
    special_tokens_map.json
    generation_config.json

用法: .venv/bin/python convert_ckpt_to_safetensors.py
"""

import json
import os
import shutil

import torch
from safetensors.torch import save_file

import sentencepiece as spm

# -----------------------------------------------------------------------------
# 路径
# -----------------------------------------------------------------------------
CKPT_PATH = "./ckpt_best.pt"
TOKENIZER_PATH = "./tokenizer.model"
OUT_DIR = "./vllm_model"


def strip_compile_prefix(state):
    """torch.compile 会给键名加 _orig_mod. 前缀,去掉它。"""
    if any(k.startswith("_orig_mod.") for k in state):
        return {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    return state


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"加载 checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    raw_cfg = ckpt.get("config", {})

    sd = strip_compile_prefix(ckpt["model"])
    n = len(sd)
    print(f"  读取到 {n} 个张量,dtype={set(t.dtype for t in sd.values())}")
    print(f"  是否含 lm_head.weight: {'lm_head.weight' in sd}")
    print(f"  是否含 wpe.weight:    {'wpe.weight' in sd}")

    # 转 bf16
    bf16_sd = {k: v.to(torch.bfloat16).contiguous() for k, v in sd.items()}

    safetensors_path = os.path.join(OUT_DIR, "model.safetensors")
    print(f"写出 {safetensors_path} (bf16, {n} 个张量)")
    save_file(bf16_sd, safetensors_path, metadata={"format": "pt"})

    # 释放
    del sd, bf16_sd, ckpt

    # -------------------------------------------------------------------------
    # config.json —— 复用 HF GPT2Config,只把 architectures 改成我们的插件架构名
    # -------------------------------------------------------------------------
    vocab_size = raw_cfg.get("vocab_size", 64000)
    d_model = raw_cfg.get("d_model", 2048)
    n_layer = raw_cfg.get("n_layer", 24)
    n_head = raw_cfg.get("n_head", 16)
    d_ff = raw_cfg.get("d_ff", 4 * d_model)
    block_size = raw_cfg.get("block_size", 4096)

    config = {
        "architectures": ["GPT3Custom"],
        "model_type": "gpt2",
        "vocab_size": vocab_size,
        "n_embd": d_model,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_inner": d_ff,
        "n_positions": block_size,
        "n_ctx": block_size,
        "layer_norm_epsilon": 1e-5,
        "activation_function": "gelu_new",
        "tie_word_embeddings": True,
        "add_cross_attention": False,
        "scale_attn_by_inverse_layer_idx": False,
        "reorder_and_upcast_attn": False,
        "torch_dtype": "bfloat16",
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 2,            # 用 eos 当 pad(本项目无单独 pad 需求)
        "transformers_version": "4.46.0",
    }
    with open(os.path.join(OUT_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print("写出 config.json")

    # -------------------------------------------------------------------------
    # tokenizer —— SentencePiece BPE,用 LlamaTokenizer 包装
    # 训练时仅在文档末尾追加 eos,不在开头加 bos,故 add_bos_token=false
    # 特殊 token 字符串从 sp 模型动态读取,避免写死错
    # -------------------------------------------------------------------------
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    unk_piece = sp.id_to_piece(sp.unk_id())     # 0 -> "<unk>"
    bos_piece = sp.id_to_piece(sp.bos_id())     # 1 -> "<s>"
    eos_piece = sp.id_to_piece(sp.eos_id())     # 2 -> "</s>"
    pad_piece = sp.id_to_piece(sp.pad_id())     # 3 -> "<pad>"
    print(f"  sp 特殊 piece: unk={unk_piece!r} bos={bos_piece!r} "
          f"eos={eos_piece!r} pad={pad_piece!r}  (vocab={sp.vocab_size()})")

    shutil.copyfile(TOKENIZER_PATH, os.path.join(OUT_DIR, "tokenizer.model"))

    tokenizer_config = {
        "tokenizer_class": "LlamaTokenizer",
        "model_max_length": block_size,
        "add_bos_token": False,
        "add_eos_token": False,
        "use_default_system_prompt": False,
        "legacy": False,
        "bos_token": bos_piece,
        "eos_token": eos_piece,
        "unk_token": unk_piece,
        "pad_token": pad_piece,
    }
    with open(os.path.join(OUT_DIR, "tokenizer_config.json"), "w",
              encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)

    special_tokens_map = {
        "bos_token": bos_piece,
        "eos_token": eos_piece,
        "unk_token": unk_piece,
        "pad_token": pad_piece,
    }
    with open(os.path.join(OUT_DIR, "special_tokens_map.json"), "w",
              encoding="utf-8") as f:
        json.dump(special_tokens_map, f, indent=2, ensure_ascii=False)
    print("写出 tokenizer.model / tokenizer_config.json / special_tokens_map.json")

    generation_config = {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 2,
        "do_sample": True,
        "transformers_version": "4.46.0",
    }
    with open(os.path.join(OUT_DIR, "generation_config.json"), "w",
              encoding="utf-8") as f:
        json.dump(generation_config, f, indent=2, ensure_ascii=False)
    print("写出 generation_config.json")

    print(f"\n完成。产物目录: {OUT_DIR}/")
    print("下一步: uv pip install -e .  然后  vllm serve ./vllm_model "
          "--dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.9")


if __name__ == "__main__":
    main()
