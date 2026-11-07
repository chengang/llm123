# -*- coding: utf-8 -*-
"""
GPT3Custom —— 与 ch7-step3-gpt-train.py 中 GPT3 网络结构一一对应的 vLLM 模型。

网络结构(经典 GPT-2/GPT-3 decoder):
  - wte : token embedding (vocab=64000, d_model=2048)
  - wpe : 绝对学习位置编码 (block_size=4096)
  - 24 个 Block: pre-LayerNorm + 因果自注意力(c_attn 融合 QKV / c_proj)+ MLP(c_fc / c_proj,gelu tanh 近似)
  - ln_f : 末层 LayerNorm
  - lm_head : 与 wte 权重共享(tie_embeddings)

权重布局与 ckpt_best.pt 完全一致:
  - 用的是 nn.Linear(非 HF Conv1D),故权重为标准 [out, in],无需转置
  - 键名扁平(wte / wpe / h.{i}.ln_1 / h.{i}.attn.c_attn / ... / ln_f / lm_head),无 transformer. 前缀
因此 load_weights 直接用 AutoWeightsLoader 加载,无需 WeightsMapper。
"""

from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    make_empty_intermediate_tensors_factory,
    make_layers,
)
from vllm.sequence import IntermediateTensors


class GPT3Attention(nn.Module):
    """因果多头自注意力:c_attn 融合 QKV -> Attention -> c_proj。"""

    def __init__(
        self,
        hidden_size: int,
        total_num_heads: int,
        layer_norm_epsilon: float,
        activation_function: str,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        total_num_heads = total_num_heads
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        assert total_num_heads % tensor_model_parallel_world_size == 0, (
            f"num_heads {total_num_heads} 不能被 TP world size "
            f"{tensor_model_parallel_world_size} 整除"
        )
        self.num_heads = total_num_heads // tensor_model_parallel_world_size
        self.head_dim = self.hidden_size // total_num_heads
        self.scale = self.head_dim ** -0.5

        # 与 ckpt 命名对齐:h.{i}.attn.c_attn / h.{i}.attn.c_proj
        self.c_attn = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            total_num_heads,          # 总头数,QKV 各占一份数据头
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.c_attn",
        )
        self.c_proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.c_proj",
        )
        # Attention 本身无权重;vLLM 在此实现 paged KV cache + 因果掩码
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            scale=self.scale,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.c_attn(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        attn_output = self.attn(q, k, v)
        attn_output, _ = self.c_proj(attn_output)
        return attn_output


class GPT3MLP(nn.Module):
    """c_fc -> gelu(tanh 近似) -> c_proj。"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation_function: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.c_fc = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.c_fc",
        )
        self.c_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.c_proj",
        )
        self.act = get_act_fn(activation_function)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.c_proj(hidden_states)
        return hidden_states


class GPT3Block(nn.Module):
    """pre-LayerNorm 残差块:x = x + attn(ln_1(x)); x = x + mlp(ln_2(x))。"""

    def __init__(
        self,
        hidden_size: int,
        total_num_heads: int,
        intermediate_size: int,
        layer_norm_epsilon: float,
        activation_function: str,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.attn = GPT3Attention(
            hidden_size,
            total_num_heads,
            layer_norm_epsilon,
            activation_function,
            cache_config,
            quant_config,
            prefix=f"{prefix}.attn",
        )
        self.ln_2 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.mlp = GPT3MLP(
            hidden_size,
            intermediate_size,
            activation_function,
            quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        hidden_states = self.attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@support_torch_compile
class GPT3CustomForCausalLM(nn.Module, SupportsPP):
    """
    顶层 LM 模型,子模块命名与 ckpt_best.pt 严格对齐:
      wte / wpe / h.{i}.* / ln_f / lm_head
    因此权重可直接按名加载,无需重命名、无需 Conv1D 转置。
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config

        hidden_size = config.hidden_size            # GPT2Config.n_embd == 2048
        total_num_heads = config.num_attention_heads  # 16
        num_layers = config.num_hidden_layers        # 24
        vocab_size = config.vocab_size               # 64000
        max_positions = config.max_position_embeddings  # 4096 (== n_positions)
        n_inner = config.n_inner if config.n_inner is not None else 4 * hidden_size
        layer_norm_epsilon = config.layer_norm_epsilon  # 1e-5
        activation_function = config.activation_function  # "gelu_new"
        tie = getattr(config, "tie_word_embeddings", True)

        # token embedding
        self.wte = VocabParallelEmbedding(
            vocab_size,
            hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.wte",
        )
        # 绝对位置编码:不并行(与 ckpt 的 nn.Embedding 一致)
        self.wpe = nn.Embedding(max_positions, hidden_size)

        # 24 个 block;make_layers 以关键字 prefix= 调用 layer_fn,故 lambda 形参名必须是 prefix
        self.start_layer, self.end_layer, self.h = make_layers(
            num_layers,
            lambda prefix: GPT3Block(
                hidden_size,
                total_num_heads,
                n_inner,
                layer_norm_epsilon,
                activation_function,
                cache_config,
                quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.h",
        )

        self.ln_f = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)

        self.lm_head = ParallelLMHead(
            vocab_size,
            hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.lm_head",
        )
        if tie:
            self.lm_head = self.lm_head.tie_weights(self.wte)

        self.logits_processor = LogitsProcessor(vocab_size)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.wte(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                inputs_embeds = self.embed_input_ids(input_ids)
            position_embeds = self.wpe(positions)
            hidden_states = inputs_embeds + position_embeds
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for layer in islice(self.h, self.start_layer, self.end_layer):
            hidden_states = layer(hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = self.ln_f(hidden_states)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # ckpt 用 nn.Linear(标准 [out, in]),键名与本模型子模块一一对应,
        # 因此无需 WeightsMapper、无需 Conv1D 转置,直接按名加载。
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
