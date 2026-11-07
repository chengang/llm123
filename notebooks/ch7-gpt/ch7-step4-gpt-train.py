#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

GPT-3 XL 1.3B

论文中各层交替使用 dense 与 locally banded sparse attention
本实现全部使用 dense causal attention ，为走 Flash 后端，更快

"""

import array
import gc
import json
import math
import os
import queue
import random
import shutil
import threading
import time

# 在 torch 初始化 CUDA 分配器之前设置，扩展显存段，显著减少长时间训练的碎片，榨干显存
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# 让 torch.cuda.is_available() 走 NVML 而不是 CUDA runtime，避免初始化 CUDA 上下文
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

import sentencepiece as spm

import httpx
from dotenv import load_dotenv

# =============================================================================
#                               Configurations
# =============================================================================
class Config:
    # ------------------------------------------------------------------ 路径
    data_dir = "./cci3-hq-data/data"     # 存放 *.jsonl 的目录
    tokenizer_path = "./tokenizer.model" # SentencePiece 模型（训练一次，后续复用）
    token_bin_dir = "./token_bins"       # uint16 token 分片输出目录
    ramdisk_dir = "./myramdisk"          # -> /dev/shm，用于滚动缓存 token 分片
    ckpt_dir = "./checkpoints"           # 只会存在 ckpt_last.pt 与 ckpt_best.pt

    # -------------------------- 模型定义 --------------------------
    vocab_size = 64000                   # < 65536，token 可用 uint16 存储
    n_layer = 24
    d_model = 2048
    n_head = 16                          # d_head = 2048 / 16 = 128
    d_ff = 4 * 2048                      # 8192
    block_size = 4096                    # 上下文长度。原始论文中为 2048。cci3-hq 中 17.6% 的文档长度大于 2048 ， 6.4% 大于 4096，2.1% 大于 8192 ， 故取此值
    dropout = 0.0                        # 大语料预训练用 0
    bias = True                          # GPT-2/3 的 Linear 带 bias
    tie_embeddings = True                # lm_head 与 wte 权重共享

    # -------------------------- 训练超参 --------------------------
    lr = 2.0e-4
    min_lr_ratio = 0.1                   # 余弦退火到 10% 峰值学习率
    betas = (0.9, 0.95)
    eps = 1e-8
    weight_decay = 0.1
    grad_clip = 1.0
    warmup_tokens = 375_000_000          # 论文：前 3.75 亿 token 线性 warmup
    batch_tokens = 512 * 2048            # 论文：XL 的 batch 约 100 万 token

    # 单次前向的序列条数
    # 真实管线实测（RTX PRO 6000，torch.compile，含梯度累积+验证+真实写 16GB 存档）：
    #   block_size=4096, micro_bsz=8 -> 32.6K tok/s, 95.7 GB (占 93.8%, 余 6.3 GB) ← 默认
    #   block_size=4096, micro_bsz=7 -> 32.1K tok/s, 86.8 GB (占 85.1%, 余 15.2 GB)
    # 取 8 是因为 grad_accum 正好 32、batch 等于论文的 1,048,576 token；
    # grad_accum = batch_tokens // (micro_bsz * block_size) = 32
    micro_bsz = 8                        # 单次前向的序列条数，主要靠这个来调整显存占用

    total_train_tokens = 30_000_000_000  # 总训练 token 数。论文是 3000 亿 token，约等于 150G 中文语料，数据集不够时会自动重复

    # -------------------------- RamDisk 加速数据集 --------------------------
    ramdisk_budget_bytes = 2 * 1024**3   # ramdisk 上 token 分片的大小限制，2 GiB
    num_workers = 6                      # DataLoader 工作进程数
    prefetch_factor = 4

    # -------------------------- 运行时优化 --------------------------
    device = "cuda"
    
    dtype = "bfloat16"                   # 只支持 bfloat16 / float32。不支持 float16，它需要 GradScaler 
    compile_model = True                 # torch.compile（失败会自动降级）
    seed = 1337
    resume = "auto"                      # "auto" = 自动加载 ckpt_last.pt；None = 从头；或填 ckpt 路径

    # -------------------------- 日志 --------------------------
    log_every = 10                       # 每多少个 optimizer step 打一行日志
    eval_every = 50                      # 每多少步跑一次验证
    eval_iters = 50                      # 验证时跑多少个 batch
    ckpt_every = 200                     # 每多少步写一次 ckpt_last.pt。200 步 ≈ 1.7 小时，即崩溃最多丢 1.7 小时；
    gc_every_steps = 2000                # 每多少步显式 gc.collect()，每个 epoch 也会 gc.collect()
    oom_max_retries = 5                  # 连续多少次 OOM 才报错。偶尔 OOM 会清梯度 + empty_cache + 丢掉这一步重来，而不是直接崩溃。

    # NVIDIA RTX PRO 6000 Blackwell Server Edition 官方说 bf16 是 1000 TFLOPS
    # https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/
    # 但那是它说的是 bf16 稀疏计算
    # 我们用的 SDPA 是 bf16 稠密计算，所以是 500 TFLOPS
    # 本参数仅用于在日志里，估算显卡 MFU
    gpu_peak_flops_bf16 = 500e12       


    # -------------------------- Debug 开关 --------------------------
    limit_steps = None                   # 只跑 N 个 optimizer step 后退出（None = 不限制）
    benchmark_only = False               # True = 只跑吞吐基准，不写 ckpt
    send_notify = True                   # 是否发送告警消息


# =============================================================================
#                                  Utils
# =============================================================================
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# 发告警消息到飞书
def send_notify(message):
    load_dotenv()
    webhook_url=os.getenv("FEISHU_WEBHOOK_URL")
    data = {
        "msg_type": "text",
        "content": {"text": message}
    }

    try:
        resp = httpx.post(webhook_url, json=data, timeout=5.0)
        print("飞书响应状态码:", resp.status_code)
        print("飞书响应内容:", resp.text)
    except Exception as e:
        print("发送飞书告警失败:", e)

def fmt_count(n):
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.2f}{unit}"
    return str(int(n))


def fmt_duration(seconds):
    if seconds != seconds or seconds in (float("inf"), float("-inf")) or seconds < 0:
        return "--"
    seconds = int(seconds)
    d, seconds = divmod(seconds, 86400)
    h, seconds = divmod(seconds, 3600)
    m, s = divmod(seconds, 60)
    if d:
        return f"{d}d{h:02d}h{m:02d}m"
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def atomic_torch_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# =============================================================================
#                      ramdisk 滚动预取的 IterableDataset
# =============================================================================


# mmap 读取 uint16 分片。ramdisk 上的副本若不在了（例如上一轮的清理线程刚删掉），
# 退回读磁盘上的源文件；再不行就整块读进内存。
def read_shard(path, n_tokens, fallback_path=None):
    for p in (path, fallback_path):
        if p is None or not os.path.exists(p):
            continue
        try:
            return torch.from_file(p, shared=False, size=n_tokens, dtype=torch.uint16)
        except Exception:
            with open(p, "rb") as f:
                return torch.frombuffer(bytearray(f.read()), dtype=torch.uint16)
    raise FileNotFoundError(f"分片不可读：{path}（备用 {fallback_path}）")


# 每个 DataLoader worker 负责 shard 列表的一个子集，并自带一个后台预取线程，
# 把下一批分片从磁盘复制进 ramdisk，用完即删。
# 所有 worker 在 ramdisk 上占用的总字节数受 budget_bytes 约束
class ShardedTokenDataset(IterableDataset):

    def __init__(self, shards, token_bin_dir, ramdisk_dir, block_size,
                 budget_bytes, num_workers, seed, tag):
        self.shards = list(shards)             # [(name, n_tokens), ...]
        self.token_bin_dir = token_bin_dir
        self.ramdisk_dir = ramdisk_dir
        self.block_size = block_size
        self.budget_bytes = budget_bytes
        self.num_workers = max(1, num_workers)
        self.seed = seed
        self.tag = tag
        self.epoch = 0

    @staticmethod
    def _stage(src, dst):
        tmp = dst + ".part"
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)

    def _stager(self, my_shards, uniq, ready_q, done_q, max_inflight, stop_evt):
        inflight = 0
        staged = []
        try:
            for name, ntok in my_shards:
                src = os.path.join(self.token_bin_dir, name)
                while inflight >= max_inflight:
                    p = done_q.get()
                    if p is None:
                        return
                    try:
                        os.remove(p)
                    except OSError:
                        pass
                    if p in staged:
                        staged.remove(p)
                    inflight -= 1
                if stop_evt.is_set():
                    return
                # 文件名带上进程/迭代唯一后缀，避免跨 epoch 重建 worker 时与
                # 上一轮尚未退出的清理线程撞名（会被误删）
                dst = os.path.join(self.ramdisk_dir, f"{self.tag}_{uniq}_{name}")
                try:
                    self._stage(src, dst)
                except OSError:
                    # ramdisk 满或出错：直接退回从磁盘读
                    ready_q.put((src, ntok, False, src))
                    continue
                staged.append(dst)
                inflight += 1
                ready_q.put((dst, ntok, True, src))
        finally:
            ready_q.put(None)
            for p in staged:
                try:
                    os.remove(p)
                except OSError:
                    pass

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        nw = info.num_workers if info is not None else 1

        g = random.Random(self.seed * 1_000_003 + self.epoch)
        order = list(range(len(self.shards)))
        g.shuffle(order)
        my_shards = [self.shards[i] for i in order[wid::nw]]
        if not my_shards:
            return

        avg_bytes = max(1, sum(s[1] for s in self.shards) * 2 // max(1, len(self.shards)))
        max_inflight = max(1, (self.budget_bytes // nw) // avg_bytes)
        uniq = f"p{os.getpid()}e{self.epoch}w{wid}"

        ready_q = queue.Queue()
        done_q = queue.Queue()
        stop_evt = threading.Event()
        th = threading.Thread(target=self._stager,
                              args=(my_shards, uniq, ready_q, done_q, max_inflight, stop_evt),
                              daemon=True)
        th.start()

        bs1 = self.block_size + 1
        try:
            while True:
                item = ready_q.get()
                if item is None:
                    break
                path, ntok, staged, src = item
                buf = read_shard(path, ntok, fallback_path=src)
                n_blocks = (ntok - 1) // self.block_size
                starts = list(range(n_blocks))
                g.shuffle(starts)
                try:
                    for s in starts:
                        i = s * self.block_size
                        yield buf[i:i + bs1].to(torch.int64)
                finally:
                    del buf
                    if staged:
                        done_q.put(path)
        finally:
            stop_evt.set()
            done_q.put(None)
            # 排空队列，让 stager 的 finally 清掉残留文件
            while True:
                try:
                    it = ready_q.get_nowait()
                except queue.Empty:
                    break
                if it is None:
                    break
                if it[2]:
                    try:
                        os.remove(it[0])
                    except OSError:
                        pass
            th.join(timeout=10)


# 这里的 worker 是在 CUDA 初始化之后 fork 的，所以它是安全的
def make_train_loader(cfg, dataset):
    return DataLoader(
        dataset,
        batch_size=cfg.micro_bsz,
        num_workers=cfg.num_workers,
        pin_memory=False,             # 按要求不使用 pin_memory
        persistent_workers=False,     # 每个 epoch 重建，以便重新播种
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        multiprocessing_context="fork" if cfg.num_workers > 0 else None,
        drop_last=True,
    )


# 验证集小于等于 max_val_tokens ，一次性读进内存
def load_val_tokens(cfg, manifest):
    parts = []
    for name, ntok in manifest["val_shards"]:
        path = os.path.join(cfg.token_bin_dir, name)
        if not os.path.exists(path):
            continue
        parts.append(read_shard(path, ntok).clone())
    if not parts:
        return None
    return torch.cat(parts) if len(parts) > 1 else parts[0]


# 从验证 token 流里切出确定性的、互不重叠的 batch
def val_batches(val_tokens, cfg, n_batches):
    bs1 = cfg.block_size + 1
    n_blocks = (val_tokens.numel() - 1) // cfg.block_size
    if n_blocks < cfg.micro_bsz:
        return
    stride = max(1, n_blocks // max(1, n_batches * cfg.micro_bsz))
    idx = 0
    for _ in range(n_batches):
        rows = []
        for _ in range(cfg.micro_bsz):
            if idx >= n_blocks:
                idx = 0
            start = idx * cfg.block_size
            rows.append(val_tokens[start:start + bs1].to(torch.int64))
            idx += stride
        yield torch.stack(rows)


# =============================================================================
#                             构建 GPT-3 网络
# =============================================================================
# 多头因果自注意力。用 F.scaled_dot_product_attention 手写
class CausalSelfAttention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.dropout = cfg.dropout
        # 一次算出 q, k, v
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, d_head)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = F.gelu(self.c_fc(x), approximate="tanh")
        return self.dropout(self.c_proj(x))


# GPT-3 风格 Pre-LayerNorm 的 Transformer decoder block
class Block(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model, eps=1e-5)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.d_model, eps=1e-5)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT3(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)   # 学习式绝对位置编码
        self.drop = nn.Dropout(cfg.dropout)
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model, eps=1e-5)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        # 残差出口按 1/sqrt(2*n_layer) 缩放初始化（GPT-2 论文的做法）
        std = 0.02 / math.sqrt(2 * cfg.n_layer)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=std)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wpe.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"序列长度 {T} 超过 block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    # 只对二维及以上的参数（矩阵、embedding）做 weight decay
    def configure_optimizer(self, cfg):
        decay, no_decay = [], []
        seen = set()
        for p in self.parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        log(f"  优化器参数组：decay {len(decay)} 个张量 / {fmt_count(sum(p.numel() for p in decay))} 参数，"
            f"no-decay {len(no_decay)} 个张量 / {fmt_count(sum(p.numel() for p in no_decay))} 参数")
        fused = cfg.device == "cuda"
        return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, fused=fused)

    # 前向+反向的近似 FLOPs/token ， 6*N + attention 的序列项
    def flops_per_token(self):
        c = self.cfg
        n = self.num_params(non_embedding=True)
        return 6 * n + 12 * c.n_layer * c.d_model * c.block_size


# =============================================================================
#                                 学习率调度
# =============================================================================
def lr_at(cfg, tokens_seen):
    min_lr = cfg.lr * cfg.min_lr_ratio
    if tokens_seen < cfg.warmup_tokens:
        return cfg.lr * tokens_seen / max(1, cfg.warmup_tokens)
    if tokens_seen >= cfg.total_train_tokens:
        return min_lr
    span = max(1, cfg.total_train_tokens - cfg.warmup_tokens)
    ratio = (tokens_seen - cfg.warmup_tokens) / span
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (cfg.lr - min_lr)


# =============================================================================
#                                Checkpoint
# =============================================================================
def strip_compile_prefix(state):
    if any(k.startswith("_orig_mod.") for k in state):
        return {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    return state


def config_snapshot(cfg):
    return {k: getattr(cfg, k) for k in dir(cfg)
            if not k.startswith("_") and not callable(getattr(cfg, k))}


def save_checkpoint(cfg, path, raw_model, optimizer, state, include_optimizer):
    obj = {
        "model": raw_model.state_dict(),
        "step": state["step"],
        "epoch": state["epoch"],
        "tokens_seen": state["tokens_seen"],
        "best_val_loss": state["best_val_loss"],
        "config": config_snapshot(cfg),
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
    }
    if torch.cuda.is_available():
        obj["cuda_rng"] = torch.cuda.get_rng_state()
    if include_optimizer:
        obj["optimizer"] = optimizer.state_dict()
    atomic_torch_save(obj, path)


def load_checkpoint(cfg, raw_model, optimizer, state):
    if cfg.resume is None:
        return False
    path = os.path.join(cfg.ckpt_dir, "ckpt_last.pt") if cfg.resume == "auto" else cfg.resume
    if not os.path.exists(path):
        if cfg.resume != "auto":
            raise FileNotFoundError(f"指定的 checkpoint 不存在：{path}")
        log("没有找到 checkpoint ，从零开始训练")
        return False

    log(f"从 checkpoint 恢复：{path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    raw_model.load_state_dict(strip_compile_prefix(ckpt["model"]))
    if "optimizer" in ckpt and optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        log("该 checkpoint 不含 optimizer 状态 ， best 只存了权重 ，优化器从零开始")
    state["step"] = ckpt.get("step", 0)
    state["epoch"] = ckpt.get("epoch", 0)
    state["tokens_seen"] = ckpt.get("tokens_seen", 0)
    state["best_val_loss"] = ckpt.get("best_val_loss", float("inf"))
    if "torch_rng" in ckpt:
        torch.set_rng_state(ckpt["torch_rng"].to(torch.uint8))
    if "cuda_rng" in ckpt and torch.cuda.is_available():
        torch.cuda.set_rng_state(ckpt["cuda_rng"].to(torch.uint8))
    if "python_rng" in ckpt:
        random.setstate(ckpt["python_rng"])
    log(f"  恢复到 step={state['step']} epoch={state['epoch']} "
        f"tokens_seen={fmt_count(state['tokens_seen'])} best_val={state['best_val_loss']:.4f}")
    log("  注意：数据只恢复到 epoch 粒度，会从该 epoch 的开头重新按新顺序取数据"
        "（学习率是按 tokens_seen 算的，调度不受影响）")
    del ckpt
    gc.collect()
    return True


# =============================================================================
#                                  阶段性验证
# =============================================================================
@torch.no_grad()
def evaluate(model, val_tokens, cfg, autocast_ctx, device):
    if val_tokens is None:
        return float("nan")
    model.eval()
    losses, n = 0.0, 0
    try:
        for batch in val_batches(val_tokens, cfg, cfg.eval_iters):
            batch = batch.to(device)
            x, y = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
            with autocast_ctx:
                _, loss = model(x, y)
            losses += loss.item()
            n += 1
    except torch.cuda.OutOfMemoryError:
        # 验证跑不动不该拖垮整个训练，放弃本次验证继续训
        batch = x = y = loss = None
        gc.collect()
        torch.cuda.empty_cache()
        log(f"  验证时 OOM，本次验证跳过（已完成 {n}/{cfg.eval_iters} 个 batch）")
    finally:
        model.train()
    return losses / n if n else float("nan")


# 启动前算清楚 checkpoint 到底要多少磁盘，不够就警告
def check_disk_space(cfg):
    with torch.device("meta"):
        n = GPT3(cfg).num_params()
    last = 12 * n          # fp32 权重 4n + AdamW 的 exp_avg/exp_avg_sq 各 4n
    best = 4 * n           # 只存权重
    # 原子写会先写 .tmp 再 rename，所以 ckpt_last 在切换的瞬间占双份
    peak = 2 * last + best

    for label, path in (("token_bin_dir", cfg.token_bin_dir), ("ckpt_dir", cfg.ckpt_dir)):
        try:
            free = shutil.disk_usage(path).free
        except OSError:
            continue
        log(f"  {label}={path} 剩余 {free / 1e9:.1f} GB")
        if label != "ckpt_dir":
            continue
        log(f"  checkpoint 预计占用：ckpt_last {last / 1e9:.1f} GB（权重+优化器状态）+ "
            f"ckpt_best {best / 1e9:.1f} GB（仅权重），原子写瞬时峰值 {peak / 1e9:.1f} GB")
        if free < peak:
            log("  " + "!" * 70)
            log(f"  ckpt_dir 剩余空间不足！需要 {peak / 1e9:.1f} GB，只有 {free / 1e9:.1f} GB。")
            log(f"  前一两次保存可能成功，但覆盖旧 ckpt_last 时会写满磁盘导致训练中断。")
            log(f"  请把 Config.ckpt_dir 改到更大的盘再启动。")
            log("  " + "!" * 70)


# =============================================================================
#                                 准备数据集
# =============================================================================
# 把 Config 里的 token 预算换算成 micro-step / optimizer step。纯算术，无副作用
def compute_schedule(cfg):
    grad_accum = max(1, cfg.batch_tokens // (cfg.micro_bsz * cfg.block_size))
    tokens_per_step = grad_accum * cfg.micro_bsz * cfg.block_size
    planned = max(1, cfg.total_train_tokens // tokens_per_step)
    max_steps = planned
    if cfg.limit_steps is not None:
        max_steps = min(max_steps, cfg.limit_steps)

    log("=" * 78)
    log("计算训练计划")
    log("=" * 78)
    log(f"  micro_bsz={cfg.micro_bsz} × block_size={cfg.block_size} × "
        f"grad_accum={grad_accum} = {fmt_count(tokens_per_step)} tokens/step")
    log(f"  总计划 {fmt_count(cfg.total_train_tokens)} tokens = {planned} steps")
    if max_steps != planned:
        log(f"  （Config.limit_steps={cfg.limit_steps}，本次只跑 {max_steps} 步）")

    return {"grad_accum": grad_accum, "tokens_per_step": tokens_per_step,
            "max_steps": max_steps}

# 建训练数据集对象并把验证集读进内存
def build_data(cfg, manifest, sched):
    train_ds = ShardedTokenDataset(
        manifest["train_shards"], cfg.token_bin_dir, cfg.ramdisk_dir, cfg.block_size,
        cfg.ramdisk_budget_bytes, cfg.num_workers, cfg.seed, "train")

    val_tokens = load_val_tokens(cfg, manifest)
    log(f"  验证集 {fmt_count(val_tokens.numel()) if val_tokens is not None else 0} tokens")

    total_train_tokens_avail = sum(s[1] for s in manifest["train_shards"])
    log(f"  训练集 {fmt_count(total_train_tokens_avail)} tokens "
        f"({len(manifest['train_shards'])} 片) - 一个 epoch 约 "
        f"{total_train_tokens_avail / max(1, sched['tokens_per_step']):.0f} steps")

    return train_ds, val_tokens


# =============================================================================
#                            初始化 CUDA 和网络
# =============================================================================
def setup_torch_runtime(cfg):
    log("=" * 78)
    log("初始化 CUDA")
    log("=" * 78)

    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    if cfg.device == "cuda":
        # get_device_properties 会触发 _lazy_init()，从这里开始 CUDA 上下文才真正建立
        prop = torch.cuda.get_device_properties(0)
        log(f"  GPU: {prop.name}（{prop.total_memory / 1e9:.0f} GB / "
            f"{prop.total_memory / (1 << 30):.1f} GiB 可分配）")

    # 只支持 bfloat16 和 float32
    ptdtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[cfg.dtype]
    autocast_ctx = torch.autocast(device_type="cuda", dtype=ptdtype) \
        if cfg.device == "cuda" and ptdtype != torch.float32 else torch.autocast(
            device_type="cuda", enabled=False)

    return device, autocast_ctx


# 建网络与优化器、恢复 checkpoint、torch.compile
# 返回 (model, raw_model, optimizer, state)
def build_model(cfg, device):
    log("=" * 78)
    log("初始化模型与优化器")
    log("=" * 78)

    model = GPT3(cfg).to(device)
    raw_model = model
    log(f"  参数量：{fmt_count(model.num_params())}"
        f"（不含位置编码 {fmt_count(model.num_params(True))}）")

    optimizer = model.configure_optimizer(cfg)

    state = {"step": 0, "epoch": 0, "tokens_seen": 0, "best_val_loss": float("inf")}
    load_checkpoint(cfg, raw_model, optimizer, state)

    if cfg.compile_model:
        try:
            log("  torch.compile 编译中（第一次迭代会比较慢）…")
            model = torch.compile(raw_model)
        except Exception as e:
            log(f"  torch.compile 失败，改用 eager 模式：{e}")
            model = raw_model

    return model, raw_model, optimizer, state


# =============================================================================
#                                  开始训练
# =============================================================================
def run_training(cfg, model, raw_model, optimizer, state,
                 train_ds, val_tokens, device, autocast_ctx, sched):
    grad_accum = sched["grad_accum"]
    tokens_per_step = sched["tokens_per_step"]
    max_steps = sched["max_steps"]

    log("=" * 78)
    log("开始训练")
    log("=" * 78)

    def infinite_batches(start_epoch):
        epoch = start_epoch
        while True:
            train_ds.epoch = epoch
            loader = make_train_loader(cfg, train_ds)
            n_yield = 0
            for batch in loader:
                yield epoch, batch
                n_yield += 1
            del loader
            gc.collect()               # 每轮结束释放内存
            if n_yield == 0:
                raise RuntimeError("训练数据为空，请检查 tokens 目录")
            epoch += 1

    batch_gen = infinite_batches(state["epoch"])

    # 训练主循环
    model.train()
    flops_per_token = raw_model.flops_per_token()
    ema_step_time = None
    steps_this_run = 0        # 本次进程跑了几步，用来丢掉含编译时间的第一步
    consecutive_oom = 0
    last_val_loss = float("nan")
    total_vram = torch.cuda.get_device_properties(0).total_memory if cfg.device == "cuda" else 0
    micro_iter = iter(batch_gen)

    log("  开始训练。日志字段 - ep=epoch, tok=已见 token 数, mfu=模型算力利用率")
    while state["step"] < max_steps:
        lr = lr_at(cfg, state["tokens_seen"])
        for group in optimizer.param_groups:
            group["lr"] = lr

        t0 = time.time()
        try:
            # loss 累加保持在 GPU 上，整个 step 只在最后同步一次，避免每个 micro-step 都被 .item() 打断流水线
            loss_acc = torch.zeros((), device=device)
            cur_epoch = state["epoch"]
            for _ in range(grad_accum):
                cur_epoch, batch = next(micro_iter)
                batch = batch.to(device)
                x, y = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
                with autocast_ctx:
                    _, loss = model(x, y)
                loss_acc += loss.detach()
                (loss / grad_accum).backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            train_loss = loss_acc.item() / grad_accum  # 这里做本 step 唯一的一次同步
        except torch.cuda.OutOfMemoryError:
            # 如果 OOM 则丢掉这一步（约 30 秒的数据）重来，而不是让几天的训练直接崩掉
            # 本步的 step/tokens_seen 都不计数，因为这一步等于没发生过。
            consecutive_oom += 1
            # 必须先断开这些张量的引用，empty_cache() 才有东西可回收。
            # 注意不能用 del locals()[...]，那对函数局部变量是无效的
            loss = loss_acc = batch = x = y = None
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            log(f"  OOM（第 {consecutive_oom}/{cfg.oom_max_retries} 次），已清理显存并丢弃本步重试。"
                f"若反复出现请调小 Config.micro_bsz")
            if consecutive_oom > cfg.oom_max_retries:
                log("  连续 OOM 次数超限，这不是偶发抖动，退出。")
                raise
            continue
        consecutive_oom = 0
        dt = time.time() - t0

        # 第一步的耗时含 torch.compile 的编译时间（几十秒到几分钟），拿它给 EMA 播种会让
        # 接下来上百步的 tok/s 和 ETA 都严重偏低，所以直接丢掉。
        steps_this_run += 1
        if steps_this_run > 1:
            ema_step_time = dt if ema_step_time is None else 0.9 * ema_step_time + 0.1 * dt

        state["step"] += 1
        state["epoch"] = cur_epoch
        state["tokens_seen"] += tokens_per_step

        # 计算验证 Loss
        if state["step"] % cfg.eval_every == 0 or state["step"] == max_steps:
            last_val_loss = evaluate(model, val_tokens, cfg, autocast_ctx, device)
            if last_val_loss == last_val_loss and last_val_loss < state["best_val_loss"]:
                state["best_val_loss"] = last_val_loss
                if not cfg.benchmark_only:
                    save_checkpoint(cfg, os.path.join(cfg.ckpt_dir, "ckpt_best.pt"),
                                    raw_model, optimizer, state, include_optimizer=False)
                    log(f"  ★ 新的最优 val loss {last_val_loss:.4f}，已写 ckpt_best.pt")

        # 打印日志
        if state["step"] % cfg.log_every == 0 or state["step"] == 1:
            # 第一步还没有可用的 EMA，因为包含编译时间已被丢弃，先用本步耗时计算
            est = ema_step_time if ema_step_time is not None else dt
            tok_per_s = tokens_per_step / max(1e-9, est)
            achieved = flops_per_token * tok_per_s
            mfu = achieved / cfg.gpu_peak_flops_bf16 * 100
            eta = (max_steps - state["step"]) * est

            # 显存用 GiB（二进制，÷2^30），好和 nvidia-smi 的 MiB 直接对得上；
            # 百分比是相对 CUDA 真正能分配的 total_memory，不是 nvidia-smi 的板载总量
            # 两者差一个驱动保留量，RTX Pro 6000 + Linux 约为 637 MiB
            gib = 1 << 30
            mem = torch.cuda.max_memory_allocated() / gib if cfg.device == "cuda" else 0.0
            memres = torch.cuda.max_memory_reserved() / gib if cfg.device == "cuda" else 0.0
            memtot = total_vram / gib if total_vram else 1.0
            ppl = math.exp(min(20.0, last_val_loss)) if last_val_loss == last_val_loss else float("nan")
            log(f"ep {state['epoch']} | step {state['step']}/{max_steps} "
                f"({100 * state['step'] / max_steps:5.2f}%) | "
                f"loss {train_loss:.4f} | vloss {last_val_loss:.4f} | ppl {ppl:9.2f} | "
                f"lr {lr:.3e} | gnorm {float(grad_norm):.2f} | "
                f"tok {fmt_count(state['tokens_seen'])} | "
                f"tok/s {tok_per_s:8.0f} | {achieved / 1e12:5.1f} TFLOPS | mfu {mfu:4.1f}% | "
                f"eta {fmt_duration(eta)} | "
                f"mem {mem:.1f}/{memres:.1f}/{memtot:.1f}GiB {memres / memtot * 100:.1f}%")

        # 保存 checkpoint
        if not cfg.benchmark_only and (state["step"] % cfg.ckpt_every == 0
                                       or state["step"] == max_steps):
            save_checkpoint(cfg, os.path.join(cfg.ckpt_dir, "ckpt_last.pt"),
                            raw_model, optimizer, state, include_optimizer=True)
            log(f"  已写 ckpt_last.pt (step {state['step']})")

        if state["step"] % cfg.gc_every_steps == 0:
            gc.collect()

    log(f"训练结束：{state['step']} steps，{fmt_count(state['tokens_seen'])} tokens，"
        f"best val loss {state['best_val_loss']:.4f}")
    return state


# =============================================================================
#                                   入口
# =============================================================================


# 清掉上次异常退出可能残留的分片
def cleanup_ramdisk(cfg):
    if not os.path.isdir(cfg.ramdisk_dir):
        return
    for name in os.listdir(cfg.ramdisk_dir):
        if name.endswith(".bin") or name.endswith(".bin.part"):
            try:
                os.remove(os.path.join(cfg.ramdisk_dir, name))
            except OSError:
                pass


def preflight(cfg):
    assert cfg.vocab_size <= 65536, "vocab_size 必须 <= 65536，token 才能用 uint16 存储"
    assert cfg.d_model % cfg.n_head == 0, "d_model 必须能被 n_head 整除"
    if cfg.dtype not in ("bfloat16", "float32"):
        raise ValueError(
            f"Config.dtype={cfg.dtype!r} 不支持，只能是 'bfloat16' 或 'float32'")

    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")

    for d in (cfg.token_bin_dir, cfg.ckpt_dir):
        os.makedirs(os.path.realpath(d), exist_ok=True)
    check_disk_space(cfg) 
    cleanup_ramdisk(cfg)


def load_manifest(cfg):
    path = os.path.join(cfg.token_bin_dir, "manifest.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        if m.get("vocab_size") != cfg.vocab_size or m.get("block_size_hint") != cfg.block_size:
            log(f"manifest 的 vocab_size/block_size 与当前 Config 不一致，"
                f"如果换了分词器请删除 {cfg.token_bin_dir} 后重跑")
        return m
    return {
        "vocab_size": cfg.vocab_size,
        "block_size_hint": cfg.block_size,
        "sources": {},
        "train_shards": [],
        "val_shards": [],
        "train_next_index": 0,
        "val_next_index": 0,
        "val_tokens": 0,
    }

# 无条件重建 ramdisk_dir 为指向 /dev/shm/myramdisk 的软链
def fix_ramdisk_link(cfg):
    link = cfg.ramdisk_dir
    target = "/dev/shm/myramdisk"

    # 只要存在（目录/软链/普通文件/悬空链接），统统先删掉
    if os.path.lexists(link):
        if os.path.isdir(link) and not os.path.islink(link):
            # 真实目录，可能残留 .bin 分片，整个删掉
            shutil.rmtree(link, ignore_errors=True)
        else:
            os.remove(link)  # 软链（含悬空）、普通文件
        log(f"已清理旧的 {link}")

    # 无条件重建：tmpfs 目标 + 软链
    os.makedirs(target, exist_ok=True)
    os.symlink(target, link)
    log(f"已重建软链 {link} -> {target}")



def main():
    cfg = Config
    t_start = time.time()
    log("GPT-3 XL 1.3B 单卡训练")

    fix_ramdisk_link(cfg)
    preflight(cfg)

    # 准备数据
    sched = compute_schedule(cfg)
    try:
        manifest = load_manifest(cfg)
        train_ds, val_tokens = build_data(cfg, manifest, sched)

        # 初始化 CUDA 
        device, autocast_ctx = setup_torch_runtime(cfg)

        # 初始化模型, 优化器, 从 checkpoint 恢复, torch.compile 等...
        model, raw_model, optimizer, state = build_model(cfg, device)

        # 开始训练
        run_training(cfg, model, raw_model, optimizer, state,
                     train_ds, val_tokens, device, autocast_ctx, sched)
    finally:
        cleanup_ramdisk(cfg)
        gc.collect()

    log(f"总耗时 {fmt_duration(time.time() - t_start)}")


if __name__ == "__main__":
    main()
