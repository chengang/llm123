#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

GPT-3 XL (1.3B) —— 单文件 PyTorch 训练脚本

网络结构按 GPT-3 论文（Brown et al., 2020）Table 2.1 的 "GPT-3 XL" ： 
    n_layers=24, d_model=2048, d_head=128, context=2048

注意力使用 torch.nn.functional.scaled_dot_product_attention 实现

论文中各层交替使用 dense 与 locally banded sparse attention，
本实现全部使用dense causal attention，为了走 Flash 后端，更快

六个阶段，前面几个阶段都会检测已有产物并跳过，可反复运行：

  0) preflight            —— 纯 CPU 自检：配置断言、建目录、磁盘余量、清理 ramdisk 残留
  1) ensure_tokenizer     —— 用 SentencePiece 训练 BPE 分词器 -> ./tokenizer.model
  2) ensure_tokens        —— jsonl 语料 -> uint16 token 分片 ./tokens/*.bin（增量，只处理新增文件）
  3) compute_schedule +
     build_data           —— 换算 batch/步数；建数据管线、读入验证集（纯 CPU）
  4) setup_torch_runtime  —— 播种、tf32、autocast；★ 全流程第一次初始化 CUDA
  5) build_model          —— 建网络与优化器、恢复 checkpoint、torch.compile
  6) run_training         —— 主循环；token 分片由后台线程滚动预取进 ./myramdisk（tmpfs）喂给 GPU

阶段 0-3 排在 CUDA 初始化之前是因为要避免阶段二 fork 出 20 个分词进程时从一个已经初始化 CUDA 上下文的父进程 fork。
避免 fork 瞬间若驱动的后台线程正持有内部锁，子进程会挂死。
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

# 在 torch 初始化 CUDA 分配器之前设置，可扩展显存段能显著减少长时间训练的碎片，目的是榨干显存
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
#                                  配置
# =============================================================================
class Config:
    # ------------------------------------------------------------------ 路径
    data_dir = "./cci3-hq-data/data"     # 存放 *.jsonl 的目录
    tokenizer_path = "./tokenizer.model" # SentencePiece 模型（训练一次，后续复用）
    token_dir = "./tokens"               # uint16 token 分片输出目录
    ramdisk_dir = "./myramdisk"          # -> /dev/shm，用于滚动缓存 token 分片
    ckpt_dir = "./checkpoints"           # 只会存在 ckpt_last.pt 与 ckpt_best.pt

    # --------------------------------------------------------- 分词器（阶段一）
    vocab_size = 64000                   # < 65536，token 可用 uint16 存储
    sp_model_type = "bpe"
    sp_character_coverage = 0.9995       # 中文语料建议 0.9995
    sp_byte_fallback = True              # 未登录字符回退到字节，保证无损
    # 归一化用 identity 而不是 nmt_nfkc：NFKC 会把全角逗号"，"转成半角","，
    # 但不动全角句号"。"，中文标点会被改得不一致，且 encode/decode 不再无损。
    sp_normalization = "identity"
    sp_add_dummy_prefix = False          # 中文不需要在句首补空格
    sp_remove_extra_whitespaces = False  # 保留原始空白，保证无损还原
    # 分词器训练语料的抽样量。CCI3 的文档几乎没有换行，一篇文档就是一"句"，
    # 所以必须按【字符数】限额，否则会把整个语料喂给 SentencePiece
    # （实测：不限额时 = 32 亿字符 / 8.6GB 文本 / 54GB 内存，BPE 合并慢到不可用）。
    # 3 亿字符对 64k 的中文 BPE 已经很充足。
    sp_sample_chars = 300_000_000
    sp_sample_sentences = 5_000_000      # 句子数上限（通常先撞到字符数限额）
    sp_doc_stride = 8                    # 每 8 篇文档取 1 篇，让抽样铺开到整个语料
    sp_max_sentence_bytes = 8192         # 超过此长度的句子会被切开
    sp_num_threads = 22
    sp_unk_id, sp_bos_id, sp_eos_id, sp_pad_id = 0, 1, 2, 3

    # ----------------------------------------------------- 语料分片（阶段二）
    tokenize_processes = 20              # 分词并行进程数
    tokenize_batch_docs = 512            # 每个任务包含多少篇文档
    tokenize_super_batch = 44            # 一次派发多少个任务包（控制内存占用）
    shard_bytes = 128 * 1024 * 1024      # 单个 .bin 分片大小（128 MiB ≈ 6710 万 token）
    val_doc_modulo = 1000                # 每 1000 篇文档抽 1 篇进验证集
    max_val_tokens = 5_000_000           # 验证集 token 上限

    # ---------------------------------------------------------- 模型（GPT-3 XL）
    n_layer = 24
    d_model = 2048
    n_head = 16                          # d_head = 2048 / 16 = 128
    d_ff = 4 * 2048                      # 8192
    # 上下文长度。论文用的是 2048，这里取 4096 是基于对本语料的全量统计
    # （analyze_lengths.py，167 万篇文档）：token 加权平均文档长度是 2908，
    # 2048 会截断 17.6% 的 token，4096 只截断 6.4%，代价约 8% 的训练时间。
    # 再往上收益急剧衰减（8192 只降到 2.1%，却要再多花 10% 时间）。
    block_size = 4096
    dropout = 0.0                        # 大语料预训练用 0
    bias = True                          # GPT-2/3 的 Linear 带 bias
    tie_embeddings = True                # lm_head 与 wte 权重共享

    # ------------------------------------------- 优化器与调度（GPT-3 论文 XL 行）
    lr = 2.0e-4
    min_lr_ratio = 0.1                   # 余弦退火到 10% 峰值学习率
    betas = (0.9, 0.95)
    eps = 1e-8
    weight_decay = 0.1
    grad_clip = 1.0
    warmup_tokens = 375_000_000          # 论文：前 3.75 亿 token 线性 warmup
    batch_tokens = 512 * 2048            # 论文：XL 的 batch 约 100 万 token
    # 单次前向的序列条数，是"吃满显存"的旋钮。显存基本只取决于
    # micro_bsz * block_size（每个 micro-step 的 token 数），和上下文长度本身无关，
    # 因为 SDPA 走 Flash 后端不会 materialize T×T 的注意力矩阵。
    # 注意显存单位：下面的 GB 都是十进制（÷1e9），和日志里打印的一致。
    # nvidia-smi 报的 97887 MiB 是二进制单位 = 102.64 GB，其中 637 MiB 是驱动保留、
    # CUDA 分配不到，所以程序真正的上限是 total_memory = 101.97 GB。
    # 真实管线实测（RTX PRO 6000，torch.compile，含梯度累积+验证+真实写 16GB 存档）：
    #   block_size=4096, micro_bsz=8 -> 32.6K tok/s, 95.7 GB (占 93.8%, 余 6.3 GB) ← 默认
    #   block_size=4096, micro_bsz=7 -> 32.1K tok/s, 86.8 GB (占 85.1%, 余 15.2 GB)
    # 取 8 是因为 grad_accum 正好 32、batch 精确等于论文的 1,048,576 token；
    # 速度只快 1.5%（十一天的运行省约 3.9 小时），不是主要理由。
    # 显存从第 2 步分配完 AdamW 状态后就完全不再变化（形状全程静态、
    # torch.compile 不会重编译、验证比训练省、存档不占显存），实测 12 步纹丝不动。
    # 代价是训练期间这张卡不能再跑别的东西。要留余量就调回 7，更保守用 6（约 78 GB）。
    # grad_accum = batch_tokens // (micro_bsz * block_size) = 32
    micro_bsz = 8
    # 余弦退火的终点。论文是 3000 亿 token；当前 10GB 语料只有约 20 亿 token，
    # 30B 相当于跑 15 个 epoch（会有明显重复）。语料下载得更多之后请调大这个值。
    total_train_tokens = 30_000_000_000

    # ------------------------------------------------------- 数据加载（阶段三/六）
    ramdisk_budget_bytes = 2 * 1024**3   # ramdisk 上 token 分片的总预算（2 GiB）
    num_workers = 6                      # DataLoader 工作进程数
    prefetch_factor = 4
    # 注意：按要求不设置 pin_memory（下面显式传 pin_memory=False）

    # ---------------------------------------------------------------- 运行
    device = "cuda"
    # 只支持 bfloat16 / float32。不支持 float16：它需要 GradScaler 才不会梯度下溢，
    # 而 bf16 的指数位与 fp32 相同、压根不会下溢，Blackwell 上两者算力又完全一样。
    # 填 "float16" 会在 preflight 里直接报错。
    dtype = "bfloat16"
    compile_model = True                 # torch.compile（失败会自动降级）
    seed = 1337

    log_every = 10                       # 每多少个 optimizer step 打一行日志
    eval_every = 50                      # 每多少步跑一次验证
    eval_iters = 50                      # 验证时跑多少个 batch
    # 每多少步写一次 ckpt_last.pt。200 步 ≈ 1.7 小时，即崩溃最多丢 1.7 小时；
    # 实测存一次档 20 秒，开销只有 0.03%，所以没必要为省这点开销拉长间隔。
    ckpt_every = 200
    gc_every_steps = 2000                # 每多少步显式 gc.collect()（每个 epoch 也会调用）
    # 撞到 OOM 时：清梯度 + empty_cache + 丢掉这一步重来，而不是直接崩溃。
    # 连续失败超过这个次数才真的抛出（避免掩盖"每步都 OOM"这种真问题）。
    oom_max_retries = 5

    # NVIDIA RTX PRO 6000 Blackwell Server Edition 官方说 bf16 是 1000 TFLOPS
    # 但那是稀疏计算，我们的 SDPA 要使用稠密计算，所以是 500 TFLOPS
    # https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/
    # 用于估算 MFU 的显卡 bf16 稠密算力峰值（近似值，只影响日志里的 MFU 计算）
    gpu_peak_flops_bf16 = 500e12       

    resume = "auto"                      # "auto" = 自动加载 ckpt_last.pt；None = 从头；或填 ckpt 路径

    # -------------------------------------------------------------- 调试旋钮
    limit_steps = None                   # 只跑 N 个 optimizer step 后退出（None = 不限制）
    limit_docs_per_file = None           # 每个 jsonl 只读前 N 篇文档（None = 全读）
    benchmark_only = False               # True = 只跑吞吐基准，不写 ckpt
    send_notify = True


# =============================================================================
#                                  小工具
# =============================================================================
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def send_notify(message):
    load_dotenv()
    webhook_url=os.getenv("FEISHU_WEBHOOK_URL")
    """发送纯文本告警消息到飞书"""
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


def list_jsonl_files(data_dir):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"找不到语料目录 {data_dir}")
    names = sorted(f for f in os.listdir(data_dir) if f.endswith(".jsonl"))
    if not names:
        raise FileNotFoundError(f"{data_dir} 下没有 .jsonl 文件")
    return names


def iter_docs(path, limit=None):
    """流式读一个 jsonl，产出 (doc_id, text)。"""
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text")
            if not text:
                continue
            yield obj.get("id", ""), text
            n += 1
            if limit is not None and n >= limit:
                return


def is_val_doc(doc_id, text, modulo):
    """确定性地把一小部分文档划到验证集；与语料文件数量无关。"""
    key = doc_id if doc_id else text[:64]
    h = 0
    for ch in key[:16]:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h % modulo == 0


# =============================================================================
#                       阶段一：训练 SentencePiece BPE 分词器
# =============================================================================
def split_into_sentences(text, max_bytes):
    """按换行切句；过长的行再按标点/硬切分成 <= max_bytes 的片段。"""
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if len(line.encode("utf-8")) <= max_bytes:
            yield line
            continue
        # 粗略按字符数硬切（UTF-8 中文最多 4 字节/字符）
        step = max(1, max_bytes // 4)
        for i in range(0, len(line), step):
            piece = line[i:i + step].strip()
            if piece:
                yield piece


def build_tokenizer_sample(cfg, sample_path):
    """
    从所有 jsonl 里轮转抽样，写成 SentencePiece 的训练输入。
    同时受字符数与句子数两个上限约束，并按 sp_doc_stride 跳着取，
    让样本铺开到整个语料而不是只覆盖每个文件的开头。
    """
    names = list_jsonl_files(cfg.data_dir)
    iters = [iter_docs(os.path.join(cfg.data_dir, n), cfg.limit_docs_per_file) for n in names]
    stride = max(1, cfg.sp_doc_stride)
    seen = {n: 0 for n in range(len(names))}
    written = chars = 0
    next_report = 100_000_000
    t0 = time.time()
    with open(sample_path, "w", encoding="utf-8") as out:
        while iters and chars < cfg.sp_sample_chars and written < cfg.sp_sample_sentences:
            for idx, it in list(enumerate(iters)):
                try:
                    _, text = next(it)
                except StopIteration:
                    iters.remove(it)
                    continue
                seen[idx] += 1
                if seen[idx] % stride:
                    continue
                for sent in split_into_sentences(text, cfg.sp_max_sentence_bytes):
                    out.write(sent)
                    out.write("\n")
                    written += 1
                    chars += len(sent)
                if chars >= cfg.sp_sample_chars or written >= cfg.sp_sample_sentences:
                    break
            if chars >= next_report:
                next_report += 100_000_000
                log(f"  分词器采样：{fmt_count(chars)} 字符 / {fmt_count(written)} 句")
    log(f"  分词器采样完成：{fmt_count(chars)} 字符 / {fmt_count(written)} 句，"
        f"{os.path.getsize(sample_path) / 1e6:.0f} MB，耗时 {fmt_duration(time.time() - t0)}")
    return written


def ensure_tokenizer(cfg):
    if os.path.exists(cfg.tokenizer_path):
        sp = spm.SentencePieceProcessor(model_file=cfg.tokenizer_path)
        log(f"复用已有分词器 {cfg.tokenizer_path}（vocab={sp.get_piece_size()}）")
        return sp

    log("=" * 78)
    log("阶段一：训练 SentencePiece BPE 分词器")
    log("=" * 78)
    # 采样文件放磁盘而不是 ramdisk：它是一次性的、体积接近 1GB，
    # 而 ramdisk 的预算是留给训练时滚动的 token 分片的。
    os.makedirs(cfg.token_dir, exist_ok=True)
    sample_path = os.path.join(cfg.token_dir, "sp_sample.txt")
    try:
        build_tokenizer_sample(cfg, sample_path)
        prefix = cfg.tokenizer_path[:-len(".model")] if cfg.tokenizer_path.endswith(".model") \
            else cfg.tokenizer_path
        log(f"  开始训练 BPE（vocab_size={cfg.vocab_size}），这一步可能要几十分钟…")
        t0 = time.time()
        spm.SentencePieceTrainer.train(
            input=sample_path,
            model_prefix=prefix,
            model_type=cfg.sp_model_type,
            vocab_size=cfg.vocab_size,
            character_coverage=cfg.sp_character_coverage,
            byte_fallback=cfg.sp_byte_fallback,
            normalization_rule_name=cfg.sp_normalization,
            add_dummy_prefix=cfg.sp_add_dummy_prefix,
            remove_extra_whitespaces=cfg.sp_remove_extra_whitespaces,
            input_sentence_size=cfg.sp_sample_sentences,
            shuffle_input_sentence=True,
            max_sentence_length=cfg.sp_max_sentence_bytes,
            num_threads=cfg.sp_num_threads,
            unk_id=cfg.sp_unk_id,
            bos_id=cfg.sp_bos_id,
            eos_id=cfg.sp_eos_id,
            pad_id=cfg.sp_pad_id,
            train_extremely_large_corpus=False,
        )
        log(f"  分词器训练完成，耗时 {fmt_duration(time.time() - t0)} -> {cfg.tokenizer_path}")
    finally:
        if os.path.exists(sample_path):
            os.remove(sample_path)
    gc.collect()
    return spm.SentencePieceProcessor(model_file=cfg.tokenizer_path)


# =============================================================================
#                    阶段二：语料 -> uint16 token 分片（增量）
# =============================================================================
_SP_WORKER = None
_SP_EOS = 2
_SP_MODULO = 1000


def _tok_worker_init(model_path, eos_id, modulo):
    global _SP_WORKER, _SP_EOS, _SP_MODULO
    _SP_WORKER = spm.SentencePieceProcessor(model_file=model_path)
    _SP_EOS, _SP_MODULO = eos_id, modulo


def _tok_worker(batch):
    """batch: [(doc_id, text), ...] -> [(is_val, ids+eos), ...]

    验证集判定与 eos 追加都放在 worker 里做，主进程只负责写盘（主进程是瓶颈）。
    """
    encoded = _SP_WORKER.encode([t for _, t in batch], out_type=int)
    out = []
    for (doc_id, text), ids in zip(batch, encoded):
        ids.append(_SP_EOS)
        out.append((is_val_doc(doc_id, text, _SP_MODULO), ids))
    return out


class ShardWriter:
    """把 token 流按固定字节数切成 uint16 的 .bin 分片。"""

    def __init__(self, out_dir, tag, start_index, shard_bytes):
        self.out_dir = out_dir
        self.tag = tag
        self.index = start_index
        self.shard_bytes = shard_bytes
        self.buf = array.array("H")
        self.max_tokens = shard_bytes // 2
        self.shards = []          # [(filename, n_tokens), ...]
        self.total_tokens = 0

    def add(self, ids):
        self.buf.extend(ids)
        self.total_tokens += len(ids)
        while len(self.buf) >= self.max_tokens:
            self._flush(self.max_tokens)

    def _flush(self, n):
        if n <= 0:
            return
        name = f"{self.tag}_{self.index:05d}.bin"
        path = os.path.join(self.out_dir, name)
        chunk = self.buf[:n]
        with open(path, "wb") as f:
            chunk.tofile(f)
        self.shards.append([name, n])
        self.index += 1
        del self.buf[:n]

    def close(self):
        """把剩余不满一片的 token 也落盘（用于每处理完一个源文件时做断点）。"""
        self._flush(len(self.buf))

    def take_shards(self):
        """取走并清空自上次以来新产生的分片列表。"""
        s, self.shards = self.shards, []
        return s


def load_manifest(cfg):
    path = os.path.join(cfg.token_dir, "manifest.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        if m.get("vocab_size") != cfg.vocab_size or m.get("block_size_hint") != cfg.block_size:
            log(f"⚠ manifest 的 vocab_size/block_size 与当前 Config 不一致，"
                f"如果换了分词器请删除 {cfg.token_dir} 后重跑")
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


def save_manifest(cfg, manifest):
    path = os.path.join(cfg.token_dir, "manifest.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def remove_orphan_shards(cfg, manifest):
    """删除 token_dir 里没被 manifest 记录的 .bin（上次异常中断的残留）。"""
    known = {s[0] for s in manifest["train_shards"]} | {s[0] for s in manifest["val_shards"]}
    removed = 0
    for name in os.listdir(cfg.token_dir):
        if name.endswith(".bin") and name not in known:
            try:
                os.remove(os.path.join(cfg.token_dir, name))
                removed += 1
            except OSError:
                pass
    if removed:
        log(f"  清理了 {removed} 个上次中断残留的孤儿分片")


def batched(iterable, n):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def ensure_tokens(cfg):
    """把还没处理过的 jsonl 编码成 token 分片。返回 manifest。"""
    import multiprocessing as mp

    os.makedirs(cfg.token_dir, exist_ok=True)
    manifest = load_manifest(cfg)

    names = list_jsonl_files(cfg.data_dir)
    todo = []
    for name in names:
        p = os.path.join(cfg.data_dir, name)
        st = os.stat(p)
        rec = manifest["sources"].get(name)
        if rec and rec["size"] == st.st_size and abs(rec["mtime"] - st.st_mtime) < 1e-6:
            continue
        todo.append((name, p, st))

    if not todo:
        log(f"复用已有 token 分片：train {len(manifest['train_shards'])} 片 / "
            f"{fmt_count(sum(s[1] for s in manifest['train_shards']))} tokens，"
            f"val {fmt_count(manifest['val_tokens'])} tokens")
        return manifest

    log("=" * 78)
    log(f"阶段二：分词 {len(todo)} 个新的 jsonl 文件 -> uint16 token 分片")
    log("=" * 78)

    # 清掉上次异常中断留下的、没被 manifest 记录的孤儿分片
    remove_orphan_shards(cfg, manifest)

    train_w = ShardWriter(cfg.token_dir, "train", manifest["train_next_index"], cfg.shard_bytes)
    val_w = ShardWriter(cfg.token_dir, "val", manifest["val_next_index"], cfg.shard_bytes)
    val_tokens = manifest["val_tokens"]

    # fork 出去的子进程会继承一份已废弃的 CUDA 上下文；更糟的是 fork 那一瞬若驱动的
    # 后台线程正持有内部锁，子进程里那把锁就永远解不开（罕见但真实的挂死来源）。
    # 所以分词必须在 CUDA 初始化之前跑完，这条断言把该时序不变量钉死在代码里。
    assert not torch.cuda.is_initialized(), \
        "分词的多进程 fork 必须发生在 CUDA 初始化之前，请检查 main() 里各阶段的顺序"

    ctx = mp.get_context("fork")
    t_start = time.time()
    n_docs = 0
    with ctx.Pool(cfg.tokenize_processes, initializer=_tok_worker_init,
                  initargs=(cfg.tokenizer_path, cfg.sp_eos_id, cfg.val_doc_modulo)) as pool:
        for name, path, st in todo:
            log(f"  处理 {name} ({st.st_size / 1e9:.2f} GB)…")
            t_file = time.time()
            docs = iter_docs(path, cfg.limit_docs_per_file)
            file_docs = 0
            next_report = 100_000
            for super_batch in batched(batched(docs, cfg.tokenize_batch_docs),
                                       cfg.tokenize_super_batch):
                for result in pool.map(_tok_worker, super_batch):
                    for is_val, ids in result:
                        if is_val and val_tokens < cfg.max_val_tokens:
                            val_w.add(ids)
                            val_tokens += len(ids)
                        else:
                            train_w.add(ids)
                        file_docs += 1
                if file_docs >= next_report:
                    next_report += 100_000
                    rate = file_docs / max(1e-6, time.time() - t_file)
                    log(f"    {fmt_count(file_docs)} 篇文档 | "
                        f"{fmt_count(train_w.total_tokens)} train tokens | "
                        f"{rate:.0f} 篇/s")

            # 每处理完一个源文件就把缓冲落盘并保存 manifest：
            # 语料到 500GB 时中途崩溃/断电不必从头再来。
            train_w.close()
            val_w.close()
            manifest["train_shards"].extend(train_w.take_shards())
            manifest["val_shards"].extend(val_w.take_shards())
            manifest["train_next_index"] = train_w.index
            manifest["val_next_index"] = val_w.index
            manifest["val_tokens"] = val_tokens
            manifest["sources"][name] = {"size": st.st_size, "mtime": st.st_mtime,
                                         "docs": file_docs}
            save_manifest(cfg, manifest)

            n_docs += file_docs
            log(f"  {name} 完成：{fmt_count(file_docs)} 篇，"
                f"耗时 {fmt_duration(time.time() - t_file)}")
            gc.collect()

    total_train = sum(s[1] for s in manifest["train_shards"])
    log(f"阶段二完成：{fmt_count(n_docs)} 篇文档 -> "
        f"train {fmt_count(total_train)} tokens ({len(manifest['train_shards'])} 片) + "
        f"val {fmt_count(val_tokens)} tokens，总耗时 {fmt_duration(time.time() - t_start)}")
    gc.collect()
    return manifest


# =============================================================================
#         数据管线：ramdisk 滚动预取的 IterableDataset（建于阶段三，用于阶段六）
# =============================================================================
def read_shard(path, n_tokens, fallback_path=None):
    """
    mmap 读取 uint16 分片。ramdisk 上的副本若不在了（例如上一轮的清理线程刚删掉），
    退回读磁盘上的源文件；再不行就整块读进内存。
    """
    for p in (path, fallback_path):
        if p is None or not os.path.exists(p):
            continue
        try:
            return torch.from_file(p, shared=False, size=n_tokens, dtype=torch.uint16)
        except Exception:
            with open(p, "rb") as f:
                return torch.frombuffer(bytearray(f.read()), dtype=torch.uint16)
    raise FileNotFoundError(f"分片不可读：{path}（备用 {fallback_path}）")


class ShardedTokenDataset(IterableDataset):
    """
    每个 DataLoader worker 负责 shard 列表的一个子集，并自带一个后台预取线程，
    把下一批分片从磁盘复制进 ramdisk（tmpfs），用完即删。
    所有 worker 在 ramdisk 上占用的总字节数受 budget_bytes 约束。
    """

    def __init__(self, shards, token_dir, ramdisk_dir, block_size,
                 budget_bytes, num_workers, seed, tag):
        self.shards = list(shards)             # [(name, n_tokens), ...]
        self.token_dir = token_dir
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
                src = os.path.join(self.token_dir, name)
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


def make_train_loader(cfg, dataset):
    # 这里的 worker 是在 CUDA 初始化之后 fork 的（训练进程里无法避免）。它安全的前提是
    # worker 内部绝不触碰 CUDA —— ShardedTokenDataset 只做文件拷贝和 CPU 张量切片，
    # 返回的也是普通 CPU 张量，搬到 GPU 是在主进程里做的。
    # 显式钉死 fork：Python 3.14 起 Linux 的默认 start method 会改成 forkserver，
    # 那会让每个 worker 重新 import torch（又慢又费内存），而这里 fork 才是正确且最省的。
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


def load_val_tokens(cfg, manifest):
    """验证集很小（<= max_val_tokens），一次性读进内存。"""
    parts = []
    for name, ntok in manifest["val_shards"]:
        path = os.path.join(cfg.token_dir, name)
        if not os.path.exists(path):
            continue
        parts.append(read_shard(path, ntok).clone())
    if not parts:
        return None
    return torch.cat(parts) if len(parts) > 1 else parts[0]


def val_batches(val_tokens, cfg, n_batches):
    """从验证 token 流里切出确定性的、互不重叠的 batch。"""
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
#                              GPT-3 网络结构
# =============================================================================
class CausalSelfAttention(nn.Module):
    """多头因果自注意力。用 F.scaled_dot_product_attention 手写，不用 nn.MultiheadAttention。"""

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


class Block(nn.Module):
    """Pre-LayerNorm 的 Transformer decoder block（GPT-2/GPT-3 风格）。"""

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

    def configure_optimizer(self, cfg):
        # 只对二维及以上的参数（矩阵、embedding）做 weight decay
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

    def flops_per_token(self):
        """前向+反向的近似 FLOPs/token（6*N + attention 的序列项）。"""
        c = self.cfg
        n = self.num_params(non_embedding=True)
        return 6 * n + 12 * c.n_layer * c.d_model * c.block_size


# =============================================================================
#                            学习率调度（按 token 数）
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
        log("没有找到 checkpoint，从零开始训练")
        return False

    log(f"从 checkpoint 恢复：{path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    raw_model.load_state_dict(strip_compile_prefix(ckpt["model"]))
    if "optimizer" in ckpt and optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        log("  ⚠ 该 checkpoint 不含 optimizer 状态（best 只存权重），优化器从零开始")
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
#                                  训练
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
        log(f"  ⚠ 验证时 OOM，本次验证跳过（已完成 {n}/{cfg.eval_iters} 个 batch）")
    finally:
        model.train()
    return losses / n if n else float("nan")


def check_disk_space(cfg):
    """启动前算清楚 checkpoint 到底要多少磁盘，不够就大声警告。"""
    with torch.device("meta"):
        n = GPT3(cfg).num_params()
    last = 12 * n          # fp32 权重 4n + AdamW 的 exp_avg/exp_avg_sq 各 4n
    best = 4 * n           # 只存权重
    # 原子写会先写 .tmp 再 rename，所以 ckpt_last 在切换的瞬间占双份
    peak = 2 * last + best

    for label, path in (("token_dir", cfg.token_dir), ("ckpt_dir", cfg.ckpt_dir)):
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
            log(f"  ⚠ ckpt_dir 剩余空间不足！需要 {peak / 1e9:.1f} GB，只有 {free / 1e9:.1f} GB。")
            log(f"  ⚠ 前一两次保存可能成功，但覆盖旧 ckpt_last 时会写满磁盘导致训练中断。")
            log(f"  ⚠ 请把 Config.ckpt_dir 改到更大的盘再启动。")
            log("  " + "!" * 70)


# =============================================================================
#                     阶段三：步数换算 + 数据管线（纯 CPU）
# =============================================================================
def compute_schedule(cfg):
    """把 Config 里的 token 预算换算成 micro-step / optimizer step。纯算术，无副作用。"""
    grad_accum = max(1, cfg.batch_tokens // (cfg.micro_bsz * cfg.block_size))
    tokens_per_step = grad_accum * cfg.micro_bsz * cfg.block_size
    planned = max(1, cfg.total_train_tokens // tokens_per_step)
    max_steps = planned
    if cfg.limit_steps is not None:
        max_steps = min(max_steps, cfg.limit_steps)

    log("=" * 78)
    log("阶段三：训练计划与数据管线")
    log("=" * 78)
    log(f"  micro_bsz={cfg.micro_bsz} × block_size={cfg.block_size} × "
        f"grad_accum={grad_accum} = {fmt_count(tokens_per_step)} tokens/step")
    log(f"  总计划 {fmt_count(cfg.total_train_tokens)} tokens = {planned} steps")
    if max_steps != planned:
        log(f"  （Config.limit_steps={cfg.limit_steps}，本次只跑 {max_steps} 步）")

    return {"grad_accum": grad_accum, "tokens_per_step": tokens_per_step,
            "max_steps": max_steps}


def build_data(cfg, manifest, sched):
    """
    建训练数据集对象并把验证集读进内存。全程纯 CPU，不碰 CUDA，所以刻意排在
    setup_torch_runtime 之前：分片缺失/损坏能立刻暴露，不用先白等建模型和 torch.compile。
    注意 ShardedTokenDataset 这里只是存几个字段，预取线程要到 __iter__ 在 DataLoader
    的 worker 进程里才启动。
    """
    train_ds = ShardedTokenDataset(
        manifest["train_shards"], cfg.token_dir, cfg.ramdisk_dir, cfg.block_size,
        cfg.ramdisk_budget_bytes, cfg.num_workers, cfg.seed, "train")

    val_tokens = load_val_tokens(cfg, manifest)
    log(f"  验证集 {fmt_count(val_tokens.numel()) if val_tokens is not None else 0} tokens")

    total_train_tokens_avail = sum(s[1] for s in manifest["train_shards"])
    log(f"  训练集 {fmt_count(total_train_tokens_avail)} tokens "
        f"({len(manifest['train_shards'])} 片) → 一个 epoch 约 "
        f"{total_train_tokens_avail / max(1, sched['tokens_per_step']):.0f} steps")

    return train_ds, val_tokens


# =============================================================================
#                   阶段四/五：CUDA 运行时 + 模型与优化器
# =============================================================================
def setup_torch_runtime(cfg):
    """
    播种、开 tf32、构造 autocast 上下文。★ 这是全流程第一次真正初始化 CUDA 的地方 ★
    在此之前的所有阶段（含阶段二那 20 个 fork 出来的分词进程）都必须保持 CUDA 未初始化。
    """
    log("=" * 78)
    log("阶段四：初始化 CUDA 运行时")
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

    # 只有这两种：float16 已在 preflight 里被拒（需要 GradScaler，本脚本没实现）
    ptdtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[cfg.dtype]
    autocast_ctx = torch.autocast(device_type="cuda", dtype=ptdtype) \
        if cfg.device == "cuda" and ptdtype != torch.float32 else torch.autocast(
            device_type="cuda", enabled=False)

    return device, autocast_ctx


def build_model(cfg, device):
    """建网络与优化器、恢复 checkpoint、torch.compile。返回 (model, raw_model, optimizer, state)。"""
    log("=" * 78)
    log("阶段五：模型与优化器")
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
            log(f"  ⚠ torch.compile 失败，改用 eager 模式：{e}")
            model = raw_model

    return model, raw_model, optimizer, state


# =============================================================================
#                             阶段六：训练主循环
# =============================================================================
def run_training(cfg, model, raw_model, optimizer, state,
                 train_ds, val_tokens, device, autocast_ctx, sched):
    grad_accum = sched["grad_accum"]
    tokens_per_step = sched["tokens_per_step"]
    max_steps = sched["max_steps"]

    log("=" * 78)
    log("阶段六：训练")
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

    # ------------------------------------------------------------- 训练循环
    model.train()
    flops_per_token = raw_model.flops_per_token()
    ema_step_time = None
    steps_this_run = 0        # 本次进程跑了几步，用来丢掉含编译时间的第一步
    consecutive_oom = 0
    last_val_loss = float("nan")
    total_vram = torch.cuda.get_device_properties(0).total_memory if cfg.device == "cuda" else 0
    micro_iter = iter(batch_gen)

    log("  开始训练。日志字段：ep=epoch, tok=已见 token 数, mfu=模型算力利用率")
    while state["step"] < max_steps:
        lr = lr_at(cfg, state["tokens_seen"])
        for group in optimizer.param_groups:
            group["lr"] = lr

        t0 = time.time()
        try:
            # loss 累加保持在 GPU 上，整个 step 只在最后同步一次，避免每个 micro-step
            # 都被 .item() 打断流水线
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
            # 丢掉这一步（约 30 秒的数据）重来，而不是让几天的训练直接崩掉。
            # 本步的 step/tokens_seen 都不计数，因为这一步等于没发生过。
            consecutive_oom += 1
            # 必须先断开这些张量的引用，empty_cache() 才有东西可回收。
            # （注意不能用 del locals()[...]，那对函数局部变量是无效的）
            loss = loss_acc = batch = x = y = None
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            log(f"  ⚠ OOM（第 {consecutive_oom}/{cfg.oom_max_retries} 次），已清理显存并丢弃本步重试。"
                f"若反复出现请调小 Config.micro_bsz")
            if consecutive_oom > cfg.oom_max_retries:
                log("  ✗ 连续 OOM 次数超限，这不是偶发抖动，退出。")
                raise
            continue
        consecutive_oom = 0
        dt = time.time() - t0
        # 第一步含 torch.compile 的编译时间（几十秒到几分钟），拿它给 EMA 播种会让
        # 接下来上百步的 tok/s 和 ETA 都严重偏低，所以直接丢掉。
        steps_this_run += 1
        if steps_this_run > 1:
            ema_step_time = dt if ema_step_time is None else 0.9 * ema_step_time + 0.1 * dt

        state["step"] += 1
        state["epoch"] = cur_epoch
        state["tokens_seen"] += tokens_per_step

        # -------------------------------------------------------- 验证
        if state["step"] % cfg.eval_every == 0 or state["step"] == max_steps:
            last_val_loss = evaluate(model, val_tokens, cfg, autocast_ctx, device)
            if last_val_loss == last_val_loss and last_val_loss < state["best_val_loss"]:
                state["best_val_loss"] = last_val_loss
                if not cfg.benchmark_only:
                    save_checkpoint(cfg, os.path.join(cfg.ckpt_dir, "ckpt_best.pt"),
                                    raw_model, optimizer, state, include_optimizer=False)
                    log(f"  ★ 新的最优 val loss {last_val_loss:.4f}，已写 ckpt_best.pt")

        # -------------------------------------------------------- 日志
        if state["step"] % cfg.log_every == 0 or state["step"] == 1:
            # 第一步还没有可用的 EMA（编译时间已被丢弃），先用本步耗时顶上
            est = ema_step_time if ema_step_time is not None else dt
            tok_per_s = tokens_per_step / max(1e-9, est)
            achieved = flops_per_token * tok_per_s
            mfu = achieved / cfg.gpu_peak_flops_bf16 * 100
            eta = (max_steps - state["step"]) * est
            # 显存一律用 GiB（二进制，÷2^30），好和 nvidia-smi 的 MiB 直接对得上；
            # 百分比是相对 CUDA 真正能分配的 total_memory，不是 nvidia-smi 的板载总量
            # （两者差一个驱动保留量，本机是 637 MiB）。
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

        # -------------------------------------------------------- checkpoint
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
def cleanup_ramdisk(cfg):
    """清掉上次异常退出可能残留的分片。"""
    if not os.path.isdir(cfg.ramdisk_dir):
        return
    for name in os.listdir(cfg.ramdisk_dir):
        if name.endswith(".bin") or name.endswith(".bin.part"):
            try:
                os.remove(os.path.join(cfg.ramdisk_dir, name))
            except OSError:
                pass


def preflight(cfg):
    """
    阶段零：纯 CPU 的启动自检。这里绝不能初始化 CUDA —— 阶段二要 fork 20 个分词进程。
    torch.cuda.is_available() 走 NVML（见文件头设的 PYTORCH_NVML_BASED_CUDA_CHECK），
    不会创建 primary context；get_device_name/get_device_properties 则会，所以显卡型号
    要等到阶段四 setup_torch_runtime 里再打印。
    """
    log("=" * 78)
    log("阶段零：启动自检")
    log("=" * 78)

    assert cfg.vocab_size <= 65536, "vocab_size 必须 <= 65536，token 才能用 uint16 存储"
    assert cfg.d_model % cfg.n_head == 0, "d_model 必须能被 n_head 整除"
    if cfg.dtype not in ("bfloat16", "float32"):
        # float16 的指数位只有 5 位（最小正规数 6.1e-5），深层网络里量级 1e-7~1e-9 的
        # 梯度会直接下溢成 0，必须配 GradScaler 才能训得动 —— 而本脚本刻意没有配：
        # bfloat16 的指数位和 fp32 一样是 8 位（最小正规数 1.2e-38），根本不会下溢，
        # 加 GradScaler 只会白白多一次 GPU->CPU 同步和一遍全量梯度扫描。
        # 何况 Blackwell 上 fp16 与 bf16 的张量核吞吐相同，切过去没有任何收益。
        # 与其静默地跑一个大概率发散的训练，不如在这里就停下。
        raise ValueError(
            f"Config.dtype={cfg.dtype!r} 不受支持，只能是 'bfloat16' 或 'float32'。"
            "float16 需要 GradScaler 才不会梯度下溢，本脚本没有实现；"
            "且在本机的 Blackwell 上 fp16 相对 bf16 没有速度优势，请用 bfloat16。")

    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")

    for d in (cfg.token_dir, cfg.ckpt_dir, cfg.ramdisk_dir):
        # 这里必须先 realpath 再建：ramdisk_dir 是指向 /dev/shm/myramdisk 的软链，而
        # /dev/shm 是 tmpfs，每次重启都会被清空，于是这条软链就变成悬空链接。
        # os.makedirs(exist_ok=True) 只在路径【已经是目录】时才吞掉 FileExistsError，
        # 悬空软链不算目录，所以会直接抛 FileExistsError: './myramdisk' 让训练起不来。
        os.makedirs(os.path.realpath(d), exist_ok=True)
    check_disk_space(cfg)          # 内部用 torch.device("meta") 建模型，同样不碰 CUDA
    cleanup_ramdisk(cfg)


def main():
    cfg = Config
    t_start = time.time()
    log("GPT-3 XL (1.3B) 单卡训练")

    # ---- 阶段零：纯 CPU 自检（不初始化 CUDA）
    preflight(cfg)

    # ---- 阶段一：分词器
    ensure_tokenizer(cfg)

    # ---- 阶段二：语料 -> uint16 token 分片（内部 fork 多进程，必须早于 CUDA 初始化）
    manifest = ensure_tokens(cfg)
    if not manifest["train_shards"]:
        raise RuntimeError("没有生成任何训练分片")

    # ---- 阶段三：换算 batch/步数（纯算术）+ 建数据管线（纯 CPU，仍不碰 CUDA）
    sched = compute_schedule(cfg)
    try:
        train_ds, val_tokens = build_data(cfg, manifest, sched)

        # ---- 阶段四：初始化 CUDA 运行时（从这一行起才允许碰 GPU）
        device, autocast_ctx = setup_torch_runtime(cfg)

        # ---- 阶段五：建模型 / 优化器 / 恢复 checkpoint / torch.compile
        model, raw_model, optimizer, state = build_model(cfg, device)

        # ---- 阶段六：训练主循环
        run_training(cfg, model, raw_model, optimizer, state,
                     train_ds, val_tokens, device, autocast_ctx, sched)
    finally:
        cleanup_ramdisk(cfg)
        gc.collect()

    log(f"总耗时 {fmt_duration(time.time() - t_start)}")


if __name__ == "__main__":
    main()
