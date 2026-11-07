#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import array
import json
import math
import os
import gc
import queue
import random
import shutil
import threading
import time
import torch
import sentencepiece as spm
import multiprocessing as mp

# 这段代码需要比较大的内存，最好 32 G往上
# 如果程序自己退出了。执行 echo $? 看到退出码是 137 ，那就是 SIGKILL，就是 OOM 内存不够了

# =============================================================================
#                                  配置
# =============================================================================
class Config:
    data_dir = "./cci3-hq-data/data"     # 存放 *.jsonl 的目录
    tokenizer_path = "./tokenizer.model" # SentencePiece 模型（训练一次，后续复用）
    token_bin_dir = "./token_bins"       # uint16 token 分片输出目录
    sample_filename = "sp_sample.txt"    # token 采样文件名

    # -------------------------- BPE 训练 --------------------------
    vocab_size = 64000                   # < 65536，token 可用 uint16 存储
    sp_model_type = "bpe"
    sp_character_coverage = 0.9995       # 中文语料建议 0.9995
    sp_byte_fallback = True              # 未登录字符回退到字节，保证无损
    sp_normalization = "identity"        # 归一化用 identity 而不是 nmt_nfkc：NFKC 会把全角逗号"，"转成半角","，
    sp_add_dummy_prefix = False          # 中文不需要在句首补空格
    sp_remove_extra_whitespaces = False  # 保留原始空白，保证无损还原
    sp_sample_chars = 3_000_000_000      # 分词器训练语料的抽样量 - 字符上限，至少应有 3 亿。也不能太多，太多 BPE 合并会很慢。 3 亿字符 - 1G 文本 - 7G 内存占用
    sp_sample_sentences = 900_000_000    # 分词器训练语料的抽样量 - 句子数上限，通常先撞到字符数限额
    sp_doc_stride = 8                    # 每 8 篇文档取 1 篇，让抽样铺开到整个语料
    sp_max_sentence_bytes = 8192         # 超过此长度的句子会被切开
    sp_num_threads = 22
    sp_unk_id, sp_bos_id, sp_eos_id, sp_pad_id = 0, 1, 2, 3

    # ---------------------- 语料转 token 分片 -----------------------
    tokenize_processes = 20              # 分词并行进程数
    tokenize_batch_docs = 512            # 每个任务包含多少篇文档
    tokenize_super_batch = 44            # 一次派发多少个任务包（控制内存占用）
    shard_bytes = 128 * 1024 * 1024      # 单个 .bin 分片大小（128 MiB ≈ 6710 万 token）
    val_doc_modulo = 1000                # 每 1000 篇文档抽 1 篇进验证集
    max_val_tokens = 5_000_000           # 验证集 token 上限
    limit_docs_per_file = None           # 每个 jsonl 只读前 N 篇文档（None = 全读）
    block_size = 4096                    # 上下文长度。原始论文中为 2048。cci3-hq 中 17.6% 的文档长度大于 2048 ， 6.4% 大于 4096，2.1% 大于 8192 ， 故取此值


# =============================================================================
#                                  Utils
# =============================================================================
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

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


def list_jsonl_files(data_dir):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"找不到语料目录 {data_dir}")
    names = sorted(f for f in os.listdir(data_dir) if f.endswith(".jsonl"))
    if not names:
        raise FileNotFoundError(f"{data_dir} 下没有 .jsonl 文件")
    return names


# 流式读一个 jsonl，产出 (doc_id, text)
def iter_docs(path, limit=None):
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


# 确定性地把一小部分文档划到验证集；与语料文件数量无关
def is_val_doc(doc_id, text, modulo):
    key = doc_id if doc_id else text[:64]
    h = 0
    for ch in key[:16]:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h % modulo == 0


# =============================================================================
#                       阶段一：训练 SentencePiece BPE 分词器
# =============================================================================

# 按换行切句；过长的行再按标点/硬切分成 <= max_bytes 的片段
def split_into_sentences(text, max_bytes):
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


"""
从所有 jsonl 里轮转抽样，写成 SentencePiece 的训练输入。
同时受字符数与句子数两个上限约束，并按 sp_doc_stride 跳着取，
让样本铺开到整个语料而不是只覆盖每个文件的开头。
"""
def build_tokenizer_sample(cfg, sample_path):
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


def train_tokenizer(cfg):
    if os.path.exists(cfg.tokenizer_path):
        sp = spm.SentencePieceProcessor(model_file=cfg.tokenizer_path)
        log(f"复用已有分词器 {cfg.tokenizer_path}（vocab={sp.get_piece_size()}）")
        return sp

    log("=" * 78)
    log("训练 SentencePiece BPE 分词器")
    log("=" * 78)
    sample_path = os.path.join(cfg.token_bin_dir, cfg.sample_filename)
    try:
        build_tokenizer_sample(cfg, sample_path)
        prefix = cfg.tokenizer_path[:-len(".model")] if cfg.tokenizer_path.endswith(".model") \
            else cfg.tokenizer_path
        log(f"  开始训练 BPE（vocab_size={cfg.vocab_size}），这一步可能要几十分钟...")
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


# 把 token 流按固定字节数切成 uint16 的 .bin 分片 
class ShardWriter:

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
        """把剩余不满一片的 token 也落盘，用于每处理完一个源文件时做断点。"""
        self._flush(len(self.buf))

    def take_shards(self):
        """取走并清空自上次以来新产生的分片列表。"""
        s, self.shards = self.shards, []
        return s


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


def save_manifest(cfg, manifest):
    path = os.path.join(cfg.token_bin_dir, "manifest.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def remove_orphan_shards(cfg, manifest):
    """删除 token_bin_dir 里没被 manifest 记录的 .bin（上次异常中断的残留）。"""
    known = {s[0] for s in manifest["train_shards"]} | {s[0] for s in manifest["val_shards"]}
    removed = 0
    for name in os.listdir(cfg.token_bin_dir):
        if name.endswith(".bin") and name not in known:
            try:
                os.remove(os.path.join(cfg.token_bin_dir, name))
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


# 多进程的子例程
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

# 把 jsonl 编码成 token 分片。返回 manifest
def build_token_bin(cfg):

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
    log(f"分词 {len(todo)} 个新的 jsonl 文件 -> uint16 token 分片")
    log("=" * 78)

    # 清掉上次异常中断留下的、没被 manifest 记录的孤儿分片
    remove_orphan_shards(cfg, manifest)

    train_w = ShardWriter(cfg.token_bin_dir, "train", manifest["train_next_index"], cfg.shard_bytes)
    val_w = ShardWriter(cfg.token_bin_dir, "val", manifest["val_next_index"], cfg.shard_bytes)
    val_tokens = manifest["val_tokens"]

    assert not torch.cuda.is_initialized(), "分词的多进程 fork 必须发生在 CUDA 初始化之前"

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

            # 每处理完一个源文件就把缓冲落盘并保存 manifest
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
    log(f"完成预分词：{fmt_count(n_docs)} 篇文档 -> "
        f"train {fmt_count(total_train)} tokens ({len(manifest['train_shards'])} 片) + "
        f"val {fmt_count(val_tokens)} tokens，总耗时 {fmt_duration(time.time() - t_start)}")
    gc.collect()
    return manifest


def main():
    cfg = Config
    t_start = time.time()

    log("Begin BPE Training...")

    assert cfg.vocab_size <= 65536, "vocab_size 必须 <= 65536，token 才能用 uint16 存储"
    os.makedirs(os.path.realpath(cfg.token_bin_dir), exist_ok=True)

    train_tokenizer(cfg)
    manifest = build_token_bin(cfg)

    if not manifest["train_shards"]:
        raise RuntimeError("没有生成任何训练分片")

    log(f"总耗时 {fmt_duration(time.time() - t_start)}")


if __name__ == "__main__":
    main()
