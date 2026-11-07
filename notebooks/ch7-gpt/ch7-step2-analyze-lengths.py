#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计 ./tokens 里语料的文档长度分布，用来决定 Config.block_size（上下文长度）该取多少。

原理：分词阶段每篇文档后面都追加了一个 eos，所以 token 流里相邻两个 eos 的间距
就是一篇文档的精确长度。这是全量精确统计，不是抽样。

用法（需要先跑过 gpt3_train.py 的阶段一、二，即 ./tokens 已经生成）：
    python analyze_lengths.py
"""

import json
import os
import time

import torch

import gpt3_train as G


def collect_doc_lengths(cfg, shards, device):
    """把若干分片当成一条连续的 token 流，扫出每篇文档的长度（不含 eos）。"""
    eos = cfg.sp_eos_id
    lengths = []
    prev_eos = -1          # 上一个 eos 在全局流里的位置
    offset = 0             # 当前分片起点的全局偏移
    t0 = time.time()
    for i, (name, ntok) in enumerate(shards):
        path = os.path.join(cfg.token_dir, name)
        buf = G.read_shard(path, ntok)
        idx = (buf.to(device).to(torch.int32) == eos).nonzero(as_tuple=True)[0]
        if idx.numel():
            pos = idx.to(torch.int64).cpu() + offset
            # 相邻 eos 的间距减 1 = 文档内容长度；第一个 eos 单独用 prev_eos 处理
            starts = torch.cat([torch.tensor([prev_eos], dtype=torch.int64), pos[:-1]])
            lengths.append(pos - starts - 1)
            prev_eos = int(pos[-1])
        offset += ntok
        del buf
        if (i + 1) % 10 == 0 or i + 1 == len(shards):
            print(f"    扫描 {i + 1}/{len(shards)} 片…", flush=True)
    print(f"    扫描完成，耗时 {time.time() - t0:.1f}s")
    return torch.cat(lengths) if lengths else torch.tensor([], dtype=torch.int64)


def bar(frac, width=42):
    n = int(round(frac * width))
    return "█" * n + "·" * (width - n)


def report(name, L, chars_per_token):
    n = L.numel()
    total = int(L.sum())
    Ls, _ = torch.sort(L)

    def pct(p):
        return int(Ls[min(n - 1, int(n * p / 100))])

    print()
    print("=" * 86)
    print(f"{name}：{n:,} 篇文档，共 {total / 1e9:.3f}B tokens")
    print("=" * 86)
    print(f"  最短 {int(Ls[0]):,}   最长 {int(Ls[-1]):,}   "
          f"平均 {total / n:,.0f}   中位数 {pct(50):,}   "
          f"标准差 {float(L.float().std()):,.0f}")
    # 按 token 加权的平均长度：随机抽一个 token，它所在文档的期望长度。
    # 这个数比按篇数平均更能反映"训练时实际见到的文档有多长"。
    tw = float((L.double() * L.double()).sum() / total)
    print(f"  token 加权平均长度 {tw:,.0f}"
          f"（随机取一个 token，它所在文档的期望长度；总是 >= 篇数平均）")
    print(f"  最长文档约相当于 {int(Ls[-1]) * chars_per_token:,.0f} 个汉字")

    print()
    print("  分位数（tokens）：")
    row = "   "
    for p in (1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9, 99.99):
        row += f" p{p}={pct(p):,}"
    print(row)

    print()
    print("  长度直方图：")
    buckets = [(0, 256), (256, 512), (512, 1024), (1024, 2048), (2048, 4096),
               (4096, 8192), (8192, 16384), (16384, 32768), (32768, 1 << 30)]
    for lo, hi in buckets:
        m = (L >= lo) & (L < hi)
        c = int(m.sum())
        t = int(L[m].sum())
        label = f"{lo:>6,}–{hi:<7,}" if hi < (1 << 30) else f"{lo:>6,}+       "
        print(f"    {label} {bar(c / n)} {c / n * 100:6.2f}% 篇 | "
              f"{t / total * 100:6.2f}% token")

    print()
    print("  按候选上下文长度 C 评估：")
    print("    C       完整装下的篇数  完整装下的token  被截断的token   窗口跨文档数")
    for C in (512, 1024, 2048, 4096, 8192, 16384, 32768):
        fits = L <= C
        n_fit = int(fits.sum())
        t_fit = int(L[fits].sum())
        # 长于 C 的文档里，超出 C 的那部分 token 会被切到别的窗口去
        over = L[~fits]
        t_trunc = int((over - C).sum()) if over.numel() else 0
        print(f"    {C:>6,}   {n_fit / n * 100:12.2f}%  {t_fit / total * 100:14.2f}%"
              f"  {t_trunc / total * 100:13.2f}%  {C / (total / n):11.1f} 篇")


def main():
    cfg = G.Config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mpath = os.path.join(cfg.token_dir, "manifest.json")
    if not os.path.exists(mpath):
        raise SystemExit(f"找不到 {mpath}，请先运行 gpt3_train.py 生成 token 分片")
    with open(mpath, encoding="utf-8") as f:
        manifest = json.load(f)

    print(f"设备 {device}｜词表 {manifest['vocab_size']}｜"
          f"train {len(manifest['train_shards'])} 片 + val {len(manifest['val_shards'])} 片")

    # 用分词器实测语料的字/token 比，好把 token 数换算成汉字数
    sp_path = cfg.tokenizer_path
    cpt = 1.87
    if os.path.exists(sp_path):
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=sp_path)
        names = G.list_jsonl_files(cfg.data_dir)
        with open(os.path.join(cfg.data_dir, names[0]), encoding="utf-8") as f:
            docs = [json.loads(next(f))["text"] for _ in range(300)]
        cpt = sum(len(d) for d in docs) / sum(len(x) for x in sp.encode(docs))
        print(f"字/token 比：{cpt:.2f}")

    print("\n扫描 train 分片…")
    tr = collect_doc_lengths(cfg, manifest["train_shards"], device)
    print("扫描 val 分片…")
    va = collect_doc_lengths(cfg, manifest["val_shards"], device)

    report("训练集", tr, cpt)
    if va.numel():
        report("验证集", va, cpt)

    print()
    print("=" * 86)
    print("注：文档之间只用 eos 分隔、没有跨文档注意力掩码，所以一个训练窗口会横跨")
    print("    多篇不相关的文档。上表最后一列就是平均横跨的文档数。")
    print("=" * 86)


if __name__ == "__main__":
    main()
