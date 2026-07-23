import re
import sys

# ===================== 配置项 =====================
# 基础清洗开关
REMOVE_XML_TAGS = True
FIX_STAR_GARBLED = True
CLEAN_CONTROL_CHARS = True
REMOVE_INDENT = True
STRIP_LINE_END = True
MERGE_SOFT_LINE_BREAK = True
NORMALIZE_PUNCTUATION = True
CLEAN_INNER_SPACE = True
REMOVE_NON_CONTENT = True

# 特殊标记配置
ADD_SPECIAL_TOKENS = True
# EOS 模式可选:
#   full       - 全文仅1个（不推荐）
#   paragraph  - 每个段落1个
#   sentence   - 每句话1个（推荐，密度高）
#   fixed      - 固定长度强制插入
EOS_MODE = "sentence"
FIXED_EOS_LENGTH = 80  # fixed 模式下每多少字符插一个 EOS

SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
PARA_TOKEN = "<PARA>"
# ==================================================

# 星号乱码映射表
STAR_GARBLED_MAP = {
    "*沂怯滞纯煊纸粽牛乇鹗悄歉鼋*": "我是又痛苦又紧张，特别是那个紧",
    "*杏晕易*": "有赢，所以我总",
    "**": "声：",
    "*坏健*": "听不到。",
    "*兆踊岣*": "日子会更",
    "*课莅岬矫┪堇*": "房屋搬到茅屋里",
    "*绱倒*": "风吹过来",
    "*运担*": "，对她说：",
    "*担*": "里，她说：",
    "*荼纠淳*": "草本来就",
    "*妓滴颐橇礁龊*": "始说我们两个很",
    "*吡耍*": "走了，我",
    "低押出去": "抵押出去",
}


def read_file(path: str) -> str:
    encodings = ["utf-8-sig", "utf-8", "gbk"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def remove_xml_tags(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"<!\[CDATA\[|\]\]>", "", text)
    text = re.sub(r"&[a-zA-Z]+;", "", text)
    return text


def fix_star_garbled(text: str) -> tuple[str, int, int, int]:
    total, fixed, removed = 0, 0, 0
    def replacer(match):
        nonlocal total, fixed, removed
        total += 1
        garbled_full = match.group(0)
        if garbled_full in STAR_GARBLED_MAP:
            fixed += 1
            return STAR_GARBLED_MAP[garbled_full]
        else:
            removed += 1
            return ""
    text = re.sub(r'\*[^*]+\*', replacer, text)
    for wrong, right in STAR_GARBLED_MAP.items():
        if not wrong.startswith("*"):
            cnt = text.count(wrong)
            if cnt > 0:
                text = text.replace(wrong, right)
                fixed += cnt
    return text, total, fixed, removed


def clean_control_chars(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff\ufffe]", "", text)
    return text


def strip_line_edges(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        if REMOVE_INDENT:
            line = line.lstrip(" 　\t")
        if STRIP_LINE_END:
            line = line.rstrip()
        cleaned.append(line)
    return "\n".join(cleaned)


def clean_inner_space(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        line = re.sub(r"[ 　\t]+", " ", line)
        cleaned.append(line.strip())
    return "\n".join(cleaned)


def merge_soft_line_breaks(text: str) -> str:
    paragraphs = []
    current_para = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current_para:
                paragraphs.append("".join(current_para))
                current_para = []
        else:
            current_para.append(stripped)
    if current_para:
        paragraphs.append("".join(current_para))
    return "\n\n".join(paragraphs)


def normalize_punctuation(text: str) -> str:
    punct_map = {
        ",": "，", ".": "。", "!": "！", "?": "？",
        ":": "：", ";": "；", "(": "（", ")": "）",
        "[": "【", "]": "】",
    }
    for en_p, zh_p in punct_map.items():
        text = text.replace(en_p, zh_p)
    text = re.sub(r"。{3,}", "……", text)
    text = re.sub(r"，{2,}", "，", text)
    text = re.sub(r"。{2,}", "。", text)
    text = re.sub(r"-{2,}", "——", text)
    return text


def remove_non_content(text: str) -> str:
    text = re.sub(r'^[一二三四五六七八九十百]+、\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^前言.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^---.*全书完.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def add_special_tokens(text: str) -> str:
    if EOS_MODE == "full":
        text = re.sub(r'\n{2,}', PARA_TOKEN, text)
        return SOS_TOKEN + text + EOS_TOKEN

    elif EOS_MODE == "paragraph":
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
        marked = [f"{SOS_TOKEN}{p}{EOS_TOKEN}" for p in paragraphs]
        return "\n".join(marked)

    elif EOS_MODE == "sentence":
        # 句末标点：。！？…… 后面加 EOS
        # 先把段落换行换成 <PARA>，再在句末加 EOS
        text = re.sub(r'\n{2,}', PARA_TOKEN, text)
        # 在句末标点后插入 EOS（引号、书名号等后闭合先不处理，简单有效）
        text = re.sub(r'([。！？……])', rf'\1{EOS_TOKEN}', text)
        # 全文首尾再加 SOS/EOS 兜底
        return SOS_TOKEN + text + EOS_TOKEN

    elif EOS_MODE == "fixed":
        # 先去掉所有换行，变成一长串，再每 N 字插一个 EOS
        text = re.sub(r'\s+', '', text)
        chars = list(text)
        result = []
        for i, c in enumerate(chars):
            result.append(c)
            if (i + 1) % FIXED_EOS_LENGTH == 0:
                result.append(EOS_TOKEN)
        return SOS_TOKEN + "".join(result) + EOS_TOKEN

    else:
        raise ValueError(f"未知 EOS_MODE: {EOS_MODE}")


def main():
    if len(sys.argv) < 2:
        print("用法: python clean_text.py <输入文件名>")
        print("示例: python clean_text.py 余华 活着.txt")
        sys.exit(1)

    input_file = sys.argv[1]
    name_part = input_file.rsplit(".", 1)[0]
    output_file = f"{name_part}-cleaned.txt"

    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"EOS 模式: {EOS_MODE}", end="")
    if EOS_MODE == "fixed":
        print(f"（每 {FIXED_EOS_LENGTH} 字一个）")
    else:
        print()
    print()

    text = read_file(input_file)
    print(f"原始文本长度: {len(text)} 字符")

    if REMOVE_XML_TAGS:
        text = remove_xml_tags(text)
        print("✓ 已清除XML/HTML标签")

    if FIX_STAR_GARBLED:
        text, total, fixed, removed = fix_star_garbled(text)
        print(f"✓ 星号乱码：匹配 {total} 处，修复 {fixed} 处，删除 {removed} 处")

    if CLEAN_CONTROL_CHARS:
        text = clean_control_chars(text)
        print("✓ 已清除控制字符")

    if REMOVE_INDENT or STRIP_LINE_END:
        text = strip_line_edges(text)
        print("✓ 已清理行首尾空白")

    if CLEAN_INNER_SPACE:
        text = clean_inner_space(text)
        print("✓ 已清理行内多余空白")

    if MERGE_SOFT_LINE_BREAK:
        text = merge_soft_line_breaks(text)
        print("✓ 已合并段落内硬换行")

    if NORMALIZE_PUNCTUATION:
        text = normalize_punctuation(text)
        print("✓ 已规范化标点")

    if REMOVE_NON_CONTENT:
        text = remove_non_content(text)
        print("✓ 已移除非正文标记")

    if ADD_SPECIAL_TOKENS:
        text = add_special_tokens(text)
        sos_cnt = text.count(SOS_TOKEN)
        eos_cnt = text.count(EOS_TOKEN)
        para_cnt = text.count(PARA_TOKEN)
        print(f"✓ 特殊标记添加完成")
        print(f"  {SOS_TOKEN}: {sos_cnt} 个")
        print(f"  {EOS_TOKEN}: {eos_cnt} 个")
        if para_cnt:
            print(f"  {PARA_TOKEN}: {para_cnt} 个")

    vocab_size = len(set(text))
    print(f"\n词汇表大小: {vocab_size} 字符")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"\n清洗完成，已保存到: {output_file}")
    print(f"最终长度: {len(text)} 字符")


if __name__ == "__main__":
    main()
