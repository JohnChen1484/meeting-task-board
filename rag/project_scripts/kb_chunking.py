"""切片策略与预处理（第四周）：供 scripts/kb_import_files.py 调用。

切片策略全部写在 config/kb_chunking.json，导入脚本只读配置。三种方法：
- platform：把原文件交给平台解析，按分隔符切（第三周的 A/B/D/S 方案）；
- semantic：语义切片，不依赖分隔符。原文按句子拆开，逐句算向量，相邻两句的语义距离
  大于本文档距离分布的某个百分位时断开，再用最短 / 最长字数兜底；每段补上
  「文档名｜章节｜状态」前缀，保证溯源和废止版过滤照常工作；
- 任一方法可以打开 ocr：页面没有文字层（扫描件）时，取出页面图片，交给多模态模型逐字转写。

平台的向量接口一次只返回一个向量，所以逐句调用（并发 + 本地缓存，重复运行不重复花费）。
"""
import concurrent.futures
import difflib
import hashlib
import json
import math
import os
import re
import time
from typing import Callable, Dict, List, Optional, Tuple

PREFIX = "【来源：《{title}》｜{section}｜{status}】\n"
SEGMENT_SEPARATOR = "\u0002____SEGMENT_BOUNDARY____\u0002"  # 与 import_kb.py 一致：平台不会再切
TEXT_CHUNK = {"separator": SEGMENT_SEPARATOR, "max_tokens": 2000, "remove_extra_spaces": False,
              "remove_urls_emails": False}

DEFAULT_STRATEGIES = {
    "A": {"dataset": "qinghe_pdf_a", "dir": "assets/kb_files/pdf", "ext": ".pdf", "method": "platform",
          "chunk": {"separator": "\n", "max_tokens": 500, "remove_extra_spaces": True, "remove_urls_emails": False}},
    "B": {"dataset": "qinghe_pdf_b", "dir": "assets/kb_files/pdf", "ext": ".pdf", "method": "platform",
          "chunk": {"separator": "\n\n", "max_tokens": 800, "remove_extra_spaces": True, "remove_urls_emails": False}},
    "D": {"dataset": "qinghe_docx_a", "dir": "assets/kb_files/docx", "ext": ".docx", "method": "platform",
          "chunk": {"separator": "\n", "max_tokens": 500, "remove_extra_spaces": True, "remove_urls_emails": False}},
    "S": {"dataset": "qinghe_scan", "dir": "assets/kb_files/extra", "ext": ".pdf", "method": "platform",
          "chunk": {"separator": "\n", "max_tokens": 500, "remove_extra_spaces": True, "remove_urls_emails": False}},
}
NOISE_RE = re.compile(r"^(青禾数智科技有限公司\s*内部资料\s*请勿外传|第\s*\d+\s*页|本文档为虚构示例.*)$")
META_RE = re.compile(r"(文档编号|生效日期|发布部门)[：:]")
HEADING_RE = re.compile(r"^(#{1,6}\s*\S.*|第[一二三四五六七八九十百]+章\s*\S*|[一二三四五六七八九十]+、\S.*)$")
BREAK_BEFORE_RE = re.compile(r"^(第[一二三四五六七八九十百\d]+[条章]|[•·\-*]\s|\d+[.、．]\s*|[一二三四五六七八九十]+、|#|\|)")


# --------------------------------------------------------------------------- 配置
def load_strategies(root: str) -> Dict[str, Dict]:
    path = os.path.join(root, "config", "kb_chunking.json")
    if not os.path.exists(path):
        return DEFAULT_STRATEGIES
    data = json.load(open(path, encoding="utf-8"))
    strategies = data.get("strategies") or {}
    for key, cfg in strategies.items():
        for field in ("dataset", "dir", "ext", "method"):
            if not cfg.get(field):
                raise ValueError(f"config/kb_chunking.json 方案 {key} 缺少 {field}")
        if cfg["method"] not in ("platform", "semantic"):
            raise ValueError(f"方案 {key} 的 method 只能是 platform 或 semantic")
        if cfg["method"] == "platform" and not cfg.get("chunk"):
            raise ValueError(f"方案 {key}（platform）缺少 chunk 参数")
    return strategies


def config_digest(cfg: Dict) -> str:
    """切片参数的指纹：参数一改，同一个文件也会被视为新版本重新导入。"""
    return hashlib.sha256(json.dumps(cfg, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- 取文字
def extract_pages(path: str) -> List[str]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        return [(p.extract_text() or "") for p in PdfReader(path).pages]
    if ext == ".docx":
        from docx2python import docx2python
        with docx2python(path) as doc:
            return [doc.text]
    return [open(path, encoding="utf-8").read()]


def page_images(path: str, page_index: int) -> List[Tuple[str, bytes]]:
    from pypdf import PdfReader
    page = PdfReader(path).pages[page_index]
    return [(img.name, img.data) for img in page.images]


OCR_PROMPT = (
    "请把图片中的文字逐字转写出来。要求：保持原有的标题、条款编号和段落；表格转成 Markdown 表格；"
    "看不清的字写成□，不要猜；不要补充、解释或总结；只输出转写的文字。"
)


def ocr_image(data: bytes, name: str) -> str:
    """多模态模型逐字转写一张图片。图片先放到项目对象存储，用预签名链接交给模型。"""
    from coze_coding_dev_sdk import S3SyncStorage
    from coze_coding_dev_sdk import LLMClient
    from langchain_core.messages import HumanMessage
    storage = S3SyncStorage()
    ext = os.path.splitext(name)[1].lower().lstrip(".") or "png"
    key = storage.upload_file(file_content=data, file_name=f"ocr_{hashlib.sha1(data).hexdigest()[:12]}.{ext}",
                              content_type=f"image/{'jpeg' if ext in ('jpg', 'jpeg') else ext}")
    url = storage.generate_presigned_url(key=key, expire_time=3600)
    msg = HumanMessage(content=[{"type": "text", "text": OCR_PROMPT}, {"type": "image_url", "image_url": {"url": url}}])
    reply = LLMClient().invoke(messages=[msg], temperature=0)
    content = reply.content
    if isinstance(content, list):
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content).strip()


def document_text(path: str, ocr: bool, cache_dir: str, log: Callable[[str], None] = print,
                  ocr_fn: Callable[[bytes, str], str] = ocr_image) -> Tuple[str, Dict]:
    """返回 (全文, 记录)。没有文字层的 PDF 页：ocr 打开时转写（结果缓存到 cache_dir），否则跳过并提示。"""
    pages = extract_pages(path)
    info = {"pages": len(pages), "ocr_pages": 0, "empty_pages": 0}
    out = []
    for i, text in enumerate(pages):
        if len(re.sub(r"\s+", "", text)) >= 20:
            out.append(text)
            continue
        if not ocr or not path.lower().endswith(".pdf"):
            info["empty_pages"] += 1
            log(f"    第 {i + 1} 页没有文字层（可能是扫描件），已跳过；在配置里打开 ocr 可转写")
            continue
        os.makedirs(cache_dir, exist_ok=True)
        cache = os.path.join(cache_dir, f"{os.path.splitext(os.path.basename(path))[0]}_p{i + 1}.txt")
        if os.path.exists(cache):
            out.append(open(cache, encoding="utf-8").read())
        else:
            texts = [ocr_fn(data, name) for name, data in page_images(path, i)]
            page_text = "\n".join(t for t in texts if t)
            open(cache, "w", encoding="utf-8").write(page_text)
            out.append(page_text)
        info["ocr_pages"] += 1
        log(f"    第 {i + 1} 页：OCR 转写 {len(out[-1])} 字（缓存 {os.path.relpath(cache)}）")
    return "\n".join(out), info


def ocr_accuracy(ocr_text: str, reference: str) -> float:
    """转写结果与原文的相似度（去空白、去表格符号后按字符比对，1 = 完全一致）。"""
    norm = lambda s: re.sub(r"[\s|#*>\-•]+", "", NOISE_LINE_SUB(s))
    a, b = norm(ocr_text), norm(reference)
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() if a and b else 0.0


def NOISE_LINE_SUB(text: str) -> str:  # noqa: N802
    return "\n".join(ln for ln in (text or "").splitlines() if not NOISE_RE.match(ln.strip()))


# --------------------------------------------------------------------------- 拆句
def clean_lines(text: str, title: str = "") -> List[str]:
    """去掉页眉页脚、文档标题行和「文档编号 / 生效日期 / 发布部门」这类元信息行。"""
    lines = []
    bare_title = re.sub(r"\s+", "", title)
    for raw in (text or "").splitlines():
        line = raw.strip().lstrip(">").strip()
        if not line or NOISE_RE.match(line) or META_RE.search(line):
            continue
        if bare_title and re.sub(r"^#+|\s+", "", line) == bare_title:
            continue
        lines.append(line)
    return lines


def join_lines(lines: List[str]) -> List[str]:
    """PDF 会在行宽处硬换行：上一行没以句末标点结束、下一行也不是标题 / 条款 / 列表项时，接回去。"""
    out: List[str] = []
    for line in lines:
        if (out and not re.search(r"[。！？；：:]$", out[-1]) and not BREAK_BEFORE_RE.match(line)
                and not HEADING_RE.match(out[-1]) and not HEADING_RE.match(line)):
            out[-1] += line
        else:
            out.append(line)
    return out


def split_units(text: str, title: str = "") -> List[Dict]:
    """拆成语义单元：标题单独一个；正文按句末标点（。！？；）拆句。"""
    units = []
    for line in join_lines(clean_lines(text, title)):
        if HEADING_RE.match(line):
            units.append({"kind": "heading", "text": re.sub(r"^#+\s*", "", line)})
            continue
        for sent in re.findall(r"[^。！？；]+[。！？；]?", line):
            if sent.strip():
                units.append({"kind": "text", "text": sent.strip()})
    return units


# --------------------------------------------------------------------------- 语义切片
def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 1.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def semantic_chunks(units: List[Dict], embed: Callable[[List[str]], List[List[float]]], params: Dict,
                    title: str = "") -> Tuple[List[Dict], Dict]:
    """返回 ([{section, text}], 记录)。断点 = 相邻单元语义距离超过本文档距离的第 percentile 百分位。"""
    min_chars = int(params.get("min_chars", 60))
    max_chars = int(params.get("max_chars", 450))
    pct = float(params.get("breakpoint_percentile", 85))
    on_heading = bool(params.get("break_on_heading", False))
    if not units:
        return [], {"units": 0}
    vectors = embed([u["text"] for u in units])
    dists = [1 - cosine(vectors[i], vectors[i + 1]) for i in range(len(units) - 1)]
    threshold = percentile(dists, pct)
    chunks, cur, section = [], [], ""
    reasons = {"semantic": 0, "max_chars": 0, "heading": 0}

    def flush():
        nonlocal cur
        body = "".join(u["text"] if u["kind"] == "text" else f"\n{u['text']}\n" for u in cur).strip()
        # 章节 = 段落开始时所在的章节 + 段落里新出现的章节（一段跨两章时两个都写上）
        labels = [cur[0]["section"]] if cur and cur[0]["section"] and cur[0]["kind"] == "text" else []
        labels += [u["text"] for u in cur if u["kind"] == "heading"]
        labels = [x for i, x in enumerate(labels) if x and x not in labels[:i]]
        if body:
            chunks.append({"section": "、".join(labels) or title, "text": body})
        cur = []

    for i, unit in enumerate(units):
        size = sum(len(u["text"]) for u in cur)
        if cur:
            gap = dists[i - 1]
            if unit["kind"] == "heading" and on_heading:
                reasons["heading"] += 1
                flush()
            elif size + len(unit["text"]) > max_chars:
                reasons["max_chars"] += 1
                flush()
            elif gap > threshold and size >= min_chars:
                reasons["semantic"] += 1
                flush()
        if unit["kind"] == "heading":
            section = unit["text"]
        cur.append(dict(unit, section=section))
    flush()
    return chunks, {"units": len(units), "threshold": round(threshold, 4), "breaks": reasons}


def prefixed(title: str, section: str, status: str, text: str) -> str:
    return PREFIX.format(title=title, section=section or title, status=status) + text


# --------------------------------------------------------------------------- 向量
class CachedEmbedder:
    """逐句调用平台向量接口（一次一个向量），并发 6 路，结果按文字缓存到文件，重复运行不再花费。"""

    def __init__(self, cache_path: str, workers: int = 6, client=None):
        self.cache_path = cache_path
        self.cache: Dict[str, List[float]] = {}
        if os.path.exists(cache_path):
            self.cache = json.load(open(cache_path, encoding="utf-8"))
        self.workers = workers
        self.client = client
        self.calls = 0

    def _client(self):
        if self.client is None:
            from coze_coding_dev_sdk import EmbeddingClient
            self.client = EmbeddingClient()
        return self.client

    def _one(self, text: str) -> List[float]:
        last = None
        for attempt in range(3):
            try:
                self.calls += 1
                return list(self._client().embed_texts([text]))
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(1 + attempt)
        raise RuntimeError(f"向量接口连续失败：{type(last).__name__}: {last}")

    def __call__(self, texts: List[str]) -> List[List[float]]:
        key = lambda t: hashlib.sha1(t.encode()).hexdigest()
        todo = sorted({t for t in texts if key(t) not in self.cache})
        if todo:
            with concurrent.futures.ThreadPoolExecutor(self.workers) as pool:
                for t, vec in zip(todo, pool.map(self._one, todo)):
                    self.cache[key(t)] = vec
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            json.dump(self.cache, open(self.cache_path, "w", encoding="utf-8"))
        return [self.cache[key(t)] for t in texts]
