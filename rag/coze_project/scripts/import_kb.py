#!/usr/bin/env python3
"""知识库导入脚本（幂等，按环境区分索引，可重复执行）。

流程：
1. 读取 assets/kb_docs/ 下的全部 Markdown 制度文档（原样读取，不修改文件）；
2. 按规则预切分段落：
   - 文档第一行 "# 标题" 作为《文档标题》；
   - 含「已废止」的文档状态为"已废止"，其余为"现行"；
   - 以"第X条"为一段；无条款编号的，以 "## 小标题" 下的内容为一段；
   - 一张表格作为完整一段，不拆开；
   - 单段不超过 600 字，超过时按空行再切分；
3. 每段开头加来源前缀：【来源：《文档标题》｜章节标题｜状态】；
4. 以 TEXT 方式逐段导入数据集 qinghe_kb（每段一个文档，便于生成 doc_id 映射）；
5. 生成按环境区分的 assets/kb_index.{env}.json：doc_id → 文件名、章节、状态 的映射表；
6. 幂等：以内容 sha256 为键，重复执行时跳过已存在的内容，不产生重复数据。

用法：
    python scripts/import_kb.py --env dev            # 导入开发库
    python scripts/import_kb.py --env prod           # 导入生产库
    python scripts/import_kb.py --env dev --verify   # 只检索验证，不导入
    python scripts/import_kb.py --env prod --verify  # 只检索验证，不导入
    python scripts/import_kb.py --env prod --if-missing  # 按知识库内标记判断是否已导入
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time

# 线上容器文件系统只读，禁止生成 __pycache__/.pyc 缓存文件。
sys.dont_write_bytecode = True

from coze_coding_dev_sdk import (
    ChunkConfig,
    Config,
    DataSourceType,
    KnowledgeClient,
    KnowledgeDocument,
)

WORKSPACE = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
KB_DOCS_DIR = os.path.join(WORKSPACE, "assets", "kb_docs")
DATASET = "qinghe_kb"

# --env 取值 → 平台环境变量（SDK 通过 COZE_PROJECT_ENV 选择 DEV/PROD 资源）
PROJECT_ENV_MAP = {"dev": "DEV", "prod": "PROD"}

# 预切分后每段作为单一 chunk 导入，使用一个正文中不可能出现的分隔符，
# 避免后端二次切分把「来源前缀 + 内容」拆开。
SEGMENT_SEPARATOR = "\u0002____SEGMENT_BOUNDARY____\u0002"
MAX_TOKENS = 2000
MAX_SEGMENT_CHARS = 600

CLAUSE_RE = re.compile(r"^第[一二三四五六七八九十百千万零0-9]+条")

# 文档顶部元信息（文档编号、版本、生效日期等）在第一个 "##" 标题之前，
# 归属到「文档信息」这一章节标签。
META_SECTION = "文档信息"

# --verify 的校验问题
VERIFY_QUERIES = [
    "婚假有几天",
    "P1 工单响应时限",
    "采购 5000 元审批",
]
VERIFY_TOP_K = 6

# --if-missing 的标记机制：在数据集内写入一段不可被业务检索命中的哨兵文本，
# 用它判断「当前文档内容 + 切分规则版本」是否已经导入过，从而摆脱对本地
# index 文件的依赖（线上容器文件系统是临时的，重启即丢）。
SPLIT_RULE_VERSION = "v1"
SENTINEL_PREFIX = "QINGHE_KB_SENTINEL"
SENTINEL_SEARCH_TOP_K = 20
EXISTS_SEARCH_TOP_K = 10


def resolve_workspace():
    """优先取脚本自身推导的项目根，兜底到环境变量，避免依赖调用时 cwd。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    return os.environ.get("COZE_WORKSPACE_PATH", project_dir)


def index_file_path(env):
    return os.path.join(WORKSPACE, "assets", f"kb_index.{env}.json")


def extract_title(lines):
    """取文档第一行的 # 标题作为《文档标题》。"""
    for line in lines:
        s = line.strip()
        if s:
            return re.sub(r"^#+\s*", "", s).strip()
    return "未命名文档"


def determine_status(title):
    """文档标题含「已废止」为已废止，其余为现行。"""
    return "已废止" if "已废止" in title else "现行"


def strip_blockquote(s):
    """去掉行首的 Markdown 引用标记 '>'。"""
    if s.startswith(">"):
        return s[1:].lstrip()
    return s


def split_long(content, max_len=MAX_SEGMENT_CHARS):
    """超过上限时按空行拆分，保证表格块不被拆散。"""
    if len(content) <= max_len:
        return [content]

    blocks = re.split(r"\n\s*\n", content)
    result = []
    current = ""
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if current and len(current) + len(block) + 1 > max_len:
            result.append(current)
            current = block
        else:
            current = current + "\n\n" + block if current else block
    if current:
        result.append(current)
    return result


def split_segments(body_lines):
    """把正文切分为 (章节标题, 内容) 列表。"""
    segments = []
    section = META_SECTION
    buf = []

    def flush():
        nonlocal buf
        content = "\n".join(buf).strip()
        if content:
            for part in split_long(content):
                segments.append((section, part))
        buf = []

    for raw in body_lines:
        line = raw.rstrip("\n")
        s = line.strip()

        if not s:
            buf.append("")
            continue

        if s.startswith("## "):
            flush()
            section = s[3:].strip()
            continue

        s = strip_blockquote(s)

        if CLAUSE_RE.match(s):
            flush()
            buf.append(s)
            continue

        buf.append(s)

    flush()
    return segments


def build_chunk_text(title, section, status, content):
    prefix = f"【来源：《{title}》｜{section}｜{status}】"
    return prefix + "\n" + content


def load_index(index_file):
    if os.path.exists(index_file):
        with open(index_file, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("entries"), list):
            return data
    return {"dataset": DATASET, "entries": []}


def save_index(index, index_file):
    os.makedirs(os.path.dirname(index_file), exist_ok=True)
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def compute_docs_hash():
    """按文件名排序拼接全部文档内容，连同切分规则版本号做 sha256。

    返回 64 位十六进制字符串。切分规则版本号参与计算，保证切分逻辑变更后
    该 hash 变化、从而生成新的哨兵文本，触发重新导入。
    """
    hasher = hashlib.sha256()
    doc_files = sorted(f for f in os.listdir(KB_DOCS_DIR) if f.endswith(".md"))
    for fn in doc_files:
        path = os.path.join(KB_DOCS_DIR, fn)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        # 文件名也参与，避免同内容不同文件名的差异被忽略
        hasher.update(fn.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(text.encode("utf-8"))
        hasher.update(b"\x00")
    hasher.update(SPLIT_RULE_VERSION.encode("utf-8"))
    return hasher.hexdigest()


def sentinel_text(docs_hash):
    return f"{SENTINEL_PREFIX}_{SPLIT_RULE_VERSION}_{docs_hash}"


def sentinel_already_imported(client, sentinel):
    """检索标记文本，若返回中有一段 content 与标记完全相等则视为已导入。"""
    resp = client.search(
        query=sentinel,
        table_names=[DATASET],
        top_k=SENTINEL_SEARCH_TOP_K,
        min_score=0.0,
    )
    if resp.code != 0:
        return False
    for ch in resp.chunks or []:
        if (ch.content or "").strip() == sentinel:
            return True
    return False


def _norm(text):
    """比较段落时忽略空白。"""
    return re.sub(r"\s+", "", text or "")


def segment_exists(client, chunk_text):
    """知识库里是否已有内容完全相同的一段（用这段文字本身去检索，再逐字比对）。

    检索出错时返回 False，按“不存在”处理：最坏结果是多导入一份重复段落，
    问答流程在回答前会去重；而误判成“已存在”会让新内容进不了知识库。
    """
    try:
        resp = client.search(
            query=chunk_text,
            table_names=[DATASET],
            top_k=EXISTS_SEARCH_TOP_K,
            min_score=0.0,
        )
    except Exception:  # noqa: BLE001
        return False
    if resp.code != 0:
        return False
    target = _norm(chunk_text)
    return any(_norm(ch.content) == target for ch in (resp.chunks or []))


def run_verify(client, env):
    """校验指定环境知识库是否有数据，打印每个问题返回段数与来源前缀。"""
    print("=" * 60)
    print(f"数据集：{DATASET}")
    print(f"环境：{env}（COZE_PROJECT_ENV={PROJECT_ENV_MAP[env]}）")
    print("校验模式：不导入，仅检索")
    print("-" * 60)
    for q in VERIFY_QUERIES:
        resp = client.search(
            query=q,
            table_names=[DATASET],
            top_k=VERIFY_TOP_K,
            min_score=0.0,
        )
        if resp.code != 0:
            print(f"[verify] 问题「{q}」：检索失败 code={resp.code} msg={resp.msg}")
            continue
        chunks = resp.chunks or []
        print(f"[verify] 问题「{q}」→ 返回 {len(chunks)} 段")
        for i, ch in enumerate(chunks[:2], 1):
            content = (ch.content or "").strip()
            first_line = content.split("\n", 1)[0] if content else ""
            print(f"    #{i} {first_line}")
    print("=" * 60)


def run_if_missing(client, env, dry_run=False):
    """按知识库内哨兵标记判断是否已导入；未导入则只补导入知识库里还没有的段落，再写入标记。

    该模式不读本地 index 文件，也不写 index 文件——幂等性完全由数据集内的
    哨兵文本承载，适配「线上容器文件系统临时、重启即丢」的场景。

    文档更新后哨兵会变。旧做法是整库重新导入，未改动的段落会各多出一份；
    现在逐段检查，只导入新增或改过的段落。被替换掉的旧段落无法删除（平台没有删除接口），
    由问答流程在回答前按“当前文档段落清单”筛掉。

    dry_run=True 时只报告哪些段落需要导入，不写入任何内容。
    """
    start = time.time()
    print("=" * 60)
    print(f"数据集：{DATASET}")
    print(f"环境：{env}（COZE_PROJECT_ENV={PROJECT_ENV_MAP[env]}）")
    print("模式：--if-missing（按知识库内标记判断）")
    print("-" * 60)

    docs_hash = compute_docs_hash()
    sentinel = sentinel_text(docs_hash)
    print(f"[if-missing] docs_hash={docs_hash}")
    print(f"[if-missing] sentinel={sentinel}")

    print("[if-missing] 检索标记文本，判断是否已导入...")
    if sentinel_already_imported(client, sentinel):
        elapsed = time.time() - start
        print("已导入，跳过")
        print(f"耗时：{elapsed:.1f}s")
        print("=" * 60)
        return

    chunk_config = ChunkConfig(
        separator=SEGMENT_SEPARATOR,
        max_tokens=MAX_TOKENS,
        remove_extra_spaces=False,
        remove_urls_emails=False,
    )

    doc_files = sorted(f for f in os.listdir(KB_DOCS_DIR) if f.endswith(".md"))
    per_doc = {}
    newly_imported = 0
    already_there = 0
    failed = 0

    for fn in doc_files:
        path = os.path.join(KB_DOCS_DIR, fn)
        with open(path, encoding="utf-8") as f:
            text = f.read()

        lines = text.split("\n")
        title = extract_title(lines)
        status = determine_status(title)
        body_lines = lines[1:]

        segments = split_segments(body_lines)
        per_doc[fn] = len(segments)

        for section, content in segments:
            chunk_text = build_chunk_text(title, section, status, content)
            if segment_exists(client, chunk_text):
                already_there += 1
                continue
            if dry_run:
                print(f"[if-missing] 需要导入：{fn} / {section}")
                newly_imported += 1
                continue
            doc = KnowledgeDocument(source=DataSourceType.TEXT, raw_data=chunk_text)
            resp = client.add_documents(
                documents=[doc],
                table_name=DATASET,
                chunk_config=chunk_config,
            )
            if resp.code != 0:
                print(f"[WARN] 导入失败 {fn} / {section}: {resp.msg}", file=sys.stderr)
                failed += 1
                continue
            print(f"[if-missing] 已导入：{fn} / {section}")
            newly_imported += 1

    total = sum(per_doc.values())

    if dry_run:
        print("=" * 60)
        print("试运行：没有写入任何内容")
        print(f"总段数：{total} 段；知识库里已有：{already_there} 段；需要导入：{newly_imported} 段")
        print("=" * 60)
        return

    if failed:
        print(
            f"[if-missing] 存在 {failed} 段导入失败，不写入标记（退出码非 0）",
            file=sys.stderr,
        )
        print("=" * 60)
        sys.exit(1)

    # 全部段落成功后，才写入哨兵标记。标记本身作为单独一段导入。
    sentinel_doc = KnowledgeDocument(source=DataSourceType.TEXT, raw_data=sentinel)
    resp = client.add_documents(
        documents=[sentinel_doc],
        table_name=DATASET,
        chunk_config=chunk_config,
    )
    if resp.code != 0:
        print(f"[if-missing] 写入标记失败：{resp.msg}", file=sys.stderr)
        print("=" * 60)
        sys.exit(1)

    elapsed = time.time() - start
    print("=" * 60)
    print(f"数据集：{DATASET}")
    print(f"环境：{env}（COZE_PROJECT_ENV={PROJECT_ENV_MAP[env]}）")
    print(f"文档数：{len(per_doc)} 份")
    print(f"总段数：{total} 段")
    print(f"知识库里已有：{already_there} 段（跳过）")
    print(f"本次导入：{newly_imported} 段")
    print(f"标记已写入：{sentinel}")
    print(f"耗时：{elapsed:.1f}s")
    for fn in doc_files:
        print(f"  {fn}: {per_doc[fn]} 段")
    print("=" * 60)


def main(args):
    global WORKSPACE
    WORKSPACE = resolve_workspace()

    # 关键：在任何 SDK 调用前设置 COZE_PROJECT_ENV，SDK 据此选择 DEV/PROD 资源。
    os.environ["COZE_PROJECT_ENV"] = PROJECT_ENV_MAP[args.env]

    client = KnowledgeClient(config=Config(), verbose=False)

    if args.verify:
        run_verify(client, args.env)
        return

    if args.if_missing:
        run_if_missing(client, args.env, dry_run=args.dry_run)
        return

    chunk_config = ChunkConfig(
        separator=SEGMENT_SEPARATOR,
        max_tokens=MAX_TOKENS,
        remove_extra_spaces=False,
        remove_urls_emails=False,
    )

    index_file = index_file_path(args.env)
    index = load_index(index_file)
    existing_hashes = {e["content_hash"] for e in index["entries"]}

    doc_files = sorted(
        f for f in os.listdir(KB_DOCS_DIR) if f.endswith(".md")
    )

    per_doc = {}
    newly_imported = 0
    failed = 0
    current_hashes = set()

    for fn in doc_files:
        path = os.path.join(KB_DOCS_DIR, fn)
        with open(path, encoding="utf-8") as f:
            text = f.read()

        lines = text.split("\n")
        title = extract_title(lines)
        status = determine_status(title)
        body_lines = lines[1:]

        segments = split_segments(body_lines)
        per_doc[fn] = len(segments)

        for section, content in segments:
            chunk_text = build_chunk_text(title, section, status, content)
            digest = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
            current_hashes.add(digest)

            if digest in existing_hashes:
                continue

            doc = KnowledgeDocument(source=DataSourceType.TEXT, raw_data=chunk_text)
            resp = client.add_documents(
                documents=[doc],
                table_name=DATASET,
                chunk_config=chunk_config,
            )

            if resp.code != 0:
                print(f"[WARN] 导入失败 {fn} / {section}: {resp.msg}", file=sys.stderr)
                failed += 1
                continue

            doc_id = resp.doc_ids[0] if resp.doc_ids else None
            index["entries"].append({
                "doc_id": doc_id,
                "file": fn,
                "title": title,
                "section": section,
                "status": status,
                "content_hash": digest,
            })
            existing_hashes.add(digest)
            newly_imported += 1

    save_index(index, index_file)

    total = sum(per_doc.values())
    print("=" * 60)
    print(f"数据集：{DATASET}")
    print(f"环境：{args.env}（COZE_PROJECT_ENV={PROJECT_ENV_MAP[args.env]}）")
    print(f"索引文件：{index_file}")
    print(f"文档数：{len(per_doc)} 份")
    print(f"总段数：{total} 段")
    print(f"本次新导入：{newly_imported} 段")
    print(f"索引总条数：{len(index['entries'])} 条")
    stale = [e for e in index["entries"] if e["content_hash"] not in current_hashes]
    if stale:
        print(f"旧版本段落：{len(stale)} 段（文档已更新或下线；平台不支持删除，问答流程会在回答前筛掉）")
        for e in stale:
            print(f"  旧：{e['file']} / {e['section']}")
    if failed:
        print(f"失败：{failed} 段")
    print("-" * 60)
    for fn in doc_files:
        print(f"  {fn}: {per_doc[fn]} 段")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="导入知识库到指定环境（dev/prod），或校验指定环境知识库数据。"
    )
    parser.add_argument(
        "--env",
        required=True,
        choices=("dev", "prod"),
        help="目标环境：dev 或 prod（决定 COZE_PROJECT_ENV 与索引文件名）",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="不导入，仅在指定环境用固定问题检索，验证知识库是否有数据",
    )
    parser.add_argument(
        "--if-missing",
        action="store_true",
        help="按知识库内哨兵标记判断是否已导入；未导入则全量导入并写入标记",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="与 --if-missing 一起用：只报告哪些段落需要导入，不写入",
    )
    parsed = parser.parse_args()
    main(parsed)