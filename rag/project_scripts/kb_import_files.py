#!/usr/bin/env python3
"""把 PDF / DOCX 原文件批量导入知识库，按切片方案分到不同数据集，并在数据库登记（第三周）。

扣子编程知识库的导入规则（据 coze_coding_dev_sdk 源码）：
- 只能通过代码导入（batch_import），异步完成；来源三种：TEXT 纯文本、URL 下载链接、URI 对象存储；
- PDF/DOCX 走 URL：先把文件放到可下载的位置（对象存储预签名链接，或公开仓库的 raw 链接），由平台解析；
- 切片参数只有 4 个：separator、max_tokens、remove_extra_spaces、remove_urls_emails，没有重叠/父子分段；
- SDK 没有删除文档的接口，所以“替换旧版本”只能在检索端按登记表过滤，不能物理删除。

切片方案全部写在 config/kb_chunking.json（第四周起），每个方案一个数据集，互不混合；
原有 qinghe_kb 为 C：按条款预切分的基线，由 scripts/import_kb.py 导入。method 两种：
  platform  把文件交给平台解析，按分隔符切（A/B/D/S）；
  semantic  语义切片，不依赖分隔符（本地拆句 → 逐句向量 → 相邻句语义距离突变处断开），
            每段补「文档名｜章节｜状态」前缀后以纯文本导入；可加 "ocr": true 转写扫描页。

登记表 kb_documents（项目 PostgreSQL）：同一环境 + 数据集 + 文件名 + sha256 已导入 → 跳过；
文件名相同但内容变了 → 导入新版本，旧记录标为“已替换”，检索端据此过滤旧 doc_id。

用法（在扣子项目终端，目录 /workspace/projects/projects）：
  python scripts/kb_import_files.py --env dev --strategy A --dry-run   # 只列出将要做的事
  python scripts/kb_import_files.py --env dev --strategy A             # 导入
  python scripts/kb_import_files.py --env dev --list                   # 查看登记表
  python scripts/kb_import_files.py --env dev --strategy M --dry-run --show 5   # 语义切片：看切出来的前 5 段
  python scripts/kb_import_files.py --env dev --strategy A --source github --github-base <raw 前缀>
语义切片的全部段落另存一份到 assets/eval/chunks/<数据集>.md，方便人工检查。
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from urllib.parse import quote

sys.dont_write_bytecode = True

# 以脚本所在位置推导项目根（/workspace/projects/projects），不依赖环境变量和当前目录。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kb_chunking as kc  # noqa: E402

STRATEGIES = kc.load_strategies(ROOT)
CONTENT_TYPES = {".pdf": "application/pdf",
                 ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}

DDL = """
CREATE TABLE IF NOT EXISTS kb_documents (
    id SERIAL PRIMARY KEY,
    env TEXT NOT NULL,
    dataset TEXT NOT NULL,
    file_name TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    doc_status TEXT NOT NULL,          -- 现行 / 已废止（来自文件名）
    state TEXT NOT NULL,               -- 有效 / 已替换
    doc_ids TEXT NOT NULL DEFAULT '[]',
    chunk_config TEXT NOT NULL,
    imported_at TIMESTAMP NOT NULL,
    UNIQUE (env, dataset, file_name, sha256)
)"""


class DbRegistry:
    """登记表放在项目 PostgreSQL（服务端状态，重启不丢）。"""
    kind = "数据库表 kb_documents"

    def __init__(self, env):
        from sqlalchemy import text
        from src.storage.database.db import get_engine
        self.text, self.env, self.eng = text, env, get_engine()
        with self.eng.begin() as conn:
            conn.execute(text(DDL))

    def rows(self, dataset=None, file_name=None):
        sql = "SELECT id, dataset, file_name, sha256, doc_status, state, doc_ids, imported_at FROM kb_documents WHERE env=:e"
        params = {"e": self.env}
        if dataset:
            sql += " AND dataset=:d AND file_name=:f"
            params.update(d=dataset, f=file_name)
        with self.eng.connect() as conn:
            return [dict(zip(("id", "dataset", "file_name", "sha256", "doc_status", "state", "doc_ids", "imported_at"), r))
                    for r in conn.execute(self.text(sql + " ORDER BY dataset, file_name, imported_at"), params)]

    def record(self, dataset, file_name, digest, status, doc_ids, chunk, replace_ids):
        with self.eng.begin() as conn:
            if replace_ids:
                conn.execute(self.text("UPDATE kb_documents SET state='已替换' WHERE id = ANY(:ids)"), {"ids": replace_ids})
            conn.execute(self.text(
                "INSERT INTO kb_documents (env, dataset, file_name, sha256, doc_status, state, doc_ids, chunk_config, "
                "imported_at) VALUES (:e,:d,:f,:s,:st,'有效',:ids,:c,:t)"),
                {"e": self.env, "d": dataset, "f": file_name, "s": digest, "st": status, "ids": json.dumps(doc_ids),
                 "c": json.dumps(chunk, ensure_ascii=False), "t": datetime.now()})


class FileRegistry(DbRegistry):
    """项目没有数据库时的退路：assets/kb_registry.{env}.json（开发环境可写；线上容器只读，不要依赖）。"""
    kind = "文件"

    def __init__(self, env):  # noqa: D107  pylint: disable=super-init-not-called
        self.env = env
        self.path = os.path.join(ROOT, "assets", f"kb_registry.{env}.json")
        self.kind = f"文件 {os.path.relpath(self.path, ROOT)}（未连上数据库时的退路）"
        self.data = json.load(open(self.path, encoding="utf-8")) if os.path.exists(self.path) else []

    def rows(self, dataset=None, file_name=None):
        return [r for r in self.data if not dataset or (r["dataset"] == dataset and r["file_name"] == file_name)]

    def record(self, dataset, file_name, digest, status, doc_ids, chunk, replace_ids):
        for r in self.data:
            if r["id"] in replace_ids:
                r["state"] = "已替换"
        self.data.append({"id": len(self.data) + 1, "dataset": dataset, "file_name": file_name, "sha256": digest,
                          "doc_status": status, "state": "有效", "doc_ids": json.dumps(doc_ids),
                          "chunk_config": chunk, "imported_at": datetime.now().isoformat(timespec="seconds")})
        json.dump(self.data, open(self.path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def registry(env):
    try:
        return DbRegistry(env)
    except Exception as exc:  # noqa: BLE001
        print(f"（提示）连接项目数据库失败（{type(exc).__name__}），登记表改存文件。")
        return FileRegistry(env)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def list_files(strategy):
    folder = os.path.join(ROOT, STRATEGIES[strategy]["dir"])
    if not os.path.isdir(folder):
        raise SystemExit(f"找不到文件夹：{folder}（请先把文件上传到 {STRATEGIES[strategy]['dir']}）")
    ext = STRATEGIES[strategy]["ext"]
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(ext))
    if not files:
        raise SystemExit(f"{folder} 里没有 {ext} 文件")
    return [os.path.join(folder, f) for f in files]


def download_url(path, source, github_base):
    name = os.path.basename(path)
    if source == "github":
        if not github_base:
            raise SystemExit("--source github 需要 --github-base（raw 链接前缀，到文件夹为止）")
        return github_base.rstrip("/") + "/" + quote(name)
    from coze_coding_dev_sdk import S3SyncStorage
    storage = S3SyncStorage()
    with open(path, "rb") as f:
        key = storage.upload_file(file_content=f.read(), file_name=name,
                                  content_type=CONTENT_TYPES[os.path.splitext(name)[1].lower()])
    return storage.generate_presigned_url(key=key, expire_time=3600)


def import_texts(client, dataset, texts):
    """语义切片的结果：每段一个纯文本文档，分隔符设成文中不会出现的值，平台不会再切。"""
    from coze_coding_dev_sdk import ChunkConfig, DataSourceType, KnowledgeDocument
    doc_ids = []
    for start in range(0, len(texts), 20):
        batch = [KnowledgeDocument(source=DataSourceType.TEXT, raw_data=t) for t in texts[start:start + 20]]
        last = None
        for attempt in (1, 2):
            try:
                resp = client.add_documents(documents=batch, table_name=dataset,
                                            chunk_config=ChunkConfig(**kc.TEXT_CHUNK))
                if resp.code == 0:
                    doc_ids += resp.doc_ids or []
                    break
                last = f"code={resp.code} msg={resp.msg}"
            except Exception as exc:  # noqa: BLE001
                last = f"{type(exc).__name__}: {exc}"
            if attempt == 1:
                time.sleep(2)
        else:
            raise RuntimeError(last)
    return doc_ids


def doc_title(path):
    """文档名：优先取 assets/kb_docs 里同名 Markdown 的标题（与 qinghe_kb 前缀一致），否则用文件名。"""
    stem = os.path.splitext(os.path.basename(path))[0].replace("_扫描件", "")
    md = os.path.join(ROOT, "assets", "kb_docs", stem + ".md")
    if os.path.exists(md):
        first = next((ln for ln in open(md, encoding="utf-8") if ln.strip()), "")
        return first.lstrip("#").strip(), md
    return stem, None


def semantic_file(path, cfg, status, embedder, log=print):
    """一个文件 → 带前缀的段落列表，以及记录。"""
    params = cfg.get("semantic") or {}
    title, md = doc_title(path)
    text, info = kc.document_text(path, bool(cfg.get("ocr")), os.path.join(ROOT, "assets", "kb_files", "ocr"), log)
    if info["ocr_pages"] and md:
        info["ocr_accuracy"] = round(kc.ocr_accuracy(text, open(md, encoding="utf-8").read()), 3)
        log(f"    OCR 与原文相似度：{info['ocr_accuracy']:.1%}")
    chunks, stats = kc.semantic_chunks(kc.split_units(text, title), embedder, params, title)
    info.update(stats)
    return [kc.prefixed(title, c["section"], status, c["text"]) for c in chunks], info


def import_one(client, dataset, url, chunk):
    from coze_coding_dev_sdk import ChunkConfig, DataSourceType, KnowledgeDocument
    last = None
    for attempt in (1, 2):
        try:
            resp = client.add_documents(documents=[KnowledgeDocument(source=DataSourceType.URL, url=url)],
                                        table_name=dataset, chunk_config=ChunkConfig(**chunk))
            if resp.code == 0:
                return resp.doc_ids or []
            last = f"code={resp.code} msg={resp.msg}"
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
        if attempt == 1:
            time.sleep(2)
    raise RuntimeError(last)


def cmd_list(env):
    reg = registry(env)
    rows = reg.rows()
    print(f"登记表：{reg.kind}")
    if not rows:
        print("（空）")
    for r in rows:
        print(f"{r['dataset']:<14} {r['file_name']:<36} {r['doc_status']:<4} {r['state']:<4} "
              f"doc_ids={r['doc_ids']} {r['imported_at']}")


def cmd_import(env, strategy, dry_run, source, github_base, show=0):
    cfg = STRATEGIES[strategy]
    files = list_files(strategy)
    reg = registry(env)
    client = None
    if not dry_run:
        from coze_coding_dev_sdk import Config, KnowledgeClient
        client = KnowledgeClient(config=Config(), verbose=False)
    semantic = cfg["method"] == "semantic"
    params = {k: cfg.get(k) for k in ("method", "chunk", "semantic", "ocr") if cfg.get(k) is not None}
    print(f"方案 {strategy} → 数据集 {cfg['dataset']}，切片 {json.dumps(params, ensure_ascii=False)}；登记表：{reg.kind}")
    embedder = kc.CachedEmbedder(os.path.join(ROOT, "assets", "kb_files", ".embedding_cache.json")) if semantic else None
    preview = []
    done = skipped = failed = 0
    for path in files:
        name = os.path.basename(path)
        # 内容指纹带上切片参数：参数改了，同一文件也按新版本重新导入
        digest = sha256(path) if not semantic else sha256(path) + ":" + kc.config_digest(params)
        status = "已废止" if "已废止" in name else "现行"
        existing = reg.rows(cfg["dataset"], name)
        same = any(r["sha256"] == digest for r in existing)
        older = [r["id"] for r in existing if r["sha256"] != digest and r["state"] == "有效"]
        if same:
            print(f"  跳过（已导入，内容未变）：{name}")
            skipped += 1
            continue
        action = "新版本，旧版本将标为已替换" if older else "新文件"
        texts = None
        if semantic:
            try:
                texts, info = semantic_file(path, cfg, status, embedder)
            except Exception as exc:  # noqa: BLE001
                print(f"  【切片失败】{name}：{type(exc).__name__}: {exc}")
                failed += 1
                continue
            lengths = [len(t) for t in texts]
            print(f"  {name}：{info.get('units', 0)} 句 → {len(texts)} 段，段长 {min(lengths or [0])}–{max(lengths or [0])} 字，"
                  f"断点 {info.get('breaks')}，阈值 {info.get('threshold')}")
            preview += [f"## {name} · 第 {i} 段（{len(t)} 字）\n\n```\n{t}\n```\n" for i, t in enumerate(texts, 1)]
            for t in texts[:show]:
                print("    ┌" + t.replace("\n", "\n    │") + "\n    └")
        if dry_run:
            print(f"  将导入（{action}）：{name}")
            continue
        try:
            if semantic:
                doc_ids = import_texts(client, cfg["dataset"], texts)
            else:
                doc_ids = import_one(client, cfg["dataset"], download_url(path, source, github_base), cfg["chunk"])
        except Exception as exc:  # noqa: BLE001
            print(f"  【导入失败】{name}：{exc}")
            failed += 1
            continue
        reg.record(cfg["dataset"], name, digest, status, doc_ids, params, older)
        print(f"  已提交导入（{action}）：{name} doc_ids={doc_ids}")
        done += 1
    if preview:
        out = os.path.join(ROOT, "assets", "eval", "chunks", f"{cfg['dataset']}.md")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        open(out, "w", encoding="utf-8").write(f"# {cfg['dataset']} 切片结果\n\n" + "\n".join(preview))
        print(f"  全部段落已存到 {os.path.relpath(out, ROOT)}；向量接口调用 {embedder.calls} 次（其余来自缓存）")
    print(f"完成：提交 {done}，跳过 {skipped}，失败 {failed}。导入是异步的，约 1–3 分钟后再检索。")
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", choices=("dev", "prod"), required=True)
    parser.add_argument("--strategy", choices=sorted(STRATEGIES))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--source", choices=("storage", "github"), default="storage",
                        help="storage=上传到项目对象存储后用预签名链接；github=用公开仓库的 raw 链接")
    parser.add_argument("--github-base")
    parser.add_argument("--show", type=int, default=0, help="语义切片：打印每个文件的前 N 段")
    args = parser.parse_args()
    os.environ["COZE_PROJECT_ENV"] = {"dev": "DEV", "prod": "PROD"}[args.env]
    if args.list:
        cmd_list(args.env)
        return 0
    if not args.strategy:
        parser.error("请指定 --strategy 或 --list")
    return cmd_import(args.env, args.strategy, args.dry_run, args.source, args.github_base, args.show)


if __name__ == "__main__":
    sys.exit(main())
