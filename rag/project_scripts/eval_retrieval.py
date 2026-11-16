#!/usr/bin/env python3
"""检索评测（第三周）：只测“该找的段落找到没有”，不调用大模型，几乎不花钱。

对评测集 assets/eval/rag_eval_set_v2.json 的每道题，在每个数据集各检索一次（top_k=10、min_score=0），
再离线计算不同 top_k 和相似度阈值下的指标：
- 证据召回率@k：每题的每条正确段落（gold：所在文件 + 段落里必含的任一短语），前 k 条里有没有；
- 全部命中率@k：一题的全部正确段落都在前 k 条里的题目比例；
- MRR：第一条正确段落排名的倒数，越接近 1 越靠前；
- 废止误召回率@k：版本冲突题的前 k 条里出现已废止 v1 文档的比例；
- 阈值分析：无答案题的最高分 vs 有答案题的正确段落得分，给出 min_score 取不同值时的取舍。

段落属于哪个文件：优先用段首【来源：《标题》…】前缀（qinghe_kb）；否则查登记表 kb_documents 的 doc_id；
再不行用正文与原文比对（去掉空白、页眉页脚后找相同片段）。

用法（扣子项目终端，目录 /workspace/projects/projects）：
  python scripts/eval_retrieval.py --env dev
  python scripts/eval_retrieval.py --env dev --datasets qinghe_kb qinghe_pdf_a --only S01,V03
报告写到 assets/eval/reports/retrieval_YYYYMMDD_HHMMSS.md（及同名 .json）。
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_DATASETS = ["qinghe_kb", "qinghe_pdf_a", "qinghe_pdf_b", "qinghe_docx_a"]
KS = (3, 5, 8, 10)
THRESHOLDS = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7)
SENTINEL = "QINGHE_KB_SENTINEL"
NOISE = ["青禾数智科技有限公司内部资料请勿外传", "•"]
DEPRECATED_FILE = "03_员工考勤与休假制度_v1_已废止.md"


def norm(s):
    s = re.sub(r"\s+", "", s or "")
    for n in NOISE:
        s = s.replace(n, "")
    return re.sub(r"第\d+页", "", s)


def load_docs():
    folder = os.path.join(ROOT, "assets", "kb_docs")
    docs, titles = {}, {}
    for name in sorted(os.listdir(folder)):
        if name.endswith(".md"):
            text = open(os.path.join(folder, name), encoding="utf-8").read()
            docs[name] = norm(text)
            first = next(line for line in text.splitlines() if line.strip())
            titles[re.sub(r"^#+\s*", "", first).strip()] = name
    return docs, titles


def load_registry(env):
    try:
        from sqlalchemy import text
        from src.storage.database.db import get_engine
        with get_engine().connect() as conn:
            rows = conn.execute(text("SELECT file_name, doc_ids, state FROM kb_documents WHERE env=:e"),
                                {"e": env}).fetchall()
    except Exception:  # noqa: BLE001
        path = os.path.join(ROOT, "assets", f"kb_registry.{env}.json")
        if not os.path.exists(path):
            print("（提示）没有登记表，改用正文比对判断段落来源")
            return {}
        rows = [(r["file_name"], r["doc_ids"], r["state"]) for r in json.load(open(path, encoding="utf-8"))]
    mapping = {}
    for file_name, doc_ids, state in rows:
        md = os.path.splitext(file_name)[0] + ".md"
        for d in json.loads(doc_ids or "[]"):
            mapping[str(d)] = (md, state)
    return mapping


def attribute(chunk, docs, titles, registry):
    """返回 (文件名, 依据)。"""
    m = re.match(r"\s*【来源：《(.+?)》", chunk["content"])
    if m and m.group(1) in titles:
        return titles[m.group(1)], "前缀"
    if chunk.get("doc_id") and str(chunk["doc_id"]) in registry:
        return registry[str(chunk["doc_id"])][0], "登记表"
    body = norm(chunk["content"])
    votes = {}
    for i in range(0, max(1, len(body) - 12), 12):
        piece = body[i:i + 12]
        if len(piece) < 8:
            continue
        for name, text in docs.items():
            if piece in text:
                votes[name] = votes.get(name, 0) + 1
    if votes:
        return max(votes, key=votes.get), "正文比对"
    return None, "未知"


def search(client, query, dataset):
    last = None
    for attempt in (1, 2):
        try:
            resp = client.search(query=query, table_names=[dataset], top_k=max(KS), min_score=0.0)
            if resp.code == 0:
                return [{"content": c.content or "", "score": float(c.score or 0), "doc_id": c.doc_id}
                        for c in (resp.chunks or []) if not (c.content or "").lstrip().startswith(SENTINEL)]
            last = f"code={resp.code} msg={resp.msg}"
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
        if attempt == 1:
            time.sleep(1)
    raise RuntimeError(last)


def gold_ranks(item, chunks):
    """每条 gold 第一次被满足的名次（1 起），没找到为 None。"""
    ranks = []
    for g in item.get("gold", []):
        phrases = [norm(p) for p in g["any"]]
        hit = None
        for i, c in enumerate(chunks, 1):
            if c["file"] == g["file"] and any(p in norm(c["content"]) for p in phrases):
                hit = i
                break
        ranks.append(hit)
    return ranks


def summarize(items, results):
    rows = {}
    for ds, per_q in results.items():
        with_gold = [q for q in items if q.get("gold") and per_q.get(q["id"])]
        row = {"题数": len(with_gold)}
        for k in KS:
            ev = [r for q in with_gold for r in per_q[q["id"]]["gold_ranks"]]
            row[f"证据召回@{k}"] = sum(1 for r in ev if r and r <= k) / len(ev) if ev else 0
            row[f"全部命中@{k}"] = (sum(1 for q in with_gold if all(r and r <= k for r in per_q[q["id"]]["gold_ranks"]))
                                 / len(with_gold) if with_gold else 0)
        firsts = [min([r for r in per_q[q["id"]]["gold_ranks"] if r] or [0]) for q in with_gold]
        row["MRR"] = sum(1 / f for f in firsts if f) / len(firsts) if firsts else 0
        vq = [q for q in items if q.get("must_not_cite") and per_q.get(q["id"])]
        for k in (3, 5):
            row[f"废止误召回@{k}"] = (sum(1 for q in vq if any(c["file"] == DEPRECATED_FILE
                                                           for c in per_q[q["id"]]["chunks"][:k])) / len(vq)) if vq else 0
        unknown = sum(1 for q in per_q.values() for c in q["chunks"] if c["file"] is None)
        row["来源未识别段落数"] = unknown
        lengths = [len(c["content"]) for q in per_q.values() for c in q["chunks"]]
        row["平均段长"] = sum(lengths) / len(lengths) if lengths else 0
        rows[ds] = row
    return rows


def threshold_table(items, per_q):
    answered = [q for q in items if q.get("gold") and per_q.get(q["id"])]
    noans = [q for q in items if q["expected_status"] == "not_found" and per_q.get(q["id"])]
    out = []
    for t in THRESHOLDS:
        keep = 0
        for q in answered:
            ch = per_q[q["id"]]["chunks"]
            first = min([r for r in per_q[q["id"]]["gold_ranks"] if r] or [0])
            if first and ch[first - 1]["score"] >= t:
                keep += 1
        reject = sum(1 for q in noans if not per_q[q["id"]]["chunks"] or per_q[q["id"]]["chunks"][0]["score"] < t)
        out.append((t, keep / len(answered) if answered else 0, reject / len(noans) if noans else 0))
    return out


def render(items, results, rows, datasets, env, started):
    pct = lambda v: f"{v * 100:.1f}%"
    lines = [f"# 检索评测 {started}", "", f"环境：{env}；题目 {len(items)} 道；每个数据集每题检索 1 次（top_k={max(KS)}，min_score=0）。",
             "", "## 总表", ""]
    keys = ["题数", "证据召回@3", "证据召回@5", "证据召回@8", "全部命中@5", "MRR", "废止误召回@3", "废止误召回@5", "来源未识别段落数", "平均段长"]
    lines.append("| 数据集 | " + " | ".join(keys) + " |")
    lines.append("| --- " * (len(keys) + 1) + "|")
    for ds in datasets:
        r = rows.get(ds)
        if not r:
            continue
        cells = [str(r[k]) if k in ("题数", "来源未识别段落数") else (f"{r[k]:.0f}" if k == "平均段长" else
                 (f"{r[k]:.3f}" if k == "MRR" else pct(r[k]))) for k in keys]
        lines.append(f"| {ds} | " + " | ".join(cells) + " |")
    lines += ["", "## 相似度阈值取舍（min_score）", "",
              "保留率：有答案的题，第一条正确段落得分 ≥ 阈值的比例；拒答率：无答案的题，最高分 < 阈值的比例。", ""]
    for ds in datasets:
        if ds not in results:
            continue
        lines.append(f"**{ds}**：" + "；".join(f"{t:.1f} → 保留 {pct(a)} / 拒答 {pct(b)}"
                                              for t, a, b in threshold_table(items, results[ds])))
        lines.append("")
    lines += ["## 按语义难点（证据召回@5）", ""]
    tags = sorted({t for q in items for t in q["tags"]})
    lines.append("| 难点 | " + " | ".join(datasets) + " |")
    lines.append("| --- " * (len(datasets) + 1) + "|")
    for tag in tags:
        cells = []
        for ds in datasets:
            ev = [r for q in items if tag in q["tags"] and q.get("gold") and results.get(ds, {}).get(q["id"])
                  for r in results[ds][q["id"]]["gold_ranks"]]
            cells.append(pct(sum(1 for r in ev if r and r <= 5) / len(ev)) if ev else "—")
        lines.append(f"| {tag} | " + " | ".join(cells) + " |")
    lines += ["", "## 未在前 5 条找到正确段落的题", ""]
    for ds in datasets:
        misses = [q for q in items if q.get("gold") and results.get(ds, {}).get(q["id"])
                  and not all(r and r <= 5 for r in results[ds][q["id"]]["gold_ranks"])]
        lines.append(f"**{ds}**（{len(misses)} 题）")
        for q in misses:
            top = results[ds][q["id"]]["chunks"][:3]
            desc = "；".join(f"{c['file'] or '未知'}@{c['score']:.2f}" for c in top)
            lines.append(f"- {q['id']} {q['question']} → 名次 {results[ds][q['id']]['gold_ranks']}；前 3：{desc}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", choices=("dev", "prod"), required=True)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--eval-set", default=os.path.join(ROOT, "assets", "eval", "rag_eval_set_v2.json"))
    parser.add_argument("--only", help="逗号分隔的题号")
    args = parser.parse_args()
    os.environ["COZE_PROJECT_ENV"] = {"dev": "DEV", "prod": "PROD"}[args.env]
    items = json.load(open(args.eval_set, encoding="utf-8"))["items"]
    if args.only:
        wanted = set(args.only.split(","))
        items = [q for q in items if q["id"] in wanted]
    docs, titles = load_docs()
    registry = load_registry(args.env)
    from coze_coding_dev_sdk import Config, KnowledgeClient
    client = KnowledgeClient(config=Config(), verbose=False)
    started = datetime.now().strftime("%Y%m%d_%H%M%S")
    results, failures = {}, []
    for ds in args.datasets:
        results[ds] = {}
        print(f"检索 {ds} …", flush=True)
        for q in items:
            try:
                chunks = search(client, q["question"], ds)
            except RuntimeError as exc:
                failures.append(f"{ds} {q['id']}：{exc}")
                continue
            for c in chunks:
                c["file"], c["by"] = attribute(c, docs, titles, registry)
            results[ds][q["id"]] = {"chunks": chunks, "gold_ranks": gold_ranks(q, chunks)}
        if not any(r["chunks"] for r in results[ds].values()):
            print(f"  （提示）{ds} 没有返回任何段落：数据集不存在、还在异步导入，或文件解析失败")
    rows = summarize(items, results)
    report = render(items, results, rows, args.datasets, args.env, started)
    if failures:
        report += "\n## 检索失败\n\n" + "\n".join(f"- {f}" for f in failures) + "\n"
    out_dir = os.path.join(ROOT, "assets", "eval", "reports")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"retrieval_{started}")
    open(base + ".md", "w", encoding="utf-8").write(report)
    json.dump({"rows": rows, "results": results}, open(base + ".json", "w", encoding="utf-8"), ensure_ascii=False)
    print(report.split("## 相似度阈值")[0])
    print(f"报告：{base}.md")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
