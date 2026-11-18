#!/usr/bin/env python3
"""重排评测（第四周）：比较不同重排方式送进模型的上下文，不重新检索、不生成答案。

读取一次检索评测的结果（assets/eval/reports/retrieval_*.json 里 qinghe_kb 每题的前 10 段），
对每道题走一遍工作流的 grade 节点（过滤已废止和旧版本、去重、重排、截断），统计：
- 送入段数：平均每题送进模型几段；
- 证据召回：答案需要的原文段落（gold）有多少在送入的段落里；
- 全部命中：一题需要的段落全部送入的比例；
- 上下文精确率：送入的段落里有多少是 gold 段落（越高说明无关上下文越少）；
- 平均字数：每题送入的上下文字数；
- 无答案题：平均送入段数，以及直接判为"未找到"的题数（只有 llm 且 llm_allow_empty 时会发生）。

lexical 不调用模型；llm 每题调用一次模型（约 60 次，会消耗积分）。

用法（扣子项目终端，目录 /workspace/projects/projects）：
  python scripts/eval_rerank.py                       # none 与 lexical，不花积分
  python scripts/eval_rerank.py --methods none lexical llm
  python scripts/eval_rerank.py --methods llm --allow-empty --only R01,R02,V03
报告写到 assets/eval/reports/rerank_YYYYMMDD_HHMMSS.md。
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime

sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["COZE_WORKSPACE_PATH"] = ROOT  # 扣子终端里这个变量指向外层目录，这里改成项目根


def norm(s):
    return re.sub(r"\s+", "", s or "")


def latest_retrieval():
    for path in sorted(glob.glob(os.path.join(ROOT, "assets", "eval", "reports", "retrieval_*.json")), reverse=True):
        data = json.load(open(path, encoding="utf-8"))
        if len(data.get("results", {}).get("qinghe_kb", {})) >= 50:
            return path, data["results"]["qinghe_kb"]
    raise SystemExit("找不到包含 qinghe_kb 全部题目的检索结果，请先运行 scripts/eval_retrieval.py --env dev")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=["none", "lexical"], choices=["none", "lexical", "llm"])
    ap.add_argument("--retrieval", help="检索结果 json，默认取最新一份含 qinghe_kb 全部题目的")
    ap.add_argument("--only", help="只跑这些题，逗号分隔")
    ap.add_argument("--allow-empty", action="store_true", help="llm 方式：全部判为无关时直接判定未找到")
    args = ap.parse_args(argv)

    from agents import agent as A

    path, per_q = (args.retrieval, json.load(open(args.retrieval, encoding="utf-8"))["results"]["qinghe_kb"]) \
        if args.retrieval else latest_retrieval()
    items = json.load(open(os.path.join(ROOT, "assets", "eval", "rag_eval_set_v2.json"), encoding="utf-8"))["items"]
    if args.only:
        ids = {x.strip() for x in args.only.split(",") if x.strip()}
        items = [i for i in items if i["id"] in ids]
    items = [i for i in items if i["id"] in per_q]
    base_cfg = A._retrieval_config()

    rows, details = [], {}
    for method in args.methods:
        cfg = json.loads(json.dumps(base_cfg))
        cfg["rerank"] = dict(A._rerank_config(base_cfg), method=method, llm_allow_empty=args.allow_empty)
        A._retrieval_config = lambda cfg=cfg: cfg
        started = time.time()
        ev = hit = allhit = nq = sent = rel = 0
        chars, noans_sent, noans_empty, fallbacks = [], [], 0, 0
        for it in items:
            raw = [{"doc_id": c.get("doc_id") or "", "chunk_id": "", "score": float(c["score"]),
                    "content": c["content"].replace(A.FICTIONAL_NOTE, "").strip(), "dataset": "qinghe_kb"}
                   for c in per_q[it["id"]]["chunks"][:10]]
            if method == "none":
                raw = raw[:int(cfg.get("top_k", 6))]  # 现状：只取前 top_k 段
            state = {"chunks": raw, "question": it["question"], "rewritten_query": it["question"],
                     "list_query": "宽泛问题" in it.get("tags", [])}
            out = A.node_grade(state)
            kept = out.get("chunks") or []
            info = out.get("rerank_info") or {}
            fallbacks += 1 if info.get("fallback") else 0
            chars.append(sum(len(c.get("body", "")) for c in kept))
            details.setdefault(it["id"], {})[method] = [f"{c.get('title', '')}｜{c.get('section', '')}" for c in kept]
            if not it.get("gold"):
                noans_sent.append(len(kept))
                noans_empty += 1 if not kept else 0
                continue
            nq += 1
            sent += len(kept)
            found = [any(any(norm(p) in norm(c.get("body", "")) for p in g["any"]) for c in kept) for g in it["gold"]]
            ev += len(found)
            hit += sum(found)
            allhit += all(found)
            rel += sum(any(any(norm(p) in norm(c.get("body", "")) for p in g["any"]) for g in it["gold"]) for c in kept)
        rows.append({
            "方法": method + ("（全部无关则判未找到）" if method == "llm" and args.allow_empty else ""),
            "送入段数": f"{sent / nq:.2f}" if nq else "—",
            "证据召回": f"{hit / ev:.1%}" if ev else "—",
            "全部命中": f"{allhit / nq:.1%}" if nq else "—",
            "上下文精确率": f"{rel / sent:.1%}" if sent else "—",
            "平均字数": f"{statistics.mean(chars):.0f}" if chars else "—",
            "无答案题送入段数": f"{statistics.mean(noans_sent):.2f}" if noans_sent else "—",
            "无答案题直接判未找到": f"{noans_empty}/{len(noans_sent)}" if noans_sent else "—",
            "退回关键词重排": fallbacks,
            "耗时（秒）": f"{time.time() - started:.0f}",
        })
        print(f"{method} 完成：{rows[-1]}", flush=True)

    keys = list(rows[0])
    lines = [f"# 重排评测 {datetime.now().strftime('%Y%m%d_%H%M%S')}", "",
             f"检索结果：{os.path.basename(path)}；题目 {len(items)} 道；候选为每题检索前 10 段；"
             f"none = 现状（只取前 {base_cfg.get('top_k', 6)} 段，不重排）。",
             f"重排参数：{json.dumps(A._rerank_config(base_cfg), ensure_ascii=False)}", "",
             "| " + " | ".join(keys) + " |", "|" + " --- |" * len(keys)]
    lines += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]
    lines += ["", "## 每题送入的段落", ""]
    for qid, per in details.items():
        lines.append(f"**{qid}**")
        for m, segs in per.items():
            lines.append(f"- {m}（{len(segs)} 段）：" + "；".join(segs))
        lines.append("")
    out_dir = os.path.join(ROOT, "assets", "eval", "reports")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"rerank_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines[:len(rows) + 7]))
    print(f"\n报告：{out}")


if __name__ == "__main__":
    main()
