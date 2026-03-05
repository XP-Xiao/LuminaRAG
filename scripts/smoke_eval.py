"""轻量 RAG smoke 评测：上游 SuperMew 数据集在 LuminaRAG 上的最小可跑版本.

数据集来源: https://github.com/icey1287/SuperMew (MIT), 原样拷入 evals/rag/.
只看分数、不拦门, 与上游 gates/baseline/指纹体系无关.

用法 (工作目录为仓库根, 用 lumina 环境):
    E:\\Miniconda\\envs\\lumina\\python.exe scripts/smoke_eval.py ingest  # 入库 corpus 到评测 collection
    E:\\Miniconda\\envs\\lumina\\python.exe scripts/smoke_eval.py run     # 全链路跑 20 问
    E:\\Miniconda\\envs\\lumina\\python.exe scripts/smoke_eval.py run --retrieval-only
    E:\\Miniconda\\envs\\lumina\\python.exe scripts/smoke_eval.py score --observations <obs.json>
    E:\\Miniconda\\envs\\lumina\\python.exe scripts/smoke_eval.py clean   # 删除评测数据
    E:\\Miniconda\\envs\\lumina\\python.exe scripts/smoke_eval.py all     # ingest -> run -> score -> clean

说明:
- 默认用独立 Milvus collection (EVAL_COLLECTION, 默认 luminarag_eval_smoke),
  通过 MILVUS_COLLECTION 环境变量切换, 主库零污染. 父分块在共享 PG 中,
  跑完 clean 会按文件名删除.
- 全链路 run 需要 .env 真 key (ARK/GRADE/FAST/EMBEDDING) + docker 基础设施;
  缺 key 时用 --retrieval-only 只算检索类指标.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "evals" / "rag" / "rag_smoke_v1.json"
DEFAULT_CORPUS = REPO_ROOT / "evals" / "rag" / "corpus"
EVAL_COLLECTION = "luminarag_eval_smoke"
STATUS_TO_OUTCOME = {
    "answerable": "ANSWERABLE",
    "partial": "ANSWERABLE",
    "no_knowledge": "NO_KNOWLEDGE",
    "needs_clarification": "INSUFFICIENT_EVIDENCE",
    "needs_scope_selection": "INSUFFICIENT_EVIDENCE",
    "needs_rewrite": "INSUFFICIENT_EVIDENCE",
}


# ---------------------------------------------------------------- 打分(纯函数,不依赖 backend,score 可离线跑)
def _norm_name(value: object) -> str:
    return str(value or "").strip().lower()


def _ranking_metrics(gold_ids: set[str], ranked_ids: list[str], k: int) -> dict:
    ranked = ranked_ids[:k]
    hits = [1 if cid in gold_ids else 0 for cid in ranked]
    matched = sum(hits)
    recall = matched / len(gold_ids) if gold_ids else 0.0
    mrr = next((1.0 / (i + 1) for i, h in enumerate(hits) if h), 0.0)
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_ids), k)))
    return {
        "recall": recall,
        "precision": matched / k if k else 0.0,
        "mrr": mrr,
        "ndcg": dcg / ideal if ideal else 0.0,
        "hit": 1.0 if matched else 0.0,
    }


def score_case(case: dict, obs: dict, k_values: list[int]) -> dict:
    gold_chunks = {c.get("chunk_id") for c in case.get("gold_chunks", []) if c.get("chunk_id")}
    gold_docs = {_norm_name(d.get("canonical_name") or d.get("document_id")) for d in case.get("gold_documents", [])}
    gold_docs.discard("")
    retrieved = obs.get("retrieved_chunks", []) or []
    ranked_ids = [r.get("chunk_id") for r in retrieved if r.get("chunk_id")]
    ranked_names = [_norm_name(r.get("filename") or r.get("canonical_name")) for r in retrieved]

    metrics: dict = {}
    for k in k_values:
        if gold_chunks:
            r = _ranking_metrics(gold_chunks, ranked_ids, k)
            metrics[f"recall_at_{k}"] = round(r["recall"], 6)
            metrics[f"mrr_at_{k}"] = round(r["mrr"], 6)
            metrics[f"ndcg_at_{k}"] = round(r["ndcg"], 6)
        else:
            metrics[f"recall_at_{k}"] = None
            metrics[f"mrr_at_{k}"] = None
            metrics[f"ndcg_at_{k}"] = None
        matched_docs = len(gold_docs & set(ranked_names[:k])) if gold_docs else 0
        metrics[f"document_recall_at_{k}"] = round(matched_docs / len(gold_docs), 6) if gold_docs else None
    matched_all = len(gold_chunks & set(ranked_ids))
    metrics["gold_chunk_coverage"] = round(matched_all / len(gold_chunks), 6) if gold_chunks else None

    expected = case.get("expected", {}) or {}
    checks: dict = {
        "complexity": None if expected.get("complexity") is None else expected.get("complexity") == obs.get("complexity"),
        "route": None if expected.get("route") is None else expected.get("route") == obs.get("route"),
        "outcome": None if expected.get("outcome") is None else expected.get("outcome") == obs.get("outcome"),
        "hitl": (expected.get("hitl") or "none") == (obs.get("hitl") or "none"),
    }
    decisive = [v for v in checks.values() if v is not None]
    return {
        "case_id": case.get("id"),
        "metrics": metrics,
        "checks": checks,
        "passed": all(decisive) if decisive else True,
    }


def aggregate(case_results: list[dict]) -> dict:
    summary: dict = {}
    metric_names = sorted({k for c in case_results for k in c["metrics"]})
    for name in metric_names:
        vals = [c["metrics"][name] for c in case_results if c["metrics"].get(name) is not None]
        summary[name] = round(sum(vals) / len(vals), 6) if vals else None
    for check, metric in (("complexity", "complexity_accuracy"), ("route", "route_accuracy"),
                          ("outcome", "outcome_accuracy"), ("hitl", "hitl_accuracy")):
        vals = [c["checks"][check] for c in case_results if c["checks"].get(check) is not None]
        summary[metric] = round(sum(1 for v in vals if v) / len(vals), 6) if vals else None
    summary["case_pass_rate"] = round(sum(1 for c in case_results if c["passed"]) / len(case_results), 6) if case_results else None
    return summary


def print_report(case_results: list[dict], summary: dict) -> None:
    print(f"{'case_id':36} {'pass':6} {'route ok':9} {'recall@10':10} {'doc_rec@10':11} {'mrr@10':8}")
    for c in case_results:
        m = c["metrics"]
        print(f"{c['case_id']:36} {str(c['passed']):6} {str(c['checks'].get('route')):9} "
              f"{m.get('recall_at_10')!s:10} {m.get('document_recall_at_10')!s:11} {m.get('mrr_at_10')!s:8}")
    print("\n summary:")
    for k in sorted(summary):
        print(f"  {k:28} {summary[k]}")
    failed = [c["case_id"] for c in case_results if not c["passed"]]
    print(f"\n failed ({len(failed)}): {failed or 'none'}")


# ---------------------------------------------------------------- 需 backend 的部分(延迟导入,先设 MILVUS_COLLECTION)
def _setup_backend(collection: str):
    os.environ["MILVUS_COLLECTION"] = collection
    sys.path.insert(0, str(REPO_ROOT))
    from backend.env import load_env  # noqa: E402
    load_env()
    return collection


def _eval_filenames(corpus_dir: Path) -> list[str]:
    return sorted(p.name for p in corpus_dir.glob("*.html"))


def cmd_ingest(args) -> None:
    _setup_backend(args.collection)
    from backend.indexing.document_loader import DocumentLoader  # noqa: E402
    from backend.indexing.milvus_client import get_milvus_store  # noqa: E402
    from backend.indexing.milvus_writer import MilvusWriter  # noqa: E402
    from backend.indexing.parent_chunk_store import ParentChunkStore  # noqa: E402

    loader = DocumentLoader()
    writer = MilvusWriter(milvus_manager=get_milvus_store())
    parent_store = ParentChunkStore()
    total_leaf = 0
    for filename in _eval_filenames(args.corpus):
        docs = loader.load_document(str(args.corpus / filename), filename)
        leaf = [d for d in docs if int(d.get("chunk_level", 0) or 0) == 3]
        parent = [d for d in docs if int(d.get("chunk_level", 0) or 0) in (1, 2)]
        if not leaf:
            print(f"  [skip] {filename}: 无叶子分块")
            continue
        parent_store.upsert_documents(parent)
        writer.write_documents(leaf)
        total_leaf += len(leaf)
        print(f"  [ok] {filename}: 父 {len(parent)} / 叶 {len(leaf)}")
    print(f"ingest done: {total_leaf} leaf chunks -> collection {args.collection}")


def _observe_full(question: str, run_rag_graph, ctx_factory) -> dict:
    ctx = ctx_factory()
    try:
        result = run_rag_graph(question, ctx)
    finally:
        try:
            ctx.close()
        except Exception:
            pass
    trace = result.get("rag_trace", {}) or {}
    docs = result.get("docs", []) or trace.get("retrieved_chunks", []) or []
    status = result.get("retrieval_status") or trace.get("retrieval_status")
    route = result.get("route") or trace.get("route")
    hitl = "none"
    if status == "needs_clarification":
        hitl = "clarify"
    elif status == "needs_scope_selection":
        hitl = "scope_select"
    return {
        "complexity": result.get("complexity") or trace.get("complexity"),
        "route": route,
        "outcome": STATUS_TO_OUTCOME.get(status or ""),
        "hitl": hitl,
        "retrieved_chunks": [
            {"chunk_id": d.get("chunk_id"), "filename": d.get("filename")} for d in docs if isinstance(d, dict)
        ],
    }


def _observe_retrieval_only(question: str, retrieve_documents, top_k: int) -> dict:
    out = retrieve_documents(question, top_k=top_k)
    docs = out.get("docs", []) or []
    return {
        "complexity": None, "route": None, "outcome": None, "hitl": "none",
        "retrieved_chunks": [
            {"chunk_id": d.get("chunk_id"), "filename": d.get("filename")} for d in docs if isinstance(d, dict)
        ],
    }


def cmd_run(args) -> str:
    _setup_backend(args.collection)
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    observations = []
    if args.retrieval_only:
        from backend.rag.utils import RETRIEVAL_TOP_K, retrieve_documents  # noqa: E402
        top_k = RETRIEVAL_TOP_K
        print(f"retrieval-only mode, top_k={top_k}")
        for case in dataset["cases"]:
            obs = _observe_retrieval_only(case["question"], retrieve_documents, top_k)
            observations.append({"case_id": case["id"], **obs})
            print(f"  [ok] {case['id']}: {len(obs['retrieved_chunks'])} chunks")
    else:
        from backend.chat.request_context import ChatRequestContext  # noqa: E402
        from backend.rag.pipeline import run_rag_graph  # noqa: E402
        from uuid import uuid4  # noqa: E402
        for case in dataset["cases"]:
            sid = f"smoke_{uuid4().hex[:8]}"
            obs = _observe_full(case["question"], run_rag_graph,
                                lambda: ChatRequestContext.for_sync(user_id="smoke_eval", session_id=sid))
            observations.append({"case_id": case["id"], **obs})
            print(f"  [ok] {case['id']}: route={obs['route']} outcome={obs['outcome']} chunks={len(obs['retrieved_chunks'])}")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(args.observations) if args.observations else (REPO_ROOT / ".artifacts" / f"smoke_obs_{ts}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"dataset": dataset.get("name"), "observations": observations},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"observations -> {out_path}")
    return str(out_path)


def cmd_score(args) -> None:
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    bundle = json.loads(Path(args.observations).read_text(encoding="utf-8"))
    obs_index = {o["case_id"]: o for o in bundle["observations"]}
    case_results = [score_case(c, obs_index.get(c["id"], {}), args.k) for c in dataset["cases"]]
    summary = aggregate(case_results)
    print_report(case_results, summary)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = Path(args.report) if args.report else (REPO_ROOT / ".artifacts" / f"smoke_report_{ts}.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"dataset": dataset.get("name"), "summary": summary, "cases": case_results},
                                      ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report -> {report_path}")


def cmd_clean(args) -> None:
    _setup_backend(args.collection)
    from backend.indexing.milvus_client import get_milvus_store  # noqa: E402
    from backend.indexing.parent_chunk_store import ParentChunkStore  # noqa: E402
    store = get_milvus_store()
    parent_store = ParentChunkStore()
    for filename in _eval_filenames(args.corpus):
        try:
            store.delete(f'filename == "{filename}"')
        except Exception as e:
            print(f"  [milvus skip] {filename}: {e}")
        try:
            n = parent_store.delete_by_filename(filename)
            print(f"  [ok] {filename}: parent deleted {n}")
        except Exception as e:
            print(f"  [parent skip] {filename}: {e}")
    print("clean done")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LuminaRAG 轻量 smoke 评测")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--collection", default=EVAL_COLLECTION)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest")
    r = sub.add_parser("run")
    r.add_argument("--retrieval-only", action="store_true")
    r.add_argument("--observations", default=None)
    s = sub.add_parser("score")
    s.add_argument("--observations", required=True)
    s.add_argument("--report", default=None)
    s.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    sub.add_parser("clean")
    sub.add_parser("all")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "score":
        cmd_score(args)
    elif args.command == "clean":
        cmd_clean(args)
    elif args.command == "all":
        args.retrieval_only = getattr(args, "retrieval_only", False)
        cmd_ingest(args)
        args.observations = cmd_run(args)
        args.k = [1, 3, 5, 10]
        args.report = None
        cmd_score(args)
        cmd_clean(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
