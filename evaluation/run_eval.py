"""Evaluate each agent component against its golden dataset.

    python -m evaluation.run_eval                          # all components
    python -m evaluation.run_eval -c router kb_generation  # selected components
    python -m evaluation.run_eval --limit 3                # quick smoke run

Each component runs in two phases:
  1. The component under test generates outputs, and deterministic checks
     (routing, grading and path accuracy, retrieval hit rate) are scored in Python.
     Latency (p50/p95) and token usage are recorded for every request.
  2. DeepEval LLM-as-judge metrics are scored by the NVIDIA judge model, and the
     judge's token usage is recorded.
Results are written to evaluation/results/<timestamp>/ (summary.md, summary.json, per-component JSON).
"""

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from statistics import mean

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
# The judge applies its own rate limiting and retries, so queued calls must not hit DeepEval's 180s task budget.
os.environ.setdefault("DEEPEVAL_DISABLE_TIMEOUTS", "1")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from deepeval import evaluate  # noqa: E402
from deepeval.evaluate import AsyncConfig, CacheConfig, DisplayConfig, ErrorConfig  # noqa: E402
from deepeval.metrics import (  # noqa: E402
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams as P  # noqa: E402
from langchain_core.callbacks import get_usage_metadata_callback  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.rag import workflow as wf  # noqa: E402
from app.rag.vectorstore import get_retriever  # noqa: E402
from evaluation.judge import NvidiaJudge  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets"


# ------------------------------------------------------------------ helpers

def retrieve(query: str):
    return get_retriever().invoke(query)


def chunks(docs) -> list[str]:
    return [d.page_content for d in docs]


def context_hit(context: str, docs) -> bool:
    """True if any retrieved chunk contains the start or end of the ground-truth section."""
    head, tail = context[:80], context[-80:]
    return any(head in d.page_content or tail in d.page_content for d in docs)


def correctness(judge) -> GEval:
    return GEval(
        name="Correctness",
        criteria=(
            "Judge whether the actual output is factually consistent with, and covers the key points of, "
            "the expected output. If the expected output describes a behaviour (for example a brief greeting, "
            "or directing the employee to HR), judge whether the actual output shows that behaviour. "
            "Extra relevant detail is acceptable; contradictions or missing key facts are not."
        ),
        evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT, P.EXPECTED_OUTPUT],
        model=judge,
    )


def external_disclaimer(judge) -> GEval:
    return GEval(
        name="External Disclaimer",
        criteria=(
            "The answer is based on public web information, not company policy. Judge whether the actual "
            "output clearly tells the employee that the information is external or public and should be "
            "validated with HR before being treated as company policy."
        ),
        evaluation_params=[P.ACTUAL_OUTPUT],
        model=judge,
    )


def rewrite_quality(judge) -> GEval:
    return GEval(
        name="Rewrite Quality",
        criteria=(
            "The input is a vague employee question and the actual output is a rewritten search query. "
            "Judge whether the rewrite preserves the user's intent, uses clear HR terminology suitable for "
            "knowledge-base and web search, and does not answer the question. The expected output is one "
            "example of a good rewrite; other wordings with the same intent are equally good."
        ),
        evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT, P.EXPECTED_OUTPUT],
        model=judge,
    )


# ------------------------------------------------------------------ components
# Each runner takes a golden and returns (test_case_fields, checks).
# `checks` maps a check name to True/False, or to a float score between 0 and 1.

def run_router(g):
    route = wf.route_question(wf.initial_state(g["input"]))["source_used"]
    return {"actual_output": route}, {"route_accuracy": route == g["additional_metadata"]["expected_route"]}


def run_retrieval(g):
    docs = retrieve(g["input"])
    meta = g["additional_metadata"]
    return (
        {"actual_output": "", "expected_output": g["context"][0], "retrieval_context": chunks(docs)},
        {
            "context_hit": context_hit(g["context"][0], docs),
            "source_hit": any(meta["expected_source"] in d.metadata.get("source", "") for d in docs),
        },
    )


def run_kb_grader(g):
    docs = retrieve(g["input"])
    grade = wf.grade_kb(wf.initial_state(g["input"], kb_docs=docs))["kb_grade"]
    expected = g["additional_metadata"]["expected_grade"]
    return (
        {"actual_output": grade, "retrieval_context": chunks(docs)},
        {"grade_accuracy": grade == expected, f"grade_accuracy_{expected}": grade == expected},
    )


def run_kb_generation(g):
    docs = retrieve(g["input"])
    answer = wf.generate_from_kb(wf.initial_state(g["input"], kb_docs=docs))["answer"]
    return (
        {"actual_output": answer, "retrieval_context": chunks(docs)},
        {"context_hit": context_hit(g["context"][0], docs)},
    )


def run_web_fallback(g):
    result = wf.ask(g["input"])
    return (
        {"actual_output": result["answer"], "retrieval_context": [result["web_results"]] if result["web_results"] else None},
        {"path_accuracy": result["source_used"] == "web_search", "kb_grade_weak": result["kb_grade"] == "weak"},
        result,
    )


def run_query_rewrite(g):
    rewritten = wf.rewrite_query(wf.initial_state(g["input"]))["current_query"]
    keywords = g["additional_metadata"]["expected_keywords"]
    original_docs, rewritten_docs = retrieve(g["input"]), retrieve(rewritten)
    return (
        {"actual_output": rewritten},
        {
            "keyword_coverage": sum(k.lower() in rewritten.lower() for k in keywords) / len(keywords),
            "context_hit_original": context_hit(g["context"][0], original_docs),
            "context_hit_rewritten": context_hit(g["context"][0], rewritten_docs),
        },
    )


def run_end_to_end(g):
    result = wf.ask(g["input"])
    expected = g["additional_metadata"]["expected_source_used"]
    return (
        {"actual_output": result["answer"]},
        {"path_accuracy": result["source_used"] == expected, f"path_accuracy_{expected}": result["source_used"] == expected},
        result,
    )


COMPONENTS = {
    "router": (run_router, lambda j: []),
    "retrieval": (run_retrieval, lambda j: [
        ContextualPrecisionMetric(model=j), ContextualRecallMetric(model=j), ContextualRelevancyMetric(model=j)]),
    "kb_grader": (run_kb_grader, lambda j: []),
    "kb_generation": (run_kb_generation, lambda j: [
        FaithfulnessMetric(model=j), AnswerRelevancyMetric(model=j), correctness(j)]),
    "web_fallback": (run_web_fallback, lambda j: [correctness(j), external_disclaimer(j)]),
    "query_rewrite": (run_query_rewrite, lambda j: [rewrite_quality(j)]),
    "end_to_end": (run_end_to_end, lambda j: [correctness(j)]),
}



# ------------------------------------------------------------------ stats helpers

def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def distribution(values: list[float], digits: int = 2) -> dict | None:
    if not values:
        return None
    return {
        "mean": round(mean(values), digits),
        "p50": round(percentile(values, 50), digits),
        "p95": round(percentile(values, 95), digits),
        "max": round(max(values), digits),
    }


def sum_usage(usage: dict) -> dict:
    """Collapse LangChain per-model usage metadata into one input/output/total count."""
    return {k: sum(m.get(f"{k}_tokens", 0) for m in usage.values()) for k in ("input", "output", "total")}


# ------------------------------------------------------------------ orchestration

def generate(component: str, goldens: list[dict], workers: int) -> list[dict]:
    runner = COMPONENTS[component][0]

    def one(i_g):
        i, g = i_g
        for attempt in range(3):
            start = time.perf_counter()
            try:
                with get_usage_metadata_callback() as usage:
                    fields, checks, *extra = runner(g)
                break
            except Exception as exc:  # transient API errors: back off and retry
                if attempt == 2:
                    return {"index": i, "golden": g, "error": f"{type(exc).__name__}: {exc}"}
                time.sleep(5 * (attempt + 1))
        result = extra[0] if extra else {}
        return {
            "index": i, "golden": g, "fields": fields, "checks": checks,
            "source_used": result.get("source_used"), "trace": result.get("trace"),
            "latency_s": round(time.perf_counter() - start, 3),
            "tokens": sum_usage(usage.usage_metadata),
        }

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, enumerate(goldens)))


def judge(component: str, records: list[dict], concurrency: int) -> dict:
    """Score records with DeepEval. Returns the judge's token usage."""
    ok = [r for r in records if "error" not in r]
    j = NvidiaJudge(max_parallel=concurrency)
    metrics = COMPONENTS[component][1](j)
    if not metrics or not ok:
        return sum_usage({})

    cases = [
        LLMTestCase(
            name=f"{component}-{r['index']}",
            input=r["golden"]["input"],
            actual_output=r["fields"].get("actual_output", ""),
            expected_output=r["fields"].get("expected_output", r["golden"].get("expected_output")),
            context=r["golden"].get("context"),
            retrieval_context=r["fields"].get("retrieval_context"),
        )
        for r in ok
    ]
    result = evaluate(
        cases,
        metrics,
        async_config=AsyncConfig(max_concurrent=concurrency),
        display_config=DisplayConfig(print_results=False, show_indicator=False),
        error_config=ErrorConfig(ignore_errors=True),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
    )
    by_name = {t.name: t for t in result.test_results}
    for r in ok:
        t = by_name.get(f"{component}-{r['index']}")
        r["metrics"] = {
            m.name: {"score": m.score, "success": m.success, "reason": m.reason, "error": m.error}
            for m in (t.metrics_data or [])
        } if t else {}
    return sum_usage(j.usage.usage_metadata)


def perf(records: list[dict]) -> dict:
    return {
        "latency_s": distribution([r["latency_s"] for r in records]),
        "tokens_per_request": distribution([r["tokens"]["total"] for r in records], 0),
        "input_tokens": sum(r["tokens"]["input"] for r in records),
        "output_tokens": sum(r["tokens"]["output"] for r in records),
    }


def summarise(component: str, records: list[dict]) -> dict:
    ok = [r for r in records if "error" not in r]
    summary = {"component": component, "goldens": len(records), "errors": len(records) - len(ok), "checks": {}, "metrics": {}}

    for name in sorted({k for r in ok for k in r["checks"]}):
        values = [float(r["checks"][name]) for r in ok if name in r["checks"]]
        summary["checks"][name] = {"mean": round(mean(values), 3), "n": len(values)}

    for name in sorted({k for r in ok for k in r.get("metrics", {})}):
        data = [r["metrics"][name] for r in ok if name in r.get("metrics", {})]
        scored = [d for d in data if d["score"] is not None]
        summary["metrics"][name] = {
            "mean_score": round(mean(d["score"] for d in scored), 3) if scored else None,
            "pass_rate": round(mean(bool(d["success"]) for d in scored), 3) if scored else None,
            "n": len(scored),
            "errors": len(data) - len(scored),
        }

    if ok:
        summary["performance"] = perf(ok)
        paths = sorted({r["source_used"] for r in ok if r["source_used"]})
        if paths:
            summary["performance_by_path"] = {p: {"n": sum(r["source_used"] == p for r in ok), **perf([r for r in ok if r["source_used"] == p])}
                                              for p in paths}
    return summary


def fmt(d: dict | None, key: str, unit: str = "") -> str:
    return f"{d[key]}{unit}" if d else "–"


def to_markdown(summaries: list[dict], totals: dict, settings, started: str) -> str:
    lines = [
        "# Evaluation report", "",
        f"- Run: {started} · wall time {totals['wall_time_min']} min",
        f"- Agent model: `{settings.nvidia_model}` (reasoning {'on' if settings.nvidia_reasoning else 'off'})",
        f"- Judge model: `{settings.nvidia_eval_model}` (reasoning on)",
        f"- Retrieval: top_k={settings.top_k}, embeddings `{settings.embedding_model}`", "",
        "## Quality", "",
        "| Component | Goldens | Errors | Measure | Score | Pass rate | n |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        rows = [(k, v["mean"], "", v["n"]) for k, v in s["checks"].items()]
        rows += [(k, v["mean_score"], v["pass_rate"], v["n"]) for k, v in s["metrics"].items()]
        for i, (name, score, pass_rate, n) in enumerate(rows):
            head = f"| {s['component']} | {s['goldens']} | {s['errors']} " if i == 0 else "| | | "
            lines.append(f"{head}| {name} | {score} | {pass_rate} | {n} |")

    lines += [
        "", "## Latency and tokens per request (agent only)", "",
        "| Component | Path | n | Latency p50 | Latency p95 | Latency max | Tokens p50 | Tokens p95 | Tokens mean |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        groups = [("all", s["goldens"] - s["errors"], s.get("performance"))]
        groups += [(p, v["n"], v) for p, v in s.get("performance_by_path", {}).items()] if len(s.get("performance_by_path", {})) > 1 else []
        for path, n, p in groups:
            if not p:
                continue
            lat, tok = p["latency_s"], p["tokens_per_request"]
            lines.append(f"| {s['component']} | {path} | {n} | {fmt(lat, 'p50', 's')} | {fmt(lat, 'p95', 's')} | {fmt(lat, 'max', 's')} "
                         f"| {fmt(tok, 'p50')} | {fmt(tok, 'p95')} | {fmt(tok, 'mean')} |")

    lines += [
        "", "## Token totals for the full evaluation", "",
        "| Component | Agent input | Agent output | Judge input | Judge output | Generation time | Judging time |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        p, jt, t = s.get("performance") or {}, s["judge_tokens"], s["timing_s"]
        lines.append(f"| {s['component']} | {p.get('input_tokens', 0):,} | {p.get('output_tokens', 0):,} | {jt['input']:,} | {jt['output']:,} "
                     f"| {t['generation']}s | {t['judging']}s |")
    a, j = totals["agent_tokens"], totals["judge_tokens"]
    lines += [
        f"| **Total** | **{a['input']:,}** | **{a['output']:,}** | **{j['input']:,}** | **{j['output']:,}** | | |",
        "", f"Grand total: **{a['total'] + j['total']:,} tokens** (agent {a['total']:,} + judge {j['total']:,}).",
        "", "Checks are deterministic (exact label/path match or substring hit). Metrics are DeepEval LLM-as-judge scores (threshold 0.5). "
            "Latency is wall-clock per golden for the component under test, run sequentially, including Pinecone and Tavily calls. "
            "Judge output tokens include the judge's reasoning tokens.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--components", nargs="+", choices=list(COMPONENTS), default=list(COMPONENTS))
    parser.add_argument("--limit", type=int, help="only evaluate the first N goldens per component")
    parser.add_argument("--workers", type=int, default=1, help="parallel agent calls (keep at 1 for clean latency numbers)")
    parser.add_argument("--concurrency", type=int, default=4, help="parallel judge calls")
    args = parser.parse_args()

    settings = get_settings()
    run_start = time.perf_counter()
    started = datetime.now().strftime("%Y-%m-%d %H:%M")
    out_dir = ROOT / "results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Agent: {settings.nvidia_model} | Judge: {settings.nvidia_eval_model} | Output: {out_dir}")
    get_retriever()  # load embeddings and connect to Pinecone once, before measuring

    summaries = []
    for component in args.components:
        goldens = json.loads((DATASETS / f"{component}.json").read_text(encoding="utf-8"))[: args.limit]
        print(f"\n=== {component}: generating {len(goldens)} outputs", flush=True)
        t0 = time.perf_counter()
        records = generate(component, goldens, args.workers)
        t1 = time.perf_counter()
        print(f"=== {component}: judging", flush=True)
        try:
            judge_tokens = judge(component, records, args.concurrency)
        except Exception as exc:  # keep the deterministic checks even if judging fails
            print(f"!!! judging failed for {component}: {type(exc).__name__}: {exc}")
            judge_tokens = sum_usage({})
        summary = summarise(component, records)
        summary["judge_tokens"] = judge_tokens
        summary["timing_s"] = {"generation": round(t1 - t0, 1), "judging": round(time.perf_counter() - t1, 1)}
        summaries.append(summary)
        (out_dir / f"{component}.json").write_text(
            json.dumps({"summary": summary, "records": records}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(json.dumps({k: summary[k] for k in ("checks", "metrics", "timing_s")}, indent=2), flush=True)

    def total(key):
        return {k: sum(s.get("performance", {}).get(f"{k}_tokens", 0) for s in summaries) if key == "agent"
                else sum(s["judge_tokens"][k] for s in summaries) for k in ("input", "output")}

    totals = {"agent_tokens": total("agent"), "judge_tokens": total("judge"),
              "wall_time_min": round((time.perf_counter() - run_start) / 60, 1)}
    for t in (totals["agent_tokens"], totals["judge_tokens"]):
        t["total"] = t["input"] + t["output"]

    report = to_markdown(summaries, totals, settings, started)
    (out_dir / "summary.md").write_text(report, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps({"totals": totals, "components": summaries}, indent=2), encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
