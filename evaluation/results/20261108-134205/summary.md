# Evaluation report

- Run: 2026-11-08 13:42 · wall time 1605.5 min
- Agent model: `nvidia/nemotron-3.5-lightning-30b-a3b` (reasoning off)
- Judge model: `nvidia/nemotron-3-ultra-550b-a55b` (reasoning on)
- Retrieval: top_k=4, embeddings `sentence-transformers/all-minilm-l6-v2`

## Quality

| Component | Goldens | Errors | Measure | Score | Pass rate | n |
|---|---|---|---|---|---|---|
| router | 20 | 0 | route_accuracy | 1.0 |  | 20 |
| retrieval | 18 | 0 | context_hit | 1.0 |  | 18 |
| | | | source_hit | 1.0 |  | 18 |
| | | | Contextual Precision | 0.953 | 0.938 | 16 |
| | | | Contextual Recall | 1.0 | 1 | 16 |
| | | | Contextual Relevancy | 0.11 | 0 | 15 |
| kb_grader | 18 | 0 | grade_accuracy | 1.0 |  | 18 |
| | | | grade_accuracy_good | 1.0 |  | 10 |
| | | | grade_accuracy_weak | 1.0 |  | 8 |
| kb_generation | 20 | 0 | context_hit | 1.0 |  | 20 |
| | | | Answer Relevancy | 0.854 | 0.95 | 20 |
| | | | Correctness [GEval] | 1.0 | 1 | 20 |
| | | | Faithfulness | 0.994 | 1 | 20 |
| web_fallback | 16 | 0 | kb_grade_weak | 0.875 |  | 16 |
| | | | path_accuracy | 0.875 |  | 16 |
| | | | Correctness [GEval] | 0.909 | 0.909 | 11 |
| | | | External Disclaimer [GEval] | 0.846 | 0.846 | 13 |
| query_rewrite | 15 | 0 | context_hit_original | 0.933 |  | 15 |
| | | | context_hit_rewritten | 1.0 |  | 15 |
| | | | keyword_coverage | 0.433 |  | 15 |
| | | | Rewrite Quality [GEval] | 0.783 | 0.75 | 12 |
| end_to_end | 20 | 0 | path_accuracy | 0.95 |  | 20 |
| | | | path_accuracy_direct | 1.0 |  | 4 |
| | | | path_accuracy_insufficient_evidence | 0.75 |  | 4 |
| | | | path_accuracy_knowledge_base | 1.0 |  | 8 |
| | | | path_accuracy_web_search | 1.0 |  | 4 |
| | | | Correctness [GEval] | 0.877 | 0.846 | 13 |

## Latency and tokens per request (agent only)

| Component | Path | n | Latency p50 | Latency p95 | Latency max | Tokens p50 | Tokens p95 | Tokens mean |
|---|---|---|---|---|---|---|---|---|
| router | all | 20 | 0.49s | 0.7s | 1.05s | 121 | 128 | 121.0 |
| retrieval | all | 18 | 0.39s | 1.5s | 1.5s | 0 | 0 | 0 |
| kb_grader | all | 18 | 0.98s | 2.99s | 2.99s | 827 | 853 | 823.0 |
| kb_generation | all | 20 | 2.76s | 6.35s | 24.17s | 942 | 1030 | 949.0 |
| web_fallback | all | 16 | 7.84s | 25.1s | 25.1s | 4391 | 5633 | 4227.0 |
| web_fallback | knowledge_base | 2 | 3.56s | 4.57s | 4.57s | 1864 | 2028 | 1946 |
| web_fallback | web_search | 14 | 7.97s | 25.1s | 25.1s | 4602 | 5633 | 4553.0 |
| query_rewrite | all | 15 | 1.43s | 4.8s | 4.8s | 81 | 172 | 92.0 |
| end_to_end | all | 20 | 5.01s | 11.19s | 11.77s | 1889 | 5333 | 2721.0 |
| end_to_end | direct | 4 | 1.14s | 8.03s | 8.03s | 169 | 176 | 170.0 |
| end_to_end | insufficient_evidence | 3 | 11.19s | 11.77s | 11.77s | 5333 | 5681 | 5286.0 |
| end_to_end | knowledge_base | 8 | 3.69s | 7.96s | 7.96s | 1850 | 1914 | 1862.0 |
| end_to_end | web_search | 5 | 8.97s | 9.92s | 9.92s | 4652 | 4898 | 4599.0 |

## Token totals for the full evaluation

| Component | Agent input | Agent output | Judge input | Judge output | Generation time | Judging time |
|---|---|---|---|---|---|---|
| router | 2,289 | 140 | 0 | 0 | 10.7s | 0.0s |
| retrieval | 0 | 0 | 102,228 | 103,428 | 8.1s | 88694.4s |
| kb_grader | 14,696 | 126 | 0 | 0 | 20.1s | 0.0s |
| kb_generation | 15,990 | 2,988 | 90,849 | 114,482 | 77.8s | 672.4s |
| web_fallback | 64,140 | 3,491 | 17,849 | 17,633 | 147.0s | 5164.0s |
| query_rewrite | 911 | 473 | 7,402 | 18,466 | 26.0s | 568.5s |
| end_to_end | 51,726 | 2,697 | 9,577 | 9,436 | 129.7s | 780.0s |
| **Total** | **149,752** | **9,915** | **227,905** | **263,445** | | |

Grand total: **651,017 tokens** (agent 159,667 + judge 491,350).

Checks are deterministic (exact label/path match or substring hit). Metrics are DeepEval LLM-as-judge scores (threshold 0.5). Latency is wall-clock per golden for the component under test, run sequentially, including Pinecone and Tavily calls. Judge output tokens include the judge's reasoning tokens.
