# Evaluation

Golden datasets for evaluating each component of the agent with [DeepEval](https://deepeval.com). The judge model is NVIDIA `NVIDIA_EVAL_MODEL`.

## Datasets

| File | Component | Goldens | Labels | Suggested metrics |
|---|---|---|---|---|
| `router.json` | `route_question` | 20 (12 kb / 8 direct) | `expected_route` | Accuracy |
| `retrieval.json` | Pinecone retriever | 18 | `context`, `expected_source` | Hit rate @k, Contextual Precision / Recall / Relevancy |
| `kb_grader.json` | `grade_kb` | 18 (10 good / 8 weak) | `expected_grade` | Accuracy, precision and recall on `weak` |
| `kb_generation.json` | `generate_from_kb` | 20 | `expected_output`, `context` | Faithfulness, Answer Relevancy, GEval correctness |
| `web_fallback.json` | `search_web` → `grade_web` → `generate_from_web` | 16 | `expected_output`, `criteria` | GEval correctness, GEval "flags external info" |
| `query_rewrite.json` | `rewrite_query` | 15 | `expected_output` (ideal rewrite), `expected_keywords`, `expected_source` | GEval intent preservation, retrieval hit rate after rewrite |
| `end_to_end.json` | Full graph via `ask()` | 20 (8 KB / 4 direct / 4 web / 4 insufficient) | `expected_source_used`, `expected_output` | Path accuracy, GEval correctness |

Every golden has `input` and, where relevant, `expected_output` and `context` (the ground-truth KB chunk, copied verbatim). Component-specific labels are in `additional_metadata`.

```python
from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(
    "evaluation/datasets/kb_generation.json",
    input_key_name="input",
    expected_output_key_name="expected_output",
    context_key_name="context",
    additional_metadata_key_name="additional_metadata",
)
```

## Notes

- Contexts are generated from `data/sample_kb`. After editing the knowledge base, run `python evaluation/build_datasets.py` to rebuild the datasets.
- Web-fallback and end-to-end web answers are public employment-law facts as of 2026-11. Re-check them periodically because laws change.
- Web search results vary between runs, so treat web-path scores as indicative rather than exact.
