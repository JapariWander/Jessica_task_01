"""Create a blank human entailment review sheet for an actual LLM run."""
import argparse
import csv
from pathlib import Path
from quant_rag.io import read_json
from quant_rag.generation import CLAIM_FIELDS

p = argparse.ArgumentParser()
p.add_argument("examples", type=Path)
p.add_argument("--output", type=Path, default=Path("evaluation/grounding_review.csv"))
a = p.parse_args()
data = read_json(a.examples)
rows = []
for example in data["examples"]:
    result = example["output"]
    if result["mode"] != "llm":
        raise SystemExit("Grounding assessment requires actual generated claims, not evidence-only mode")
    context = {c["chunk_id"]: c["text"] for c in example["retrieved_context"]}
    for field in CLAIM_FIELDS:
        for claim in result["answer"][field]:
            rows.append({"question_id": example["question"]["id"], "field": field,
                         "claim": claim["claim"], "citations": ";".join(claim["citations"]),
                         "evidence": "\n\n".join(context[c] for c in claim["citations"]),
                         "supported_yes_partial_no": "", "actionability_1_to_5": "", "reviewer": "", "notes": ""})
if not rows:
    raise SystemExit("No factual claims to review; not a passing grounding evaluation")
a.output.parent.mkdir(parents=True, exist_ok=True)
with a.output.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print(a.output)
