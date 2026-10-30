"""Document-level retrieval evaluation; human-only by default."""
import statistics
import time
from pathlib import Path

from .generation import evidence_only, generate
from .ingest import load_documents
from .io import ROOT, now, read_json, write_json
from .retrieval import Embedder, Retriever, TfidfEmbedder, build_index, corpus_fingerprint


def metrics(retrieved_ids, relevant_ids, k=5):
    # Rank is chunk rank, recall is unique relevant documents in top-k chunks.
    top = retrieved_ids[:k]
    relevant = set(relevant_ids)
    if not relevant:
        return {"recall_at_k": None, "mrr_at_k": None}
    return {"recall_at_k": len(set(top) & relevant) / len(relevant),
            "mrr_at_k": next((1 / rank for rank, doc in enumerate(top, 1) if doc in relevant), 0.0)}


def run_experiment(questions_path=None, labels_path=None, sizes=(256, 512, 1024), exploratory=False, output=None, backend="minilm"):
    questions = read_json(questions_path or ROOT / "evaluation/questions.json")
    labels = read_json(labels_path or ROOT / "evaluation/labels.json")
    docs = load_documents()
    valid_ids = {d["id"] for d in docs}
    for label in labels:
        if not set(label["relevant_document_ids"]) <= valid_ids:
            raise ValueError("Label refers to missing documents; complete acquisition first")
    if not exploratory and any(x.get("status") != "human_reviewed" or not x.get("reviewer") for x in labels):
        raise ValueError("Human evaluation requires human_reviewed labels and named reviewers; use --exploratory for agent-proposed labels")
    by_id = {x["question_id"]: x for x in labels}
    if len(by_id) != len(questions) or set(by_id) != {q["id"] for q in questions}:
        raise ValueError("Exactly one relevance label is required per question")
    embedder = Embedder() if backend == "minilm" else TfidfEmbedder(documents=docs)
    output = output or ROOT / f"evaluation/results-{backend}.json"
    run = {"created_at": now(), "backend": backend, "status": "exploratory_not_human" if exploratory else "human_labels",
           "corpus_sha256": corpus_fingerprint(docs), "k": 5,
           "label_snapshot": labels, "question_snapshot": questions, "sizes": []}
    for size in sizes:
        start = time.perf_counter()
        directory = build_index(size, ROOT / f"artifacts/index-{backend}-{size}", documents=docs, embedder=embedder)
        index_seconds = time.perf_counter() - start
        retriever = Retriever(directory, embedder)
        records = []
        for q in questions:
            start = time.perf_counter()
            context = retriever.search(q["query"], 5)
            elapsed = time.perf_counter() - start
            relevant = by_id[q["id"]]["relevant_document_ids"]
            records.append({**q, "relevant_document_ids": relevant,
                            "metrics": metrics([c["document_id"] for c in context], relevant),
                            "latency_seconds": elapsed, "context": context})
        summary = {}
        # Chunk size selection uses dev questions only; test stays held out.
        for split in ("dev", "test"):
            subset = [r for r in records if r["split"] == split and r["metrics"]["recall_at_k"] is not None]
            summary[split] = {"scored_questions": len(subset)}
            for metric in ("recall_at_k", "mrr_at_k"):
                summary[split][metric] = statistics.mean(r["metrics"][metric] for r in subset) if subset else None
        run["sizes"].append({"chunk_tokens": size, "index_metadata": retriever.metadata,
                             "index_seconds": index_seconds,
                             "median_query_seconds": statistics.median(r["latency_seconds"] for r in records),
                             "summary": summary, "records": records})
        write_json(output or ROOT / "evaluation/results.json", run)
    eligible = [s for s in run["sizes"] if s["summary"]["dev"]["recall_at_k"] is not None]
    if eligible:
        best = max(eligible, key=lambda s: (s["summary"]["dev"]["recall_at_k"],
                                           s["summary"]["dev"]["mrr_at_k"], -s["chunk_tokens"]))
        run["selected_chunk_tokens"] = best["chunk_tokens"]
        run["selection_rule"] = "dev Recall@5, then dev MRR@5, then smaller chunk; not tuned on test"
    write_json(output or ROOT / "evaluation/results.json", run)
    return run


def save_examples(directory, mode="evidence", output=None):
    retriever = Retriever(directory)
    questions = read_json(ROOT / "evaluation/questions.json")
    examples = []
    for q in questions[:3]:
        contexts = retriever.search(q["query"])
        result = generate(q["query"], contexts) if mode == "llm" else evidence_only(q["query"], contexts)
        examples.append({"question": q, "retrieved_context": contexts, "output": result})
    path = output or ROOT / f"examples/{mode}.json"
    write_json(path, {"created_at": now(), "mode": mode, "examples": examples})
    return path
