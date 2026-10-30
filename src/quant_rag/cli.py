import argparse
import json
import sys
from pathlib import Path

from .io import ROOT, write_json


def main():
    parser = argparse.ArgumentParser(description="Local quantitative research RAG")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("ingest")
    p.add_argument("--manifest", type=Path, default=ROOT / "data/sources.csv")
    p = sub.add_parser("index")
    p.add_argument("--size", type=int, choices=[256, 512, 1024], default=1024)
    p.add_argument("--backend", choices=["minilm", "tfidf"], default="minilm")
    p = sub.add_parser("query")
    p.add_argument("query")
    p.add_argument("--size", type=int, choices=[256, 512, 1024], default=1024)
    p.add_argument("--mode", choices=["evidence", "llm"], default="evidence")
    p.add_argument("--output", type=Path)
    p.add_argument("--backend", choices=["minilm", "tfidf"], default="minilm")
    p = sub.add_parser("evaluate")
    p.add_argument("--exploratory", action="store_true")
    p.add_argument("--backend", choices=["minilm", "tfidf"], default="minilm")
    p = sub.add_parser("examples")
    p.add_argument("--size", type=int, choices=[256, 512, 1024], default=1024)
    p.add_argument("--mode", choices=["evidence", "llm"], default="evidence")
    p.add_argument("--backend", choices=["minilm", "tfidf"], default="minilm")
    args = parser.parse_args()
    try:
        if args.command == "ingest":
            from .ingest import acquire
            results = acquire(args.manifest)
            if any(r["status"] != "ok" for r in results):
                return 1
        elif args.command == "index":
            from .retrieval import build_index, TfidfEmbedder
            from .ingest import load_documents
            docs = load_documents()
            embedder = TfidfEmbedder(documents=docs) if args.backend == "tfidf" else None
            print(build_index(args.size, ROOT / f"artifacts/index-{args.backend}-{args.size}", documents=docs, embedder=embedder))
        elif args.command == "query":
            from .retrieval import Retriever
            from .generation import evidence_only, generate
            ctx = Retriever(ROOT / f"artifacts/index-{args.backend}-{args.size}").search(args.query)
            result = generate(args.query, ctx) if args.mode == "llm" else evidence_only(args.query, ctx)
            if args.output:
                write_json(args.output, result)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif args.command == "evaluate":
            from .evaluate import run_experiment
            result = run_experiment(exploratory=args.exploratory, backend=args.backend)
            print(json.dumps({"status": result["status"], "selected_chunk_tokens": result.get("selected_chunk_tokens"),
                              "summary": [{"size": s["chunk_tokens"], **s["summary"]} for s in result["sizes"]]}, indent=2))
        elif args.command == "examples":
            from .evaluate import save_examples
            print(save_examples(ROOT / f"artifacts/index-{args.backend}-{args.size}", args.mode,
                                ROOT / f"examples/{args.backend}-{args.mode}.json"))
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
