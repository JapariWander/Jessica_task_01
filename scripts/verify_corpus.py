"""Audit local acquisition hashes and manifest completeness without network."""
import csv
from collections import Counter
from quant_rag.ingest import load_documents
from quant_rag.io import ROOT, digest, read_json

rows = list(csv.DictReader((ROOT / "data/sources.csv").open(encoding="utf-8")))
counts = Counter(r["document_type"] for r in rows)
assert counts == {"article": 12, "paper": 4}, counts
docs = {d["id"]: d for d in load_documents()}
records = {r["id"]: r for r in read_json(ROOT / "data/acquisition.json")["documents"]}
assert set(docs) == set(records) == {r["id"] for r in rows}
for row in rows:
    record, doc = records[row["id"]], docs[row["id"]]
    assert record["status"] == "ok", row["id"]
    assert record["text_sha256"] == digest(doc["text"].encode()), row["id"]
    suffix = ".pdf" if row["document_type"] == "paper" else ".md" if row["format"] == "markdown" else ".html"
    raw = (ROOT / "data/raw" / (row["id"] + suffix)).read_bytes()
    assert digest(raw) == record["raw_sha256"], row["id"]
    assert len(doc["text"].split()) >= 250
    assert row["url"] == doc["url"] and row["license_note"]
print("PASS: 12 articles + 4 papers, unique documents, source metadata, raw/text SHA-256 hashes and minimum body lengths")
