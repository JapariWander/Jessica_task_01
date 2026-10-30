import json
from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from quant_rag.evaluate import metrics
from quant_rag.generation import GenerationError, generate, validate_answer
from quant_rag.ingest import clean_html, clean_markdown
from quant_rag.io import read_json
from quant_rag.retrieval import Embedder, Retriever, build_index, chunk_documents


def tokenizer():
    t = Tokenizer(WordLevel({"[UNK]": 0, **{f"w{i}": i+1 for i in range(1300)}}, unk_token="[UNK]"))
    t.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=t, unk_token="[UNK]")


def doc(id="a", text=None):
    return {"id": id, "text": text or " ".join(f"w{i}" for i in range(1200)),
            "title": id, "url": "https://example.org/"+id, "topic": "test"}


@pytest.mark.parametrize("size", [256, 512, 1024])
def test_chunk_full_coverage_and_exact_provenance(size):
    d = doc()
    chunks = chunk_documents([d], tokenizer(), size)
    covered = set()
    for c in chunks:
        assert c["token_end"]-c["token_start"] <= size
        assert c["text"] == d["text"][c["char_start"]:c["char_end"]]
        covered.update(range(c["token_start"], c["token_end"]))
    assert covered == set(range(1200))
    assert chunks[-1]["token_end"] == 1200


def test_html_excludes_navigation_preserves_code_and_metadata():
    text, meta = clean_html(b'<title>T</title><nav>ads</nav><article><h1>Strategy</h1><p>Signal</p><pre>x = 1\ny = 2</pre></article>')
    assert "ads" not in text and "x = 1\ny = 2" in text
    assert meta["observed_title"] == "T"


def test_original_markdown_removes_frontmatter_and_images():
    text, _ = clean_markdown(b'---\ntitle: test\n---\nBuy lagged signals.\n![x](plot.png)\n```python\nx = 1\n```')
    assert "title:" not in text and "plot.png" not in text and "x = 1" in text


def test_metrics_deduplicate_documents_without_promoting_chunk_ranks():
    assert metrics(["a", "a", "b"], ["a", "b"]) == {"recall_at_k": 1, "mrr_at_k": 1}
    assert metrics(["x", "a"], ["a"])["mrr_at_k"] == 0.5
    assert metrics(["x"], ["a"])["recall_at_k"] == 0
    assert metrics(["x"], [])["recall_at_k"] is None


def answer():
    return {"strategy_logic": [{"claim": "A", "citations": ["a:0-3"]}], "comparison": [],
            "required_data": [], "signal_and_rules": [], "pseudocode": ["proposal: lag signals"],
            "risks": [], "limitations": [], "insufficient_evidence": False}


def test_citation_validation_does_not_claim_entailment():
    result = validate_answer(answer(), [{"chunk_id": "a:0-3"}])
    assert result["semantic_entailment"].startswith("not_evaluated")


@pytest.mark.parametrize("cites", [[], ["invented:0-9"], "a:0-3"])
def test_missing_or_fabricated_citations_rejected(cites):
    a = answer(); a["strategy_logic"][0]["citations"] = cites
    with pytest.raises(GenerationError):
        validate_answer(a, [{"chunk_id": "a:0-3"}])


def test_api_contract_and_secrets_not_in_return(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "unit-test-secret")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    def post(url, **kwargs):
        assert url.endswith("/chat/completions")
        assert kwargs["headers"]["Authorization"] == "Bearer unit-test-secret"
        assert kwargs["allow_redirects"] is False
        return SimpleNamespace(status_code=200, json=lambda: {"choices": [{"message": {"content": json.dumps(answer())}}]})
    result = generate("explain", [{"chunk_id": "a:0-3", "text": "evidence"}], SimpleNamespace(post=post))
    assert result["mode"] == "llm" and "unit-test-secret" not in json.dumps(result)


def test_api_failure_body_is_not_exposed(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "unit-test-secret")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    with pytest.raises(GenerationError, match="HTTP 401"):
        generate("query", [{"chunk_id": "a"}], SimpleNamespace(post=lambda *a, **k: SimpleNamespace(status_code=401)))


def test_no_credentials_requires_explicit_evidence_mode(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(GenerationError, match="LLM_API_KEY"):
        generate("query", [{"chunk_id": "a"}])


def test_real_faiss_save_reload_rank_and_provenance(tmp_path):
    class E:
        name = "test-deterministic"; window = 254
        model = [SimpleNamespace(auto_model=SimpleNamespace(config=SimpleNamespace(_commit_hash=None)))]
        tokenizer = tokenizer()
        def encode(self, texts):
            return np.array([[1, 0] if "w1" in t else [0, 1] for t in texts], dtype="float32")
    docs = [doc("relevant", "w1 w2"), doc("other", "w8 w9")]
    directory = build_index(256, tmp_path / "index", documents=docs, embedder=E())
    r = Retriever(directory, E())
    found = r.search("w1", 5)
    assert found[0]["document_id"] == "relevant"
    assert len(found) == 2 and found[0]["url"] == docs[0]["url"]
    assert read_json(directory / "metadata.json")["documents"] == 2


def test_long_embedding_uses_tail_beyond_model_window():
    e = Embedder.__new__(Embedder)
    e.tokenizer = tokenizer()
    e.window = 254
    seen = []
    class Model:
        def encode(self, windows, **kwargs):
            seen.extend(windows)
            return np.array([[0., 1.] if "w1199" in w else [1., 0.] for w in windows])
    e.model = Model()
    vector = e.encode([doc()["text"]])
    assert len(seen) == 5
    assert all(len(e.tokens(w)) <= 254 for w in seen)
    assert vector[0, 1] > 0, "Tail evidence must affect the pooled vector"
    assert np.linalg.norm(vector[0]) == pytest.approx(1)


def test_proposed_labels_cannot_pass_as_human(tmp_path, monkeypatch):
    from quant_rag import evaluate
    questions = tmp_path / "q.json"; labels = tmp_path / "l.json"
    questions.write_text(json.dumps([{"id": "q1", "query": "a", "split": "dev"}]))
    labels.write_text(json.dumps([{"question_id": "q1", "relevant_document_ids": ["a"], "status": "agent_proposed"}]))
    monkeypatch.setattr(evaluate, "load_documents", lambda: [doc()])
    with pytest.raises(ValueError, match="Human evaluation"):
        evaluate.run_experiment(questions, labels)
