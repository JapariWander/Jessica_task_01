"""Functional UI smoke test against the actual local corpus and MiniLM index."""
from streamlit.testing.v1 import AppTest
from quant_rag.io import ROOT

at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
assert not at.exception, at.exception
at.selectbox[0].select(1024)
at.selectbox[1].select("minilm")
at.button[0].click().run()
assert not at.exception, at.exception
assert len(at.expander) == 5
result, contexts = at.session_state["result"]
assert result["mode"] == "evidence_only" and contexts
assert all(c["url"].startswith("https://") and c["text"] for c in contexts)
print("PASS: Streamlit query -> MiniLM -> FAISS -> source evidence and downloadable output")

review = AppTest.from_file(str(ROOT / "pages/1_相关性标注.py"), default_timeout=30).run()
assert not review.exception, review.exception
review.button[0].click().run()
assert not review.exception and len(review.error) == 1
print("PASS: annotation page refuses unnamed/unconfirmed human review")
