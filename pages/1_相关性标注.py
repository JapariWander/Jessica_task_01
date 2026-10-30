"""Human review UI: agent proposals are never auto-promoted to human labels."""
import streamlit as st
from quant_rag.ingest import load_documents
from quant_rag.io import ROOT, now, read_json, write_json

st.title("人工相关性标注")
st.write("请阅读问题及语料，选出能够直接支持回答的文档。候选标签由代理提出，尚非人工标注。")
questions = read_json(ROOT / "evaluation/questions.json")
labels = read_json(ROOT / "evaluation/labels.json")
documents = load_documents()
q = st.selectbox("问题", questions, format_func=lambda x: f"{x['id']} · {x['query']}")
label = next(x for x in labels if x["question_id"] == q["id"])
reviewer = st.text_input("审核者姓名或标识")
selected = st.multiselect("相关文档（无充分证据时可为空）", [d["id"] for d in documents],
                          default=label["relevant_document_ids"],
                          format_func=lambda i: next(d["title"] for d in documents if d["id"] == i), key=q["id"])
reason = st.text_area("标注理由及重要证据位置", label["rationale"], key="reason-"+q["id"])
reviewed = st.checkbox("我已实际审核本问题的相关性，不只是接受代理候选标签。", key="check-"+q["id"])
if st.button("保存人工标注"):
    if not reviewer.strip() or not reviewed or not reason.strip():
        st.error("请填写审核者、理由，并确认实际审核。")
    else:
        label.update(relevant_document_ids=selected, status="human_reviewed", reviewer=reviewer.strip(),
                     rationale=reason.strip(), reviewed_at=now())
        write_json(ROOT / "evaluation/labels.json", labels)
        st.success("已保存。全部问题审核后，可运行 quant-rag evaluate。")
for doc in documents:
    with st.expander(f"{doc['id']} · {doc['title']}"):
        st.write(doc["url"])
        st.text(doc["text"])
