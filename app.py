import json

import streamlit as st

from quant_rag.generation import GenerationError, evidence_only, generate
from quant_rag.io import ROOT
from quant_rag.retrieval import Retriever

st.set_page_config(page_title="Quant Research RAG", layout="wide")
st.title("量化研究助手 / Quant Research RAG")
st.caption("检索研究证据，构思可回测的策略。回答不代表回测结果。")
size = st.sidebar.selectbox("Chunk size (tokens)", [256, 512, 1024], index=2)
backend = st.sidebar.selectbox("检索模型", ["minilm", "tfidf"])
if backend == "tfidf":
    st.sidebar.caption("TF-IDF 是词汇检索基线，与语义 embedding 的结果分别报告。")
mode = st.sidebar.radio("回答模式", ["原文证据（无需 API）", "结构化 LLM 回答"])
st.sidebar.caption("LLM 凭据从进程环境读取；不会在此界面保存。")
query = st.text_area("研究问题 / Research question", "Compare momentum and mean reversion signals and their implementation risks.")


@st.cache_resource
def load_retriever(path, version):
    return Retriever(path)


if st.button("检索与分析", type="primary"):
    path = ROOT / f"artifacts/index-{backend}-{size}"
    if not (path / "metadata.json").exists():
        st.error(f"索引尚未构建。请先运行 quant-rag ingest 和 quant-rag index --backend {backend} --size {size}。")
    elif not query.strip():
        st.warning("请输入研究问题。")
    else:
        try:
            with st.spinner("检索研究资料…"):
                retriever = load_retriever(str(path), (path / "metadata.json").stat().st_mtime_ns)
                contexts = retriever.search(query)
            result = evidence_only(query, contexts) if mode.startswith("原文") else generate(query, contexts)
            st.session_state["result"] = (result, contexts)
        except (GenerationError, OSError, ValueError, RuntimeError) as exc:
            st.error(str(exc))

if "result" in st.session_state:
    result, contexts = st.session_state["result"]
    if result["mode"] == "evidence_only":
        st.info("以下为检索原文，没有调用 LLM，也没有生成或验证策略。")
    else:
        st.subheader("结构化研究输出")
        st.json(result["answer"])
        st.caption("引用 ID 已检查；论断是否被原文支持仍需审核。")
    st.subheader("检索上下文 / Retrieved evidence")
    for c in contexts:
        with st.expander(f"{c['chunk_id']} · {c['title']} · cosine {c['score']:.3f}"):
            st.write(c["text"])
            st.write("来源：", c["url"])
    st.download_button("下载问题、上下文和输出", json.dumps(result, ensure_ascii=False, indent=2), "research-result.json", "application/json")
