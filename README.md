# Quant Research RAG · 本地量化研究助手

从公开量化文章和论文检索证据，辅助策略解释、方法比较和回测构思。
默认技术栈：Python、Sentence Transformers、FAISS、Streamlit；生成使用可配置的
OpenAI-compatible Chat Completions API。本地证据检索不需要 LLM 密钥。

## 安装

需要 Python 3.11+，建议 3.12。当前云环境目录为 `/workspace/Jessica_task_01`。

```bash
cd /workspace/Jessica_task_01
bash scripts/install.sh
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
```

`requirements.lock.txt` 固定已验证环境中的依赖，包括 CPU 版 PyTorch。
该锁文件适用于已验证的 Linux/Python 3.12 环境；其他平台可安装 `pip install -e '.[dev]'`
并自行验证。模型首次运行需要 Hugging Face 及其权重 CDN 的网络访问。后续可使用缓存，
或将 `EMBEDDING_MODEL` 指向本地 Sentence Transformer 模型目录。

## 数据与索引

```bash
.venv/bin/quant-rag ingest
.venv/bin/quant-rag index --size 256
.venv/bin/quant-rag index --size 512
.venv/bin/quant-rag index --size 1024
```

`data/sources.csv` 包含 12 篇真实公开博客文章和 4 篇 arXiv 论文。
文章来源固定至作者发布仓库的 Git 提交；原始阅读地址、作者、日期、主题和许可均保留。
采集检查 robots.txt，不绕过付费墙；失败会返回非零退出码。
清洗后的 Markdown/JSON 位于 `data/processed`，原始文件位于 `data/raw`。
详情与版权限制见 [data/DATASET.md](data/DATASET.md)。

```bash
.venv/bin/quant-rag query "Compare momentum and cross-sectional mean reversion signals." --size 1024
bash scripts/start.sh
```

CLI 默认输出真实原文证据，不伪装成 LLM 答案。Streamlit 界面可切换分块尺寸、检索模型、
回答模式，查看完整来源片段，并下载查询结果。界面中的人工标注页可审核相关性标签。
服务默认使用 8501 端口；云环境是否提供外部访问由平台决定。

## 配置结构化生成

在安全的环境设置中提供 `LLM_API_KEY`、`LLM_MODEL`。
`LLM_BASE_URL` 默认 `https://api.openai.com/v1`；其他兼容供应商需设置相应地址和网络访问。
密钥只从进程环境读取，不写入代码、输出或 UI。不要在聊天中粘贴密钥。
`.env.example` 仅说明变量，不自动加载，也不含凭据。

```bash
.venv/bin/quant-rag query "Explain cointegration pairs trading and propose implementation steps." \
  --size 1024 --mode llm --output /tmp/research-answer.json
```

输出包括：策略逻辑、比较、数据要求、信号与规则、拟议伪代码、风险、局限和证据不足标识。
事实性论断必须引用真实 `document_id:tokenStart-tokenEnd`，系统拒绝缺失或未知引用。
引用 ID 检查不是语义正确性证明；仍需人工检查原文是否支持论断。
没有密钥/模型时，LLM 模式明确失败，不自动用模板冒充生成结果。
HTTP 仅允许本地服务；远程 API 必须使用 HTTPS，重定向不会携带凭据继续请求。

## 分块实验与评估

```bash
# 默认拒绝未经人工审核的标签
.venv/bin/quant-rag evaluate
# 明确允许代理候选标签，仅作探索实验
.venv/bin/quant-rag evaluate --exploratory
# 单独的词汇检索基线
.venv/bin/quant-rag evaluate --backend tfidf --exploratory
```

24 个问题分为 12 个开发问题和 12 个测试问题；开发集、测试集各一个问题没有已确认的支持证据，
不计入 Recall/MRR。一次候选标签纠错记录在 `evaluation/label_audit.json`，不是人工评审。
使用相同语料比较 256/512/1024 WordPiece tokens、12.5% overlap、top-5。
默认 MiniLM 只有 256-token 模型窗口，因此长块采用完整子窗口向量聚合，避免静默截断。
TF-IDF 是额外基线，不会被标为 Sentence Transformers 结果。

`evaluation/labels.json` 当前为 **agent_proposed**，不是人工标注。
请在标注页阅读资料、修改标签、填写审核者和理由；全部完成后再运行不带
`--exploratory` 的正式评估。开发集选择参数，测试集只报告结果。
完整协议、指标定义与失败分析方法见 [evaluation/PROTOCOL.md](evaluation/PROTOCOL.md)。

实际完整实验结果：`evaluation/results-minilm.json`、`evaluation/results-tfidf.json`。
不含原文的可提交摘要：`evaluation/summary.json`；设计与发现见 [report.md](report.md)。

## 示例与引用支撑审核

```bash
# 将 --size 改成实验选出的尺寸
.venv/bin/quant-rag examples --size 1024
# 配置 API 后，保存真实生成的答案及其检索上下文
.venv/bin/quant-rag examples --size 1024 --mode llm
.venv/bin/python scripts/export_grounding_review.py examples/minilm-llm.json
```

示例包含策略解释、方法比较、实现构思三个问题。真实运行的证据输出保存在
`examples/minilm-evidence.json`，见 [examples/README.md](examples/README.md)。
全文、完整上下文和模型权重保留在本地并被 Git 忽略；源码、来源清单、元信息和指标摘要
可单独提交。不要把未取得再分发许可的全文直接加入公开仓库。

## 验证与项目结构

```bash
.venv/bin/pytest -q
.venv/bin/python -m pip check
```

- `src/quant_rag/ingest.py`：下载、清洗、来源与哈希记录。
- `src/quant_rag/retrieval.py`：token 分块、完整窗口 embedding、FAISS、TF-IDF 基线。
- `src/quant_rag/generation.py`：API、结构化输出、引用检查和证据模式。
- `src/quant_rag/evaluate.py`：三尺寸对比、标签校验、上下文与示例保存。
- `app.py`、`pages/`：研究界面和人工标注。
- `evaluation/`、`examples/`、`report.md`：评估、示例与报告。

这个项目支持研究构思；它没有实现行情回测引擎，也没有证明任何策略盈利或已节省研究时间。

## GitHub 发布内容

`evaluation/articles/` 和 `evaluation/papers/` 的公开文本仅包含标题和作者。全文、原始 PDF、完整检索上下文、向量索引与模型缓存不上传。`data/sources.csv` 保留复现采集所需的来源清单；下载及索引在运行环境内执行。
