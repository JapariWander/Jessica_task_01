# 实际示例

以下三个问题由 `quant-rag examples` 实际检索，完整片段、来源、分数和输出写入
`minilm-evidence.json` / `tfidf-evidence.json`，保存在当前工作区。文件包含第三方原文，
因此默认不进入 Git；可通过命令重新生成。

1. Explain time-series mean reversion and how a half-life estimate can inform a trading signal.
2. Compare cross-sectional mean reversion with the momentum ranking strategy from Stocks on the Move.
3. Propose pseudocode for cointegration-based pairs trading, including estimation and execution risks.

这些文件的输出模式是 `evidence_only`：实际原文摘录，**不是生成的策略答案**。
目前未配置生成 API，因而没有真实 LLM 生成示例，也没有语义引用支撑率。
配置密钥和模型后运行 `quant-rag examples --mode llm --size <选定尺寸>`，再用
`scripts/export_grounding_review.py` 导出人工逐论断审核表。

例如可追溯地检查：问题 1 是否同时找到均值回归的估计说明及成本论文；问题 2
是否同时找到两类方法；问题 3 的拟议步骤是否超出原文证据。完整实验文件保留了
所有 24 个问题、三种尺寸的检索上下文，便于复核遗漏与无关片段。
