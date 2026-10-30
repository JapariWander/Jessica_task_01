# Corpus and provenance

`sources.csv` is the authoritative 16-source manifest: 12 articles and 4 publicly
readable arXiv preprints. The articles come from Letian Wang's *Quantitative Trading
and Systematic Investing* blog and Teddy Koker's blog, using their public publishing
repositories. `fetch_url` pins each article to an exact publisher Git commit;
`url` records the original reading URL. This is acquisition of an openly published
source, not a paywall or access-control bypass. QuantStart returned 503 and was
replaced with these verified sources; no unavailable article is counted.

Topics cover time-series/cross-sectional mean reversion, cointegration, momentum,
portfolio/factor methods, PCA yield-curve factors, GARCH, leverage, volatility
management, and survivorship bias. The article treatment of "factors" includes
portfolio risk and PCA rather than a comprehensive equity value/size-factor survey;
the Bayesian factor-investing paper supplies a direct factor-investing treatment.
The corpus is educational and small; it is not representative of all quant research.

Run `quant-rag ingest`. `acquisition.json` records successful retrieval, word counts,
raw and clean-text SHA-256 hashes, retrieval time, PDF pages, and observed metadata.
`processed/*.md` are readable clean texts; matching JSON files hold the indexed
body and metadata. `raw/` retains source bytes locally for audit. Source title,
author, and publication date are in the manifest; observed PDF author names/dates
were verified against arXiv landing-page citation metadata. Publication dates are
initial preprint dates, not necessarily the latest revision date. The acquisition
log's fetched URL and hashes identify the particular PDF acquired.

Extraction removes HTML navigation, scripts, forms and image markup; it retains
headings, code, paragraphs and page boundaries. Charts/images are not interpreted.
PDF formulas, table layout, and references can remain noisy and need visual checks.
This is text extraction, not a claim of lossless mathematical conversion.

## Rights

Public accessibility is not a blanket redistribution license. Letian Wang's pages
state CC BY-NC-SA 4.0 unless otherwise noted. Preserve attribution, noncommercial
conditions and share-alike obligations for any applicable redistribution. Teddy
Koker's repository LICENSE credits its Minima theme; it is not treated as an
explicit license to all authored article text. The four arXiv pages state arXiv's
nonexclusive distribution license; that permits arXiv distribution, not arbitrary
third-party republication. Consult rights holders before republishing full texts.

Raw/processed full texts, model weights, full-context evaluation traces and example
JSON are ignored by Git. They are available in this local workspace, and reproducible
with the acquisition/experiment commands. Safe metadata, summary metrics, code,
and reports are tracked separately. Do not blindly add ignored third-party content
to a public repository. No repository has been published or pushed by this project.
