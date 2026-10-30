# Evaluation protocol

## Labels and splits

24 fixed questions: q01–q12 development; q13–q24 held-out test. English tasks cover
explanation, comparison, implementation, and costs/biases; q24 probes Chinese to
English retrieval. q12 and q15 have no established corpus-supported answer and are
excluded from recall/MRR (neither zero nor successful retrievals). They still need
qualitative abstention checks. No reliable abstention accuracy is claimed from
these two negative queries. There are 11 scored dev and 11 scored test questions.

`label_audit.json` records a correction made after exploratory results exposed a
label-quality problem: a10 describes position-count and volatility filters, not
signal smoothing. q15 was preserved and its false-positive relevance label removed.
Before-audit runs are retained locally. Retrieval parameters were not tuned on this
test finding; all final scores are still exploratory and need independent human
review. This audit is not a clean, preregistered final test study.

`labels.json` contains **agent-proposed document relevance**, not human annotations.
These are broad document-level judgments; they do not certify that every retrieved
chunk from that document answers the question. In the Streamlit annotation page,
a human must review the corpus, correct relevant IDs, give a rationale, and identify
the reviewer. Do not mark labels `human_reviewed` unless this happened. Review all
24 before a human-label run. For stronger evaluation, use two independent reviewers,
resolve disagreements, and record agreement. No such agreement has been measured.

`quant-rag evaluate` refuses incomplete human labels. `--exploratory` explicitly
allows provisional labels. Results snapshot both the question set and labels to
prevent subsequent edits from silently changing what was evaluated.

## Retrieval experiment

MiniLM WordPiece tokenizer, chunk sizes 256, 512, 1024 tokens, overlap 12.5%, exact
character offsets into cleaned documents, cosine similarity via L2-normalized
vectors and FAISS IndexFlatIP. All sizes use the same corpus, questions, model, and
k=5. MiniLM's 256-token model limit is handled by embedding nontruncated subwindows
and averaging their normalized vectors, then normalizing again. This allows all
large-chunk text to contribute, but can dilute specific signals. The experiment
therefore compares complete retrieval configurations, not a pure isolated causal
effect of chunk size. Chunk sizes also change the total evidence-token budget at
fixed k; generation-quality comparisons should additionally control that budget.

The explicit TF-IDF baseline fits up to 4096 word/unigram-bigram features on the
same full documents before splitting, and also uses FAISS cosine search. It uses
the same WordPiece chunk boundaries. It is a lexical baseline, not Sentence
Transformers and not a cross-lingual model. Zero query vectors return no matches.

Recall@5 = distinct relevant documents found in the five highest-scoring chunks /
all labeled relevant documents. MRR@5 uses the first relevant chunk's actual rank.
Duplicate chunks from one document do not increase recall. A document-level hit
is weaker than a relevant-passage judgment; inspect the saved contexts as well.
Select by dev Recall@5, then dev MRR@5, then smaller size. Test scores are reported
once selection is made; do not choose the size with the best test score.

Timing includes query embedding and FAISS retrieval, excludes LLM and model load;
CPU single runs are diagnostic only, not stable performance benchmarks. Indexing
time includes chunking, embedding, and writes, but excludes model initialization.

## Grounding and actionability

Configure the API and run `quant-rag examples --mode llm --size <selected-size>`.
Export its actual factual claims with `scripts/export_grounding_review.py`.
For each claim, a human marks:

- **yes**: cited excerpt entails the complete claim;
- **partial**: excerpt supports only part or omits an important qualification;
- **no**: unsupported, contradicted, or irrelevant evidence.

Report yes / all reviewed claims, partial and no separately, and count outputs
rejected by schema/citation checks. Never use valid citation IDs as entailment
scores. Rate actionability 1–5: 1 = vague prose; 3 = usable signal/data description
with missing execution detail; 5 = clear proposed pseudocode, timing, costs, and
out-of-sample checks. Keep source claims separate from untested implementation
proposals. Pseudocode is not a completed or validated backtest.

For q12, check whether the model explicitly declines unsupported findings. Review
possible prompt injection in source text and missing evidence for comparisons.
The system provides no tool execution to the LLM, but prompt-only separation is
not a proof of injection resistance.

## Research-time benefit

Use a paired comparison: same questions and corpus, manual search versus assistant,
record time to a reviewer-accepted cited research outline, randomize task order,
and separately log factual errors. No user study has been run; time savings and
investment profitability must not be claimed as measured outcomes.
