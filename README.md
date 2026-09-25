# Evaluating a LangGraph + Qdrant Hybrid RAG Pipeline with DeepEval

A companion notebook to `LangGraph_Qdrant_Hybrid_Rerank.ipynb`, focused entirely on **evaluating** a Retrieval-Augmented Generation (RAG) pipeline using the open-source [DeepEval](https://github.com/confident-ai/deepeval) framework — because building a RAG pipeline is only half the job; knowing whether it's actually *good* is the other half.

**Environment:** Google Colab
**Judge / generation LLM:** OpenAI (via API key stored in Colab Secrets) — DeepEval uses an LLM internally to score most metrics

---

### 📌 Table of Contents

* [What is this project?](#-what-is-this-project)
* [Why LLM Evaluation?](#-why-llm-evaluation)
* [The Evaluation Problem — Why It's Hard](#-the-evaluation-problem--why-its-hard)
* [Metrics Demonstrated](#-metrics-demonstrated)
* [How it Works — Full Pipeline](#-how-it-works--full-pipeline)
* [Flowchart](#-flowchart)
* [Project Structure](#-project-structure)
* [Tech Stack & Tools](#-tech-stack--tools)
* [Component Deep Dive](#-component-deep-dive)
* [All DeepEval Metrics — Complete Reference](#-all-deepeval-metrics--complete-reference)
* [Setup & Installation](#-setup--installation)
* [Running on Google Colab](#-running-on-google-colab)
* [Configuration & Customization](#-configuration--customization)
* [Sample Outputs](#-sample-outputs)
* [Limitations](#-limitations)
* [How Limitations Can Be Resolved](#-how-limitations-can-be-resolved)
* [Key Concepts for Beginners](#-key-concepts-for-beginners)
* [DeepEval vs Other Evaluation Frameworks](#-deepeval-vs-other-evaluation-frameworks)
* [Real-World Use Cases](#-real-world-use-cases)

---

## 📌 What is this project?

This notebook rebuilds a **minimal, self-contained version** of a hybrid RAG pipeline (dense + sparse retrieval, cross-encoder reranking, LLM generation) purely so it can generate real outputs — and then puts those outputs under a microscope using DeepEval.

Concretely, it covers:
- DeepEval fundamentals: `LLMTestCase`, metrics, and `evaluate()`
- Rebuilding a minimal RAG pipeline (LangGraph + Qdrant hybrid search + reranking) to produce real answers to evaluate
- Building a small hand-curated evaluation dataset (queries + reference answers)
- Core **RAG metrics**: Faithfulness, Answer Relevancy, Contextual Precision, Contextual Recall, Contextual Relevancy
- The **Hallucination** metric
- **LLM-as-a-judge** scoring via `GEval` for custom, criteria-based checks
- Aggregating everything into a readable scorecard
- Notes on taking this evaluation workflow from a notebook into CI (regression testing)

This is not a tutorial on *building* a RAG system — it assumes that part exists — it's a tutorial on **proving your RAG system works, and knowing exactly where it breaks when it doesn't.**

## 📌 Why LLM Evaluation?

Anyone can eyeball five example answers from a RAG demo and say "looks good." That doesn't scale, isn't reproducible, and tells you nothing about *why* an answer is wrong when it is.

LLM evaluation frameworks like DeepEval exist because:
- **Manual review doesn't scale.** You can't hand-check every answer every time you tweak a prompt, swap an embedding model, or change `top_k`.
- **RAG failures are compound.** A bad final answer could be caused by retrieval, generation, or both — and you need metrics that isolate which.
- **Regressions are silent.** Without numeric tracking, a change that quietly degrades answer quality by 15% ships unnoticed.
- **"Good" is subjective but still measurable.** LLM-as-a-judge techniques (like `GEval`) let you encode fuzzy, human-style judgment ("is this answer clear and non-hedgy?") into a repeatable, scorable check.

## 📌 The Evaluation Problem — Why It's Hard

A RAG pipeline has at least **two independent moving parts**, and each can fail on its own:

1. **Retrieval** — did the system fetch the right context for the query?
2. **Generation** — given that context, did the LLM produce an answer that's faithful to it *and* actually answers the question?

A generic "is this a good answer?" score can't tell you which part broke. If the answer is wrong, was it because:
- retrieval missed the right document (a **retrieval** failure), or
- the LLM had the right context but hallucinated anyway (a **generation** failure)?

This is exactly why DeepEval splits its metrics cleanly along that boundary instead of giving one opaque quality score — see [Metrics Demonstrated](#-metrics-demonstrated) below.

## 📌 Metrics Demonstrated

The notebook exercises **eight** evaluation signals, split across three categories:

| Category | Metric | What it catches |
|---|---|---|
| Generation | `FaithfulnessMetric` | Claims in the answer that aren't supported by the retrieved context (hallucination *relative to context*) |
| Generation | `AnswerRelevancyMetric` | Correct-sounding but off-topic answers that don't actually address the question |
| Retrieval | `ContextualPrecisionMetric` | Relevant chunks buried under irrelevant ones (ranking quality of retrieval) |
| Retrieval | `ContextualRecallMetric` | Retrieval that's too narrow — missing information needed for the reference answer |
| Retrieval | `ContextualRelevancyMetric` | Retrieval that's too broad/noisy overall |
| Hallucination | `HallucinationMetric` | Factual contradiction against a fixed ground-truth document (distinct from Faithfulness, which checks against *retrieved* context specifically) |
| LLM-as-a-judge | `GEval` — "Correctness" | Whether the answer is factually consistent with the reference answer, without requiring exact wording |
| LLM-as-a-judge | `GEval` — "Instructional Clarity" | Whether the answer is clear, concise, and free of hedging/filler for a learner audience |

## 📌 How it Works — Full Pipeline

End-to-end, the notebook runs through these stages:

1. **Environment setup** — install DeepEval, LangGraph, LangChain, Qdrant client, FastEmbed, and Sentence-Transformers; load the OpenAI API key from Colab Secrets.
2. **Rebuild a minimal RAG pipeline** — a condensed 3-node LangGraph graph (`retrieve → rerank → generate`) backed by an in-memory Qdrant collection, so the notebook is self-contained and produces *real* outputs rather than hand-written examples.
3. **Build an evaluation dataset** — six hand-curated `(query, expected_output)` pairs covering LangGraph, Qdrant, hybrid search, and reranking topics.
4. **Run the pipeline** on each query to collect `actual_output` (the generated answer) and `retrieval_context` (the reranked chunks actually used).
5. **Wrap each result in an `LLMTestCase`** — DeepEval's standard evaluation unit.
6. **Score with core RAG metrics** — Faithfulness, Answer Relevancy, Contextual Precision/Recall/Relevancy — via `evaluate()`.
7. **Score with custom `GEval` judges** — Correctness and Instructional Clarity.
8. **Aggregate everything into a pandas scorecard** — one row per query, one column per metric, plus per-metric averages to spot systematic weak points.
9. **(Notes) Move to CI** — a sketch of how the same test cases and metrics become a `pytest` suite (`deepeval test run`) for regression gating.

## 📌 Flowchart

```
                 ┌────────────────────┐
                 │   Evaluation Set    │
                 │ (query + reference) │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌────────────────────┐
                 │   RAG Pipeline      │
                 │ (LangGraph graph)   │
                 │                     │
                 │  retrieve  ──►      │  Hybrid search (dense + sparse, RRF fusion)
                 │  rerank    ──►      │  CrossEncoder reranking → top 3 chunks
                 │  generate           │  LLM answer grounded in reranked context
                 └──────────┬──────────┘
                            │  actual_output + retrieval_context
                            ▼
                 ┌────────────────────┐
                 │   LLMTestCase       │
                 │ input / actual_out  │
                 │ expected_out /      │
                 │ retrieval_context   │
                 └──────────┬──────────┘
                            │
              ┌─────────────┼──────────────┐
              ▼             ▼              ▼
      Retrieval metrics  Generation    LLM-as-a-judge
      (Precision/Recall/   metrics       (GEval:
       Relevancy)       (Faithfulness/   Correctness,
                         Answer Relevancy, Clarity)
                         Hallucination)
              │             │              │
              └─────────────┼──────────────┘
                            ▼
                 ┌────────────────────┐
                 │  Pandas Scorecard   │
                 │ per-query & per-    │
                 │ metric averages     │
                 └────────────────────┘
```

## 📌 Project Structure

```
.
├── DeepEval_RAG_Evaluation.ipynb        # This notebook (evaluation-only)
├── LangGraph_Qdrant_Hybrid_Rerank.ipynb # Companion notebook: builds the full RAG pipeline
└── README.md                            # You are here
```

## 📌 Tech Stack & Tools

| Layer | Tool |
|---|---|
| Evaluation framework | [DeepEval](https://github.com/confident-ai/deepeval) |
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) (`StateGraph`) |
| LLM access | LangChain (`ChatOpenAI`) |
| Vector database | [Qdrant](https://qdrant.tech/) (in-memory client) |
| Dense embeddings | FastEmbed — `BAAI/bge-small-en-v1.5` |
| Sparse embeddings | FastEmbed — `Qdrant/bm25` |
| Reranker | Sentence-Transformers `CrossEncoder` — `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Judge / generation LLM | OpenAI (default judge model: `gpt-4o-mini`) |
| Results wrangling | pandas |
| Runtime | Google Colab |

## 📌 Component Deep Dive

**Hybrid retrieval (`hybrid_search`)** — Embeds the query with both the dense (`bge-small-en-v1.5`) and sparse (BM25-style) models, issues two `Prefetch` queries against Qdrant, and merges the two ranked lists using **Reciprocal Rank Fusion (RRF)**, returning the fused top-k results.

**LangGraph state (`RAGState`)** — A `TypedDict` carrying `query`, `retrieved_docs`, `reranked_docs`, and `answer` through three nodes:
- `retrieve_node` — runs hybrid search, returns the top 10 candidates
- `rerank_node` — scores each (query, doc) pair with the CrossEncoder and keeps the top 3
- `generate_node` — builds a context-grounded prompt ("Answer using only the context below...") and calls the LLM

**Evaluation dataset** — Six queries spanning LangGraph vs. LCEL, hybrid search mechanics, open-source rerankers, RRF fusion rationale, and Qdrant fundamentals, each paired with a hand-written reference answer.

**`LLMTestCase` construction** — For every query, the pipeline's real `actual_output` and `retrieval_context` are combined with the dataset's `expected_output` to build the test case DeepEval scores against.

**Metric objects** — Each metric (`FaithfulnessMetric(threshold=0.7)`, etc.) is instantiated once and reused across all test cases, either through `evaluate()` for batch scoring or `.measure(test_case)` for inspecting a single example's `.score` and `.reason`.

**`GEval` custom judges** — `correctness_judge` and `clarity_judge` are defined with plain-English `criteria` strings and a list of `LLMTestCaseParams` telling DeepEval which fields of the test case to show the judge LLM.

**Scorecard aggregation** — Loops over every test case and every metric (`rag_metrics + judge_metrics`), calls `.measure()`, and collects scores into a pandas `DataFrame` — one row per query — plus a per-metric average to highlight systematic weak spots (e.g., "Contextual Recall is consistently the lowest score").

## 📌 All DeepEval Metrics — Complete Reference

| Metric | Type | Compares | Threshold used |
|---|---|---|---|
| `FaithfulnessMetric` | Generation | `actual_output` vs `retrieval_context` | 0.7 |
| `AnswerRelevancyMetric` | Generation | `actual_output` vs `input` | 0.7 |
| `ContextualPrecisionMetric` | Retrieval | `retrieval_context` ranking vs `expected_output` | 0.7 |
| `ContextualRecallMetric` | Retrieval | `retrieval_context` coverage vs `expected_output` | 0.7 |
| `ContextualRelevancyMetric` | Retrieval | `retrieval_context` vs `input` | 0.7 |
| `HallucinationMetric` | Factuality | `actual_output` vs a fixed reference context | — |
| `GEval` (custom) | LLM-as-a-judge | Any fields you choose, scored against your own plain-English criteria | 0.7 |

Every metric produces a `0–1` score, a pass/fail against its `threshold`, and a natural-language `reason` explaining the score — the `reason` field is what makes debugging a low score tractable instead of just a number in a spreadsheet.

## 📌 Setup & Installation

```bash
pip install deepeval langgraph langchain langchain-openai langchain-core \
    qdrant-client fastembed sentence-transformers
```

You'll also need an **OpenAI API key**, since DeepEval's default metrics use `gpt-4o-mini` internally as the judge model, and the pipeline's `generate_node` uses an OpenAI chat model to produce answers.

## 📌 Running on Google Colab

1. Open the notebook in Google Colab.
2. Click the 🔑 **Secrets** panel in the left sidebar.
3. Add a new secret named `OPENAI_API_KEY` with your OpenAI key as the value, and toggle **notebook access** on.
4. Run all cells top to bottom — the first code cell installs dependencies, the second loads the key via `google.colab.userdata`.

If you already have a compiled `graph` from the companion `LangGraph_Qdrant_Hybrid_Rerank.ipynb` notebook in the same runtime, you can skip the pipeline-rebuild section and reuse it directly.

## 📌 Configuration & Customization

- **Swap the judge model** — pass `model=` to any metric constructor, e.g. `FaithfulnessMetric(threshold=0.7, model="gpt-4o")` for a stricter judge.
- **Fully open-source judging** — DeepEval supports plugging in local models via its `DeepEvalBaseLLM` interface, trading judge consistency/quality for zero OpenAI dependency.
- **Adjust thresholds** — every metric's `threshold` controls the pass/fail cutoff independently of the raw score.
- **Add your own `GEval` criteria** — define any qualitative axis in plain English: penalize missing topic citations, penalize jargon for a beginner audience, reward answers that admit insufficient context, etc.
- **Tune retrieval** — `top_k` and `prefetch_k` in `hybrid_search`, and the number of chunks kept in `rerank_node` (currently top 3), directly affect Contextual Precision/Recall/Relevancy scores.

## 📌 Sample Outputs

The evaluation dataset includes queries such as:

- *"What is the difference between LangGraph and a LangChain LCEL chain?"*
- *"How does hybrid search combine dense and sparse retrieval?"*
- *"What open-source model can be used for reranking retrieved documents?"*
- *"Why would I use RRF fusion instead of dense search alone?"*
- *"What is Qdrant and what makes it different from a regular database?"*
- *"What programming language is Qdrant written in?"*

For each, the notebook prints the generated answer, the number of retrieval-context chunks used, and — once scored — a per-query, per-metric row in the final pandas scorecard, plus a sorted table of average scores per metric so the weakest evaluation dimension is immediately visible.

## 📌 Limitations

- **Tiny eval set.** Six queries is enough to demonstrate the workflow, not enough to draw statistically meaningful conclusions about pipeline quality.
- **Judge-model dependency.** Every built-in metric and every `GEval` judge relies on an LLM (`gpt-4o-mini` by default) to score outputs — judge inconsistency or bias becomes evaluation noise.
- **Cost and latency.** Every metric call is itself an LLM call; scoring 6 examples × 7 metrics already means dozens of API calls, which scales linearly (or worse) with eval-set size.
- **In-memory vector store.** The Qdrant collection lives in RAM and disappears when the Colab runtime ends — fine for demos, not for a persistent eval harness.
- **Static reference answers.** Hand-written `expected_output` values don't update as the underlying corpus changes, risking drift between the eval set and reality.
- **No retrieval-only ablations.** The notebook evaluates the full pipeline end-to-end; it doesn't isolate dense-only vs. sparse-only vs. hybrid retrieval quality on its own.

## 📌 How Limitations Can Be Resolved

- **Grow the eval set** to dozens/hundreds of examples, ideally sourced from real user queries or logged production traffic, and keep it **held out** from prompt/retrieval tuning to avoid overfitting to your own tests.
- **Diversify or self-host judges** — compare scores across multiple judge models, or use DeepEval's `DeepEvalBaseLLM` interface to run fully open-source judging where consistency matters more than judge sophistication.
- **Batch and cache** metric calls, and consider running only the cheapest/fastest metrics on every commit while reserving the full suite for nightly or pre-release runs.
- **Move to a persistent Qdrant deployment** (Qdrant Cloud or a self-hosted server) once the eval harness needs to run repeatedly rather than once per Colab session.
- **Refresh reference answers** whenever the underlying document corpus changes, and version the eval dataset alongside the corpus.
- **Add retrieval-only experiments** (dense vs. sparse vs. hybrid, with/without reranking) to pair with `ContextualPrecision`/`ContextualRecall` trends and isolate whether regressions come from retrieval or generation.

## 📌 Key Concepts for Beginners

| Term | Plain-English meaning |
|---|---|
| **RAG (Retrieval-Augmented Generation)** | Instead of relying only on what an LLM memorized during training, first fetch relevant documents, then ask the LLM to answer using that fetched context. |
| **Dense retrieval** | Searching by *meaning* — text is converted into numeric vectors, and similar meanings end up close together, even without shared words. |
| **Sparse retrieval (BM25)** | Searching by *keywords* — like a smarter version of Ctrl+F that weighs how rare/important each term is. |
| **Hybrid search** | Running dense and sparse retrieval together and merging the results, so you catch both "conceptually similar" and "exact keyword match" hits. |
| **Reciprocal Rank Fusion (RRF)** | A simple, robust way to merge two ranked lists into one, based on each item's rank position rather than raw scores. |
| **Reranking** | A second, more expensive pass that jointly scores each (query, document) pair to produce a more accurate final ordering than the initial retrieval alone. |
| **Faithfulness / Hallucination** | Whether the model's answer sticks to what the retrieved documents actually said, instead of making things up. |
| **LLM-as-a-judge** | Using one LLM to *grade* another LLM's output against criteria you write in plain English, instead of a hand-coded scoring function. |

## 📌 DeepEval vs Other Evaluation Frameworks

| Framework | Focus | Notable strengths | Trade-offs |
|---|---|---|---|
| **DeepEval** | RAG & LLM app testing | `pytest`-style workflow (`deepeval test run`), rich RAG-specific metrics out of the box, flexible `GEval` LLM-as-a-judge, optional hosted dashboard (Confident AI) | Metric quality tied to judge-LLM quality; another dependency to install and manage |
| **RAGAS** | RAG-specific metrics | Similar retrieval/generation metric split (faithfulness, context precision/recall); popular in the RAG community | Smaller general-purpose LLM-app tooling than DeepEval; less CI/pytest tooling |
| **TruLens** | LLM app observability + eval | Strong tracing/observability integration, feedback functions, good for monitoring in production | More observability-first than test-first; steeper setup for pure notebook evals |
| **LangSmith** | Tracing, evaluation, and datasets for LangChain/LangGraph apps | Deep native integration if you're already in the LangChain ecosystem, dataset management, human review UI | Tied closely to LangChain/LangGraph; less framework-agnostic |
| **OpenAI Evals** | General LLM evaluation | Backed by OpenAI, good for simple correctness-style evals | Less RAG-specific tooling (no built-in contextual precision/recall style metrics) |

DeepEval's main differentiators for this notebook's use case are (1) purpose-built RAG metrics that split cleanly along the retrieval/generation boundary, and (2) a `pytest`-native workflow that lets the exact same test cases gate CI, not just run once in a notebook.

## 📌 Real-World Use Cases

- **Pre-merge CI gating** — run `deepeval test run` on a held-out eval set for every pull request that touches prompts, retrieval, or reranking, and block merges on faithfulness/relevancy regressions.
- **Model or embedder swaps** — quantify whether switching `bge-small-en-v1.5` for a different embedding model, or `gpt-4o-mini` for another LLM, actually improves answers before rolling it out.
- **Retrieval tuning** — use Contextual Precision/Recall/Relevancy trends to decide whether to widen `top_k`, add reranking, or adjust the hybrid fusion strategy.
- **Product-specific quality bars** — encode custom `GEval` criteria for things generic metrics miss: tone, jargon level for a target audience, required disclaimers, or "must say when context is insufficient."
- **Vendor/framework evaluation** — compare RAG configurations (dense-only vs. hybrid, with/without reranking) on the same eval set to make an evidence-based architecture decision instead of a hunch-based one.
- **Ongoing quality monitoring** — track average metric scores over time per model/prompt/retrieval version to catch slow quality drift (e.g., 0.91 → 0.78 average) that a single pass/fail run wouldn't flag.

---

*Companion to `LangGraph_Qdrant_Hybrid_Rerank.ipynb`, which builds the full hybrid retrieval + reranking pipeline this notebook evaluates.*
