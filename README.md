# From Retrieval to Citation: The Effectiveness of Generative Engine Optimization Beyond the Generation Stage

**Authors:** Leonard Rampf, Niklas Keckeisen  
**Advisors:** Michail Batikas  
**Institution:** Nova School of Business and Economics  
**Submission:** May 2026

---

## Overview

This project empirically tests whether content-based Generative Engine Optimization (GEO) edits improve a document's visibility in AI-generated responses. Using the C-SEO Bench retail dataset (Puerto et al., 2025), we apply 21 GEO methods to 500 product descriptions and measure their effect on two stages of a Generative Search Engine pipeline:

- **H1** — Content-based GEO edits significantly alter a document’s retrieval rank within the retrieval stage of a GSE.
- **H2** — Content-based GEO edits significantly enhance a document’s citation visibility within the LLM-generated response.

---

## Pipeline

```
1. Preparation   →   Extract competitor documents from the C-SEO Bench dataset
2. Manipulation  →   Apply 21 GEO methods to each target document (GPT-4o-mini)
3. Retrieval     →   Submit edited documents to Agent Search, record retrieval ranks
4. Generation    →   Generate LLM responses with inline citations (GPT-4o-mini) and test H1 via Wilcoxon signed-rank test + Sign Test
5. Evaluation    →   Test H2 via Wilcoxon signed-rank test + Sign Test
```

---

## Project Structure

```
geo-experiment/
├── data/retail/
│   ├── dataset/
│   │   ├── selected_docs.json                             # Target document indices (Puerto et al.)
│   │   ├── 1_preparation_competitor_docs_v1.json          # Competitor documents per query
│   │   ├── 2_manipulation_selected_docs_A/B/C/D_v1.json  # Subset files per notebook
│   │   └── 2_manipulation_selected_docs_v1.json           # Merged manipulation output
│   ├── retrieval/
│   │   └── 20260413_retrieval_results_geo_v2.csv          # Retrieval results per query-method
│   └── generation/
│       ├── 4_generation_results_A/B/C/D_v4.parquet        # Per-notebook generation results
│       └── 4_generation_results_v4.parquet                # Merged generation results
├── notebook/
│   ├── 1_preparation/      1_preparation_data.ipynb
│   ├── 2_manipulation/     2_manipulation_combined_edits_A/B/C/D.ipynb
│   ├── 3_retrieval_stage/
│   ├── 4_generation_stage/ 4_generation_parallel_A/B/C/D_v4.ipynb
│   └── 5_evaluation/       5_evaluation_main_v4.ipynb
│                           5_evaluation_sensitivity_v2.ipynb
├── src/
│   ├── llms/               OpenAI API wrapper
│   └── methods/            GEO method implementations
├── config.json             API keys (not committed)
└── requirements.txt
```

---

## Notebooks

| Notebook | Description |
|---|---|
| `1_preparation_data.ipynb` | Downloads C-SEO Bench, extracts competitor documents |
| `2_manipulation_combined_edits_A–D.ipynb` | Applies 11 combined GEO methods in parallel (GPT-4o-mini) |
| `3_retrieval_stage/` | Submits documents to Agent Search, records retrieval ranks |
| `4_generation_parallel_A–D_v4.ipynb` | Generates LLM responses with inline citations in parallel |
| `5_evaluation_main_v4.ipynb` | Main analysis: H2 (N = 402 complete queries) |
| `5_evaluation_sensitivity_v2.ipynb` | Sensitivity analysis without completeness filter (N = 500) |

---

## GEO Methods

10 single edits (from Puerto et al., 2025) + 11 combined edits (this work). Combined edits apply single edits sequentially: F = Fluency, C = Citations, Q = Quotes, S = Statistics.

| Type | Methods |
|---|---|
| Single | Fluency, Citations, Quotes, Statistics, Authoritative, UniqueWords, TechnicalTerms, SimpleLanguage, LLMstxt, ContentImprovement |
| Combined (2-way) | FC, FQ, FS, CQ, CS, QS |
| Combined (3-way) | FCQ, FCS, FQS, CQS |
| Combined (4-way) | FCQS |

### Naming Convention

The thesis uses descriptive names; notebooks and code use shorthand identifiers:

| Thesis Name | Notebook / Code Name |
|---|---|
| Cite Sources | `Citations(doc)` |
| Quotation Addition | `Quotes(doc)` |
| Statistics Addition | `Statistics(doc)` |
| Fluency Optimization | `Fluency(doc)` |
| Simple Language | `SimpleLanguage(doc)` |
| Technical Terms | `TechnicalTerms(doc)` |
| Unique Words | `UniqueWords(doc)` |
| Authoritative Style | `Authoritative(doc)` |
| Content Improvement | `ContentImprovement(doc)` |
| LLM Guidance | `LLMstxt(doc)` |
| CQ | `CQ(doc)` |
| CS | `CS(doc)` |
| QS | `QS(doc)` |
| CQS | `CQS(doc)` |
| FC | `FC(doc)` |
| FQ | `FQ(doc)` |
| FS | `FS(doc)` |
| FCQ | `FCQ(doc)` |
| FCS | `FCS(doc)` |
| FQS | `FQS(doc)` |
| FCQS | `FCQS(doc)` |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Create a `config.json` in the project root:

```json
{
  "OPENAI_API_KEY": "your-openai-api-key"
}
```

### 3. Agent Search (Retrieval Stage)

The retrieval stage uses the Google Cloud Agent Search (`discoveryengine_v1` Python client). In addition to the Python package in `requirements.txt`, the **Google Cloud CLI** must be installed and authenticated:

```bash
# Install: https://cloud.google.com/sdk/docs/install
gcloud auth application-default login
```

**App configuration**
- Project ID: `project-6dc8c84c-e76e-4519-bb0`replace with your own ID if needed 
- Engine ID: `retrieval-stage-geo-v2_1775725808007`replace with your own ID if needed 
- App type: Custom search (general), location: global
- Enterprise edition features, generative responses, and all re-ranking/boost/bury controls disabled
- Serving config: `default_config` with `RANK_BY_EMBEDDING` and Google-managed text embeddings
- Connected to 4 different data stores below

**Data stores** — corpus distributed across four structured data stores (27,500 documents each):
- `geo-experiment-batch1-v2_1775680563433`
- `experiment-geo-batch2-v2_1775681604535`
- `experiment-geo-batch3-v2_1775725626254`
- `experiment-geo-batch4-v2_1775725685864`

**Schema** — three fields per document:
- `text` — Searchable, Retrievable
- `query_id` — Indexable, Retrievable (enables per-unit filter at request time)
- `original_query` — Indexable, Retrievable

**Request parameters**
- `page_size = 50`, `relevance_threshold = LOWEST`
- `filter = f'query_id: ANY("{query_id_filter}")'` — isolates each experimental unit
- No spelling correction, query expansion, or query rewriting applied

**Ingestion**
- Source: JSONL from Google Cloud Storage, one-time synchronisation, global location
- "Exclude from generative AI features" enabled to prevent platform models from altering document text
- Ingestion date: 09/04/2026

---

## Data

The dataset is the retail split of **C-SEO Bench** (Puerto et al., 2025), available on HuggingFace:  
[`parameterlab/c-seo-bench`](https://huggingface.co/datasets/parameterlab/c-seo-bench)

500 queries × 10 product descriptions. Target document indices are provided by Puerto et al. via the [C-SEO Bench GitHub repository](https://github.com/parameterlab/c-seo-bench).

---

## Reference

Puerto, S. et al. (2025). *C-SEO Bench: A Benchmark for Competitive Search Engine Optimization.*
