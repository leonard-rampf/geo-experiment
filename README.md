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
│   │   ├── 2_manipulation_selected_docs_A_v1.json         # Manipulation output — subset A
│   │   ├── 2_manipulation_selected_docs_B_v1.json         # Manipulation output — subset B
│   │   ├── 2_manipulation_selected_docs_C_v1.json         # Manipulation output — subset C
│   │   ├── 2_manipulation_selected_docs_D_v1.json         # Manipulation output — subset D
│   │   └── 2_manipulation_selected_docs_v1.json           # Merged manipulation output
│   ├── retrieval/
│   │   ├── 20260413_vertex_geoedits_v1.jsonl              # JSONL corpus uploaded to Agent Search
│   │   ├── 20260413_retrieval_results_geo_v1.csv          # Raw retrieval results (all runs appended)
│   │   ├── 20260413_retrieval_results_geo_v2.csv          # Clean retrieval results (deduplicated)
│   │   ├── 20260413_retrieval_alert_log_geo_v1.csv        # Alert log — Run 1
│   │   ├── 20260413_retrieval_alert_log_geo_v2.csv        # Alert log — Run 2
│   │   ├── 20260413_retrieval_alert_log_geo_v3.csv        # Alert log — Run 3
│   │   └── 20260413_retrieval_alert_investigation_results_v1.csv  # Empty-query verification results
│   └── generation/
│       ├── 4_generation_results_A_v4.parquet              # Per-notebook generation results
│       ├── 4_generation_results_B_v4.parquet
│       ├── 4_generation_results_C_v4.parquet
│       ├── 4_generation_results_D_v4.parquet
│       ├── 4_generation_results_v4.parquet                # Merged generation results
│       └── 4_generation_runs_without_target_v3.json       # Generation runs excluding target document
├── notebook/
│   ├── 1_preparation/
│   │   ├── 0_power_calculation.ipynb                      # A priori sample size calculation
│   │   └── 1_preparation_data.ipynb                       # C-SEO Bench extraction & competitor docs
│   ├── 2_manipulation/
│   │   ├── 2_manipulation_combined_edits_A.ipynb          # GEO edits — subset A (parallel)
│   │   ├── 2_manipulation_combined_edits_B.ipynb          # GEO edits — subset B (parallel)
│   │   ├── 2_manipulation_combined_edits_C.ipynb          # GEO edits — subset C (parallel)
│   │   ├── 2_manipulation_combined_edits_D.ipynb          # GEO edits — subset D (parallel)
│   │   └── 3_transform_to_jsonl.ipynb                     # Transform documents to Agent Search JSONL
│   ├── 3_retrieval_stage/
│   │   └── 4_retrieval_stage_geo.ipynb                    # Retrieval experiment & fault-tolerant pipeline
│   ├── 4_generation_stage/
│   │   ├── 4_generation_parallel_A_v4.ipynb               # LLM response generation — subset A
│   │   ├── 4_generation_parallel_B_v4.ipynb               # LLM response generation — subset B
│   │   ├── 4_generation_parallel_C_v4.ipynb               # LLM response generation — subset C
│   │   └── 4_generation_parallel_D_v4.ipynb               # LLM response generation — subset D
│   └── 5_evaluation/
│       ├── 5_retrieval_analysis.ipynb                     # H1 analysis: retrieval rank differences
│       ├── 5_evaluation_main_v4.ipynb                     # H2 main analysis 
│       └── 5_evaluation_sensitivity_v2.ipynb              # H2 sensitivity analysis 
├── src/
│   ├── llms/
│   │   ├── __init__.py
│   │   ├── llm_interface.py                               # Abstract LLM interface
│   │   └── openai.py                                      # OpenAI API wrapper
│   └── methods/
│       ├── __init__.py
│       ├── citation_boosting.py                           # Citation-based GEO methods
│       └── optimization_methods.py                        # Remaining GEO method implementations
├── config.json                                            
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
