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
3. Retrieval     →   Submit edited documents to Vertex AI Search, record retrieval ranks
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
| `3_retrieval_stage/` | Submits documents to Vertex AI Search, records retrieval ranks |
| `4_generation_parallel_A–D_v4.ipynb` | Generates LLM responses with inline citations in parallel |
| `5_evaluation_main_v4.ipynb` | Main analysis: H2 (N = 402 complete queries) |
| `5_evaluation_sensitivity_v2.ipynb` | Sensitivity analysis without completeness filter (N = 500) |

---

## GEO Methods

10 single edits (from Puerto et al., 2025) + 11 combined edits (this work).

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

### 3. Vertex AI (Retrieval Stage)

The retrieval stage requires a Google Cloud project with Vertex AI Search configured. See the retrieval stage notebooks for setup details.

---

## Data

The dataset is the retail split of **C-SEO Bench** (Puerto et al., 2025), available on HuggingFace:  
[`parameterlab/c-seo-bench`](https://huggingface.co/datasets/parameterlab/c-seo-bench)

500 queries × 10 product descriptions. Target document indices are provided by Puerto et al. via the [C-SEO Bench GitHub repository](https://github.com/parameterlab/c-seo-bench).

---

## Reference

Puerto, S. et al. (2025). *C-SEO Bench: A Benchmark for Competitive Search Engine Optimization.*
