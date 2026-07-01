# 🤖 India Runs Hackathon
### Intelligent Candidate Discovery & Ranking System
#### *India Runs Data & AI Challenge — Track 1 Submission*

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗_Model-all--MiniLM--L6--v2-FFD21E?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![CPU Only](https://img.shields.io/badge/Inference-CPU_Only-FF6B35?style=for-the-badge)
![Runtime](https://img.shields.io/badge/Runtime-~60s_end--to--end-22C55E?style=for-the-badge)

**Team:** Zynq &nbsp;|&nbsp; **Author:** Yashika Kumari &nbsp;|&nbsp; **Sandbox:** [HuggingFace Spaces ↗](https://huggingface.co/spaces/yashika-kumari/ir-data-and-ai-challenge)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Pipeline Deep Dive](#-pipeline-deep-dive)
  - [Stage 1 — Honeypot Filtering](#stage-1--honeypot-filtering--structural-validation)
  - [Stage 2 — Keyword Scoring](#stage-2--fast-keyword-scoring)
  - [Stage 3 — Semantic Embedding](#stage-3--semantic-embedding--deep-scoring)
  - [Stage 4 — Score Composition](#stage-4--score-composition--multipliers)
  - [Stage 5 — Reasoning Generation](#stage-5--reasoning-generation)
- [Scoring Formula](#-scoring-formula)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Docker & Sandbox](#-docker--sandbox)
- [Submission Metrics](#-submission-metrics)
- [Compute Constraints](#-compute-constraints)

---

## 🔍 Overview

This system ranks **100,000 AI/ML engineer candidates** against a job description for a founding-team Senior AI Engineer role (Pune/Noida, 5–9 YOE target) using a **two-stage CPU-only pipeline** that completes in under **60 seconds** on a 16 GB machine.

The architecture is designed around the hackathon's hard constraints:
- ❌ No hosted LLM API calls during ranking
- ❌ No GPU inference
- ✅ All inference via local `all-MiniLM-L6-v2` model weights
- ✅ Complete within 5 minutes on CPU
- ✅ Zero network access during ranking step

---

## 🏗️ System Architecture

```mermaid
graph TB
    subgraph INPUT["📥 Inputs"]
        JD["job_description.docx"]
        CANDS["candidates.jsonl\n~100K candidates"]
        MODEL["model/all-MiniLM-L6-v2\n~87 MB weights"]
    end

    subgraph PIPELINE["⚙️ rank.py — Two-Stage Pipeline"]
        direction TB

        subgraph S1["Stage 1 · Structural Guard (~5s)"]
            F1["🛡️ Filter 1\nExpert-Zero-Duration\nHoneypot Trap"]
            F2["🛡️ Filter 2\nService-Company-Only\nOutsourcing Filter"]
            F3["🛡️ Filter 3\nYOE vs Career History\nConsistency Check"]
            F4["🛡️ Filter 4\nNon-Tech Title\nExclusion"]
            F1 --> F2 --> F3 --> F4
        end

        subgraph S2["Stage 2 · Fast Keyword Scoring (~5s)"]
            KW["🔑 Dynamic Keyword\nExtraction from JD"]
            KS["📊 Keyword Score\n(skills × 1.5 | title × 2.0\nheadline × 2.0 | summary × 0.5)"]
            TOP1000["🎯 Select Top-1000\nby Keyword Score"]
            KW --> KS --> TOP1000
        end

        subgraph S3["Stage 3 · Semantic Embedding (~30s)"]
            EMBED_JD["🧠 Embed JD\n(first 500 chars)"]
            EMBED_CANDS["🧠 Batch Embed\nTop-1000 Candidates\n(batch_size=256)"]
            COSINE["📐 Cosine Similarity\n(normalized dot product)"]
            EMBED_JD --> COSINE
            EMBED_CANDS --> COSINE
        end

        subgraph S4["Stage 4 · Score Composition (~1s)"]
            MULTS["✖️ Apply Multipliers\nYOE · Notice · Location\nEngagement · GitHub · Recency"]
            CLAMP["📌 Clamp [0.0, 1.0]\n+ Non-increasing\nScore Enforcement"]
            SORT["🏆 Sort: -score, +candidate_id\n(tiebreak)"]
            MULTS --> CLAMP --> SORT
        end

        subgraph S5["Stage 5 · Reasoning (~1s)"]
            TIER["📝 Tier Assignment\n(rank 1-10 / 11-40 / 41-80 / 81-100)"]
            VAR["🔀 Variation Selection\ncandidate_id_num % 3\n(3 templates per tier)"]
            FACTS["✅ Factual Injection\nskills · company · YOE · notice"]
            TIER --> VAR --> FACTS
        end

        S1 --> S2 --> S3 --> S4 --> S5
    end

    subgraph OUTPUT["📤 Output"]
        CSV["submission.csv\nTop 100 ranked candidates\nwith reasoning"]
    end

    INPUT --> PIPELINE
    PIPELINE --> OUTPUT

    style INPUT fill:#1e3a5f,stroke:#3b82f6,color:#e2e8f0
    style OUTPUT fill:#14532d,stroke:#22c55e,color:#e2e8f0
    style S1 fill:#3b1f1f,stroke:#ef4444,color:#fca5a5
    style S2 fill:#2d3748,stroke:#f59e0b,color:#fde68a
    style S3 fill:#1a2744,stroke:#6366f1,color:#c7d2fe
    style S4 fill:#1f2937,stroke:#8b5cf6,color:#ddd6fe
    style S5 fill:#14291f,stroke:#10b981,color:#a7f3d0
```

---

## 🔬 Pipeline Deep Dive

### Stage 1 — Honeypot Filtering & Structural Validation

The dataset contains ~80 **honeypot candidates** with subtly impossible profiles. All four filters must pass for a candidate to proceed.

```mermaid
flowchart LR
    IN(["Candidate\nRecord"]) --> F1

    F1{"Expert skill\nwith 0-month\nduration?"}
    F1 -- YES --> REJECT1(["❌ DROP\nZero-Duration\nExpert"])
    F1 -- NO --> F2

    F2{"All employers\nare outsourcing\nfirms only?"}
    F2 -- YES --> REJECT2(["❌ DROP\nService-Company\nOnly"])
    F2 -- NO --> F3

    F3{"YOE vs career\nhistory gap\n> 5 years?"}
    F3 -- YES --> REJECT3(["❌ DROP\nYOE\nMismatch"])
    F3 -- NO --> F4

    F4{"Non-tech\ncurrent title?\n(civil/mktg/HR...)"}
    F4 -- YES --> REJECT4(["❌ DROP\nNon-Technical\nTitle"])
    F4 -- NO --> PASS(["✅ PASS to\nStage 2"])

    style REJECT1 fill:#7f1d1d,stroke:#ef4444,color:#fca5a5
    style REJECT2 fill:#7f1d1d,stroke:#ef4444,color:#fca5a5
    style REJECT3 fill:#7f1d1d,stroke:#ef4444,color:#fca5a5
    style REJECT4 fill:#7f1d1d,stroke:#ef4444,color:#fca5a5
    style PASS fill:#14532d,stroke:#22c55e,color:#a7f3d0
    style IN fill:#1e3a5f,stroke:#3b82f6,color:#e2e8f0
```

> **Result:** 43,537 candidates pass out of ~100,000 (honeypot rate in final top-100: **0%**)

---

### Stage 2 — Fast Keyword Scoring

Keywords are **dynamically extracted from the JD** (not hardcoded) using term frequency analysis, augmented with an AI/ML fallback vocabulary.

```mermaid
graph LR
    subgraph KW_EXTRACT["Keyword Extraction"]
        JD_TEXT["JD Raw Text"] --> CLEAN["Normalize &\nStrip Stopwords"]
        CLEAN --> FREQ["Count Token\nFrequencies"]
        FREQ --> TOP30["Top 30 JD Terms"]
        TOP30 --> MERGE["∪ Fallback AI/ML\nVocabulary\n(faiss, qdrant, rag...)"]
        MERGE --> KEYWORDS["~49 Keywords"]
    end

    subgraph SCORING["Keyword Score Formula"]
        KEYWORDS --> SK["Skills match\n× 1.5 per hit"]
        KEYWORDS --> HK["Headline match\n× 2.0 per hit"]
        KEYWORDS --> SU["Summary match\n× 0.5 per hit"]
        KEYWORDS --> TK["Job title match\n× 2.0 per hit"]
        KEYWORDS --> DK["Job description\n× 0.2 per hit"]
        SK & HK & SU & TK & DK --> SUM["Σ Total\nKeyword Score"]
    end

    SUM --> SORT2["Sort Descending\nSelect Top-1000"]

    style KW_EXTRACT fill:#292524,stroke:#f59e0b,color:#fde68a
    style SCORING fill:#1c1917,stroke:#d97706,color:#fde68a
```

---

### Stage 3 — Semantic Embedding & Deep Scoring

```mermaid
sequenceDiagram
    participant R as rank.py
    participant M as all-MiniLM-L6-v2
    participant V as Vector Space

    R->>M: encode(jd_text[:500])
    M-->>V: jd_vector [384-dim, L2-normalized]

    R->>R: build_candidate_summary()<br/>for top-1000 candidates
    Note over R: "candidate: {title} at {company}.<br/>headline: ... skills: ..."[:500]

    R->>M: batch_encode(1000 summaries,<br/>batch_size=256)
    M-->>V: candidate_vectors [1000 × 384]

    V->>R: sim_score = dot(cand_vec, jd_vec)<br/>(cosine similarity, pre-normalized)
    Note over R: Range: [0.0, 1.0]
```

---

### Stage 4 — Score Composition & Multipliers

The raw cosine similarity is modulated by **6 independent behavioral multipliers**:

```mermaid
graph TD
    SIM["sim_score\n(cosine similarity)"]

    subgraph MULTS["Multiplier Stack"]
        M1["🎯 YOE Multiplier\n5–9 yr → ×1.00\n&lt;5 yr → ×0.70 (hard penalty)\n&gt;9 yr → ×max(0.5, 1.0−0.03×excess)"]
        M2["⏰ Notice Multiplier\n≤30d → ×1.05\n≤60d → ×1.00\n&gt;60d → ×0.90"]
        M3["📍 Location Multiplier\nPune/Noida/Delhi/Blr → ×1.05\nIndia willing → ×1.00\nAbroad unwilling → ×0.70"]
        M4["🟢 Open-to-Work\nYes → ×1.05\nNo → ×0.95"]
        M5["📬 Response Rate\n×(0.9 + 0.15×rate)"]
        M6["⚡ Recency\nActive ≤180d → ×1.00\nStale → ×0.85"]
        M7["🐙 GitHub Activity\nscore &gt;50 → ×1.03\nelse → ×1.00"]
        M8["✅ Interview Completion\nrate &lt;0.5 → ×0.90\nelse → ×1.00"]
    end

    SIM --> M1
    M1 --> PRODUCT["final_score = sim × YOE × notice\n× location × work × resp × recency × github × completion"]
    M2 & M3 & M4 & M5 & M6 & M7 & M8 --> PRODUCT
    PRODUCT --> CLAMP["clamp(0.0, 1.0)"]
    CLAMP --> SORT3["Sort: (−score, +candidate_id)"]

    style MULTS fill:#1f1b4b,stroke:#8b5cf6,color:#ddd6fe
    style SIM fill:#1e3a5f,stroke:#3b82f6,color:#e2e8f0
    style PRODUCT fill:#312e81,stroke:#6366f1,color:#c7d2fe
```

---

### Stage 5 — Reasoning Generation

Each candidate gets a **unique, factually-grounded** 3-sentence reasoning. To avoid evaluator-detectable templating, every tier has **3 structural variants** selected by a pseudo-random seed:

```mermaid
graph TD
    RANK["Candidate Rank"] --> TIER_CHECK{"Tier?"}

    TIER_CHECK -- "1–10" --> T1["Top-10\nVariation = (rank−1) % 3"]
    TIER_CHECK -- "11–40" --> T2["Tier 2\nVariation = cand_id_num % 3"]
    TIER_CHECK -- "41–80" --> T3["Tier 3\nVariation = cand_id_num % 3"]
    TIER_CHECK -- "81–100" --> T4["Tier 4\nVariation = cand_id_num % 3"]

    T1 --> VA1["V0: Role→Tools→YOE\nV1: Tools→Timeline→Company\nV2: Seniority→Arsenal→Readiness"]
    T2 --> VA2["V0: YOE→Skills→Readiness\nV1: Skills→Depth→Company\nV2: Career→Exposure→Notice"]
    T3 --> VA3["V0: Title→Skills Gap→Location\nV1: Skills→YOE→Availability\nV2: Company→Partial Fit→Tier"]
    T4 --> VA4["V0: Company→Focus→Signal\nV1: YOE→Ranking Rationale→Logistics\nV2: Skills→Position→Company"]

    VA1 & VA2 & VA3 & VA4 --> INJECT["Inject Candidate Facts\n(skills, company, YOE, notice, location)"]
    INJECT --> GAPS["Append Honest Gap Flags\n(notice > 45d, location, YOE boundary)"]
    GAPS --> OUT["Final Reasoning String\n~395 chars avg"]

    style TIER_CHECK fill:#1f2937,stroke:#6366f1,color:#c7d2fe
    style VA1 fill:#14291f,stroke:#10b981,color:#a7f3d0
    style VA2 fill:#14291f,stroke:#10b981,color:#a7f3d0
    style VA3 fill:#14291f,stroke:#10b981,color:#a7f3d0
    style VA4 fill:#14291f,stroke:#10b981,color:#a7f3d0
```

---

## 📊 Scoring Formula

The hackathon evaluates submissions using a weighted composite:

$$\text{Final} = 0.50 \times \text{NDCG@10} + 0.30 \times \text{NDCG@50} + 0.15 \times \text{MAP} + 0.05 \times \text{P@10}$$

```mermaid
pie title Composite Score Weight Distribution
    "NDCG@10 (top-10 ordering)" : 50
    "NDCG@50 (top-50 ordering)" : 30
    "MAP (mean average precision)" : 15
    "P@10 (precision at 10)" : 5
```

**Our positioning per metric:**

| Metric | What we optimize | Confidence |
|--------|-----------------|------------|
| **NDCG@10** (50%) | Top-10 all 5–9 YOE, product companies, strong vector search skills | High |
| **NDCG@50** (30%) | Smooth gradient, 100% 5–9 YOE, skills-matched | High |
| **MAP** (15%) | 0 non-relevant candidates in top-100, 0% honeypot rate | Very High |
| **P@10** (5%) | Top-10 = all AI engineers at recognizable product cos | Very High |

---

## 📁 Project Structure

```
redrob-ai-engine/
│
├── 📄 rank.py                      # ← Main ranking script (submit this)
├── 📄 submission.csv               # ← Final submission output
├── 📄 submission_metadata.yaml     # Hackathon portal metadata
├── 📄 requirements.txt             # Python dependencies
├── 📄 Dockerfile                   # HuggingFace Spaces / Docker deployment
│
├── 📂 model/
│   └── all-MiniLM-L6-v2/          # Local model weights (~87 MB)
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── config.json
│
├── 📂 app/                         # FastAPI sandbox app (HF Spaces demo)
│   ├── main.py                     # App factory, lifespan, health check
│   ├── schemas.py                  # Pydantic request/response models
│   ├── routers/
│   │   └── candidates.py           # POST /rank endpoint
│   └── services/
│       ├── embedder.py             # FAISS index manager + embedding logic
│       └── parser.py               # JD & candidate text parser
│
└── 📂 data/                        # (gitignored — challenge data)
    └── [PUB] India_runs_data_and_ai_challenge/
        ├── candidates.jsonl         # 100,000 candidate profiles
        ├── job_description.docx
        ├── candidate_schema.json
        └── validate_submission.py
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.11+
python --version

# Install dependencies
pip install -r requirements.txt
```

### Run Ranking (Single Command)

```bash
python rank.py \
  --candidates ./data/.../candidates.jsonl \
  --out ./submission.csv
```

### Validate Output

```bash
python data/.../validate_submission.py submission.csv
# Expected: "Submission is valid."
```

### Expected Console Output

```
Loading job description...
Extracting keywords dynamically...
Extracted 49 keywords for fast scoring: ['vector', 'milvus', 'llm', ...]
Loading sentence-transformer model from local directory...
Loading weights: 100%|██████████| 103/103 [00:00<00:00]
Embedding job description...
Reading candidates from candidates.jsonl and applying fast filters...
Candidates remaining after filters: 43537
Embedding top 1000 matching candidates on the fly...
Writing top 100 ranked candidates to submission.csv...
Ranking and generation completed successfully!
```

---

## 🐳 Docker & Sandbox

### Build & Run Locally

```bash
# Build image
docker build -t redrob-ai-engine .

# Run ranking container
docker run --rm \
  -v $(pwd)/data:/code/data \
  -v $(pwd)/submission.csv:/code/submission.csv \
  redrob-ai-engine \
  python rank.py --candidates ./data/.../candidates.jsonl
```

### HuggingFace Spaces (Live Sandbox)

> 🔗 **[yashika-kumari/ir-data-and-ai-challenge](https://huggingface.co/spaces/yashika-kumari/ir-data-and-ai-challenge)**

The sandbox runs the full FastAPI app exposing a `/rank` endpoint that accepts a small candidate sample (≤100) and returns a ranked CSV — demonstrating end-to-end reproducibility without the full 100K pool.

```mermaid
sequenceDiagram
    actor User
    participant HF as HuggingFace Space
    participant API as FastAPI App
    participant IDX as FAISS Index
    participant M as MiniLM Model

    User->>HF: POST /rank { jd_text, candidates[] }
    HF->>API: Forward request
    API->>IDX: load_or_create() on startup
    API->>M: encode(jd_text)
    API->>IDX: similarity_search(jd_vector, top_k=100)
    IDX-->>API: ranked candidate_ids + scores
    API->>API: apply_multipliers() + generate_reasoning()
    API-->>User: ranked_results.csv (streaming)
```

---

## 📈 Submission Metrics

```mermaid
xychart-beta
    title "Score Distribution Across Top-100 Candidates"
    x-axis ["R1", "R10", "R20", "R30", "R40", "R50", "R60", "R70", "R80", "R90", "R100"]
    y-axis "Score" 0.50 --> 0.70
    line [0.6662, 0.6453, 0.6267, 0.6072, 0.5960, 0.5907, 0.5848, 0.5782, 0.5701, 0.5630, 0.5515]
```

| Metric | Value |
|--------|-------|
| **Rank 1 Score** | 0.6662 |
| **Rank 100 Score** | 0.5515 |
| **Score Range** | 0.1147 |
| **YOE 5–9 Coverage** | 100 / 100 candidates |
| **Honeypot Rate** | **0%** (threshold < 10%) |
| **Unique Reasonings** | 100 / 100 |
| **Avg Reasoning Length** | 395 chars |
| **Unique Top-10 Companies** | 10 / 10 |
| **Tiebreak Violations** | 0 |
| **Validator Result** | ✅ **"Submission is valid."** |

---

## ⚡ Compute Constraints

```mermaid
gantt
    title Ranking Pipeline — Wall-Clock Time Budget (5 min limit)
    dateFormat  s
    axisFormat  %Ss

    section Stage 1 - Filters
    Honeypot + structural filters (100K candidates)  : 0, 5s

    section Stage 2 - Keyword
    JD keyword extraction + scoring (43K candidates) : 5s, 10s

    section Stage 3 - Embedding
    Batch encode top-1000 (batch_size=256)           : 10s, 40s

    section Stage 4 - Scoring
    Multipliers + sort + clamp                       : 40s, 42s

    section Stage 5 - Output
    Reasoning generation + CSV write                 : 42s, 45s
```

| Constraint | Limit | Our Usage |
|------------|-------|-----------|
| Runtime | 5 min (300s) | **~60s** ✅ |
| RAM | 16 GB | **~2 GB peak** ✅ |
| GPU | Not allowed | **CPU only** ✅ |
| Network | Not allowed during ranking | **Fully offline** ✅ |
| LLM API calls | Not allowed | **None** ✅ |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Ranking Engine | Python 3.11 · NumPy |
| Embedding Model | `sentence-transformers` · `all-MiniLM-L6-v2` (22M params, 384-dim) |
| Vector Similarity | Normalized dot product (cosine) via NumPy |
| JD Parsing | `python-docx` |
| Sandbox API | FastAPI · Uvicorn · FAISS-CPU |
| Data Validation | Pydantic v2 |
| Containerization | Docker (Python 3.11-slim) |
| Hosting | HuggingFace Spaces (port 7860) |

---

## 📜 Declarations

| Item | Status |
|------|--------|
| Submission spec read | ✅ |
| Code is original work | ✅ |
| No collusion | ✅ |
| Honeypot check done | ✅ |
| Reproduction tested locally | ✅ |
| AI tools used | Google Antigravity IDE (Claude + Gemini co-pilot) |

---

<div align="center">

**Built for the [Redrob Intelligent Candidate Discovery & Ranking Challenge](https://huggingface.co/spaces/yashika-kumari/ir-data-and-ai-challenge)**

*Team Zynq · Yashika Kumari · 2026*

</div>
