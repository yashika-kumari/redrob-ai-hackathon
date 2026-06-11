# Redrob Hackathon AI Engine - Master System State & Agent Directives

## 🎯 1. Core Vision & Hackathon Compliance
- **Target Event:** Redrob AI Hackathon ("INDIA RUNS").
- **Core Track:** Track 1 (Data & AI Challenge) - Intelligent Candidate Discovery.
- **Objective:** Build an "AI Brain" for candidate discovery using dense vector embeddings and parallel Cosine Similarity matching. Simple, raw keyword-matching filters are strictly forbidden.
- **Submission Requirements:** 1) Well-organized GitHub repository, 2) System Architecture README Blueprint, 3) Predefined ranked JSON/CSV output file.
- **Cost Constraint:** 100% free open-source software ecosystem. Zero external paid APIs, zero cloud compute, zero cloud vector databases.

## 🛠️ 2. Immutable Technology Stack
- **API Engine:** FastAPI (Python 3.11+, strictly asynchronous using async/await patterns).
- **AI Core:** Localized inference via `sentence-transformers` (`all-MiniLM-L6-v2`).
- **Vector Space Database:** In-memory local `FAISS` (Facebook AI Similarity Search) index.
- **Document Parsers:** `PyPDF2` and `python-docx` for structured local file ingestion.
- **Frontend Layer:** Single-Page Interface using HTML5, Vanilla JavaScript, and Tailwind CSS (inspired by minimalist design layouts from Landbook).

## 🛡️ 3. Non-SQL Vector Vulnerability Mitigation Matrix
Since this system lacks a standard SQL database, standard SQL injection is irrelevant. The Agent must explicitly code defenses against these active AI pipeline exploits:
1. **Path Traversal Attacks:** Sanitize all incoming file names. Strip relative components (`../`) or absolute system root elements to prevent malicious document writes over configuration paths.
2. **Decompression/Parsing Bombs:** Enforce strict middleware and payload validation. Reject file streams larger than 5MB *before* loading file arrays into CPU memory to completely avoid Denial of Service (DoS) conditions.
3. **Indirect Prompt Injection:** Isolate text parsed from multi-format resumes. Ensure candidate skills are strictly run against the local structural JSON mapper schema to strip malicious string commands embedded within documents.

## 📈 4. Strict Granular Git Commit & Workspace Protocol
The Agent is strictly prohibited from batching code changes or performing single-dump repository commits. For every phase, the Agent MUST follow this micro-commit loop:


## 📈 4. Strict Granular Git Commit & Workspace Protocol
The Agent is strictly prohibited from batching code changes or performing single-dump repository commits. For every phase, the Agent MUST follow this micro-commit loop:

```text
[Write/Modify Single File] 
        │
        ▼
[Execute Local Validation Check (Syntax/Linter)]
        │
        ▼
[Stash Changes Separately via Terminal] -> git add <target_file>
        │
        ▼
[Execute Micro-Commit with Semantic Tag] -> git commit -m "feat/fix: <message>"
```

### Required Git History Lifecycle Mapping:
- **Phase 1 Commit 1:** `chore: setup project directory structures and establish local git ignores`
- **Phase 1 Commit 2:** `chore: instantiate baseline open-source software dependencies in requirements.txt`
- **Phase 1 Commit 3:** `feat: implement async FastAPI core application instance and global exception middleware`
- **Phase 1 Commit 4:** `feat: establish structural Pydantic data schemas for validation contracts`
- **Phase 2 Commit 1:** `feat: create secure, path-traversal insulated multi-format resume ingestion parser`

## 🚀 5. Current Task State: Phase 1 Baseline Activation
The project is currently sitting at a completely clean initialized directory level. The Agent's current direct task is to construct the files for **Phase 1** natively, running the Git micro-commit loops exactly as described in Section 4.