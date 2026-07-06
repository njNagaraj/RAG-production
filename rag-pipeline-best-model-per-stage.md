# RAG Pipeline — Best Model / Tool per Stage (2026)

> A production reference mapping each RAG stage to its best-in-class model or tool, tuned for an
> internal employee assistant that answers over **company knowledge** *and* pulls **live records**
> from products like Zoho People / Zoho Expense, delivered over WhatsApp.
>
> Landscape note: this space moves monthly. Embedder, vector store, reranker, and generator are
> intentionally **swappable behind interfaces**. Benchmark every pick on *your own data* — public
> leaderboards are a starting point, not a verdict. (And MTEB v2 scores are not comparable to v1.)

---

## The one decision that shapes everything

Your assistant has **two retrieval modes**, and the orchestrator routes between them:

| Question type | Example | Mechanism |
|---|---|---|
| Unstructured knowledge | "How many casual leaves do I get per year?" | Vector search over policy docs (RAG, stages 1–11) |
| Structured per-user record | "How many casual leaves do **I** have left?" | Live, authenticated Zoho API call scoped to that employee |

You can't embed a number that changes daily, and you must never let one employee's vectors
retrieve another's balance. So "best model per stage" below covers the RAG path; the live-data
path is a tool-calling layer (see the Zoho section).

---

## At-a-glance: best pick per stage

| # | Stage | Primary pick | Strong alternatives |
|---|---|---|---|
| 1 | Data ingestion | Source connectors + job queue (no model) | — |
| 2 | Parsing & cleaning | **Docling** (IBM, open, VLM layout) | LlamaParse (managed), Unstructured, Mistral OCR (scans) |
| 3 | Chunking | Structure-aware + **contextual retrieval** | Semantic chunking |
| 4 | Embedding generation | **BGE-M3** (self-host, multilingual, hybrid) | Voyage-3-large, Cohere embed-v4, Gemini Embedding |
| 5 | Vector storage / index | **pgvector on Postgres** | Qdrant, Pinecone (managed) |
| 6 | Query processing | **Claude Haiku 4.5** (contextualize + multi-query) | Gemini Flash, GPT-5 nano |
| 7 | Query embedding | Same model as stage 4 (**BGE-M3**) | — |
| 8 | Retrieval | **Hybrid: dense + BM25, fused with RRF** | Dense-only (not recommended) |
| 9 | Re-ranking | **Cohere Rerank 4** | BGE-reranker-v2-m3 (self-host), Voyage rerank-2.5 |
| 10 | Context augmentation | Prompt assembly + citations (no model) | — |
| 11 | Generation | **Claude Sonnet 4.6** (Haiku 4.5 cheap / Opus 4.8 hard) | GPT-5 series, Gemini 3 Pro, DeepSeek (cost) |
| 12 | Post-processing | **Vectara HHEM / RAGAS** faithfulness check | MiniCheck, LLM-as-judge |
| 13 | Response to user | WhatsApp Cloud API send (no model) | — |

---

## Stage 1 — Data ingestion

**What:** collect documents (PDFs, web pages, DB rows, API payloads, files).
**Model/tool:** no ML model here — this is connectors + a job queue.

- Split sources at the source: **policy/handbook/FAQ docs → RAG index**; **live records (leave
  balances, claim status, payslips) → never indexed, fetched at query time from Zoho.**
- Use an async ingestion worker (Celery / `arq` + Redis). Track `source`, `version`,
  `effective_date` per document so you can re-index on policy changes.

---

## Stage 2 — Document parsing & cleaning

**What:** extract clean text/tables from raw docs; strip boilerplate, HTML, duplicates.
**Parsing quality caps the entire pipeline** — a flattened table becomes disconnected numbers in
your vector store, and the LLM will confidently answer wrong.

| Pick | When |
|---|---|
| **Docling** (IBM, open source, Linux Foundation) | Default. Self-hosted, strong layout + table analysis; ships **Granite-Docling-258M** VLM (Apache 2.0), single-pass parse to structured DocTags/Markdown. Best for data residency. |
| **LlamaParse** (managed) | Cleanest Markdown from visually complex PDFs; vision-model based; API-only (no self-host). |
| **Unstructured** | Most versatile; emits typed elements (Title, Table, NarrativeText) ideal for chunking; 30+ formats. |
| **Mistral OCR / Azure Document Intelligence / AWS Textract** | Scanned/image PDFs needing OCR. |

> Test your **worst tables** (per-diem grids, grade-wise limits) before committing — no parser
> nails every merged-cell table. Keep tables intact as structured rows.

---

## Stage 3 — Chunking

**What:** split documents into retrievable units (~200–1000 tokens).
**Model/tool:** structure-aware splitting + an LLM enrichment step.

- Chunk on **structure** (headings/sections), ~200–500 tokens, ~15% overlap; keep each table +
  its headers as one chunk.
- Add **contextual retrieval** (Anthropic pattern): prepend a one-line, LLM-generated note of
  where each chunk sits in its parent doc *before embedding*. Measurably cuts retrieval failures;
  cheap with prompt caching. Use **Claude Haiku 4.5** for this enrichment.

---

## Stage 4 — Embedding generation

**What:** convert each chunk to a vector.
**Choose by language first** — English + Tamil + transliterated text means prioritize multilingual.

| Pick | When |
|---|---|
| **BGE-M3** (self-host, MIT) | Default. 100+ languages; emits **dense + sparse** vectors → hybrid search from one model; no per-token bill. |
| **Voyage `voyage-3-large`** | Managed, retrieval-optimized, top-tier quality. |
| **Cohere `embed-v4`** | Managed, strong multilingual, Matryoshka dimensions. |
| **Gemini Embedding** | Long context + Matryoshka; good if you're in Google's ecosystem; multimodal. |

> Fine-tuning the embedder on your domain can add +10–30% retrieval quality if generic performance
> is short — start pre-trained, fine-tune only if needed.

---

## Stage 5 — Vector storage / indexing

**What:** store + index embeddings for similarity search.
You're under ~10M vectors (company docs), where the guidance is near-unanimous.

| Pick | When |
|---|---|
| **pgvector on Postgres** | Default. Handles ~50M vectors; reuses the Postgres you already need for chat history, employee↔phone map, encrypted Zoho tokens, audit logs — **one system**, transactional, easy row-level security. |
| **Qdrant** | Best builder balance if you outgrow pgvector: Rust core, strong metadata filtering, cheap self-host. |
| **Pinecone** | Zero-ops managed; pay a premium. |

Use **HNSW** indexing. Store rich metadata per vector for filtering (department, doc version, language).

---

## Stage 6 — User query processing

**What:** rewrite / expand / classify the query before retrieval.
**Model/tool:** a small, fast LLM — **Claude Haiku 4.5** (alts: Gemini Flash, GPT-5 nano). Keep it
sub-100–200ms; run rewriters in parallel.

Three techniques, in priority order for a WhatsApp chat bot:

1. **Contextualization (do this first).** ~60% of conversational follow-ups have unresolved
   references ("and *its* status?"). Rewrite each turn into a standalone query using chat history.
   This is the single biggest fix for multi-turn WhatsApp threads.
2. **Multi-query expansion.** Generate ~3 paraphrases, retrieve against all, merge. Often the
   biggest single recall lift on conversational corpora; lower-risk than HyDE.
3. **HyDE (optional).** Generate a hypothetical answer and embed *that* — strong on technical
   corpora, but the hallucinated draft can skew retrieval, so prefer multi-query unless you've
   measured HyDE winning on your data.
4. **Routing.** The same model also classifies: knowledge question (→ RAG) vs personal-record
   question (→ Zoho tool). This is the orchestrator's core job.

---

## Stage 7 — Query embedding

**What:** embed the (rewritten) query.
**Model/tool:** **the exact same embedder as stage 4 (BGE-M3)** — query and corpus vectors must
share one space, or similarity is meaningless. With multi-query, embed each rewrite.

---

## Stage 8 — Retrieval

**What:** find the most relevant chunks.
**Hybrid is non-negotiable in 2026.** HR/finance queries are full of exact tokens (employee IDs,
claim numbers, grades like "L3") that dense vectors miss, and semantics that BM25 misses.

- Run **dense (BGE-M3) + BM25/sparse in parallel**, fuse with **Reciprocal Rank Fusion (RRF)**.
- Over-fetch: pull **dense top-50 + BM25 top-50** so the reranker has signal to work with.
- pgvector does dense; pair with Postgres `tsvector`/full-text for lexical, or use BGE-M3's sparse output.

---

## Stage 9 — Re-ranking

**What:** reorder candidates with a precise cross-encoder; keep only the best.
**The cheapest accuracy win in the pipeline — typically +15–40% retrieval accuracy**, which
directly cuts hallucinations. Pattern: retrieve ~20–50 → rerank → **send top 5–8** to the LLM.

| Pick | When |
|---|---|
| **Cohere Rerank 4** | Strongest managed default; English + 100+ languages; ~80–150ms; lowest setup cost. |
| **BGE-reranker-v2-m3** (self-host) | Best convenient open-weight; single GPU; pairs naturally with BGE-M3. |
| **Voyage rerank-2.5** | Best quality/latency balance; strong code/finance/legal variants. |

> Some newer rerankers (e.g. Zerank-2) top neutral leaderboards but ship **CC-BY-NC**
> (non-commercial) weights — they need a paid license for your use.

---

## Stage 10 — Context augmentation

**What:** assemble retrieved chunks + the question into the LLM prompt.
**Model/tool:** no model — disciplined prompt engineering.

- Order top chunks by rerank score; include **per-chunk metadata** (`source_doc`, `effective_date`,
  `url`) so the generator can cite and so superseded policy versions are caught.
- Instruct the model: answer **only** from provided context (or tool results); say "I don't have
  that in our policies — I'll flag it to HR" rather than guess.
- Add **semantic caching**: if a new query is >0.95 similar to a cached one, return the cached
  answer — cuts LLM calls 30–50% on a high-volume employee bot.

---

## Stage 11 — Generation

**What:** the LLM writes the grounded answer.
For RAG, three capabilities matter: **faithfulness** (grounds in context, doesn't invent),
**instruction-following** (formats + cites as told), and **tool-use reliability** (drives Zoho calls).

| Pick | When |
|---|---|
| **Claude Sonnet 4.6** | **Default answer model.** Leads grounded-RAG faithfulness (~1.9% hallucination on retrieved context) and "I don't know" calibration (~91%) — the metric that matters most for an HR/finance bot. ~95% tool-use reliability. |
| **Claude Haiku 4.5** | High-volume / simple lookups + the orchestrator (routing, query rewrite, contextual-retrieval enrichment). Fast and cheap. |
| **Claude Opus 4.8** | Hard multi-step reasoning on demand (e.g. "compare my Q2 travel claims against grade policy, flag rejections"). Invoked only when the router flags complexity. |
| GPT-5 series | Strongest function-calling reliability (~97%) — viable if your tool layer is large/agentic. |
| Gemini 3 Pro | Multi-million-token context + multimodal; lets you skip RAG for small knowledge bases or do long-doc analysis. |
| DeepSeek (open) | 85–90% cheaper; fine for internal-only, risk-tolerant use. |

**Tiered routing** (Haiku → Sonnet → Opus by complexity, with a confidence escalation check)
typically cuts cost ~50–60% vs running the top model on everything.

> Counter-intuitive but important: **turn reasoning/thinking mode OFF for source-faithful
> generation.** Reasoning models add inferences beyond the source and *raise* hallucination on
> grounded tasks. Enable reasoning only for genuine multi-hop analysis (the Opus path).

---

## Stage 12 — Post-processing

**What:** format, cite, validate facts, apply guardrails.
**Model/tool:** a faithfulness checker on high-stakes answers (anything about money/entitlements).

| Pick | When |
|---|---|
| **Vectara HHEM** / **Open-RAG-Eval** | Hallucination/groundedness scoring; HHEM is purpose-built and fast. |
| **RAGAS** | Faithfulness, answer-relevancy, context-precision/recall metrics. Production targets: Faithfulness >0.9, Answer Relevancy >0.85, Context Precision >0.8. |
| **MiniCheck** | Lightweight grounding fact-check against the source chunk. |
| **LLM-as-judge** (Claude Opus / GPT-5) | Custom rubric scoring, offline eval. |

Also here: enforce **numbers come from tool calls or table chunks, never free-text**; attach
citations; run output guardrails (PII, scope — keep the bot task-specific, see WhatsApp note).

---

## Stage 13 — Response to user

**What:** deliver the answer over WhatsApp.
**Model/tool:** no model — **Meta WhatsApp Cloud API** send endpoint.

- **Async pattern (mandatory):** webhook receives → enqueue job → respond **HTTP 200 immediately**
  → process → send reply. >~20s and Meta retries, causing duplicate messages.
- **24-hour rule:** free-form replies only within 24h of the employee's last message; outside that,
  use **pre-approved templates**.
- **Auth:** permanent token via a Business Manager **System User** (default token expires in 24h);
  verify inbound `X-Hub-Signature-256`.
- **Policy:** since **Jan 15 2026**, WhatsApp bans *general-purpose* AI bots — only **task-specific**
  bots (HR/expense lookups qualify). Keep the assistant scoped to company info; don't let it act as
  open ChatGPT, or you risk the number.

---

## The live-data path: Zoho tool layer (not a RAG stage, but where the value is)

Per-user questions bypass stages 4–11 and hit Zoho directly via tools the orchestrator calls:
`get_leave_balance(employee_id)`, `get_expense_claim_status(employee_id, claim_id)`,
`list_pending_approvals(manager_id)`, `get_payslip(employee_id, month)`, etc.

Two hard requirements:

1. **Identity binding (security boundary).** On webhook receipt, resolve the WhatsApp number to a
   **verified** Zoho employee record (one-time opt-in), then inject `employee_id` **server-side**
   into every tool call. The LLM never chooses *whose* data to fetch. This stops employee A reading
   employee B's salary. Treat it as a security control, not a convenience.
2. **Zoho OAuth.** OAuth 2.0 + refresh tokens, region-specific DC (likely `.in`). Store tokens
   encrypted, handle refresh, rate-limit. Pull the current module docs (People vs Expense) when
   wiring so field names and scopes are current.

---

## Recommended end-to-end stack

| Layer | Choice |
|---|---|
| API / webhook | **FastAPI** (async-native) |
| Queue / state | **Celery** or **arq** + **Redis** (24h conversation TTL) |
| Datastore + vectors | **Postgres + pgvector** |
| Parsing | **Docling** (+ Mistral OCR for scans) |
| Embeddings | **BGE-M3** (self-host) |
| Query rewrite / route | **Claude Haiku 4.5** |
| Reranker | **Cohere Rerank 4** or **BGE-reranker-v2-m3** |
| Generation | **Claude Sonnet 4.6** (+ Haiku 4.5 cheap / Opus 4.8 hard) |
| Faithfulness check | **RAGAS** + **Vectara HHEM** |
| Tracing / eval | **Langfuse** |
| Messaging | **Meta WhatsApp Cloud API** (direct) |
| Business data | **Zoho People + Expense** (OAuth, India DC, server-side identity binding) |

---

## Build order

1. **Loop + identity** — WhatsApp webhook → verify → echo. Proves async loop + number→employee map.
2. **Knowledge RAG** — Docling → chunk → BGE-M3 → hybrid retrieve → Cohere rerank → Sonnet answer
   with citations. Knowledge questions only.
3. **First live tool** — orchestrator + `get_leave_balance`, identity binding enforced end-to-end.
4. **Expand + harden** — more tools, RAGAS eval set (50–100 real questions), guardrails, semantic
   cache, rate limits, templates, monitoring → roll out.

---

## What gets indexed vs fetched live

| Indexed (RAG) | Fetched live (Zoho tools) |
|---|---|
| Leave policy, holiday rules | Your remaining leave balance |
| Expense policy, per-diem tables | Your specific claim's status |
| Handbooks, onboarding, FAQs | Your payslip, your pending approvals |
| Same for everyone | Specific to one person / changes daily |
