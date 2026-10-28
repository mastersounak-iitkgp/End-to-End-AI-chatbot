# Critical Architectural Changes (`changes.md`)

To guarantee **zero API rate-limit errors**, **100% free execution**, **crash-free 200 MB PDF ingestion**, and **high retrieval accuracy**, the following mandatory modifications must be made to `plan.md` before implementation.

---

## 1. Rate-Limit & Cost Verification (CRITICAL)

### 🔴 Issue 1: Hugging Face Inference API Will Fail on 200 MB PDFs
* **The Flaw in `plan.md`:** Section 3 states that `sentence-transformers/all-MiniLM-L6-v2` will be called primarily via `HuggingFaceEndpointEmbeddings` (HF Inference API) with local `sentence-transformers` only as a fallback.
* **Why It Fails:** A 200 MB text-heavy PDF can easily contain **50,000 to 150,000+ chunks**. Sending 100,000 chunks over HTTP to the free Hugging Face Inference API will immediately trigger `HTTP 429 (Too Many Requests)` or `HTTP 504 (Gateway Timeout)`, causing the app to crash or hang for hours.
* **The Fix:** Flip the architecture. Use **Local CPU Embeddings (`HuggingFaceEmbeddings` via `langchain-huggingface`) as your PRIMARY and ONLY embedding engine.**
  * `all-MiniLM-L6-v2` is tiny (~80 MB in RAM).
  * Running it locally on CPU with batching (`batch_size=64` or `128`) is **100% free, requires no API key, has zero rate limits, and cannot fail due to network timeouts.**

### 🟡 Issue 2: Mistral AI Free Tier Rate Limits During Agentic Loops
* **The Flaw in `plan.md`:** When LangGraph routes from `chat` -> `tools` -> `chat`, it makes **two sequential LLM calls** within a single user turn. If the LLM decides to call multiple tools or reformulate a query, it makes 3+ calls in seconds.
* **Why It Fails:** Mistral AI's free tier (`La Plateforme` Experiment plan) enforces strict **1 Request Per Second (RPS)** and token-per-minute limits. Back-to-back tool return calls frequently trigger `HTTP 429` errors.
* **The Fix:**
  1. Configure `ChatMistralAI` with built-in exponential backoff retries: `max_retries=5` and `timeout=60`.
  2. Add a lightweight token/time throttle (e.g., `time.sleep(1.1)`) inside the `ToolNode` wrapper before returning execution to the `chat` node.

---

## 2. Memory, Stability & Security Fixes (Preventing User Errors)

### 🔴 Issue 3: `PyPDFLoader` Will Crash on 200 MB PDFs & Scanned Files
* **The Flaw in `plan.md`:** Section 3 proposes `pypdf` / `PyPDFLoader` to parse 200 MB PDFs.
* **Why It Fails:**
  1. `pypdf` is pure Python, slow on massive files, and loads the entire document structure into memory, often spiking RAM above 1–2 GB for a 200 MB file.
  2. **The "Image-Heavy" Trap:** Most 200 MB PDFs are large because of embedded high-res images or scanned pages. If a user uploads a 200 MB scanned PDF, `pypdf` extracts `""` (empty strings), and `FAISS.from_documents([])` will crash with `ValueError: Cannot create index from empty list`.
* **The Fix:**
  1. Replace `pypdf` with **`PyMuPDF` (`fitz` / `PyMuPDFLoader`)**. It is written in C, processes pages **15x–20x faster**, and uses a fraction of the RAM.
  2. **Add an Empty-Text Guardrail:** Filter out pages where `len(page.page_content.strip()) < 20`. If the entire PDF yields 0 valid chunks, catch it gracefully and display a Streamlit warning: *"⚠️ This PDF appears to contain scanned images rather than selectable text. Please upload a text-searchable PDF."*
  3. **Chunk Cap Guardrail:** If a 200 MB PDF has pure text (e.g., 5,000+ pages), increase `chunk_size` dynamically (e.g., from `850` to `2000`) so indexing finishes in seconds rather than 15 minutes.

### 🔴 Issue 4: Missing `is_separator_regex=True` Breaks Your Text Splitter
* **The Flaw in `plan.md`:** Section 3 specifies:
  ```python
  separators = ["\n\n", "\n", "(?<=\. )", " ", ""]
  ```
* **Why It Fails:** In LangChain's `RecursiveCharacterTextSplitter`, `is_separator_regex` defaults to `False`. Without setting `is_separator_regex=True`, LangChain searches for the **literal characters** `(?<=\. )` in the PDF text, completely skipping sentence-level splitting! Furthermore, once `is_separator_regex=True` is enabled, `\n\n` and `\n` work fine, and lookbehind `(?<=\. )` properly splits after periods.
* **The Fix:** Explicitly initialize the splitter with `is_separator_regex=True`:
  ```python
  splitter = RecursiveCharacterTextSplitter(
      chunk_size=850,
      chunk_overlap=150, # Standardized (plan had 150 in diagram, 120 in text)
      separators=["\n\n", "\n", r"(?<=\. )", " ", ""],
      is_separator_regex=True
  )
  ```

### 🔴 Issue 5: LangGraph Tool Cannot Access `thread_id` By Default
* **The Flaw in `plan.md`:** Section 4 defines `@tool def search_uploaded_documents(query: str) -> str:` and states that the tool *"Checks if a vector index exists for the current `thread_id`"*.
* **Why It Fails:** If you only pass `query: str`, the tool has no idea which user or `thread_id` called it! And if you ask the LLM to pass `thread_id`, the LLM will hallucinate it.
* **The Fix:** Use LangChain's `RunnableConfig` injection inside the tool signature so LangGraph automatically passes the active `thread_id` behind the scenes without exposing it to the LLM:
  ```python
  from langchain_core.runnables import RunnableConfig
  from langchain_core.tools import tool

  @tool
  def search_uploaded_documents(query: str, config: RunnableConfig) -> str:
      """Search the uploaded PDF document for relevant information, facts, or data."""
      thread_id = config.get("configurable", {}).get("thread_id", "default")
      # Load vectorstore from data/vectorstores/{thread_id}/
  ```

### 🔴 Issue 6: Streamlit Re-Indexing Loop & FAISS Deserialization Crash
* **The Flaw in `plan.md`:**
  1. In Streamlit, `st.file_uploader` returns the uploaded file object on **every single UI rerun** (every time the user sends a chat message). Without state tracking, your app will re-read, re-chunk, and re-embed the 200 MB PDF on every single chat message!
  2. When reloading a persisted FAISS index from `data/vectorstores/{thread_id}/`, newer versions of `langchain-community` throw a `ValueError` unless `allow_dangerous_deserialization=True` is explicitly passed.
* **The Fix:**
  1. **Hash Check in `st.session_state`:** Compute a fast SHA-256 hash of the uploaded file bytes (`hashlib.sha256(file.getvalue()).hexdigest()`). Only run the ingestion pipeline if that hash is not already indexed for the current `thread_id`.
  2. **Safe Local Load:** Since your app generates its own `.pkl` files locally, pass `FAISS.load_local(folder_path, embeddings, allow_dangerous_deserialization=True)` when reloading thread indexes.

---

## 3. Retrieval Accuracy Upgrades (Maximizing Relevance)

Your plan currently relies on pure dense vector search (`all-MiniLM-L6-v2`) with `k=5` and `LongContextReorder`. While good, `all-MiniLM-L6-v2` struggles with **exact keyword matching** (e.g., contract clauses like `"Section 4.2"`, part numbers, exact names, financial figures, or dates).

To substantially increase retrieval accuracy while keeping the stack **100% free and local**, implement this 3-stage pipeline:

```
User Query ──► [1. Hybrid Search: FAISS (MMR, k=10) + BM25 (k=10)]
           ──► [2. Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2) -> Top 5]
           ──► [3. LongContextReorder (Fixes Lost-in-the-Middle)]
           ──► Returned to Mistral AI
```

### 🟢 Accuracy Upgrade 1: Hybrid Search (`EnsembleRetriever` = FAISS + BM25)
* **Why:** Dense embeddings (`FAISS`) capture semantic meaning, while sparse lexical search (`BM25Retriever`) captures exact keywords, numbers, and section headers.
* **How to Implement:**
  * Alongside saving the FAISS index, save the split chunks (`docs.pkl`) per thread.
  * Combine `BM25Retriever.from_documents(docs, k=10)` (weight `0.35`) and `faiss_store.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 25})` (weight `0.65`) using `EnsembleRetriever`.
  * **Cost:** 100% Free, runs in $<10\text{ ms}$ locally using the `rank_bm25` package.

### 🟢 Accuracy Upgrade 2: Local Cross-Encoder Re-Ranking (`FlashRank` or `HuggingFaceCrossEncoder`)
* **Why:** Bi-encoders (`all-MiniLM-L6-v2`) compress an entire 850-character chunk into a single 384-dim vector, losing fine-grained detail. A Cross-Encoder looks at the `Query + Chunk` together token-by-token and scores true relevance.
* **How to Implement:**
  * Fetch the top 15–20 candidates from the Hybrid Retriever.
  * Pass them through a lightweight local Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers` or `flashrank`) to select the **true top 5 chunks**.
  * Apply a **Minimum Relevance Score Threshold** so completely irrelevant chunks are dropped rather than confusing the LLM.
  * **Cost:** 100% Free, adds only ~150ms on CPU, dramatically reduces hallucinations.

### 🟢 Accuracy Upgrade 3: Inject Page Context Into Chunk Text Before Embedding
* **The Flaw in `plan.md`:** `plan.md` stores `source` and `page` only in the `metadata` dictionary and adds them to the text *after* retrieval.
* **Why It Hurts Accuracy:** If a user asks *"What is summarized on page 12?"* or *"What does the Q3 Financial Report say about revenue?"*, the vector embedding of the raw chunk text doesn't contain `"Page 12"` or `"Q3 Financial Report"`.
* **The Fix:** Prepend a structured header directly into `chunk.page_content` **before** embedding:
  ```python
  chunk.page_content = f"[Source: {file_name} | Page: {page_num}]\n{chunk.page_content}"
  ```

---

## 4. Updated Dependencies (`requirements.txt`)

Replace Section 6 of `plan.md` with the following modern, non-deprecated packages:

```txt
pymupdf>=1.24.0                 # Replaces pypdf for 20x faster, low-RAM 200MB PDF parsing
faiss-cpu>=1.8.0                # Local vector database
sentence-transformers>=3.0.0    # Local embeddings & Cross-Encoder reranking (Zero rate limits)
langchain-huggingface>=0.1.0    # Modern replacement for deprecated community HF classes
langchain-community>=0.3.0      # For FAISS, BM25Retriever, EnsembleRetriever, LongContextReorder
rank_bm25>=0.2.2                # Required for Hybrid Keyword + Semantic Search
```

---

## 5. Summary Checklist of Changes to Apply to `plan.md`

| Component | Original `plan.md` | Required Change in `changes.md` |
| :--- | :--- | :--- |
| **Embedding Execution** | HF Inference API primary, local fallback | **Local `HuggingFaceEmbeddings` primary** (prevents HTTP 429 on large PDFs) |
| **PDF Parser** | `pypdf` / `PyPDFLoader` | **`PyMuPDF` (`fitz`)** + empty/scanned PDF guardrail + dynamic chunk cap |
| **Text Splitter** | `chunk_size=900/850`, missing regex flag | Standardize `chunk_size=850, overlap=150` + **`is_separator_regex=True`** |
| **Retrieval Strategy** | FAISS top-$k$ only | **Hybrid Search (FAISS MMR + BM25) + Cross-Encoder Reranking** |
| **Thread Isolation in Tool** | `@tool def search...(query: str)` | Inject **`config: RunnableConfig`** to safely read `thread_id` |
| **Streamlit Upload State** | Re-runs on every interaction | Track **file SHA-256 hash in `st.session_state`** to index only once |
| **Mistral Rate Limits** | Standard tool call loop | Add **`max_retries=5`** on `ChatMistralAI` + 1s rate-limit buffer |