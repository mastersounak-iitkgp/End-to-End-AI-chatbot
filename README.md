# 🤖 AI Chatbot with LangGraph, Tools & Advanced Hybrid RAG

A multi-session conversational AI assistant built with **LangGraph**, **Mistral AI**, and **Streamlit**. Features persistent conversation memory, automated tool calling (live web search, real-time stock quotes, safe math calculation), and an enterprise-grade 3-stage **Retrieval-Augmented Generation (RAG)** pipeline for document Q&A with local embeddings.

---

## 🌟 Key Features

### 🧠 LangGraph Agent & State Management
- **Cyclical Graph Execution**: Orchestrates dynamic interaction flows (`START -> chat -> [tools_condition] -> tools -> chat -> END`).
- **Persistent SQLite Memory**: Uses LangGraph's `SqliteSaver` checkpointer (`chatbot_memory.db`) to retain multi-turn context and full chat history across application restarts.
- **Multi-Session Management**: Create new chats, switch between past conversations, rename threads, and customize per-session system instructions.
- **Real-Time Token & Tool Streaming**: Streams response tokens in real-time alongside visual tool execution indicators.

### 🛠️ Built-In Agent Tools
1. **📄 Document Search (RAG)** (`search_uploaded_documents`): Queries uploaded PDF documents, providing answers with exact page and document citations.
2. **🌐 DuckDuckGo Web Search** (`duckduckgo_search`): Fetches live web information, breaking news, sports scores, and real-time facts.
3. **🧮 Math Calculator** (`calculate_math`): Safe, AST-based expression evaluator supporting arithmetic, powers, and trigonometry without insecure `eval()` calls.
4. **📈 Stock Price Checker** (`check_stock_price`): Real-time market data, valuation metrics, 24-hour price changes, and trading ranges via `yfinance`.

### 📚 Advanced 3-Stage RAG Pipeline
- **Fast Extraction**: Low-memory PDF text parsing via PyMuPDF with scanned/image-only document detection.
- **100% Free Local Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` locally on CPU (no external API calls or rate limits).
- **Stage 1 (Hybrid Search)**: Combines dense vector retrieval (FAISS with Maximal Marginal Relevance) and sparse lexical search (BM25Okapi) using Weighted Reciprocal Rank Fusion (RRF).
- **Stage 2 (Cross-Encoder Re-Ranking)**: Re-scores candidate passages with `cross-encoder/ms-marco-MiniLM-L-6-v2` for high contextual precision.
- **Stage 3 (Long-Context Reordering)**: Mitigates the "Lost in the Middle" attention degradation in LLMs by placing the most relevant passages at the prompt boundaries.
- **Thread Isolation**: Vector indices and uploaded documents are isolated per conversation session under `data/vectorstores/{thread_id}`.

### 🎨 Modern Glassmorphic UI
- Premium dark-themed UI built with custom CSS.
- Collapsible tool execution cards displaying invocation arguments and raw tool outputs.
- Sidebar controls for PDF uploads, index status, conversation switching, and custom system prompt adjustments.

---

## 🏗️ Architecture

```
                      ┌─────────────────────────┐
                      │    Streamlit Frontend   │
                      │        (app.py)         │
                      └────────────┬────────────┘
                                   │ User Input / Stream
                                   ▼
                      ┌─────────────────────────┐
                      │    LangGraph Workflow   │
                      │   (chatbot_backend.py)  │
                      └────────────┬────────────┘
                                   │
                ┌──────────────────┴──────────────────┐
                ▼                                     ▼
     ┌──────────────────────┐              ┌──────────────────────┐
     │      Mistral AI      │              │  SQLite Checkpointer │
     │ (ministral-8b-latest)│              │ (chatbot_memory.db)  │
     └──────────┬───────────┘              └──────────────────────┘
                │
                ├───────────────────────────────────────────────────┐
                ▼                                                   ▼
     ┌──────────────────────┐                            ┌──────────────────────┐
     │    External Tools    │                            │    RAG Engine        │
     │ 🌐 DuckDuckGo Search │                            │   (rag_engine.py)    │
     │ 🧮 Math Calculator   │                            ├──────────────────────┤
     │ 📈 Stock Checker     │                            │ • PyMuPDF Parser     │
     └──────────────────────┘                            │ • FAISS + BM25 (RRF) │
                                                         │ • Cross-Encoder Rank │
                                                         │ • LongContextReorder │
                                                         └──────────────────────┘
```

---

## 📁 Project Structure

```plaintext
AI Chatbot/
├── app.py                   # Streamlit frontend & interactive UI
├── chatbot_backend.py       # LangGraph agent, tool definitions & streaming pipeline
├── rag_engine.py            # PDF processing, hybrid search, reranker & vector store
├── requirements.txt         # Project dependencies
├── .env.example             # Template for environment variables (API keys)
├── .gitignore               # Excludes secrets, caches, and local databases
├── data/                    # Storage directory (git-ignored runtime data)
│   ├── uploads/             # Session-specific uploaded PDF documents
│   └── vectorstores/        # FAISS indexes and BM25 caches per session
└── README.md                # Project documentation
```

---

## ⚙️ Prerequisites

- **Python**: Version `3.10` or higher (`3.11` / `3.12` / `3.13` supported).
- **Mistral AI API Key**: Sign up and generate a free API key at [Mistral AI Console](https://console.mistral.ai/).
- *(Optional)* Hugging Face Token / Google API Key for additional models or integrations.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd "AI Chatbot"
```

### 2. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> [!NOTE]
> On the first run, the local embedding model (`all-MiniLM-L6-v2`) and cross-encoder reranker weights (~160MB total) will be automatically downloaded and cached locally by Hugging Face.

### 4. Configure Environment Variables
Copy the `.env.example` file to `.env`:

```bash
# Windows (PowerShell)
Copy-Item .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` in your text editor and add your API keys:
```env
# Required for chatbot intelligence
MISTRAL_API_KEY="your-mistral-api-key-here"

# Optional / External integrations
GOOGLE_API_KEY=""
HF_TOKEN=""
HUGGINGFACEHUB_API_TOKEN=""
```

---

## 🖥️ Running the Application

Launch the Streamlit interface:

```bash
streamlit run app.py
```

Once started, the application will be accessible in your web browser at:
```
Local URL: http://localhost:8501
```

---

## 📖 How to Use

1. **Start a Conversation**: Type any question into the chat input box at the bottom.
2. **Web Search**: Ask questions regarding recent events (e.g., *"What were the top tech headlines today?"* or *"What is the score of the latest cricket match?"*).
3. **Stock Prices**: Inquire about financial assets (e.g., *"What is the current stock price of Apple (AAPL)?"* or *"How is NVDA trading today?"*).
4. **Calculations**: Ask mathematical queries (e.g., *"Calculate compound interest on $10,000 at 7% for 5 years"*).
5. **PDF Document Q&A (RAG)**:
   - Open the **Sidebar** and expand **"📄 Upload PDF Document"**.
   - Upload any document (contracts, research papers, reports).
   - Once indexed, ask questions like *"Summarize section 3 of the uploaded document"* or *"What are the liability terms?"*.
   - The assistant will perform hybrid search and reranking, citing page numbers directly.
6. **Manage Sessions**:
   - Click **"+ New Chat"** in the sidebar to start a clean thread.
   - Switch back to any historical chat at any time without losing state.
   - Adjust the **System Prompt** in the sidebar to modify the persona or instructions.

---

## 🔒 Security & Safe Pushing to GitHub

Sensitive credentials and dynamic session data are kept out of Git version control via [`.gitignore`](file:///c:/Users/maste/OneDrive/Desktop/AI%20Chatbot/.gitignore):
- Secrets (`.env`, `.env.*`)
- Virtual environments (`.venv/`, `venv/`)
- SQLite databases (`*.db`, `*.db-wal`, `*.db-shm`)
- Cached embeddings and uploaded PDFs (`data/uploads/*`, `data/vectorstores/*`)

Before pushing to a public repository:
```bash
git status
```
Confirm that `.env` is listed under ignored files and **not** staged for commit.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

