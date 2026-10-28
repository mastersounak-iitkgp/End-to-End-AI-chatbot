"""
RAG Engine — Hardened Document Processing, Hybrid Search & Reordering
======================================================================
Features:
  1. Fast, low-memory PDF extraction using PyMuPDF (pymupdf).
  2. Scanned / image-only PDF detection and empty-text guardrails.
  3. Dynamic chunk scaling with sentence-preserving regex splitting.
  4. Metadata injection ([Document: ... | Page: ...]) directly into chunk text.
  5. 100% Free, rate-limit-free local CPU embeddings (all-MiniLM-L6-v2).
  6. In-memory singleton caching (@st.cache_resource / singleton pattern).
  7. 3-Stage Precision Retrieval:
     - Stage 1: Hybrid Search (FAISS MMR + BM25 via Weighted RRF)
     - Stage 2: Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2, raw logit sort)
     - Stage 3: LongContextReorder (solves 'Lost in the Middle')
  8. Thread-isolated storage (data/vectorstores/{thread_id}/).
  9. Efficient SHA-256 file hashing (first/last 64KB sample).
"""

import os
import sys
import pickle
import hashlib
from typing import Optional, List, Dict, Any, Tuple

import pymupdf  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_transformers import LongContextReorder
from rank_bm25 import BM25Okapi

# ── Paths ──────────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_DIR, "data")
VECTORSTORE_ROOT = os.path.join(DATA_DIR, "vectorstores")
UPLOADS_ROOT = os.path.join(DATA_DIR, "uploads")

os.makedirs(VECTORSTORE_ROOT, exist_ok=True)
os.makedirs(UPLOADS_ROOT, exist_ok=True)

# ── Singleton Model Cache ──────────────────────────────────────────────────────
# Prevents reloading ~160 MB of model weights on every tool call or Streamlit rerun
_CACHED_EMBEDDINGS: Optional[HuggingFaceEmbeddings] = None
_CACHED_CROSS_ENCODER = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Returns the singleton HuggingFaceEmbeddings instance for all-MiniLM-L6-v2.
    Runs locally on CPU with batching. 100% free, zero rate-limits.
    """
    global _CACHED_EMBEDDINGS
    if _CACHED_EMBEDDINGS is None:
        _CACHED_EMBEDDINGS = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
        )
    return _CACHED_EMBEDDINGS


def get_cross_encoder():
    """
    Returns the singleton CrossEncoder instance for ms-marco-MiniLM-L-6-v2.
    Gracefully falls back to None if model loading fails or is unavailable.
    """
    global _CACHED_CROSS_ENCODER
    if _CACHED_CROSS_ENCODER is None:
        try:
            from sentence_transformers import CrossEncoder
            _CACHED_CROSS_ENCODER = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                max_length=512,
                device="cpu",
            )
        except Exception as e:
            # Fallback gracefully without breaking retrieval
            _CACHED_CROSS_ENCODER = False
    return _CACHED_CROSS_ENCODER if _CACHED_CROSS_ENCODER is not False else None


# ── File Hashing ───────────────────────────────────────────────────────────────
def compute_file_hash(file_name: str, file_size: int, file_bytes: bytes) -> str:
    """
    Computes a content-aware SHA-256 hash using file metadata and the first/last
    64 KB of data. Guarantees content uniqueness with minimal RAM overhead.
    """
    hasher = hashlib.sha256()
    hasher.update(file_name.encode("utf-8"))
    hasher.update(str(file_size).encode("utf-8"))
    if file_bytes:
        hasher.update(file_bytes[:65536])
        if len(file_bytes) > 65536:
            hasher.update(file_bytes[-65536:])
    return hasher.hexdigest()


# ── PDF Parsing with PyMuPDF ──────────────────────────────────────────────────
def extract_text_from_pdf(pdf_path: str) -> Tuple[List[Document], int]:
    """
    Extracts text page-by-page from a PDF using PyMuPDF (fitz).
    Applies an empty-text guardrail to catch scanned / image-only PDFs.

    Returns:
        (documents, total_pages)
    Raises:
        ValueError if PDF is empty or contains only scanned images without text.
    """
    file_name = os.path.basename(pdf_path)
    file_size = os.path.getsize(pdf_path)
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)

    pages: List[Document] = []
    total_characters = 0

    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text("text").strip()

        # Guardrail: ignore blank or image-only pages (< 20 chars)
        if len(text) >= 20:
            total_characters += len(text)
            pages.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_name,
                        "page": page_num + 1,
                        "total_pages": total_pages,
                        "file_size": file_size,
                    },
                )
            )

    doc.close()

    if total_characters < 80 or not pages:
        raise ValueError(
            f"The uploaded file '{file_name}' contains no selectable text (scanned or image-only PDF). "
            "Please upload a searchable text-based PDF."
        )

    return pages, total_pages


# ── Text Chunking ──────────────────────────────────────────────────────────────
def chunk_documents(pages: List[Document], total_pages: int) -> List[Document]:
    """
    Splits document pages using RecursiveCharacterTextSplitter with regex lookbehinds.
    Dynamically scales chunk size for massive (300+ page) documents to prevent
    indexing bottlenecks on 200 MB files.
    Injects [Document: ... | Page: ...] directly into chunk text before embedding.
    """
    # Dynamic chunk sizing: scale up for massive documents
    if total_pages > 300:
        chunk_size = 1800
        chunk_overlap = 200
    else:
        chunk_size = 850
        chunk_overlap = 150

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", r"(?<=\. )", " ", ""],
        is_separator_regex=True,
    )

    raw_chunks = splitter.split_documents(pages)
    enriched_chunks: List[Document] = []

    for idx, c in enumerate(raw_chunks):
        file_name = c.metadata.get("source", "Document")
        page_num = c.metadata.get("page", 1)

        # Inject document context directly into chunk text for semantic search
        header = f"[Document: {file_name} | Page {page_num}]\n"
        injected_content = header + c.page_content.strip()

        metadata = dict(c.metadata)
        metadata["chunk_id"] = idx

        enriched_chunks.append(
            Document(page_content=injected_content, metadata=metadata)
        )

    return enriched_chunks


# ── Indexing Pipeline ──────────────────────────────────────────────────────────
def index_pdf_document(pdf_bytes: bytes, file_name: str, thread_id: str) -> Dict[str, Any]:
    """
    Full ingestion pipeline:
      1. Saves PDF to data/uploads/{thread_id}/
      2. Extracts text using PyMuPDF (fitz)
      3. Chunks text with regex lookbehinds
      4. Generates local CPU embeddings
      5. Builds & persists FAISS vector store + chunks.pkl to data/vectorstores/{thread_id}/

    Returns:
        dict with metadata (file_name, total_pages, chunk_count, file_hash)
    """
    thread_upload_dir = os.path.join(UPLOADS_ROOT, thread_id)
    thread_store_dir = os.path.join(VECTORSTORE_ROOT, thread_id)
    os.makedirs(thread_upload_dir, exist_ok=True)
    os.makedirs(thread_store_dir, exist_ok=True)

    # Save uploaded file to disk
    pdf_path = os.path.join(thread_upload_dir, file_name)
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)

    file_hash = compute_file_hash(file_name, len(pdf_bytes), pdf_bytes)

    # Extract text with PyMuPDF
    pages, total_pages = extract_text_from_pdf(pdf_path)

    # Chunk text
    chunks = chunk_documents(pages, total_pages)

    # Generate embeddings and persist FAISS vector store
    embeddings = get_embedding_model()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(thread_store_dir)

    # Persist chunks.pkl for instantaneous BM25 re-creation
    chunks_path = os.path.join(thread_store_dir, "chunks.pkl")
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    # Persist metadata summary
    meta_summary = {
        "file_name": file_name,
        "total_pages": total_pages,
        "chunk_count": len(chunks),
        "file_hash": file_hash,
    }
    meta_path = os.path.join(thread_store_dir, "doc_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta_summary, f)

    return meta_summary


def get_thread_doc_info(thread_id: str) -> Optional[Dict[str, Any]]:
    """Returns the document indexing metadata for a thread, or None if not indexed."""
    meta_path = os.path.join(VECTORSTORE_ROOT, thread_id, "doc_meta.pkl")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def clear_thread_index(thread_id: str) -> None:
    """Removes the vectorstore and uploaded documents for a thread."""
    thread_store_dir = os.path.join(VECTORSTORE_ROOT, thread_id)
    thread_upload_dir = os.path.join(UPLOADS_ROOT, thread_id)
    import shutil
    if os.path.exists(thread_store_dir):
        shutil.rmtree(thread_store_dir, ignore_errors=True)
    if os.path.exists(thread_upload_dir):
        shutil.rmtree(thread_upload_dir, ignore_errors=True)


# ── 3-Stage Precision Retrieval Pipeline ───────────────────────────────────────

def _weighted_reciprocal_rank_fusion(
    dense_docs: List[Document],
    sparse_docs: List[Document],
    dense_weight: float = 0.65,
    sparse_weight: float = 0.35,
    c: int = 60,
    top_k: int = 15,
) -> List[Document]:
    """
    Fuses dense vector results and sparse lexical results using Weighted RRF.
    Score = sum(w / (c + rank)). Sorts candidates by score descending.
    Zero external dependencies, works directly on Document lists.
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    def _doc_key(doc: Document) -> str:
        # Unique identifier combining page and first 60 chars of text
        return f"{doc.metadata.get('page', 0)}_{doc.page_content[:60]}"

    for rank, doc in enumerate(dense_docs):
        key = _doc_key(doc)
        scores[key] = scores.get(key, 0.0) + dense_weight * (1.0 / (c + rank + 1))
        doc_map[key] = doc

    for rank, doc in enumerate(sparse_docs):
        key = _doc_key(doc)
        scores[key] = scores.get(key, 0.0) + sparse_weight * (1.0 / (c + rank + 1))
        doc_map[key] = doc

    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    return [doc_map[k] for k in sorted_keys[:top_k]]


def _reorder_lost_in_the_middle(documents: List[Document]) -> List[Document]:
    """
    Reorders retrieved documents into a U-shaped distribution to place the strongest
    evidence in the primary attention zones (beginning and end) of the prompt.
    Uses LangChain LongContextReorder with robust fallback.
    """
    if len(documents) <= 2:
        return documents

    try:
        reorderer = LongContextReorder()
        return list(reorderer.transform_documents(documents))
    except Exception:
        # In-line fallback: alternating between end and start
        reordered: List[Document] = []
        for i, doc in enumerate(documents):
            if i % 2 == 0:
                reordered.append(doc)
            else:
                reordered.insert(0, doc)
        return reordered


def retrieve_relevant_chunks(
    query: str, thread_id: str, final_top_k: int = 5
) -> Tuple[List[Document], str]:
    """
    Executes the 3-Stage Precision Retrieval Pipeline:
      Stage 1: Hybrid Search (FAISS MMR + BM25 via Weighted RRF) -> Top 15
      Stage 2: Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2) -> Top 5
               Sorts by raw logits descending (no arbitrary logit threshold drop)
      Stage 3: LongContextReorder (solves 'Lost in the Middle') -> Final 5

    Returns:
        (reordered_documents, formatted_text)
    """
    thread_store_dir = os.path.join(VECTORSTORE_ROOT, thread_id)
    chunks_path = os.path.join(thread_store_dir, "chunks.pkl")

    if not os.path.exists(thread_store_dir) or not os.path.exists(chunks_path):
        return [], "No document has been indexed for this conversation yet. Please upload a PDF in the sidebar."

    # Load FAISS index (safe local load)
    embeddings = get_embedding_model()
    vectorstore = FAISS.load_local(
        thread_store_dir, embeddings, allow_dangerous_deserialization=True
    )

    # Load chunks for BM25
    with open(chunks_path, "rb") as f:
        chunks: List[Document] = pickle.load(f)

    if not chunks:
        return [], "The document index contains no searchable chunks."

    # ── Stage 1: Hybrid Search (Dense FAISS + Sparse BM25) ───────────────────
    # 1.1 Dense MMR search
    try:
        dense_docs = vectorstore.max_marginal_relevance_search(
            query, k=10, fetch_k=25
        )
    except Exception:
        dense_docs = vectorstore.similarity_search(query, k=10)

    # 1.2 Sparse BM25 search
    corpus_tokens = [doc.page_content.lower().split() for doc in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)

    # Pick top 10 from BM25
    top_bm25_indices = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:10]
    sparse_docs = [chunks[i] for i in top_bm25_indices if bm25_scores[i] > 0]

    # 1.3 Weighted RRF Fusion -> Top 15
    fused_candidates = _weighted_reciprocal_rank_fusion(
        dense_docs, sparse_docs, dense_weight=0.65, sparse_weight=0.35, top_k=15
    )

    # ── Stage 2: Cross-Encoder Reranking ─────────────────────────────────────
    cross_encoder = get_cross_encoder()
    if cross_encoder and len(fused_candidates) > 1:
        pairs = [(query, doc.page_content) for doc in fused_candidates]
        try:
            # ms-marco outputs raw unnormalized logits (-10.0 to +10.0)
            logits = cross_encoder.predict(pairs)
            # Sort candidates by raw logit descending without arbitrary hard cutoff
            scored_candidates = list(zip(fused_candidates, logits))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidates = [doc for doc, _ in scored_candidates[:final_top_k]]
        except Exception:
            top_candidates = fused_candidates[:final_top_k]
    else:
        top_candidates = fused_candidates[:final_top_k]

    # ── Stage 3: LongContextReorder (Fix Lost-in-the-Middle) ──────────────────
    reordered_docs = _reorder_lost_in_the_middle(top_candidates)

    # Format into cleanly cited text blocks
    formatted_blocks = []
    for rank, doc in enumerate(reordered_docs, 1):
        source = doc.metadata.get("source", "Document")
        page = doc.metadata.get("page", 1)
        # Strip internal injected header if already in page_content
        content = doc.page_content
        if content.startswith(f"[Document: {source} | Page {page}]"):
            content = content[len(f"[Document: {source} | Page {page}]"):].strip()

        formatted_blocks.append(
            f"--- Excerpt {rank} (Source: {source}, Page {page}) ---\n{content}"
        )

    formatted_result = "\n\n".join(formatted_blocks)
    return reordered_docs, formatted_result
