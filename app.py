# app.py
import os
import logging
import pathway as pw
from typing import Optional

# Try to import the most common classes from Pathway LLM xpack
try:
    from pathway.xpacks.llm.parsers import UnstructuredParser
except Exception:
    try:
        from pathway.xpacks.llm.parsers import ParseUnstructured as UnstructuredParser
        logging.warning("Using deprecated ParseUnstructured as UnstructuredParser.")
    except Exception:
        UnstructuredParser = None
        logging.warning("UnstructuredParser not available in this Pathway build.")

try:
    from pathway.xpacks.llm.splitters import TokenCountSplitter
except Exception:
    TokenCountSplitter = None
    logging.warning("TokenCountSplitter not found in pathway.xpacks.llm.splitters; check your package version.")

try:
    from pathway.xpacks.llm.embedders import SentenceTransformerEmbedder
except Exception:
    SentenceTransformerEmbedder = None
    logging.warning("SentenceTransformerEmbedder not present in this Pathway build.")

from pathway.stdlib.indexing import BruteForceKnnFactory
from pathway.xpacks.llm.document_store import DocumentStore
from pathway.xpacks.llm.servers import DocumentStoreServer

# ----------------------
# Configuration
# ----------------------
DATA_FILE = os.getenv("DATA_FILE", "./feed.jsonl")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
SENTENCE_TRANSFORMERS_MODEL = os.getenv("SENTENCE_TRANSFORMERS_MODEL", "all-MiniLM-L6-v2")

# Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pathway-rag")

# ----------------------
# Ensure dependencies are available
# ----------------------
if SentenceTransformerEmbedder is None:
    raise RuntimeError(
        "SentenceTransformerEmbedder not available. "
        "Install 'sentence-transformers' and use a Pathway build with the LLM xpack."
    )

# ----------------------
# Read streaming JSONL input
# Schema with only the fields that exist in feed.jsonl
# ----------------------
class NewsArticleSchema(pw.Schema):
    headline: str
    body: str

log.info(f"Reading JSONL data from {DATA_FILE}")
news_stream = pw.io.jsonlines.read(
    DATA_FILE,
    schema=NewsArticleSchema,
    mode="streaming",
    autocommit_duration_ms=500,
)

# Create initial documents table with text and metadata columns preserved
# Extract string from tuple if body is a tuple (some Pathway versions return tuples)
documents = news_stream.select(
    text=pw.apply(lambda body: body[0] if isinstance(body, tuple) else body, pw.this.body),
    headline=pw.this.headline,
)

log.info("Documents table created with columns: text, headline")

# ----------------------
# Parser & Splitter
# ----------------------
# Skip UnstructuredParser for now to preserve columns
# Use the raw documents as single chunks
log.info("Using raw bodies as chunks (no parsing)")
chunks = documents

# ----------------------
# Token-based splitter (optional)
# ----------------------
if TokenCountSplitter is not None:
    try:
        log.info("Attempting to use TokenCountSplitter")
        splitter = TokenCountSplitter(min_tokens=50, max_tokens=400, encoding_name="cl100k_base")
        
        # Apply splitter - this returns chunks but we need to preserve headline
        chunks_with_split = chunks.select(
            chunk=splitter(pw.this.text),
            headline=pw.this.headline
        )
        
        # Flatten chunk arrays into one row per chunk
        # Extract string from tuple if chunk is a tuple
        chunks = chunks_with_split.flatten(pw.this.chunk).select(
            text=pw.apply(lambda chunk: chunk[0] if isinstance(chunk, tuple) else chunk, pw.this.chunk),
            headline=pw.this.headline
        )
        log.info("TokenCountSplitter applied successfully")
    except Exception as e:
        log.warning(f"TokenCountSplitter failed: {e}. Using full bodies as chunks.")
        chunks = documents
else:
    log.info("TokenCountSplitter not present; using full bodies as chunks")

# ----------------------
# Embedding setup
# ----------------------
embedder = SentenceTransformerEmbedder(
    model=SENTENCE_TRANSFORMERS_MODEL, 
    call_kwargs={"show_progress_bar": False}
)

try:
    emb_dim = embedder.get_embedding_dimension()
    log.info(f"Embedder dimension: {emb_dim}")
except Exception as e:
    log.warning(f"Could not detect embed dimension: {e}. Defaulting to 384.")
    emb_dim = 384

# Build KNN factory
knn_factory = BruteForceKnnFactory(
    reserved_space=1000,
    embedder=embedder,
    metric=pw.engine.BruteForceKnnMetricKind.COS,
    dimensions=emb_dim,
)

# ----------------------
# Prepare documents for DocumentStore
# ----------------------
log.info(f"Chunks columns before metadata creation: {chunks.schema if hasattr(chunks, 'schema') else 'unknown'}")

# Create docs_for_store with data and _metadata columns
# Ensure data is a plain string, not a tuple (DocumentStore requires plain strings)
try:
    docs_for_store = chunks.select(
        data=pw.apply(lambda text: text[0] if isinstance(text, tuple) else str(text), pw.this.text),
        _metadata=pw.apply(
            lambda headline: {"headline": str(headline)},
            pw.this.headline
        )
    )
    log.info("Successfully created docs_for_store with headline metadata")
except Exception as e:
    log.warning(f"Failed to create metadata: {e}. Using data-only format.")
    docs_for_store = chunks.select(
        data=pw.apply(lambda text: text[0] if isinstance(text, tuple) else str(text), pw.this.text)
    )

log.info(f"docs_for_store schema: {docs_for_store.schema if hasattr(docs_for_store, 'schema') else 'unknown'}")

# ----------------------
# Build DocumentStore
# ----------------------
# Don't pass parser/splitter to DocumentStore since we already processed the data
document_store = DocumentStore(
    docs=docs_for_store,
    retriever_factory=knn_factory,
)

log.info("DocumentStore created successfully")

# ----------------------
# Start REST server
# ----------------------
server = DocumentStoreServer(host=HOST, port=PORT, document_store=document_store)

log.info(f"Starting DocumentStoreServer on {HOST}:{PORT}")
try:
    server.run(threaded=True)
except TypeError:
    try:
        server.serve(threaded=True)
    except Exception as err:
        log.warning(f"server.run/serve failed: {err}")

# ----------------------
# Start Pathway engine
# ----------------------
log.info("Starting Pathway engine (pw.run())")
pw.run()