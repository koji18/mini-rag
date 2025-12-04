"""
Configuration file for mini-RAG system.

All implementation policies and constants are defined here.
"""

import os
from pathlib import Path
from typing import List, Dict, Any

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
INDEX_DIR = DATA_DIR / "index"
CACHE_DIR = DATA_DIR / "cache"

# Create directories if they don't exist
for dir_path in [DATA_DIR, DOCS_DIR, INDEX_DIR, CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 1. Embedding Model Configuration
# ============================================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
EMBEDDING_DEVICE = "cpu"
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_NORMALIZE = True


# ============================================================================
# 2. Document Chunking Strategy
# ============================================================================

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
PARAGRAPH_SIZE = 2048
CHUNKING_STRATEGY = "hierarchical"

SENTENCE_DELIMITERS = [
    "。", "！", "？", "\n", ".", "!", "?",
]

MIN_CHUNK_SIZE = 50


# ============================================================================
# 3. Index Configuration
# ============================================================================

INDEX_TYPE = "file"
INDEX_SAVE_PATH = INDEX_DIR / "rag_index.pkl"
INDEX_METADATA_PATH = INDEX_DIR / "rag_metadata.json"
INDEX_EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"

INDEX_AUTO_SAVE = True
INDEX_SAVE_INTERVAL = 100


# ============================================================================
# 4. Retrieval & Ranking Configuration
# ============================================================================

SIMILARITY_METRIC = "cosine"
SIMILARITY_THRESHOLD = 0.3
TOP_K = 3

ENABLE_RANKING = True
RANK_BY_RECENCY = False
RANK_BY_SOURCE = False


# ============================================================================
# 5. LLM Integration Policy
# ============================================================================

LLM_TYPE = "template"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_TEMPERATURE = 0.3
OPENAI_MAX_TOKENS = 500

LLM_TIMEOUT = 30

DEFAULT_NO_RESULTS_MESSAGE = (
    "申し訳ございません。提供された情報に基づく回答が見つかりませんでした。"
)

DEFAULT_ERROR_MESSAGE = (
    "エラーが発生しました。もう一度お試しください。"
)


# ============================================================================
# 6. Error Handling Policy
# ============================================================================

ERROR_HANDLING_MODE = "graceful"

MAX_RETRIES = 3
RETRY_BACKOFF = "exponential"
RETRY_INITIAL_DELAY = 0.5

VALIDATE_INPUTS = True
VALIDATE_EMBEDDINGS = True

LOG_LEVEL = "INFO"
LOG_FILE = PROJECT_ROOT / "logs" / "rag.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 7. Performance & Caching Configuration
# ============================================================================

ENABLE_CACHE = True
CACHE_BACKEND = "file"
CACHE_DIR_PATH = CACHE_DIR

EMBEDDING_CACHE_ENABLED = True
EMBEDDING_CACHE_SIZE = 10000
EMBEDDING_CACHE_TTL = 86400

RETRIEVAL_CACHE_ENABLED = True
RETRIEVAL_CACHE_SIZE = 5000
RETRIEVAL_CACHE_TTL = 3600

BATCH_SIZE = 32
NUM_WORKERS = 4

ENABLE_PROFILING = False
PROFILE_SLOW_QUERIES = True
SLOW_QUERY_THRESHOLD = 1.0


# ============================================================================
# 8. File System Configuration
# ============================================================================

SUPPORTED_FORMATS = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".rst": "text/x-rst",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".csv": "text/csv",
    ".json": "application/json",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
}

RECURSIVE_SEARCH = True
MAX_FILE_SIZE_MB = 100
ENCODING_AUTO_DETECT = True

PDF_EXTRACT_TEXT = True
PDF_EXTRACT_IMAGES = True

IMAGE_OCR_ENABLED = True
OCR_LANGUAGE = "jpn+eng"

REMOVE_MARKDOWN_LINKS = True
REMOVE_HTML_TAGS = True
NORMALIZE_WHITESPACE = True
REMOVE_SPECIAL_CHARACTERS = False


# ============================================================================
# Additional Configuration
# ============================================================================

AUTO_REBUILD_INDEX = False
WATCH_DOCS_DIRECTORY = False

API_HOST = "0.0.0.0"
API_PORT = 8000
API_DEBUG = False

DATABASE_TYPE = None
DATABASE_URL = None


# ============================================================================
# Utility Functions
# ============================================================================

def get_config_summary() -> Dict[str, Any]:
    """Get a summary of current configuration."""
    return {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "index_type": INDEX_TYPE,
        "similarity_metric": SIMILARITY_METRIC,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "top_k": TOP_K,
        "llm_type": LLM_TYPE,
        "error_handling": ERROR_HANDLING_MODE,
        "caching_enabled": ENABLE_CACHE,
        "supported_formats": list(SUPPORTED_FORMATS.keys()),
    }


def validate_config() -> List[str]:
    """Validate configuration settings. Return list of warnings."""
    warnings = []

    if not DOCS_DIR.exists():
        warnings.append(f"Documents directory not found: {DOCS_DIR}")

    if LLM_TYPE == "openai" and not OPENAI_API_KEY:
        warnings.append("LLM_TYPE is 'openai' but OPENAI_API_KEY is not set")

    if CHUNK_SIZE < 50:
        warnings.append(f"CHUNK_SIZE is too small: {CHUNK_SIZE}")

    if CHUNK_SIZE > 10000:
        warnings.append(f"CHUNK_SIZE is very large: {CHUNK_SIZE}")

    if not (0 <= SIMILARITY_THRESHOLD <= 1):
        warnings.append(f"SIMILARITY_THRESHOLD out of range: {SIMILARITY_THRESHOLD}")

    if TOP_K < 1:
        warnings.append(f"TOP_K should be at least 1: {TOP_K}")

    return warnings


if __name__ == "__main__":
    print("=" * 60)
    print("Mini-RAG Configuration Summary")
    print("=" * 60)

    for key, value in get_config_summary().items():
        print(f"{key:.<30} {value}")

    print("\n" + "=" * 60)
    print("Validation Warnings:")
    print("=" * 60)

    warnings = validate_config()
    if warnings:
        for warning in warnings:
            print(f"Warning: {warning}")
    else:
        print("No configuration warnings")

    print("=" * 60)
