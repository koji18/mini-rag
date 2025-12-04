#!/usr/bin/env python3
"""
Mini-RAG Bootstrap Script

プロジェクトの初期化とデモンストレーション用のメインスクリプト
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import (
    get_config_summary,
    validate_config,
    PROJECT_ROOT,
    DOCS_DIR,
    LOG_LEVEL,
)

# ============================================================================
# Logging Setup
# ============================================================================

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Project Initialization
# ============================================================================

def check_dependencies() -> bool:
    """Check if all required dependencies are installed."""
    logger.info("Checking dependencies...")

    required_packages = {
        "numpy": "NumPy",
        "sklearn": "scikit-learn",
        "sentence_transformers": "Sentence Transformers",
    }

    missing = []
    for module, name in required_packages.items():
        try:
            __import__(module)
            logger.debug(f"✓ {name} is installed")
        except ImportError:
            logger.warning(f"✗ {name} is NOT installed")
            missing.append(name)

    if missing:
        logger.error(f"Missing packages: {', '.join(missing)}")
        logger.error(f"Install with: pip install -r requirements.txt")
        return False

    logger.info("✓ All required dependencies are installed")
    return True


def check_directories() -> bool:
    """Check if required directories exist."""
    logger.info("Checking project structure...")

    required_dirs = [
        DOCS_DIR,
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "tests",
    ]

    for dir_path in required_dirs:
        if dir_path.exists():
            logger.debug(f"✓ {dir_path.relative_to(PROJECT_ROOT)} exists")
        else:
            logger.warning(f"✗ {dir_path.relative_to(PROJECT_ROOT)} not found")
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"  Created: {dir_path.relative_to(PROJECT_ROOT)}")

    logger.info("✓ Project structure is valid")
    return True


def check_configuration() -> bool:
    """Check configuration settings."""
    logger.info("Validating configuration...")

    warnings = validate_config()

    if warnings:
        for warning in warnings:
            logger.warning(f"⚠️  {warning}")
        logger.info("Configuration has warnings but is usable")
        return True

    logger.info("✓ Configuration is valid")
    return True


def print_welcome() -> None:
    """Print welcome message."""
    print("\n" + "=" * 70)
    print("█▀ █▀ █▀█ █░█ █▀▀ █░░ █▀▄ █▀▀ █▀▄   ▀█▀ █▀▀ █▀ ▀█▀ █▀▀ █▀█")
    print("█░ █░░ █▀▄ █▀█ █░░ █░░ █░█ █░░ █░█   █░█ █▀▀ ▀▄  █░ █▀▀ █░▄")
    print("▀▀░ ▀▀▀ ▀░▀ ▀░▀ ▀▀▀ ▀▀▀ ▀▀░ ▀▀▀ ▀▀░   ▀░▀ ▀▀▀ ▀▀░ ▀░ ▀▀▀ ▀░▀")
    print("=" * 70)
    print("Welcome to Mini-RAG: A Simple Retrieval-Augmented Generation System")
    print("=" * 70 + "\n")


def print_config_summary() -> None:
    """Print configuration summary."""
    print("\n" + "─" * 70)
    print("Configuration Summary:")
    print("─" * 70)

    config = get_config_summary()
    for key, value in config.items():
        if isinstance(value, list):
            print(f"  {key:.<40} {len(value)} formats")
            for fmt in value[:5]:
                print(f"    - {fmt}")
            if len(value) > 5:
                print(f"    ... and {len(value) - 5} more")
        else:
            print(f"  {key:.<40} {value}")

    print("─" * 70 + "\n")


def print_quick_start() -> None:
    """Print quick start guide."""
    print("\n" + "─" * 70)
    print("Quick Start Guide:")
    print("─" * 70)
    print("""
1. Install dependencies:
   $ pip install -r requirements.txt

2. Prepare your documents:
   $ Place documents in ./data/docs/
   $ Supported formats: txt, md, rst, pdf, png, jpg, etc.

3. Run tests to verify installation:
   $ pytest tests/ -v

4. Try the CLI:
   $ python -m src.cli ingest ./data/docs
   $ python -m src.cli query "Your question here"

5. View configuration:
   $ python src/config.py

More information:
   - See IMPLEMENTATION_ROADMAP.md for detailed implementation plan
   - See tests/README.md for testing information
   - See README.MD for project overview
""")
    print("─" * 70 + "\n")


def print_implementation_guide() -> None:
    """Print implementation guide."""
    print("\n" + "─" * 70)
    print("Implementation Guide:")
    print("─" * 70)
    print("""
Phase 1: MVP Implementation
  1. src/embeddings.py    - Embedding model and similarity calculation
  2. src/ingest.py        - Document loading and chunking
  3. src/retriever.py     - Document retrieval and indexing
  4. src/rag.py           - Main RAG pipeline (template-based)
  5. src/cli.py           - Command-line interface

Phase 2: File Format Support
  6. Extend ingest.py for PDF, images, and other formats

Phase 3: LLM Integration
  7. Extend rag.py with OpenAI GPT integration

Phase 4: Performance Optimization
  8. Implement advanced caching and indexing

See IMPLEMENTATION_ROADMAP.md for detailed roadmap and API specifications.
""")
    print("─" * 70 + "\n")


def run_demo() -> None:
    """Run a simple demonstration."""
    print("\n" + "─" * 70)
    print("Attempting to Run Demo...")
    print("─" * 70)

    try:
        # Try to import modules (they may not be implemented yet)
        logger.info("Checking module availability...")

        # Test if config can be loaded
        from src.config import EMBEDDING_MODEL, CHUNK_SIZE
        logger.info(f"✓ Configuration loaded successfully")
        logger.info(f"  - Embedding Model: {EMBEDDING_MODEL}")
        logger.info(f"  - Chunk Size: {CHUNK_SIZE}")

        # Try to import embeddings
        try:
            from src.embeddings import EmbeddingManager
            logger.info("✓ Embeddings module is available")

            # Try to initialize
            manager = EmbeddingManager()
            logger.info("✓ EmbeddingManager initialized successfully")

            # Try to embed a sample text
            sample_text = "Pythonは汎用プログラミング言語です。"
            embedding = manager.embed_text(sample_text)
            logger.info(f"✓ Sample embedding generated (dimension: {len(embedding)})")

        except Exception as e:
            logger.warning(f"Embeddings demo failed: {e}")
            logger.info("Modules may not be fully implemented yet")

    except Exception as e:
        logger.warning(f"Demo failed: {e}")
        logger.info("Some modules may not be implemented yet")

    print("─" * 70 + "\n")


def main() -> int:
    """Main entry point."""
    print_welcome()

    # Initialize project
    logger.info("Initializing Mini-RAG project...\n")

    # Check dependencies
    if not check_dependencies():
        logger.error("❌ Dependency check failed")
        print("\n⚠️  Please install missing dependencies:")
        print("   pip install -r requirements.txt\n")
        return 1

    # Check directories
    if not check_directories():
        logger.error("❌ Directory check failed")
        return 1

    # Check configuration
    if not check_configuration():
        logger.error("❌ Configuration check failed")
        return 1

    # Print configuration summary
    print_config_summary()

    # Try to run demo
    run_demo()

    # Print guides
    print_quick_start()
    print_implementation_guide()

    print("=" * 70)
    print("✅ Mini-RAG is ready! Start implementing Phase 1 modules.")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
