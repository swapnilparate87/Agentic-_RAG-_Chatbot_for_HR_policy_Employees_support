"""Index a directory of HR documents into Pinecone.

Usage: python ingest.py [DIRECTORY]   (defaults to KNOWLEDGE_BASE_DIR)
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from app.core.config import get_settings  # noqa: E402
from app.services.ingestion import SUPPORTED, ingest_files  # noqa: E402

folder = Path(sys.argv[1]) if len(sys.argv) > 1 else get_settings().knowledge_base_dir
files = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED)
if not files:
    sys.exit(f"No supported documents ({', '.join(sorted(SUPPORTED))}) found in {folder}")

chunks = ingest_files(files)
print(f"Indexed {len(files)} files → {chunks} chunks into Pinecone")
