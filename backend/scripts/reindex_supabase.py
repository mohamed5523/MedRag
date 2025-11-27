#!/usr/bin/env python3
"""CLI helper to reindex all Supabase documents into Weaviate."""

import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

# Ensure backend package is importable when running `python backend/scripts/...`
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

load_dotenv(ROOT / ".env")

from app.services.supabase_reindexer import reindex_all_supabase_documents  # noqa: E402


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    await reindex_all_supabase_documents(delay_seconds=0.0)


if __name__ == "__main__":
    asyncio.run(main())


