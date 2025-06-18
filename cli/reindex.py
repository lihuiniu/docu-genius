import argparse
import asyncio
import logging
from typing import List

from index_manager import IndexingManager
from vector_store.milvus_client import MilvusVectorStore
from vector_store.redis_cache import RedisVectorCache
from storage.s3_client import S3Client  # or AzureBlobClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("batch-reindex")

vector_db = MilvusVectorStore()
cache_db = RedisVectorCache()
storage_client = S3Client()
indexing_manager = IndexingManager(vector_db, cache_db, storage_client)

async def batch_reindex(doc_ids: List[str], full: bool = False, dry_run: bool = False, retry_failed: bool = False):
    failed_docs = []

    for doc_id in doc_ids:
        logger.info(f"\n--- Processing {doc_id} ---")
        try:
            if dry_run:
                version = await indexing_manager.get_document_version(doc_id)
                state = await indexing_manager.get_indexing_state(doc_id)
                logger.info(f"[DRY RUN] {doc_id} => version: {version}, already indexed chunks: {sorted(state.get('chunks_indexed', []))}")
                continue

            if full:
                await indexing_manager.clear_index_for_document(doc_id)

            await indexing_manager.reindex_document(doc_id)
            progress = await indexing_manager.get_indexing_progress(doc_id)
            logger.info(f"✅ {doc_id} reindex complete: {progress}")

        except Exception as e:
            logger.error(f"❌ Failed to reindex {doc_id}: {e}")
            failed_docs.append(doc_id)

    if retry_failed and failed_docs:
        logger.info("\nRetrying failed documents...")
        await batch_reindex(failed_docs, full=full, dry_run=False, retry_failed=False)

    return failed_docs

def load_doc_ids_from_file(path: str) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def parse_args():
    parser = argparse.ArgumentParser(description="Batch Reindex Documents")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--doc-ids", nargs="+", help="List of document IDs to reindex")
    group.add_argument("--doc-file", help="Path to file with one document ID per line")

    parser.add_argument("--full", action="store_true", help="Clear and rebuild index from scratch")
    parser.add_argument("--dry-run", action="store_true", help="Print document index state without performing indexing")
    parser.add_argument("--retry", action="store_true", help="Retry failed documents once")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    docs = args.doc_ids if args.doc_ids else load_doc_ids_from_file(args.doc_file)
    failed = asyncio.run(batch_reindex(docs, full=args.full, dry_run=args.dry_run, retry_failed=args.retry))

    if failed:
        logger.warning(f"\nSome documents failed reindex: {failed}")


# Example usage:
# Reindex one or more docs (resume if previously interrupted)
python cli/reindex.py --doc-ids doc123 doc456

# Full reindex (clear index before rebuilding)
python cli/reindex.py --doc-ids doc123 --full

# From file
python cli/reindex.py --doc-file doc_list.txt

# Where doc_list.txt is:
# doc123
# doc456
# doc789