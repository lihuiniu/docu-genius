# Project files are being generated...
# Due to the large scope of this request, the project will include the following modules:
# 1. core/ summarizer with OpenAI + tokenizer + multi-threaded chunking
# 2. vector_store/ Milvus integration + Redis 8.0 VectorSet cache
# 3. api/ FastAPI REST endpoints
# 4. pipeline/ reindexing logic
# 5. seq2seq/ alternative model-based summarizer
# 6. tests/ pytest + mock
# 7. infra/ Dockerfile + Helm charts + K8s YAMLs
# 8. README.md with full architecture

# The full implementation is too large for one step.
# We'll now generate each component step-by-step.

# Step 1: Create the base summarization class with OpenAI SDK v1.x and tokenizer-aware chunking.

# File: core/summarizer.py
from typing import List, Dict
from openai import OpenAI
import tiktoken
import threading
import logging
import time

# Initialize logger
logger = logging.getLogger("OpenAISummarizer")
logging.basicConfig(level=logging.INFO)

class OpenAISummarizer:
    """
     Step 1 complete: Created OpenAISummarizer with multi-threaded chunking
     and token-aware summarization using OpenAI v1 SDK.
    """
    def __init__(self, model: str, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.tokenizer = tiktoken.encoding_for_model(model)

    def chunk_text(self, text: str, max_tokens: int = 2000) -> List[str]:
        paragraphs = text.split('\n\n')
        chunks, current_chunk, current_len = [], [], 0

        for para in paragraphs:
            token_len = len(self.tokenizer.encode(para))
            if current_len + token_len > max_tokens:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_len = token_len
            else:
                current_chunk.append(para)
                current_len += token_len

        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        logger.info(f"Chunked into {len(chunks)} parts")
        return chunks

    def summarize_chunk(self, chunk: str, system_prompt: str = "Summarize the following section:") -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": chunk}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error summarizing chunk: {e}")
            return ""

    def summarize_document(self, text: str, threads: int = 5) -> List[str]:
        chunks = self.chunk_text(text)
        results = [None] * len(chunks)

        def worker(i):
            results[i] = self.summarize_chunk(chunks[i])
            logger.info(f"Finished summary for chunk {i+1}/{len(chunks)}")

        thread_pool = []
        for i in range(len(chunks)):
            t = threading.Thread(target=worker, args=(i,))
            thread_pool.append(t)
            t.start()
            if len(thread_pool) >= threads:
                for t in thread_pool: t.join()
                thread_pool = []

        for t in thread_pool: t.join()  # Final flush
        return results