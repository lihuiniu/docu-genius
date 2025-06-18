# utils.py
import asyncio
from typing import List, Dict, Any
from enum import Enum

class ChunkStrategy(Enum):
    PARAGRAPH = "paragraph"
    FIXED_CHAR = "fixed_char"
    SENTENCE = "sentence"
def chunk_by_paragraph(text: str, max_chunk_size: int = 1000) -> List[str]:
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def chunk_by_char_size(text: str, chunk_size: int = 1000) -> List[str]:
    words = text.split()
    chunks = []
    current_chunk = []

    current_len = 0
    for word in words:
        word_len = len(word) + 1  # space or separator
        if current_len + word_len > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_len = word_len
        else:
            current_chunk.append(word)
            current_len += word_len

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

import re

def split_into_sentences(text: str) -> List[str]:
    # Simple regex-based sentence splitter (imperfect, but works for many cases)
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    return sentences

def chunk_by_sentence(text: str, max_chunk_size: int = 1000) -> List[str]:
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = ""

    for sent in sentences:
        if len(current_chunk) + len(sent) + 1 <= max_chunk_size:
            current_chunk += (sent + " ")
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sent + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

from typing import Callable

def chunk_by_token_count(text: str, max_tokens: int, tokenizer: Callable[[str], List[str]]) -> List[str]:
    tokens = tokenizer(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = " ".join(chunk_tokens)  # or detokenize if needed
        chunks.append(chunk_text)
        start = end
    return chunks
def chunk_by_sentence_with_max_size(text: str, max_chunk_size: int = 1000) -> List[str]:
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = ""

    for sent in sentences:
        if len(current_chunk) + len(sent) + 1 <= max_chunk_size:
            current_chunk += (sent + " ")
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sent + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

async def run_sync_in_threadpool(func, *args, **kwargs):
    """
    Helper to run a synchronous function in thread pool from async code.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

def merge_metadata(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two metadata dicts, with override taking precedence.
    """
    merged = base.copy()
    merged.update(override)
    return merged

def generate_vector_id(document_id: str, chunk_index: int) -> str:
    """
    Generate a unique vector id for a chunk.
    """
    return f"{document_id}_chunk_{chunk_index}"

def chunk_text(text: str,
               strategy: ChunkStrategy = ChunkStrategy.PARAGRAPH,
               max_chunk_size: int = 1000,
               tokenizer: Optional[Callable[[str], List[str]]] = None) -> List[str]:

    if strategy == ChunkStrategy.PARAGRAPH:
        return chunk_by_paragraph(text, max_chunk_size)
    elif strategy == ChunkStrategy.FIXED_CHAR:
        return chunk_by_char_size(text, max_chunk_size)
    elif strategy == ChunkStrategy.SENTENCE:
        return chunk_by_sentence(text, max_chunk_size)
    elif strategy == ChunkStrategy.TOKEN:
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided for TOKEN chunking strategy")
        return chunk_by_token_count(text, max_chunk_size, tokenizer)
    else:
        raise ValueError(f"Unsupported chunking strategy: {strategy}")