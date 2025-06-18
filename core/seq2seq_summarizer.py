import logging
import threading
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

logger = logging.getLogger("Seq2SeqSummarizer")
logging.basicConfig(level=logging.INFO)


class Seq2SeqSummarizer:
    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        device: str = "auto",
        max_tokens: int = 1024,
        quantized: bool = False
    ):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.max_tokens = max_tokens
        logger.info(f"Loading model {model_name} on {self.device}")

        if quantized:
            logger.info("Loading quantized model is currently not supported directly in HuggingFace pipelines. Skipping quantization...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

        self.summarizer = pipeline(
            "summarization",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == torch.device("cuda") else -1
        )

    def _resolve_device(self, device: str):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def chunk_text(self, text: str) -> List[str]:
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_len = 0

        for para in paragraphs:
            tokens = self.tokenizer.encode(para, truncation=False, add_special_tokens=False)
            if current_len + len(tokens) > self.max_tokens:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_len = len(tokens)
            else:
                current_chunk.append(para)
                current_len += len(tokens)

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        logger.info(f"Split into {len(chunks)} chunks")
        return chunks

    def summarize_chunk(self, chunk: str) -> str:
        try:
            result = self.summarizer(chunk, max_length=256, min_length=30, do_sample=False)
            return result[0]['summary_text']
        except Exception as e:
            logger.error(f"Error in summarizing chunk: {e}")
            return ""

    def summarize_document(self, text: str, threads: int = 5) -> List[str]:
        chunks = self.chunk_text(text)
        results = [None] * len(chunks)

        def worker(i):
            results[i] = self.summarize_chunk(chunks[i])
            logger.info(f"Finished chunk {i+1}/{len(chunks)}")

        pool = []
        for i in range(len(chunks)):
            t = threading.Thread(target=worker, args=(i,))
            pool.append(t)
            t.start()
            if len(pool) >= threads:
                for t in pool: t.join()
                pool = []
        for t in pool: t.join()

        return results

# LangGraph node interface
def seq2seq_summarize_node(state):
    text = state["text"]
    summarizer = Seq2SeqSummarizer(
        model_name=state.get("model_name", "facebook/bart-large-cnn"),
        device=state.get("device", "auto"),
        quantized=state.get("quantized", False)
    )
    summaries = summarizer.summarize_document(text)
    return {"summaries": summaries, **state}
