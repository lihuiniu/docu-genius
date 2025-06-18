# seq2seq/seq2seq_summarizer.py
import logging
import asyncio
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

logger = logging.getLogger("Seq2SeqSummarizer")
logging.basicConfig(level=logging.INFO)

class Seq2SeqSummarizer(BaseSummarizer):
    def __init__(
        self,
        model_name: str = "t5-small",
        max_tokens: int = 512,
        use_gpu: bool = False,
        quantize: bool = False,
    ):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model '{model_name}' on device {self.device} (quantize={quantize})")

        # Optional quantize support using bitsandbytes if requested
        if quantize:
            try:
                from bitsandbytes import load_in_8bit_model
                self.model = load_in_8bit_model(model_name).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            except ImportError:
                logger.warning("bitsandbytes not installed; loading full precision model")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.max_tokens = max_tokens

    def chunk_text(self, text: str) -> List[str]:
        tokens = self.tokenizer.encode(text)
        chunks = []
        for i in range(0, len(tokens), self.max_tokens):
            chunk_ids = tokens[i : i + self.max_tokens]
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunks.append(chunk_text)
        logger.info(f"Split text into {len(chunks)} chunks of max {self.max_tokens} tokens")
        return chunks

    async def summarize_chunk(self, chunk: str) -> str:
        inputs = self.tokenizer.encode(
            chunk,
            return_tensors="pt",
            max_length=self.max_tokens,
            truncation=True,
        ).to(self.device)

        try:
            outputs = await asyncio.to_thread(
                lambda: self.model.generate(
                    inputs,
                    max_length=150,
                    num_beams=4,
                    early_stopping=True,
                )
            )
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            return ""

    async def summarize_document(self, text: str) -> str:
        chunks = self.chunk_text(text)
        coros = [self.summarize_chunk(c) for c in chunks]
        summaries = await asyncio.gather(*coros)
        combined_summary = "\n\n".join(filter(None, summaries))
        return combined_summary
