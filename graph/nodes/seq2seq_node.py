# graph/nodes/seq2seq_node.py
import asyncio
from seq2seq.seq2seq_summarizer import Seq2SeqSummarizer

class Seq2SeqSummarizerNode:
    def __init__(self, config: dict):
        model_name = config.get("model_name", "t5-small")
        use_gpu = config.get("use_gpu", False)
        quantize = config.get("quantize", False)
        max_tokens = config.get("max_tokens", 512)
        self.summarizer = Seq2SeqSummarizer(
            model_name=model_name,
            use_gpu=use_gpu,
            quantize=quantize,
            max_tokens=max_tokens,
        )

    async def run(self, input_text: str) -> str:
        summary = await self.summarizer.summarize_document(input_text)
        return summary
