from core.base import BaseSummarizer
from core.openai_summarizer import OpenAISummarizer
from core.langchain_summarizer import LangchainSummarizer
from core.seq2seq_summarizer import Seq2SeqSummarizer

def SummarizerFactory(engine: str, model: str, api_key: str = None) -> BaseSummarizer:
    engine = engine.lower()

    if engine == "openai":
        return OpenAISummarizer(model=model, api_key=api_key)
    elif engine == "langchain":
        return LangchainSummarizer(model_name=model, api_key=api_key)
    elif engine == "seq2seq":
        return Seq2SeqSummarizer(model_name_or_path=model)
    else:
        raise ValueError(f"Unsupported engine: {engine}")
