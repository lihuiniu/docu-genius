from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger("LangchainSummarizer")

class LangchainSummarizer:
    """
    LLM-based summarizer using LangChain's LLMChain and ChatPromptTemplate.
    Uses RecursiveCharacterTextSplitter to chunk documents.
    """
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: str = None):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.3,
            api_key=api_key,
        )

        self.prompt = ChatPromptTemplate.from_template(
            "Summarize the following section:\n\n{chunk}"
        )

        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def chunk_text(self, text: str) -> List[str]:
        try:
            chunks = self.splitter.split_text(text)
            logger.info(f"Split into {len(chunks)} chunks using LangChain splitter")
            return chunks
        except Exception as e:
            logger.warning(f"Failed to chunk text: {e}")
            return [text]

    def summarize_chunk(self, chunk: str) -> str:
        try:
            response = self.chain.run({"chunk": chunk})
            return response.strip()
        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            return ""

    def summarize_document(self, text: str) -> List[str]:
        chunks = self.chunk_text(text)
        summaries = []

        for i, chunk in enumerate(chunks):
            summary = self.summarize_chunk(chunk)
            logger.info(f"Chunk {i+1}/{len(chunks)} summarized.")
            summaries.append(summary)

        return summaries

# Example Usage (in CLI or LangGraph)
summarizer = LangchainSummarizer(model_name="gpt-4", api_key="sk-...")
text = "..."  # Large input text
summary_chunks = summarizer.summarize_document(text)

for i, s in enumerate(summary_chunks):
    print(f"--- Chunk {i+1} ---\n{s}\n")