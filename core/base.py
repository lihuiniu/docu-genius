from abc import ABC, abstractmethod
from typing import List

class BaseSummarizer(ABC):
    @abstractmethod
    def chunk_text(self, text: str) -> List[str]:
        pass

    @abstractmethod
    def summarize_chunk(self, chunk: str) -> str:
        pass

    @abstractmethod
    def summarize_document(self, text: str) -> List[str]:
        pass
