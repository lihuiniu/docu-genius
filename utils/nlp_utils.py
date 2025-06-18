# utils/nlp_utils.py
from typing import List
import re
from openai_api import OpenAIClient

def extract_keywords_fallback(text: str, max_keywords: int = 10) -> List[str]:
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    freq = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1
    sorted_keywords = sorted(freq, key=freq.get, reverse=True)
    return sorted_keywords[:max_keywords]

def extract_keywords_llm(text: str, client: OpenAIClient, max_keywords: int = 10) -> List[str]:
    prompt = (
        f"Extract up to {max_keywords} concise, relevant keywords or key phrases from the following text:\n\n{text[:2000]}\n\n"
        "Return them as a comma-separated list."
    )
    response = client.chat(prompt)
    if isinstance(response, str):
        return [kw.strip() for kw in response.split(",") if kw.strip()]
    return []

def extract_keywords(text: str, mode: str = "fallback", client: OpenAIClient = None, max_keywords: int = 10) -> List[str]:
    if mode == "llm" and client:
        try:
            return extract_keywords_llm(text, client, max_keywords)
        except Exception as e:
            print(f"[Keyword Extract LLM fallback]: {e}")
    return extract_keywords_fallback(text, max_keywords)
