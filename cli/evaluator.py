# evaluator.py
class SummaryEvaluator:
    
    async def score_summaries(self, summaries: List[str]) -> List[float]:
        return [self.heuristic_score(text) for text in summaries]

    def heuristic_score(self, text: str) -> float:
        if not text or len(text) < 30:
            return 0.2
        if "lorem ipsum" in text.lower():
            return 0.1
        return min(1.0, len(text) / 300)
