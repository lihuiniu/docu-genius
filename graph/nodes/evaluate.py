# graph/nodes/evaluate.py
import logging
from typing import List, Dict, Any
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

logger = logging.getLogger("evaluate")

# Define the prompt template for evaluation
EVALUATION_PROMPT = """
You are an expert evaluator. Given the following document chunk summary, rate its quality on a scale from 1 to 10 and provide a short justification.

Summary:
{summary}

Please respond in JSON format with:
{{
  "score": int,
  "reason": str
}}
"""

prompt = PromptTemplate(
    input_variables=["summary"],
    template=EVALUATION_PROMPT
)

# Initialize the LLMChain
llm = OpenAI(temperature=0)  # or your preferred LLM config
evaluation_chain = LLMChain(llm=llm, prompt=prompt)


async def evaluate_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Evaluate the quality of summaries using an LLMChain.
    Each chunk is a dict with at least a 'summary' key.

    Returns a list of chunk dicts with added 'evaluation' key:
    {
      "score": int,
      "reason": str,
      "needs_resummarize": bool
    }
    """
    results = []
    for chunk in chunks:
        summary = chunk.get("summary", "")
        if not summary:
            logger.warning("Empty summary, marking as low quality")
            evaluation = {"score": 0, "reason": "Empty summary", "needs_resummarize": True}
        else:
            try:
                response = evaluation_chain.run(summary=summary)
                # Expect response JSON string, parse it
                import json
                eval_data = json.loads(response)
                score = eval_data.get("score", 0)
                reason = eval_data.get("reason", "")
                needs_resummarize = score < 5  # threshold for resummarizing

                evaluation = {
                    "score": score,
                    "reason": reason,
                    "needs_resummarize": needs_resummarize,
                }
            except Exception as e:
                logger.error(f"LLM evaluation failed: {e}")
                # fallback: mark as needing resummarize
                evaluation = {"score": 0, "reason": "Evaluation error", "needs_resummarize": True}

        chunk["evaluation"] = evaluation
        results.append(chunk)

    return results
