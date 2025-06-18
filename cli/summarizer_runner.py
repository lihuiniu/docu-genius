import argparse
from core.summarizer_factory import SummarizerFactory

parser = argparse.ArgumentParser()
parser.add_argument("--engine", choices=["openai", "langchain", "seq2seq"], required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--api-key", type=str, default=None)
parser.add_argument("--file", type=str, required=True)

args = parser.parse_args()

with open(args.file, "r", encoding="utf-8") as f:
    text = f.read()

summarizer = SummarizerFactory(args.engine, args.model, args.api_key)
summaries = summarizer.summarize_document(text)

for i, s in enumerate(summaries):
    print(f"\n--- Chunk {i+1} ---\n{s}")
