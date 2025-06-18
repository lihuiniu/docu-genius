# cli/seq2seq_optimize_runner.py
import argparse
import asyncio
import yaml
from seq2seq.seq2seq_summarizer import Seq2SeqSummarizer

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

async def main():
    parser = argparse.ArgumentParser(description="Seq2Seq Optimize Summarizer CLI")
    parser.add_argument("input_file", help="Text file to summarize")
    parser.add_argument(
        "-c", "--config", default="config/seq2seq_optimize_config.yaml", help="Path to config YAML"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    summarizer = Seq2SeqSummarizer(
        model_name=config.get("model_name", "t5-small"),
        use_gpu=config.get("use_gpu", False),
        quantize=config.get("quantize", False),
        max_tokens=config.get("max_tokens", 512),
    )

    with open(args.input_file, "r", encoding="utf-8") as f:
        text = f.read()

    summary = await summarizer.summarize_document(text)
    print("\n===== Summary =====\n")
    print(summary)

if __name__ == "__main__":
    asyncio.run(main())
