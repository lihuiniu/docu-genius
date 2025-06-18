# File: cli/seq2seq_runner.py

import argparse
import yaml
import logging
from pathlib import Path
from seq2seq.summarizer import Seq2SeqSummarizer

# Logger config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Seq2SeqCLI")

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Run Seq2Seq Summarization via CLI")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--input", type=str, required=True, help="Path to input text file")
    parser.add_argument("--output", type=str, required=True, help="Path to save summarized output")

    args = parser.parse_args()
    config = load_config(args.config)

    # Load text
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return
    text = input_path.read_text(encoding='utf-8')

    summarizer = Seq2SeqSummarizer(
        model_name=config['model_name'],
        device=config.get('device', 'cpu'),
        use_quantization=config.get('use_quantization', False),
        chunk_strategy=config.get('chunk_strategy', 'paragraph')
    )

    results = summarizer.summarize_document(text, threads=config.get('threads', 4))

    output_path = Path(args.output)
    output_path.write_text('\n\n'.join(results), encoding='utf-8')
    logger.info(f"Saved summarized output to: {output_path}")

if __name__ == "__main__":
    main()
