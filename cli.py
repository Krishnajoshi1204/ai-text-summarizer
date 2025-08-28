# cli.py
import argparse
from summarizer import AbstractiveSummarizer, ExtractiveSummarizer
from pathlib import Path

parser = argparse.ArgumentParser(description='AI Text Summarizer CLI')
parser.add_argument('--file', '-f', action='append', help='Text file to summarize (can pass multiple)')
parser.add_argument('--text', '-t', help='Raw text to summarize')
parser.add_argument('--mode', choices=['abstractive', 'extractive'], default='abstractive')
parser.add_argument('--model', default='facebook/bart-large-cnn', help='Transformer model name (for abstractive)')
parser.add_argument('--sentences', type=int, default=3, help='Number of sentences for extractive summarizer')
args = parser.parse_args()

texts = []
if args.file:
    for f in args.file:
        p = Path(f)
        texts.append(p.read_text(encoding='utf-8'))
elif args.text:
    texts.append(args.text)
else:
    print('No input provided. Use --file or --text')
    exit(1)

if args.mode == 'abstractive':
    s = AbstractiveSummarizer(model_name=args.model)
    for i, t in enumerate(texts, 1):
        print(f\"--- Summary {i} (abstractive) ---\")
        print(s.summarize(t))
        print()
else:
    s = ExtractiveSummarizer()
    for i, t in enumerate(texts, 1):
        print(f\"--- Summary {i} (extractive) ---\")
        print(s.summarize(t, sentences_count=args.sentences))
        print()
