# AI-Powered Text Summarizer

This repository contains a Python application for summarizing large texts using AI/NLP techniques. It supports both **abstractive** (transformer-based) and **extractive** (LexRank) summarization, batch processing, a small Streamlit web UI, and optional ROUGE evaluation.

---

## Project Structure

- `README.md` — this documentation (also included here).
- `requirements.txt` — Python dependencies.
- `summarizer.py` — core library: abstractive & extractive summarizers, batch helpers, chunking, and evaluation helpers.
- `cli.py` — CLI wrapper to summarize text files or raw text from command line.
- `streamlit_app.py` — lightweight Streamlit web interface.
- `evaluate.py` — simple ROUGE evaluation script.
- `.gitignore`

---

## Quick Notes / Design Decisions

- **Abstractive summarization** uses Hugging Face Transformers (`facebook/bart-large-cnn` by default, with an option for `t5-small` for low-resource setups). The code uses a chunking strategy for long texts and a 2-stage summarization (chunk -> per-chunk summary -> final summary) to handle long documents.
- **Extractive summarization** uses Sumy's LexRank (unsupervised graph-based summarizer) to return the top N sentences.
- **Batch mode**: the CLI and library accept a list of texts/files and summarize them sequentially.
- **Evaluation**: `rouge_score` is used to compute ROUGE-1/2/L between generated and reference summaries (if references provided).
- **Web UI**: Streamlit app for interactive use; upload multiple files or paste text.

---

## Installation

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # on Windows use: venv\\Scripts\\activate
```

2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Download NLTK punkt data (used by Sumy tokenizer):
```bash
python -c "import nltk; nltk.download('punkt')"
```

---

## Usage Examples

### CLI

```bash
# single file abstractive
python cli.py --file article1.txt --mode abstractive --model facebook/bart-large-cnn

# multiple files extractive
python cli.py --file a.txt --file b.txt --mode extractive --sentences 4

# direct text
python cli.py --text "Long article content..." --mode abstractive
```

### Streamlit

```bash
streamlit run streamlit_app.py
```

Then visit the URL shown by Streamlit (usually http://localhost:8501).

---

## Notes, Caveats & Tips

- **Model size:** `facebook/bart-large-cnn` is high-quality but large (~1.6GB). If you have limited RAM or no GPU, prefer `t5-small` or `sshleifer/distilbart-cnn-12-6` to reduce memory and download time.
- **Chunking:** The library uses character-based chunking, which is robust but not token-aware. For production, consider token-based chunking with the model tokenizer to better respect model token limits.
- **Performance:** Downloading models takes time; keep patience the first run. GPU will speed up summarization greatly. To use GPU, install `torch` with CUDA support and set `device=0` when instantiating `AbstractiveSummarizer`.
- **Evaluation:** ROUGE is a rough automatic metric—human evaluation is best for final quality checks.
