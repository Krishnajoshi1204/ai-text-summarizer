# summarizer.py
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from rouge_score import rouge_scorer
from typing import List, Optional
import math
from tqdm import tqdm

class AbstractiveSummarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn", device: int = -1):
        \"\"\"device=-1 uses CPU; set device=0 for first GPU if available\"\"\"
        self.model_name = model_name
        self.device = device
        # lazy init
        self._pipe = None

    def _ensure_pipeline(self):
        if self._pipe is None:
            # Use pipeline (it will download model if not present)
            self._pipe = pipeline("summarization", model=self.model_name, device=self.device)

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        # naive character-based chunking (keeps words intact where possible)
        text = text.strip()
        if len(text) <= chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            # try to avoid splitting mid-word
            if end < len(text):
                # extend to next space if short
                next_space = text.find(' ', end)
                if next_space != -1 and next_space - start <= chunk_size + 200:
                    end = next_space
                    chunk = text[start:end]
            chunks.append(chunk.strip())
            start = end - overlap
            if start < 0:
                start = 0
            if start >= len(text):
                break
        return chunks

    def summarize(self, text: str, max_length: int = 130, min_length: int = 30) -> str:
        self._ensure_pipeline()
        chunks = self._chunk_text(text)
        if len(chunks) == 1:
            out = self._pipe(text, max_length=max_length, min_length=min_length, truncation=True)
            return out[0]['summary_text'].strip()
        # stage 1: summarize chunks
        chunk_summaries = []
        for chunk in tqdm(chunks, desc="Summarizing chunks"):
            out = self._pipe(chunk, max_length=max_length, min_length=min_length, truncation=True)
            chunk_summaries.append(out[0]['summary_text'].strip())
        # stage 2: summarize concatenated chunk summaries
        concat = ' '.join(chunk_summaries)
        final = self._pipe(concat, max_length=max_length, min_length=min_length, truncation=True)
        return final[0]['summary_text'].strip()


class ExtractiveSummarizer:
    def __init__(self, language: str = 'english'):
        self.language = language
        self.summarizer = LexRankSummarizer()

    def summarize(self, text: str, sentences_count: int = 3) -> str:
        parser = PlaintextParser.from_string(text, Tokenizer(self.language))
        summary_sentences = self.summarizer(parser.document, sentences_count)
        return ' '.join(str(s) for s in summary_sentences)


# Batch helpers

def batch_summarize_abstractive(texts: List[str], model_name: str = "facebook/bart-large-cnn", device: int = -1):
    s = AbstractiveSummarizer(model_name=model_name, device=device)
    results = []
    for txt in texts:
        results.append(s.summarize(txt))
    return results


def batch_summarize_extractive(texts: List[str], sentences_count: int = 3):
    s = ExtractiveSummarizer()
    return [s.summarize(t, sentences_count) for t in texts]


# Evaluation helper (ROUGE)

def rouge_eval(references: List[str], hypotheses: List[str]):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(r, h) for r, h in zip(references, hypotheses)]
    # average
    avg = {}
    for key in ['rouge1', 'rouge2', 'rougeL']:
        avg[key] = {
            'precision': sum(s[key].precision for s in scores) / len(scores),
            'recall': sum(s[key].recall for s in scores) / len(scores),
            'fmeasure': sum(s[key].fmeasure for s in scores) / len(scores),
        }
    return avg


if __name__ == '__main__':
    # Simple quick test
    short = \"\"\"
    The quick brown fox jumps over the lazy dog. This is a sample document used to show summarization. The code is a toy example.\n\"\"\"
    abs_s = AbstractiveSummarizer(model_name='t5-small', device=-1)
    print(abs_s.summarize(short, max_length=40, min_length=5))
    ext_s = ExtractiveSummarizer()
    print(ext_s.summarize(short, sentences_count=2))
