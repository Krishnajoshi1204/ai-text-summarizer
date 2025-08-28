# evaluate.py
from summarizer import rouge_eval

# Example usage: provide lists of references and hypotheses
refs = ["This is the gold summary of document one."]
hyps = ["This is the generated summary for doc one."]
print(rouge_eval(refs, hyps))
