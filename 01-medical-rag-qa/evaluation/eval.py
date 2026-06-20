"""RAG answer-quality metrics: BLEU, ROUGE and supporting-document coverage.

BLEU  - n-gram overlap between generated and reference answer.
ROUGE - recall-oriented overlap (summary-style evaluation).
Coverage - fraction of generated tokens grounded in the retrieved documents.
"""
import logging

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def char_tokens(text):
    return [c for c in text if c.strip()]


def evaluate(reference, generated, documents):
    # Character-level BLEU (Chinese has no whitespace tokenization)
    smoothing = SmoothingFunction()
    bleu = sentence_bleu(
        [list(reference)], list(generated), smoothing_function=smoothing.method1
    )
    rouge1_f = Rouge().get_scores(generated, reference)[0]["rouge-1"]["f"]

    gen_chars = set(char_tokens(generated))
    doc_chars = set(char_tokens("".join(documents)))
    coverage = len(gen_chars & doc_chars) / len(gen_chars)

    return {"bleu": bleu, "rouge1_f": rouge1_f, "coverage": coverage}


if __name__ == "__main__":
    reference = "您的汽车保险可赔偿医疗费用、车辆维修费，以及第三方损害赔偿。"
    generated = "您的保单通常涵盖车祸后的医疗费用、车辆损失，以及对第三方的赔偿。"
    documents = [
        "根据保险条款，医疗费用和车辆损失在车祸理赔中可以获得赔偿。",
        "如果您对第三方造成损害，保险也会提供相应的赔付。",
    ]
    scores = evaluate(reference, generated, documents)
    logger.info(
        "BLEU: %.3f | ROUGE-1 F1: %.3f | coverage: %.2f",
        scores["bleu"],
        scores["rouge1_f"],
        scores["coverage"],
    )
