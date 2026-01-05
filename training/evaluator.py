from typing import List, Dict

import sacrebleu
from bert_score import score as bert_score


class TranslationEvaluator:

    def __init__(
        self,
        bert_model: str = "xlm-roberta-base",
        device: str = "cpu"
    ):
        self.bert_model = bert_model
        self.device = device

    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        assert len(predictions) == len(references)

        # sacreBLEU expects list of hypotheses and list of reference lists
        refs = [references]

        # BLEU
        bleu = sacrebleu.corpus_bleu(predictions, refs)
        # chrF++
        chrf = sacrebleu.corpus_chrf(
            predictions,
            refs,
            word_order=2  # chrF++
        )
        # BERTScore
        P, R, F1 = bert_score(
            predictions,
            references,
            model_type=self.bert_model,
            device=self.device,
            lang="vi"
        )

        return {
            "bleu": bleu.score,
            "chrf++": chrf.score,
            "bert_score_precision": P.mean().item(),
            "bert_score_recall": R.mean().item(),
            "bert_score_f1": F1.mean().item(),
        }


if __name__ == "__main__":
    
    predictions = ["Tôi đang theo học AI ở trường."]
    references = ["Tôi đang học trí tuệ nhân tạo tại trường đại học."]
    
    evaluator = TranslationEvaluator(device="cpu" )
    scores = evaluator.evaluate(predictions, references)

    for k, v in scores.items():
        print(f"{k}: {v:.4f}")