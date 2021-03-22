from typing import List, Tuple, Dict
import torch
from allennlp.training.metrics import Average
from allennlp.training.metrics import Metric


class HitsAt10(Average):
    def __call__(self, value):
        rank = value

        if rank <= 10:
            self._total_value += 1
        self._count += 1


class HitsAt1(Average):
    def __call__(self, value):
        rank = value

        if rank <= 1:
            self._total_value += 1
        self._count += 1


class HitsAt3(Average):
    def __call__(self, value):
        rank = value

        if rank <= 3:
            self._total_value += 1
        self._count += 1


class F1WithThreshold(Metric):
    def __init__(self, flip_sign: bool = True) -> None:
        super().__init__()
        self._scores: List = []
        self._labels: List = []
        self.flip_sign = flip_sign
        self._threshold = None

    def reset(self) -> None:
        self._scores = []
        self._labels = []

    def __call__(self, scores: torch.Tensor, labels: torch.Tensor) -> None:
        if len(scores.shape) != 1:
            raise ValueError("Scores should be 1D")

        if len(labels.shape) != 1:
            raise ValueError("Labesl should be 1D")

        if scores.shape != labels.shape:
            raise ValueError("Shape of score should be same as labels")
        temp_scores = scores.detach().cpu()

        if self.flip_sign:
            temp_scores = -1 * temp_scores
        current_scores = temp_scores.tolist()
        self._scores.extend(current_scores)
        current_labels = labels.detach().cpu().tolist()
        self._labels.extend(current_labels)

    def compute_best_threshold_and_f1(
            self) -> Tuple[float, float, float, float]:
        # Assumes that lower scores have to be classified as pos

        total_pos = sum(self._labels)
        sorted_scores_and_labels = sorted(zip(self._scores, self._labels))
        true_pos = 0.0
        false_pos = 0.0
        best_thresh = 0.0
        best_f1 = 0.0
        best_precision = 0.0
        best_recall = 0.0

        for score, label in sorted_scores_and_labels:
            true_pos += label
            false_pos += (1.0 - label)
            precision = true_pos / (true_pos + false_pos + 1e-8)
            recall = true_pos / total_pos
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            if f1 > best_f1:
                best_thresh = score
                best_precision = precision
                best_recall = recall
                best_f1 = f1

        if self.flip_sign:
            best_thresh = -1 * best_thresh
        self._sorted_scores_and_labels = sorted_scores_and_labels
        self._threshold = best_thresh

        return best_thresh, best_f1, best_precision, best_recall

    def recompute_f1_using_threshold(self):
        tp = 0.0
        fp = 0.0
        fn = 0.0

        for score, label in zip(self._scores, self._labels):
            pass

    def get_metric(self, reset: bool) -> Dict:
        # this is an expensive operation,
        # lets do it only once when reset is true

        if reset:
            thresh, f1, precision, recall = self.compute_best_threshold_and_f1(
            )
            self.reset()
        else:
            thresh, f1, precision, recall = (0, 0, 0, 0)

        return {
            'threshold': thresh,
            'fscore': f1,
            'precision': precision,
            'recall': recall
        }
