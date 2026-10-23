"""
Contains implementations for model scoring and evaluation
"""

from dataset import IGNORE_INDEX


def _safe_div(top: float, bottom: float) -> float:
    """
    Ensures scoring never crashes due to a ZeroDivisionError
    :param top:     Numerator
    :param bottom:  Denominator
    :return:        Either top/bottom or 0.0 if bottom is 0
    """
    if bottom == 0:
        return 0.0
    return top / bottom

def confusion_counts(predictions, masks):
    """
    Computes a dictionary containing counts for true positives, false positives, false negatives, and true negatives
    For this case the sunlit class is treated as positive (class index 1)
    :param predictions: Class IDs, shape [batch_size, height, width]
    :param masks:       Class IDs, shape [batch_size, height, width]
    :return:            Dictionary containing TP, FP, FN, TN counts
    """
    valid = masks != IGNORE_INDEX  # Ignore pixels that were the result of padding
    return {
        "tp": ((predictions == 1) & (masks == 1) & valid).sum().item(),  # The model correctly predicted the positive class
        "fp": ((predictions == 1) & (masks == 0) & valid).sum().item(),  # The model incorrectly predicted the positive class
        "fn": ((predictions == 0) & (masks == 1) & valid).sum().item(),  # The model incorrectly predicted the negative class
        "tn": ((predictions == 0) & (masks == 0) & valid).sum().item(),  # The model correctly predicted the negative class
    }

def scores_from_counts(tp: int, fp: int, fn: int, tn:int):
    """
    Takes the counts (tp, fp, fn, tn) and calculates various metrics to be used in model evaluation
    :param tp:  Count of true positives
    :param fp:  Count of false positives
    :param fn:  Count of false negatives
    :param tn:  Count of true negatives
    :return:    Dictionary containing scores: precision, recall, fpr, f1, sunlit IOU, non-sunlit IOU, mean IOU, accuracy
    """

    # Calculate all the scores (https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers)
    # See _safe_div definition above, we need to guard against ZeroDivisionError
    accuracy =          _safe_div((tp + tn), (tp + fp + fn + tn))
    precision =         _safe_div(tp, (tp + fp))
    recall =            _safe_div(tp, (tp + fn))
    fpr =               _safe_div(fp, (fp + tn))
    f1 =                _safe_div(2 * (precision * recall), (precision + recall))
    sunlit_iou =        _safe_div(tp, (tp + fp + fn))
    non_sunlit_iou =    _safe_div(tn, (tn + fp + fn))
    mean_iou =          _safe_div((sunlit_iou + non_sunlit_iou), 2)

    # Return the scores as a dictionary
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "f1": f1,
        "sunlit_iou": sunlit_iou,
        "non_sunlit_iou": non_sunlit_iou,
        "mean_iou": mean_iou,
    }

