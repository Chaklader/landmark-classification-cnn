"""
The MIT License

Copyright (c) 2019, Pavel Yakubovskiy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

(taken from https://github.com/qubvel/segmentation_models.pytorch)

Various metrics based on Type I and Type II errors.

References:
    https://en.wikipedia.org/wiki/Confusion_matrix


Example:

    .. code-block:: python

        import segmentation_models_pytorch as smp

        # lets assume we have multilabel prediction for 3 classes
        output = torch.rand([10, 3, 256, 256])
        target = torch.rand([10, 3, 256, 256]).round().long()

        # first compute statistics for true positives, false positives, false negative and
        # true negative "pixels"
        tp, fp, fn, tn = smp.metrics.get_stats(output, target, mode='multilabel', threshold=0.5)

        # then compute metrics with required reduction (see metric docs)
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")

"""
import torch
import warnings
from typing import Optional, List, Tuple, Union


__all__ = [
    "get_stats",
    "fbeta_score",
    "f1_score",
    "iou_score",
    "accuracy",
    "precision",
    "recall",
    "sensitivity",
    "specificity",
    "balanced_accuracy",
    "positive_predictive_value",
    "negative_predictive_value",
    "false_negative_rate",
    "false_positive_rate",
    "false_discovery_rate",
    "false_omission_rate",
    "positive_likelihood_ratio",
    "negative_likelihood_ratio",
]


###################################################################################################
# Statistics computation (true positives, false positives, false negatives, false positives)
###################################################################################################


def get_stats(
    output: Union[torch.LongTensor, torch.FloatTensor],
    target: torch.LongTensor,
    mode: str,
    ignore_index: Optional[int] = None,
    threshold: Optional[Union[float, List[float]]] = None,
    num_classes: Optional[int] = None,
) -> Tuple[torch.LongTensor]:
    """Compute true positive, false positive, false negative, true negative 'pixels'
    for each image and each class.

    Args:
        output (Union[torch.LongTensor, torch.FloatTensor]): Model output with following
            shapes and types depending on the specified ``mode``:

            'binary'
                shape (N, 1, ...) and ``torch.LongTensor`` or ``torch.FloatTensor``

            'multilabel'
                shape (N, C, ...) and ``torch.LongTensor`` or ``torch.FloatTensor``

            'multiclass'
                shape (N, ...) and ``torch.LongTensor``

        target (torch.LongTensor): Targets with following shapes depending on the specified ``mode``:

            'binary'
                shape (N, 1, ...)

            'multilabel'
                shape (N, C, ...)

            'multiclass'
                shape (N, ...)

        mode (str): One of ``'binary'`` | ``'multilabel'`` | ``'multiclass'``
        ignore_index (Optional[int]): Label to ignore on for metric computation.
            **Not** supproted for ``'binary'`` and ``'multilabel'`` modes.  Defaults to None.
        threshold (Optional[float, List[float]]): Binarization threshold for
            ``output`` in case of ``'binary'`` or ``'multilabel'`` modes. Defaults to None.
        num_classes (Optional[int]): Number of classes, necessary attribute
            only for ``'multiclass'`` mode. Class values should be in range 0..(num_classes - 1).
            If ``ignore_index`` is specified it should be outside the classes range, e.g. ``-1`` or
            ``255``.

    Raises:
        ValueError: in case of misconfiguration.

    Returns:
        Tuple[torch.LongTensor]: true_positive, false_positive, false_negative,
            true_negative tensors (N, C) shape each.

    """

    if torch.is_floating_point(target):
        raise ValueError(f"Target should be one of the integer types, got {target.dtype}.")

    if torch.is_floating_point(output) and threshold is None:
        raise ValueError(
            f"Output should be one of the integer types if ``threshold`` is not None, got {output.dtype}."
        )

    if torch.is_floating_point(output) and mode == "multiclass":
        raise ValueError(f"For ``multiclass`` mode ``target`` should be one of the integer types, got {output.dtype}.")

    if mode not in {"binary", "multiclass", "multilabel"}:
        raise ValueError(f"``mode`` should be in ['binary', 'multiclass', 'multilabel'], got mode={mode}.")

    if mode == "multiclass" and threshold is not None:
        raise ValueError("``threshold`` parameter does not supported for this 'multiclass' mode")

    if output.shape != target.shape:
        raise ValueError(
            "Dimensions should match, but ``output`` shape is not equal to ``target`` "
            + f"shape, {output.shape} != {target.shape}"
        )

    if mode != "multiclass" and ignore_index is not None:
        raise ValueError(f"``ignore_index`` parameter is not supproted for '{mode}' mode")

    if mode == "multiclass" and num_classes is None:
        raise ValueError("``num_classes`` attribute should be not ``None`` for 'multiclass' mode.")

    if ignore_index is not None and 0 <= ignore_index <= num_classes - 1:
        raise ValueError(
            f"``ignore_index`` should be outside the class values range, but got class values in range "
            f"0..{num_classes - 1} and ``ignore_index={ignore_index}``. Hint: if you have ``ignore_index = 0``"
            f"consirder subtracting ``1`` from your target and model output to make ``ignore_index = -1``"
            f"and relevant class values started from ``0``."
        )

    if mode == "multiclass":
        tp, fp, fn, tn = _get_stats_multiclass(output, target, num_classes, ignore_index)
    else:
        if threshold is not None:
            output = torch.where(output >= threshold, 1, 0)
            target = torch.where(target >= threshold, 1, 0)
        tp, fp, fn, tn = _get_stats_multilabel(output, target)

    return tp, fp, fn, tn


@torch.no_grad()
def _get_stats_multiclass(
    output: torch.LongTensor,
    target: torch.LongTensor,
    num_classes: int,
    ignore_index: Optional[int],
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    """
    Calculate confusion matrix statistics for multiclass classification.
    
    Computes True Positives (TP), False Positives (FP), False Negatives (FN),
    and True Negatives (TN) for each class in a multiclass setting.
    
    Args:
        output (torch.LongTensor): Predicted class labels with shape (batch_size, *dims)
        target (torch.LongTensor): Ground truth class labels with shape (batch_size, *dims)
        num_classes (int): Total number of classes in the classification problem
        ignore_index (Optional[int]): Class index to ignore in metric calculation.
            Pixels/samples with this label are excluded from all statistics.
    
    Returns:
        Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
            - tp_count: True positives per class, shape (batch_size, num_classes)
            - fp_count: False positives per class, shape (batch_size, num_classes)
            - fn_count: False negatives per class, shape (batch_size, num_classes)
            - tn_count: True negatives per class, shape (batch_size, num_classes)
    
    Note:
        Uses histogram-based counting for efficient computation of confusion matrix
        statistics across all classes simultaneously.
    """

    batch_size, *dims = output.shape
    num_elements = torch.prod(torch.tensor(dims)).long()

    if ignore_index is not None:
        ignore = target == ignore_index
        output = torch.where(ignore, -1, output)
        target = torch.where(ignore, -1, target)
        ignore_per_sample = ignore.view(batch_size, -1).sum(1)

    tp_count = torch.zeros(batch_size, num_classes, dtype=torch.long)
    fp_count = torch.zeros(batch_size, num_classes, dtype=torch.long)
    fn_count = torch.zeros(batch_size, num_classes, dtype=torch.long)
    tn_count = torch.zeros(batch_size, num_classes, dtype=torch.long)

    for i in range(batch_size):
        target_i = target[i]
        output_i = output[i]
        mask = output_i == target_i
        matched = torch.where(mask, target_i, -1)
        tp = torch.histc(matched.float(), bins=num_classes, min=0, max=num_classes - 1)
        fp = torch.histc(output_i.float(), bins=num_classes, min=0, max=num_classes - 1) - tp
        fn = torch.histc(target_i.float(), bins=num_classes, min=0, max=num_classes - 1) - tp
        tn = num_elements - tp - fp - fn
        if ignore_index is not None:
            tn = tn - ignore_per_sample[i]
        tp_count[i] = tp.long()
        fp_count[i] = fp.long()
        fn_count[i] = fn.long()
        tn_count[i] = tn.long()

    return tp_count, fp_count, fn_count, tn_count


@torch.no_grad()
def _get_stats_multilabel(
    output: torch.LongTensor,
    target: torch.LongTensor,
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    """
    Calculate confusion matrix statistics for multilabel classification.
    
    Computes TP, FP, FN, and TN for each label in a multilabel setting where
    each sample can belong to multiple classes simultaneously.
    
    Args:
        output (torch.LongTensor): Predicted binary labels with shape 
            (batch_size, num_classes, *dims). Values should be 0 or 1.
        target (torch.LongTensor): Ground truth binary labels with shape
            (batch_size, num_classes, *dims). Values should be 0 or 1.
    
    Returns:
        Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
            - tp: True positives per label, shape (batch_size, num_classes)
            - fp: False positives per label, shape (batch_size, num_classes)
            - fn: False negatives per label, shape (batch_size, num_classes)
            - tn: True negatives per label, shape (batch_size, num_classes)
    
    Note:
        Each label is treated independently, allowing for multiple positive
        labels per sample in multilabel classification scenarios.
    """

    batch_size, num_classes, *dims = target.shape
    output = output.view(batch_size, num_classes, -1)
    target = target.view(batch_size, num_classes, -1)

    tp = (output * target).sum(2)
    fp = output.sum(2) - tp
    fn = target.sum(2) - tp
    tn = torch.prod(torch.tensor(dims)) - (tp + fp + fn)

    return tp, fp, fn, tn


###################################################################################################
# Metrics computation
###################################################################################################


def _handle_zero_division(x, zero_division):
    """
    Handle division by zero in metric calculations.
    
    Replaces NaN values (resulting from zero division) with specified values
    and optionally issues warnings.
    
    Args:
        x (torch.Tensor): Tensor potentially containing NaN values from division by zero
        zero_division (Union[str, float]): How to handle zero division:
            - "warn": Replace NaN with 0 and issue warning
            - float value: Replace NaN with this value
    
    Returns:
        torch.Tensor: Input tensor with NaN values replaced
    
    Note:
        Common in metrics like precision/recall when no positive predictions
        or ground truth labels exist for a class.
    """
    nans = torch.isnan(x)
    if torch.any(nans) and zero_division == "warn":
        warnings.warn("Zero division in metric calculation!")
    value = zero_division if zero_division != "warn" else 0
    value = torch.tensor(value, dtype=x.dtype).to(x.device)
    x = torch.where(nans, value, x)
    return x


def _compute_metric(
    metric_fn,
    tp,
    fp,
    fn,
    tn,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division="warn",
    **metric_kwargs,
) -> float:
    """
    Generic metric computation with various reduction strategies.
    
    Applies a metric function to confusion matrix statistics with different
    aggregation methods across classes and samples.
    
    Args:
        metric_fn (callable): Function that computes metric from (tp, fp, fn, tn)
        tp (torch.Tensor): True positives with shape (batch_size, num_classes)
        fp (torch.Tensor): False positives with shape (batch_size, num_classes)
        fn (torch.Tensor): False negatives with shape (batch_size, num_classes)
        tn (torch.Tensor): True negatives with shape (batch_size, num_classes)
        reduction (Optional[str]): How to aggregate across classes/samples:
            - "micro": Pool all classes together before computing metric
            - "macro": Compute metric per class, then average
            - "weighted": Compute metric per class, then weighted average
            - "micro-imagewise": Compute metric per sample, then average
            - None: Return per-class metrics without aggregation
        class_weights (Optional[List[float]]): Weights for each class in weighted reduction
        zero_division (Union[str, float]): How to handle division by zero
        **metric_kwargs: Additional arguments passed to metric_fn
    
    Returns:
        float or torch.Tensor: Computed metric value(s)
    
    Raises:
        ValueError: If weighted reduction is requested without class_weights
    
    Note:
        Different reduction strategies are useful for different scenarios:
        - Micro: Good for imbalanced datasets, emphasizes frequent classes
        - Macro: Treats all classes equally regardless of frequency
        - Weighted: Balances between micro and macro based on class frequency
    """

    if class_weights is None and reduction is not None and "weighted" in reduction:
        raise ValueError(f"Class weights should be provided for `{reduction}` reduction")

    class_weights = class_weights if class_weights is not None else 1.0
    class_weights = torch.tensor(class_weights).to(tp.device)
    class_weights = class_weights / class_weights.sum()

    if reduction == "micro":
        tp = tp.sum()
        fp = fp.sum()
        fn = fn.sum()
        tn = tn.sum()
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)

    elif reduction == "macro" or reduction == "weighted":
        tp = tp.sum(0)
        fp = fp.sum(0)
        fn = fn.sum(0)
        tn = tn.sum(0)
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)
        score = _handle_zero_division(score, zero_division)
        score = (score * class_weights).mean()

    elif reduction == "micro-imagewise":
        tp = tp.sum(1)
        fp = fp.sum(1)
        fn = fn.sum(1)
        tn = tn.sum(1)
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)
        score = _handle_zero_division(score, zero_division)
        score = score.mean()

    elif reduction == "macro-imagewise" or reduction == "weighted-imagewise":
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)
        score = _handle_zero_division(score, zero_division)
        score = (score.mean(0) * class_weights).mean()

    elif reduction == "none" or reduction is None:
        score = metric_fn(tp, fp, fn, tn, **metric_kwargs)
        score = _handle_zero_division(score, zero_division)

    else:
        raise ValueError(
            "`reduction` should be in [micro, macro, weighted, micro-imagewise,"
            + "macro-imagesize, weighted-imagewise, none, None]"
        )

    return score


# Logic for metric computation, all metrics are with the same interface


def _fbeta_score(tp, fp, fn, tn, beta=1):
    """
    Compute F-beta score from confusion matrix statistics.
    
    F-beta score is the weighted harmonic mean of precision and recall,
    where beta controls the relative importance of recall vs precision.
    
    Args:
        tp (torch.Tensor): True positives
        fp (torch.Tensor): False positives  
        fn (torch.Tensor): False negatives
        tn (torch.Tensor): True negatives (unused in calculation)
        beta (float): Weight of recall in harmonic mean. beta=1 gives F1 score,
                     beta<1 emphasizes precision, beta>1 emphasizes recall
    
    Returns:
        torch.Tensor: F-beta scores
    
    Formula:
        F_β = (1 + β²) × TP / ((1 + β²) × TP + β² × FN + FP)
    """
    beta_tp = (1 + beta ** 2) * tp
    beta_fn = (beta ** 2) * fn
    score = beta_tp / (beta_tp + beta_fn + fp)
    return score


def _iou_score(tp, fp, fn, tn):
    """
    Compute Intersection over Union (IoU) score, also known as Jaccard index.
    
    IoU measures the overlap between predicted and ground truth regions.
    Commonly used in object detection and segmentation tasks.
    
    Args:
        tp (torch.Tensor): True positives
        fp (torch.Tensor): False positives
        fn (torch.Tensor): False negatives
        tn (torch.Tensor): True negatives (unused in calculation)
    
    Returns:
        torch.Tensor: IoU scores in range [0, 1]
    
    Formula:
        IoU = TP / (TP + FP + FN)
    """
    return tp / (tp + fp + fn)


def _accuracy(tp, fp, fn, tn):
    """
    Compute classification accuracy.
    
    Accuracy is the fraction of predictions that match the ground truth labels.
    
    Args:
        tp (torch.Tensor): True positives
        fp (torch.Tensor): False positives
        fn (torch.Tensor): False negatives
        tn (torch.Tensor): True negatives
    
    Returns:
        torch.Tensor: Accuracy scores in range [0, 1]
    
    Formula:
        Accuracy = (TP + TN) / (TP + FP + FN + TN)
    """
    return (tp + tn) / (tp + fp + fn + tn)


def _sensitivity(tp, fp, fn, tn):
    """
    Compute sensitivity (recall, true positive rate).
    
    Sensitivity measures the proportion of actual positives correctly identified.
    Also known as recall, hit rate, or true positive rate (TPR).
    
    Args:
        tp (torch.Tensor): True positives
        fp (torch.Tensor): False positives (unused in calculation)
        fn (torch.Tensor): False negatives
        tn (torch.Tensor): True negatives (unused in calculation)
    
    Returns:
        torch.Tensor: Sensitivity scores in range [0, 1]
    
    Formula:
        Sensitivity = TP / (TP + FN)
    """
    return tp / (tp + fn)


def _specificity(tp, fp, fn, tn):
    """
    Compute specificity (true negative rate).
    
    Specificity measures the proportion of actual negatives correctly identified.
    Also known as true negative rate (TNR) or selectivity.
    
    Args:
        tp (torch.Tensor): True positives (unused in calculation)
        fp (torch.Tensor): False positives
        fn (torch.Tensor): False negatives (unused in calculation)
        tn (torch.Tensor): True negatives
    
    Returns:
        torch.Tensor: Specificity scores in range [0, 1]
    
    Formula:
        Specificity = TN / (TN + FP)
    """
    return tn / (tn + fp)


def _balanced_accuracy(tp, fp, fn, tn):
    """
    Compute balanced accuracy.
    
    Balanced accuracy is the arithmetic mean of sensitivity and specificity.
    Useful for imbalanced datasets where regular accuracy can be misleading.
    
    Args:
        tp (torch.Tensor): True positives
        fp (torch.Tensor): False positives
        fn (torch.Tensor): False negatives
        tn (torch.Tensor): True negatives
    
    Returns:
        torch.Tensor: Balanced accuracy scores in range [0, 1]
    
    Formula:
        Balanced Accuracy = (Sensitivity + Specificity) / 2
    """
    return (_sensitivity(tp, fp, fn, tn) + _specificity(tp, fp, fn, tn)) / 2


def _positive_predictive_value(tp, fp, fn, tn):
    """
    Compute positive predictive value (precision).
    
    PPV measures the proportion of positive predictions that are actually correct.
    Also known as precision.
    
    Args:
        tp (torch.Tensor): True positives
        fp (torch.Tensor): False positives
        fn (torch.Tensor): False negatives (unused in calculation)
        tn (torch.Tensor): True negatives (unused in calculation)
    
    Returns:
        torch.Tensor: PPV scores in range [0, 1]
    
    Formula:
        PPV = TP / (TP + FP)
    """
    return tp / (tp + fp)


def _negative_predictive_value(tp, fp, fn, tn):
    """
    Compute negative predictive value.
    
    NPV measures the proportion of negative predictions that are actually correct.
    
    Args:
        tp (torch.Tensor): True positives (unused in calculation)
        fp (torch.Tensor): False positives (unused in calculation)
        fn (torch.Tensor): False negatives
        tn (torch.Tensor): True negatives
    
    Returns:
        torch.Tensor: NPV scores in range [0, 1]
    
    Formula:
        NPV = TN / (TN + FN)
    """
    return tn / (tn + fn)


def _false_negative_rate(tp, fp, fn, tn):
    """
    Compute false negative rate (miss rate).
    
    FNR measures the proportion of actual positives that were incorrectly
    classified as negative. Also known as miss rate.
    
    Args:
        tp (torch.Tensor): True positives
        fp (torch.Tensor): False positives (unused in calculation)
        fn (torch.Tensor): False negatives
        tn (torch.Tensor): True negatives (unused in calculation)
    
    Returns:
        torch.Tensor: FNR scores in range [0, 1]
    
    Formula:
        FNR = FN / (FN + TP) = 1 - Sensitivity
    """
    return fn / (fn + tp)


def _false_positive_rate(tp, fp, fn, tn):
    """
    Compute false positive rate (fall-out).
    
    FPR measures the proportion of actual negatives that were incorrectly
    classified as positive. Also known as fall-out or false alarm rate.
    
    Args:
        tp (torch.Tensor): True positives (unused in calculation)
        fp (torch.Tensor): False positives
        fn (torch.Tensor): False negatives (unused in calculation)
        tn (torch.Tensor): True negatives
    
    Returns:
        torch.Tensor: FPR scores in range [0, 1]
    
    Formula:
        FPR = FP / (FP + TN) = 1 - Specificity
    """
    return fp / (fp + tn)


def _false_discovery_rate(tp, fp, fn, tn):
    """
    Compute false discovery rate.
    
    FDR measures the proportion of positive predictions that are actually incorrect.
    
    Args:
        tp (torch.Tensor): True positives
        fp (torch.Tensor): False positives
        fn (torch.Tensor): False negatives (unused in calculation)
        tn (torch.Tensor): True negatives (unused in calculation)
    
    Returns:
        torch.Tensor: FDR scores in range [0, 1]
    
    Formula:
        FDR = FP / (FP + TP) = 1 - Precision
    """
    return 1 - _positive_predictive_value(tp, fp, fn, tn)


def _false_omission_rate(tp, fp, fn, tn):
    """
    Compute false omission rate.
    
    FOR measures the proportion of negative predictions that are actually incorrect.
    
    Args:
        tp (torch.Tensor): True positives (unused in calculation)
        fp (torch.Tensor): False positives (unused in calculation)
        fn (torch.Tensor): False negatives
        tn (torch.Tensor): True negatives
    
    Returns:
        torch.Tensor: FOR scores in range [0, 1]
    
    Formula:
        FOR = FN / (FN + TN) = 1 - NPV
    """
    return 1 - _negative_predictive_value(tp, fp, fn, tn)


def _positive_likelihood_ratio(tp, fp, fn, tn):
    """
    Compute positive likelihood ratio (LR+).
    
    LR+ indicates how much more likely a positive test result is in patients
    with the condition compared to those without.
    
    Args:
        tp (torch.Tensor): True positives
        fp (torch.Tensor): False positives
        fn (torch.Tensor): False negatives
        tn (torch.Tensor): True negatives
    
    Returns:
        torch.Tensor: LR+ scores (range [1, ∞] for good tests)
    
    Formula:
        LR+ = Sensitivity / (1 - Specificity) = TPR / FPR
    """
    return _sensitivity(tp, fp, fn, tn) / _false_positive_rate(tp, fp, fn, tn)


def _negative_likelihood_ratio(tp, fp, fn, tn):
    """
    Compute negative likelihood ratio (LR-).
    
    LR- indicates how much less likely a negative test result is in patients
    with the condition compared to those without.
    
    Args:
        tp (torch.Tensor): True positives
        fp (torch.Tensor): False positives
        fn (torch.Tensor): False negatives
        tn (torch.Tensor): True negatives
    
    Returns:
        torch.Tensor: LR- scores (range [0, 1] for good tests)
    
    Formula:
        LR- = (1 - Sensitivity) / Specificity = FNR / TNR
    """
    return _false_negative_rate(tp, fp, fn, tn) / _specificity(tp, fp, fn, tn)


def fbeta_score(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    beta: float = 1.0,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """
    Compute F-beta score with various aggregation strategies.
    
    F-beta score is the weighted harmonic mean of precision and recall,
    where beta parameter controls the relative importance of recall vs precision.
    
    Args:
        tp (torch.LongTensor): True positive counts, shape (N, C)
        fp (torch.LongTensor): False positive counts, shape (N, C)
        fn (torch.LongTensor): False negative counts, shape (N, C)
        tn (torch.LongTensor): True negative counts, shape (N, C)
        beta (float): Weight of recall in harmonic mean. Default 1.0 (F1 score)
        reduction (Optional[str]): Aggregation method across classes/samples
        class_weights (Optional[List[float]]): Weights for weighted reduction
        zero_division (Union[str, float]): Value for division by zero cases
    
    Returns:
        torch.Tensor: F-beta scores
    
    Note:
        - beta=1: F1 score (equal weight to precision and recall)
        - beta<1: Emphasizes precision over recall
        - beta>1: Emphasizes recall over precision
    """
    return _compute_metric(
        _fbeta_score,
        tp,
        fp,
        fn,
        tn,
        beta=beta,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def f1_score(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """
    Compute F1 score (harmonic mean of precision and recall).
    
    F1 score is a special case of F-beta score with beta=1, giving equal
    weight to precision and recall. Commonly used for binary classification.
    
    Args:
        tp (torch.LongTensor): True positive counts, shape (N, C)
        fp (torch.LongTensor): False positive counts, shape (N, C)
        fn (torch.LongTensor): False negative counts, shape (N, C)
        tn (torch.LongTensor): True negative counts, shape (N, C)
        reduction (Optional[str]): Aggregation method across classes/samples
        class_weights (Optional[List[float]]): Weights for weighted reduction
        zero_division (Union[str, float]): Value for division by zero cases
    
    Returns:
        torch.Tensor: F1 scores in range [0, 1]
    
    Formula:
        F1 = 2 * (precision * recall) / (precision + recall)
    """
    return _compute_metric(
        _fbeta_score,
        tp,
        fp,
        fn,
        tn,
        beta=1.0,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def iou_score(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """
    Compute Intersection over Union (IoU) score, also known as Jaccard index.
    
    IoU measures the overlap between predicted and ground truth regions.
    Widely used in object detection, segmentation, and computer vision tasks.
    
    Args:
        tp (torch.LongTensor): True positive counts, shape (N, C)
        fp (torch.LongTensor): False positive counts, shape (N, C)
        fn (torch.LongTensor): False negative counts, shape (N, C)
        tn (torch.LongTensor): True negative counts, shape (N, C)
        reduction (Optional[str]): Aggregation method across classes/samples
        class_weights (Optional[List[float]]): Weights for weighted reduction
        zero_division (Union[str, float]): Value for division by zero cases
    
    Returns:
        torch.Tensor: IoU scores in range [0, 1]
    
    Note:
        IoU = 1 indicates perfect overlap, IoU = 0 indicates no overlap.
        Commonly used threshold is IoU > 0.5 for positive detection.
    """
    return _compute_metric(
        _iou_score,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def accuracy(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """
    Compute classification accuracy.
    
    Accuracy measures the fraction of predictions that match ground truth labels.
    Simple and intuitive metric, but can be misleading for imbalanced datasets.
    
    Args:
        tp (torch.LongTensor): True positive counts, shape (N, C)
        fp (torch.LongTensor): False positive counts, shape (N, C)
        fn (torch.LongTensor): False negative counts, shape (N, C)
        tn (torch.LongTensor): True negative counts, shape (N, C)
        reduction (Optional[str]): Aggregation method across classes/samples
        class_weights (Optional[List[float]]): Weights for weighted reduction
        zero_division (Union[str, float]): Value for division by zero cases
    
    Returns:
        torch.Tensor: Accuracy scores in range [0, 1]
    
    Note:
        For imbalanced datasets, consider using balanced_accuracy,
        precision, recall, or F1 score instead.
    """
    return _compute_metric(
        _accuracy,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def sensitivity(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """
    Compute sensitivity (recall, true positive rate).
    
    Sensitivity measures the proportion of actual positives correctly identified.
    Critical metric when missing positive cases has high cost (e.g., medical diagnosis).
    
    Args:
        tp (torch.LongTensor): True positive counts, shape (N, C)
        fp (torch.LongTensor): False positive counts, shape (N, C)
        fn (torch.LongTensor): False negative counts, shape (N, C)
        tn (torch.LongTensor): True negative counts, shape (N, C)
        reduction (Optional[str]): Aggregation method across classes/samples
        class_weights (Optional[List[float]]): Weights for weighted reduction
        zero_division (Union[str, float]): Value for division by zero cases
    
    Returns:
        torch.Tensor: Sensitivity scores in range [0, 1]
    
    Aliases:
        Also known as recall, hit rate, or true positive rate (TPR).
    """
    return _compute_metric(
        _sensitivity,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def specificity(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """
    Compute specificity (true negative rate, selectivity).
    
    Specificity measures the proportion of actual negatives correctly identified.
    Critical metric when avoiding false positives is important (e.g., spam detection).
    
    Args:
        tp (torch.LongTensor): True positive counts, shape (N, C)
        fp (torch.LongTensor): False positive counts, shape (N, C)
        fn (torch.LongTensor): False negative counts, shape (N, C)
        tn (torch.LongTensor): True negative counts, shape (N, C)
        reduction (Optional[str]): Aggregation method across classes/samples
        class_weights (Optional[List[float]]): Weights for weighted reduction
        zero_division (Union[str, float]): Value for division by zero cases
    
    Returns:
        torch.Tensor: Specificity scores in range [0, 1]
    
    Aliases:
        Also known as true negative rate (TNR) or selectivity.
    """
    return _compute_metric(
        _specificity,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def balanced_accuracy(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """
    Compute balanced accuracy (arithmetic mean of sensitivity and specificity).
    
    Balanced accuracy is particularly useful for imbalanced datasets where
    regular accuracy can be misleading due to class distribution skew.
    
    Args:
        tp (torch.LongTensor): True positive counts, shape (N, C)
        fp (torch.LongTensor): False positive counts, shape (N, C)
        fn (torch.LongTensor): False negative counts, shape (N, C)
        tn (torch.LongTensor): True negative counts, shape (N, C)
        reduction (Optional[str]): Aggregation method across classes/samples
        class_weights (Optional[List[float]]): Weights for weighted reduction
        zero_division (Union[str, float]): Value for division by zero cases
    
    Returns:
        torch.Tensor: Balanced accuracy scores in range [0, 1]
    
    Note:
        Balanced accuracy = (Sensitivity + Specificity) / 2
        Ranges from 0.5 (random classifier) to 1.0 (perfect classifier).
    """
    return _compute_metric(
        _balanced_accuracy,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def positive_predictive_value(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """
    Compute positive predictive value (precision).
    
    PPV measures the proportion of positive predictions that are actually correct.
    Important when the cost of false positives is high.
    
    Args:
        tp (torch.LongTensor): True positive counts, shape (N, C)
        fp (torch.LongTensor): False positive counts, shape (N, C)
        fn (torch.LongTensor): False negative counts, shape (N, C)
        tn (torch.LongTensor): True negative counts, shape (N, C)
        reduction (Optional[str]): Aggregation method across classes/samples
        class_weights (Optional[List[float]]): Weights for weighted reduction
        zero_division (Union[str, float]): Value for division by zero cases
    
    Returns:
        torch.Tensor: PPV scores in range [0, 1]
    
    Aliases:
        Also known as precision. Commonly used with recall in F1 score.
    """
    return _compute_metric(
        _positive_predictive_value,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def negative_predictive_value(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """
    Compute negative predictive value.
    
    NPV measures the proportion of negative predictions that are actually correct.
    Useful for assessing the reliability of negative test results.
    
    Args:
        tp (torch.LongTensor): True positive counts, shape (N, C)
        fp (torch.LongTensor): False positive counts, shape (N, C)
        fn (torch.LongTensor): False negative counts, shape (N, C)
        tn (torch.LongTensor): True negative counts, shape (N, C)
        reduction (Optional[str]): Aggregation method across classes/samples
        class_weights (Optional[List[float]]): Weights for weighted reduction
        zero_division (Union[str, float]): Value for division by zero cases
    
    Returns:
        torch.Tensor: NPV scores in range [0, 1]
    
    Note:
        High NPV indicates that negative predictions are reliable.
        Complement to positive predictive value (precision).
    """
    return _compute_metric(
        _negative_predictive_value,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def false_negative_rate(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """
    Compute false negative rate (miss rate).
    
    FNR measures the proportion of actual positives that were incorrectly
    classified as negative. Critical in medical diagnosis and safety applications.
    
    Args:
        tp (torch.LongTensor): True positive counts, shape (N, C)
        fp (torch.LongTensor): False positive counts, shape (N, C)
        fn (torch.LongTensor): False negative counts, shape (N, C)
        tn (torch.LongTensor): True negative counts, shape (N, C)
        reduction (Optional[str]): Aggregation method across classes/samples
        class_weights (Optional[List[float]]): Weights for weighted reduction
        zero_division (Union[str, float]): Value for division by zero cases
    
    Returns:
        torch.Tensor: FNR scores in range [0, 1]
    
    Note:
        FNR = 1 - Sensitivity (recall). Lower values indicate better performance.
        Also known as miss rate or Type II error rate.
    """
    return _compute_metric(
        _false_negative_rate,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def false_positive_rate(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """
    Compute false positive rate (fall-out, false alarm rate).
    
    FPR measures the proportion of actual negatives that were incorrectly
    classified as positive. Important in spam detection and security systems.
    
    Args:
        tp (torch.LongTensor): True positive counts, shape (N, C)
        fp (torch.LongTensor): False positive counts, shape (N, C)
        fn (torch.LongTensor): False negative counts, shape (N, C)
        tn (torch.LongTensor): True negative counts, shape (N, C)
        reduction (Optional[str]): Aggregation method across classes/samples
        class_weights (Optional[List[float]]): Weights for weighted reduction
        zero_division (Union[str, float]): Value for division by zero cases
    
    Returns:
        torch.Tensor: FPR scores in range [0, 1]
    
    Note:
        FPR = 1 - Specificity. Lower values indicate better performance.
        Used in ROC curve analysis (x-axis) paired with TPR (y-axis).
    """
    return _compute_metric(
        _false_positive_rate,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def false_discovery_rate(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """
    Compute false discovery rate.
    
    FDR measures the proportion of positive predictions that are actually incorrect.
    Important in multiple hypothesis testing and information retrieval.
    
    Args:
        tp (torch.LongTensor): True positive counts, shape (N, C)
        fp (torch.LongTensor): False positive counts, shape (N, C)
        fn (torch.LongTensor): False negative counts, shape (N, C)
        tn (torch.LongTensor): True negative counts, shape (N, C)
        reduction (Optional[str]): Aggregation method across classes/samples
        class_weights (Optional[List[float]]): Weights for weighted reduction
        zero_division (Union[str, float]): Value for division by zero cases
    
    Returns:
        torch.Tensor: FDR scores in range [0, 1]
    
    Note:
        FDR = 1 - Precision (PPV). Lower values indicate better performance.
        Commonly controlled in statistical multiple testing procedures.
    """
    return _compute_metric(
        _false_discovery_rate,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def false_omission_rate(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """
    Compute false omission rate.
    
    FOR measures the proportion of negative predictions that are actually incorrect.
    Useful for assessing the reliability of negative screening results.
    
    Args:
        tp (torch.LongTensor): True positive counts, shape (N, C)
        fp (torch.LongTensor): False positive counts, shape (N, C)
        fn (torch.LongTensor): False negative counts, shape (N, C)
        tn (torch.LongTensor): True negative counts, shape (N, C)
        reduction (Optional[str]): Aggregation method across classes/samples
        class_weights (Optional[List[float]]): Weights for weighted reduction
        zero_division (Union[str, float]): Value for division by zero cases
    
    Returns:
        torch.Tensor: FOR scores in range [0, 1]
    
    Note:
        FOR = 1 - NPV. Lower values indicate better performance.
        Complement to false discovery rate (FDR).
    """
    return _compute_metric(
        _false_omission_rate,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def positive_likelihood_ratio(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """
    Compute positive likelihood ratio (LR+).
    
    LR+ indicates how much more likely a positive test result is in subjects
    with the condition compared to those without. Used in diagnostic testing.
    
    Args:
        tp (torch.LongTensor): True positive counts, shape (N, C)
        fp (torch.LongTensor): False positive counts, shape (N, C)
        fn (torch.LongTensor): False negative counts, shape (N, C)
        tn (torch.LongTensor): True negative counts, shape (N, C)
        reduction (Optional[str]): Aggregation method across classes/samples
        class_weights (Optional[List[float]]): Weights for weighted reduction
        zero_division (Union[str, float]): Value for division by zero cases
    
    Returns:
        torch.Tensor: LR+ scores (range [1, ∞] for useful tests)
    
    Note:
        LR+ = Sensitivity / (1 - Specificity) = TPR / FPR
        Values > 10 indicate strong evidence for positive diagnosis.
    """
    return _compute_metric(
        _positive_likelihood_ratio,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


def negative_likelihood_ratio(
    tp: torch.LongTensor,
    fp: torch.LongTensor,
    fn: torch.LongTensor,
    tn: torch.LongTensor,
    reduction: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
    zero_division: Union[str, float] = 1.0,
) -> torch.Tensor:
    """
    Compute negative likelihood ratio (LR-).
    
    LR- indicates how much less likely a negative test result is in subjects
    with the condition compared to those without. Used in diagnostic testing.
    
    Args:
        tp (torch.LongTensor): True positive counts, shape (N, C)
        fp (torch.LongTensor): False positive counts, shape (N, C)
        fn (torch.LongTensor): False negative counts, shape (N, C)
        tn (torch.LongTensor): True negative counts, shape (N, C)
        reduction (Optional[str]): Aggregation method across classes/samples
        class_weights (Optional[List[float]]): Weights for weighted reduction
        zero_division (Union[str, float]): Value for division by zero cases
    
    Returns:
        torch.Tensor: LR- scores (range [0, 1] for useful tests)
    
    Note:
        LR- = (1 - Sensitivity) / Specificity = FNR / TNR
        Values < 0.1 indicate strong evidence against positive diagnosis.
    """
    return _compute_metric(
        _negative_likelihood_ratio,
        tp,
        fp,
        fn,
        tn,
        reduction=reduction,
        class_weights=class_weights,
        zero_division=zero_division,
    )


_doc = """
    Common documentation template for all metric functions.

    Args:
        tp (torch.LongTensor): True positive counts with shape (N, C) where:
            - N is the number of samples/images in the batch
            - C is the number of classes
        fp (torch.LongTensor): False positive counts with shape (N, C)
        fn (torch.LongTensor): False negative counts with shape (N, C)  
        tn (torch.LongTensor): True negative counts with shape (N, C)
        reduction (Optional[str]): Aggregation strategy across classes and samples:
            - "micro": Pool all classes together, then compute metric
            - "macro": Compute metric per class, then unweighted average
            - "weighted": Compute metric per class, then weighted average by class frequency
            - "micro-imagewise": Compute metric per sample, then average across samples
            - "macro-imagewise": Compute metric per class per sample, then average
            - "weighted-imagewise": Weighted average per sample, then average across samples
            - None: Return per-class metrics without aggregation
        class_weights (Optional[List[float]]): Class-specific weights for weighted reduction.
            Must be provided when using "weighted" or "weighted-imagewise" reduction.
            Should sum to 1.0 and have length equal to number of classes.
        zero_division (Union[str, float]): Handling strategy for division by zero:
            - "warn": Replace NaN with 0 and issue warning (default for most metrics)
            - float value: Replace NaN with this specific value
            - Common values: 0.0 (conservative), 1.0 (optimistic)

    Returns:
        torch.Tensor: Computed metric values. Shape depends on reduction:
            - With reduction: scalar tensor
            - Without reduction: tensor of shape (C,) with per-class metrics

    Note:
        All metrics are computed from confusion matrix statistics (TP, FP, FN, TN).
        Different reduction strategies are appropriate for different use cases:
        - Use "micro" for overall performance across all classes
        - Use "macro" when all classes are equally important
        - Use "weighted" when class frequency should influence the metric
        - Use imagewise variants for per-sample analysis

    Examples:
        >>> # Binary classification example
        >>> tp = torch.tensor([[10], [8]])  # 2 samples, 1 class
        >>> fp = torch.tensor([[2], [1]])
        >>> fn = torch.tensor([[1], [3]])
        >>> tn = torch.tensor([[5], [6]])
        >>> 
        >>> # Compute F1 score with different reductions
        >>> f1_micro = f1_score(tp, fp, fn, tn, reduction="micro")
        >>> f1_macro = f1_score(tp, fp, fn, tn, reduction="macro")
        >>> f1_per_class = f1_score(tp, fp, fn, tn, reduction=None)
"""

fbeta_score.__doc__ += _doc
f1_score.__doc__ += _doc
iou_score.__doc__ += _doc
accuracy.__doc__ += _doc
sensitivity.__doc__ += _doc
specificity.__doc__ += _doc
balanced_accuracy.__doc__ += _doc
positive_predictive_value.__doc__ += _doc
negative_predictive_value.__doc__ += _doc
false_negative_rate.__doc__ += _doc
false_positive_rate.__doc__ += _doc
false_discovery_rate.__doc__ += _doc
false_omission_rate.__doc__ += _doc
positive_likelihood_ratio.__doc__ += _doc
negative_likelihood_ratio.__doc__ += _doc

precision = positive_predictive_value
recall = sensitivity