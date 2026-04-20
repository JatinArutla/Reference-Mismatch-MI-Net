
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def classification_metrics(y_true, y_pred):
    return {
        'acc': float(accuracy_score(y_true, y_pred) * 100.0),
        'bal_acc': float(balanced_accuracy_score(y_true, y_pred) * 100.0),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro') * 100.0),
    }
