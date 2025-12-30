"""
Evaluation metrics for football prediction.

Implements RPS (primary), calibration error, accuracy, and log loss.
"""

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, f1_score


def ranked_probability_score(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> float:
    """
    Calculate Ranked Probability Score (RPS).
    
    RPS measures how well-calibrated probability predictions are
    for ordered outcomes. Lower is better.
    
    Formula: RPS = (1/K) * sum((CDF_predicted - CDF_actual)^2)
    
    Args:
        y_true: True class labels (0, 1, 2)
        y_proba: Predicted probabilities, shape (n_samples, 3)
    
    Returns:
        Mean RPS across all samples (lower is better)
    """
    n_samples = len(y_true)
    n_classes = y_proba.shape[1]
    
    # Create one-hot encoding of true labels
    y_onehot = np.zeros((n_samples, n_classes))
    y_onehot[np.arange(n_samples), y_true.astype(int)] = 1
    
    # Calculate cumulative distributions
    cdf_pred = np.cumsum(y_proba, axis=1)
    cdf_true = np.cumsum(y_onehot, axis=1)
    
    # RPS for each sample
    rps_per_sample = np.mean((cdf_pred - cdf_true) ** 2, axis=1)
    
    return float(np.mean(rps_per_sample))


def calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """
    Calculate Expected and Maximum Calibration Error.
    
    Args:
        y_true: True class labels
        y_proba: Predicted probabilities
        n_bins: Number of bins for calibration
    
    Returns:
        Dict with ECE and MCE
    """
    n_samples = len(y_true)
    
    # Get predicted class and confidence
    pred_class = np.argmax(y_proba, axis=1)
    confidence = np.max(y_proba, axis=1)
    correct = (pred_class == y_true).astype(float)
    
    # Bin by confidence
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    
    for i in range(n_bins):
        in_bin = (confidence > bin_edges[i]) & (confidence <= bin_edges[i + 1])
        n_in_bin = in_bin.sum()
        
        if n_in_bin > 0:
            avg_confidence = confidence[in_bin].mean()
            avg_accuracy = correct[in_bin].mean()
            gap = abs(avg_accuracy - avg_confidence)
            
            ece += (n_in_bin / n_samples) * gap
            mce = max(mce, gap)
    
    return {"ece": ece, "mce": mce}


def calculate_all_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """
    Calculate all evaluation metrics.
    
    Returns:
        Dict with RPS, accuracy, log_loss, calibration errors, F1 scores
    """
    y_pred = np.argmax(y_proba, axis=1)
    
    # Core metrics
    rps = ranked_probability_score(y_true, y_proba)
    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, y_proba, labels=[0, 1, 2])
    
    # Calibration
    cal = calibration_error(y_true, y_proba)
    
    # Per-class F1
    f1_home = f1_score(y_true, y_pred, labels=[0], average='macro', zero_division=0)
    f1_draw = f1_score(y_true, y_pred, labels=[1], average='macro', zero_division=0)
    f1_away = f1_score(y_true, y_pred, labels=[2], average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    return {
        "rps": rps,
        "accuracy": acc,
        "log_loss": ll,
        "ece": cal["ece"],
        "mce": cal["mce"],
        "f1_home": f1_home,
        "f1_draw": f1_draw,
        "f1_away": f1_away,
        "f1_macro": f1_macro,
    }
