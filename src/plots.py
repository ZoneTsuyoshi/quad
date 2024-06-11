import os, pathlib
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics

def plot_precision_recall_curve(logger, result_dir: pathlib.Path, binary_labels: np.ndarray, scores: np.ndarray, name: str):
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(binary_labels, scores)
    f1_score = 2 * precision * recall / (precision + recall)
    best_idx = np.argmax(f1_score)
    logger.log_metrics({f"{name}_{metric}": value[best_idx] for metric, value in zip(["precision", "recall", "f1_score", "threshold"], [precision, recall, f1_score, thresholds])})
    auprc = sklearn.metrics.auc(recall, precision)
    logger.log_metric(f"{name}_auprc", auprc)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect("equal")
    
    fig.savefig(os.path.join(result_dir, f"{name}_precision_recall_curve.pdf"), bbox_inches="tight")
    logger.log_figure(f"{name}_precision_recall_curve", fig)
    plt.close(fig)


def plot_roc_curve(logger, result_dir: pathlib.Path, binary_labels: np.ndarray, scores: np.ndarray, name: str):
    fpr, tpr, _ = sklearn.metrics.roc_curve(binary_labels, scores)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    logger.log_metric(f"{name}_roc_auc", roc_auc)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect("equal")
    
    fig.savefig(os.path.join(result_dir, f"{name}_roc_curve.pdf"), bbox_inches="tight")
    logger.log_figure(f"{name}_roc_curve", fig)
    plt.close(fig)


def plot_period_transition(logger, result_dir: pathlib.Path, p_true: np.ndarray, p_est: np.ndarray):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(p_true, label="True")
    ax.plot(p_est, label="Estimated")
    ax.set_xlabel("Period")
    ax.set_ylabel("Probability")
    ax.legend()
    
    fig.savefig(os.path.join(result_dir, "period_transition.pdf"), bbox_inches="tight")
    logger.log_figure("period_transition", fig)
    plt.close(fig)