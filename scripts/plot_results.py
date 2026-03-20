#!/usr/bin/env python3
"""
Tau Classification Results Analysis

Analyze the output from OmniLearned evaluation.

Primary task (2-class): Tau vs QCD only. Class 0=QCD, 1=Tau. 
Electron jets are excluded from primary metrics.

Auxiliary tasks: 
- decay_mode (7-class: 1p0n, 1p1n, 1pXn, 3p0n, 3pXn, Other, NotSet)
- electron_vs_qcd (QCD vs Electron)
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    classification_report,
    accuracy_score,
)
from pathlib import Path
import argparse

import mplhep as hep
hep.style.use(hep.style.ATLAS)

# plt.rcParams['font.size'] = 12


# ---------------------------------------------------------------------------
# Shared plotting helpers
# ---------------------------------------------------------------------------

def setup_plot(xlabel="", ylabel="", title="", xlim=None, ylim=None):
    """Create a single-canvas figure with common formatting applied.

    Returns (fig, ax). Caller is responsible for adding data and calling
    save_plot() when done.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    return fig, ax


def save_plot(fig, path, dpi=300):
    """Save and close a figure."""
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_path="results/outputs__tau_0.npz"):
    """Load evaluation results from NPZ file."""
    data = np.load(results_path)
    print("Available arrays:", list(data.keys()))

    predictions = data['prediction']  # Shape: [N_jets, 3]
    true_labels = data['pid']         # Shape: [N_jets]

    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Labels shape: {true_labels.shape}")
    print(f"Total jets: {len(true_labels)}")

    return data, predictions, true_labels


# ---------------------------------------------------------------------------
# Primary task
# ---------------------------------------------------------------------------

def analyze_primary_task(predictions, true_labels, class_names):
    """Analyze primary 3-class classification task."""
    print("\n" + "=" * 60)
    print("PRIMARY TASK: Tau vs QCD vs Electron (3-class)")
    print("=" * 60)

    pred_labels = np.argmax(predictions, axis=1)

    print("\nSample distribution:")
    for i, name in enumerate(class_names):
        n_true = np.sum(true_labels == i)
        n_pred = np.sum(pred_labels == i)
        print(f"  {name}: {n_true} true, {n_pred} predicted")

    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    print("\nPer-class Accuracy:")
    for i, name in enumerate(class_names):
        mask = true_labels == i
        if mask.sum() > 0:
            class_acc = (pred_labels[mask] == i).mean()
            print(f"  {name}: {class_acc:.4f} ({class_acc*100:.2f}%)")

    print("\nDetailed Classification Report:")
    print(classification_report(true_labels, pred_labels, target_names=class_names, digits=4))

    return pred_labels, accuracy


# ---------------------------------------------------------------------------
# Confusion matrices — one file per matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(true_labels, pred_labels, class_names, output_dir):
    """Save raw-count and normalised confusion matrices as separate files."""
    print("\nPlotting confusion matrices...")
    cm = confusion_matrix(true_labels, pred_labels)

    # Raw counts
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, cmap='Blues', values_format='d')
    ax.grid(False)
    ax.set_title('Counts')
    save_plot(fig, f'{output_dir}/confusion_matrix_counts.png')

    # Normalised
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm_norm, display_labels=class_names).plot(
        ax=ax, cmap='Blues', values_format='.3f')
    ax.grid(False)
    ax.set_title('Normalised by True Label')
    save_plot(fig, f'{output_dir}/confusion_matrix_normalised.png')


# ---------------------------------------------------------------------------
# ROC curves — one file per class
# ---------------------------------------------------------------------------

def plot_roc_curves(predictions, true_labels, class_names, output_dir):
    """Save one ROC curve per class (one-vs-rest)."""
    print("Plotting ROC curves...")
    colors = ['#e41a1c', '#377eb8', '#4daf4a']

    for i, (name, color) in enumerate(zip(class_names, colors)):
        y_true_binary = (true_labels == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, predictions[:, i])
        roc_auc = auc(fpr, tpr)

        fig, ax = setup_plot(
            xlabel='False Positive Rate',
            ylabel='True Positive Rate',
            title=f'{name} vs Rest',
            xlim=[0.0, 1.0],
            ylim=[0.0, 1.0],
        )
        ax.plot(fpr, tpr, color=color, lw=2, label=f'AUC = {roc_auc:.4f}')
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.legend(loc='lower right')
        save_plot(fig, f'{output_dir}/roc_{name.lower()}_vs_rest.png')


# ---------------------------------------------------------------------------
# Tau vs QCD
# ---------------------------------------------------------------------------

def plot_tau_vs_qcd(predictions, true_labels, output_dir):
    """Save Tau vs QCD ROC curve and background-rejection curve."""
    print("Plotting Tau vs QCD analysis...")
    tau_qcd_mask = (true_labels == 0) | (true_labels == 1)
    tau_qcd_labels = true_labels[tau_qcd_mask]
    tau_qcd_probs = predictions[tau_qcd_mask]

    tau_score = tau_qcd_probs[:, 1]
    is_tau = (tau_qcd_labels == 1).astype(int)
    fpr, tpr, _ = roc_curve(is_tau, tau_score)
    roc_auc = auc(fpr, tpr)

    # ROC curve
    fig, ax = setup_plot(
        xlabel='False Positive Rate',
        ylabel='True Positive Rate',
        title='Tau vs QCD',
        xlim=[0.0, 1.0],
        ylim=[0.0, 1.0],
    )
    ax.plot(fpr, tpr, 'b-', lw=2, label=f'AUC = {roc_auc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.legend(loc='lower right')
    save_plot(fig, f'{output_dir}/tau_vs_qcd_roc.png')

    # Background rejection
    rejection = np.where(fpr > 0, 1.0 / fpr, np.inf)
    valid = rejection < 1e6

    fig, ax = setup_plot(
        xlabel='Tau Efficiency',
        ylabel='QCD Rejection',
        title='Tau vs QCD',
        xlim=[0, 1],
    )
    ax.semilogy(tpr[valid], rejection[valid], 'b-', lw=2)
    save_plot(fig, f'{output_dir}/tau_vs_qcd_rejection.png')

    # Print working points
    print("\nTau vs QCD Working Points:")
    print(f"  AUC: {roc_auc:.4f}")
    print("\n  Tau efficiency -> QCD rejection:")
    for target_eff in [0.5, 0.6, 0.7, 0.8, 0.9]:
        idx = np.argmin(np.abs(tpr - target_eff))
        if fpr[idx] > 0:
            print(f"    {target_eff*100:.0f}% -> {1/fpr[idx]:.1f}x rejection")


# ---------------------------------------------------------------------------
# Score distributions
# ---------------------------------------------------------------------------

def plot_score_distributions(predictions, true_labels, class_names, output_dir):
    """Save one score-distribution plot per class."""
    print("Plotting score distributions...")
    colors = ['#e41a1c', '#377eb8', '#4daf4a']

    for i, name in enumerate(class_names):
        fig, ax = setup_plot(
            xlabel=f'{name} Score',
            ylabel='Density',
            title='',
            xlim=[0, 1],
        )
        for j, (true_name, color) in enumerate(zip(class_names, colors)):
            mask = true_labels == j
            ax.hist(predictions[mask, i], bins=50, alpha=0.5,
                    label=f'True {true_name}', color=color, density=True)
        ax.legend()
        save_plot(fig, f'{output_dir}/score_dist_{name.lower()}.png')


# ---------------------------------------------------------------------------
# Decay mode
# ---------------------------------------------------------------------------

def plot_decay_mode_score_distributions(dm_pred_valid, dm_true_valid, prong_names, output_dir):
    """Save one score-distribution plot per decay-mode class."""
    print("Plotting decay mode score distributions...")
    colors_dm = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (name, color) in enumerate(zip(prong_names, colors_dm)):
        fig, ax = setup_plot(
            xlabel=f'{name} Score',
            ylabel='Density',
            title='',
            xlim=[0, 1],
        )
        for j, (true_name, true_color) in enumerate(zip(prong_names, colors_dm)):
            mask = dm_true_valid == j
            if mask.any():
                ax.hist(dm_pred_valid[mask, i], bins=50, alpha=0.5,
                        label=f'True {true_name}', color=true_color, density=True)
        ax.legend(fontsize=7)
        save_plot(fig, f'{output_dir}/decay_mode_score_dist_{name}.png')


def analyze_decay_mode(data, true_labels, output_dir):
    """Analyze 5-class decay mode predictions."""
    if 'aux_decay_mode_pred' not in data.keys():
        print("\nNo auxiliary decay_mode predictions found.")
        return None

    print("\n" + "=" * 60)
    print("AUXILIARY TASK 1: Decay Mode (5-class)")
    print("=" * 60)

    decay_mode_pred = data['aux_decay_mode_pred']
    decay_mode_true = data['decay_mode']

    tau_mask = true_labels == 1
    valid_mask = tau_mask & (decay_mode_true >= 0) & (decay_mode_true < 5)

    print(f"\nTotal tau jets: {tau_mask.sum()}")
    print(f"Tau jets with valid decay mode (0-4): {valid_mask.sum()}")

    dm_pred_valid = decay_mode_pred[valid_mask]
    dm_true_valid = decay_mode_true[valid_mask]
    dm_pred_labels = np.argmax(dm_pred_valid, axis=1)

    prong_names = ['1p0n', '1p1n', '1pXn', '3p0n', '3pXn']
    n_classes_plot = 5

    print("\nDecay mode distribution:")
    for i, name in enumerate(prong_names):
        print(f"  {name}: {(dm_true_valid == i).sum()}")

    dm_accuracy = accuracy_score(dm_true_valid, dm_pred_labels)
    print(f"\nDecay Mode Accuracy: {dm_accuracy:.4f} ({dm_accuracy*100:.2f}%)")

    print("\nDecay Mode Classification Report:")
    present_labels = sorted(set(dm_true_valid) | set(dm_pred_labels))
    present_names = [prong_names[i] for i in present_labels if i < len(prong_names)]
    print(classification_report(dm_true_valid, dm_pred_labels,
                                labels=present_labels, target_names=present_names, digits=4))

    plot_decay_mode_score_distributions(dm_pred_valid, dm_true_valid, prong_names, output_dir)

    # Confusion matrices
    print("Plotting decay mode confusion matrices...")
    cm_dm = confusion_matrix(dm_true_valid, dm_pred_labels, labels=list(range(n_classes_plot)))

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm_dm, display_labels=prong_names).plot(
        ax=ax, cmap='Oranges', values_format='d')
    ax.grid(False)
    ax.set_title('Counts')
    save_plot(fig, f'{output_dir}/decay_mode_confusion_counts.png')

    row_sums = cm_dm.sum(axis=1)[:, np.newaxis]
    cm_dm_norm = np.divide(cm_dm.astype('float'), row_sums, where=row_sums != 0,
                           out=np.zeros_like(cm_dm, dtype=float))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm_dm_norm, display_labels=prong_names).plot(
        ax=ax, cmap='Oranges', values_format='.3f')
    ax.grid(False)
    ax.set_title('Normalised by Truth')
    save_plot(fig, f'{output_dir}/decay_mode_confusion_normalised.png')

    # ROC curves
    print("Plotting decay mode ROC curves...")
    colors_dm = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    print("\nDecay Mode AUC (one-vs-rest):")

    for i, (name, color) in enumerate(zip(prong_names, colors_dm)):
        y_true_binary = (dm_true_valid == i).astype(int)
        if y_true_binary.sum() == 0 or y_true_binary.sum() == len(y_true_binary):
            print(f"  {name}: N/A (no samples)")
            continue

        fpr_dm, tpr_dm, _ = roc_curve(y_true_binary, dm_pred_valid[:, i])
        auc_dm = auc(fpr_dm, tpr_dm)

        fig, ax = setup_plot(
            xlabel='False Positive Rate',
            ylabel='True Positive Rate',
            title=f'{name}',
            xlim=[0.0, 1.0],
            ylim=[0.0, 1.0],
        )
        ax.plot(fpr_dm, tpr_dm, color=color, lw=2, label=f'AUC = {auc_dm:.4f}')
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.legend(loc='lower right')
        save_plot(fig, f'{output_dir}/decay_mode_roc_{name}.png')

        print(f"  {name}: {auc_dm:.4f}")

    return dm_accuracy


# ---------------------------------------------------------------------------
# Electron vs QCD
# ---------------------------------------------------------------------------

def analyze_electron_vs_qcd(data, true_labels, output_dir):
    """Analyze electron vs QCD auxiliary task."""
    if 'aux_electron_vs_qcd_pred' not in data.keys():
        print("\nNo electron_vs_qcd auxiliary predictions found.")
        return None

    print("\n" + "=" * 60)
    print("AUXILIARY TASK 2: Electron vs QCD")
    print("=" * 60)

    evq_pred = data['aux_electron_vs_qcd_pred']
    evq_mask = (true_labels == 0) | (true_labels == 2)
    evq_pred_valid = evq_pred[evq_mask]
    evq_true = (true_labels[evq_mask] == 2).astype(int)

    print(f"\nElectron vs QCD samples: {evq_mask.sum()}")
    print(f"  QCD jets: {(true_labels[evq_mask] == 0).sum()}")
    print(f"  Electron jets: {(true_labels[evq_mask] == 2).sum()}")

    evq_score = evq_pred_valid[:, 1]
    fpr_evq, tpr_evq, _ = roc_curve(evq_true, evq_score)
    auc_evq = auc(fpr_evq, tpr_evq)

    print("Plotting electron vs QCD analysis...")

    # ROC curve
    fig, ax = setup_plot(
        xlabel='False Positive Rate',
        ylabel='True Positive Rate',
        title='Electron vs QCD',
        xlim=[0.0, 1.0],
        ylim=[0.0, 1.0],
    )
    ax.plot(fpr_evq, tpr_evq, 'green', lw=2, label=f'AUC = {auc_evq:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.legend(loc='lower right')
    save_plot(fig, f'{output_dir}/electron_vs_qcd_roc.png')

    # Score distribution
    fig, ax = setup_plot(
        xlabel='Electron Score',
        ylabel='Density',
        title='Electron vs QCD',
        xlim=[0, 1],
    )
    ax.hist(evq_score[evq_true == 0], bins=50, alpha=0.6,
            label='True QCD', color='red', density=True)
    ax.hist(evq_score[evq_true == 1], bins=50, alpha=0.6,
            label='True Electron', color='green', density=True)
    ax.legend()
    save_plot(fig, f'{output_dir}/electron_vs_qcd_score_dist.png')

    evq_acc = accuracy_score(evq_true, (evq_score > 0.5).astype(int))
    print(f"\nElectron vs QCD Accuracy: {evq_acc:.4f}")
    print(f"Electron vs QCD AUC: {auc_evq:.4f}")

    return auc_evq


# ---------------------------------------------------------------------------
# Regression tasks
# ---------------------------------------------------------------------------

def analyze_regression_task(data, output_dir):
    """Analyze regression auxiliary tasks."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    regression_tasks = {}
    for key in data.keys():
        print(key)
        if key.startswith('aux_') and key.endswith('_pred'):
            task_name = key.replace('aux_', '').replace('_pred', '')
            if data[key].ndim == 1 or data[key].shape[1] == 1:
                regression_tasks[task_name] = data[key].flatten()

    if not regression_tasks:
        print("\nNo regression auxiliary predictions found.")
        return {}

    print("\n" + "=" * 60)
    print("AUXILIARY REGRESSION TASKS")
    print("=" * 60)

    results = {}

    for task_name, predictions in regression_tasks.items():
        truth_key = f"aux_{task_name}_true"
        if truth_key not in data.keys():
            print(f"\nWarning: No ground truth found for {task_name}, skipping.")
            continue

        ground_truth = data[truth_key].flatten()

        if len(predictions) != len(ground_truth):
            print(f"\nWarning: Shape mismatch for {task_name}, skipping.")
            continue

        mae = mean_absolute_error(ground_truth, predictions)
        mse = mean_squared_error(ground_truth, predictions)
        rmse = np.sqrt(mse)
        correlation = np.corrcoef(ground_truth, predictions)[0, 1]

        ss_res = np.sum((ground_truth - predictions) ** 2)
        ss_tot = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        print(f"\n{task_name.upper()}:")
        print(f"  MAE:           {mae:.6f}")
        print(f"  RMSE:          {rmse:.6f}")
        print(f"  Correlation:   {correlation:.4f}")
        print(f"  R²:            {r_squared:.4f}")

        results[task_name] = {
            'mae': mae, 'rmse': rmse,
            'correlation': correlation, 'r_squared': r_squared,
        }

        _plot_regression_task(ground_truth, predictions, task_name, output_dir)

    return results


def _plot_regression_task(ground_truth, predictions, task_name, output_dir):
    """Save one plot per regression diagnostic."""
    residuals = predictions - ground_truth
    min_val = min(ground_truth.min(), predictions.min())
    max_val = max(ground_truth.max(), predictions.max())

    # Prediction vs ground truth
    fig, ax = setup_plot(
        xlabel='Ground Truth',
        ylabel='Prediction',
        title=f'{task_name}: Prediction vs Ground Truth',
    )
    ax.scatter(ground_truth, predictions, alpha=0.5, s=10)
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    ax.legend()
    save_plot(fig, f'{output_dir}/{task_name}_pred_vs_truth.png')

    # Residuals vs ground truth
    fig, ax = setup_plot(
        xlabel='Ground Truth',
        ylabel='Residuals',
        title=f'{task_name}: Residual Plot',
    )
    ax.scatter(ground_truth, residuals, alpha=0.5, s=10)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    save_plot(fig, f'{output_dir}/{task_name}_residuals.png')

    # Residual distribution
    fig, ax = setup_plot(
        xlabel='Residuals',
        ylabel='Frequency',
        title=f'{task_name}: Residual Distribution',
    )
    ax.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', lw=2)
    save_plot(fig, f'{output_dir}/{task_name}_residual_dist.png')

    # Prediction distribution
    fig, ax = setup_plot(
        xlabel=f'{task_name} Value',
        ylabel='Density',
        title=f'{task_name}: Prediction Distribution',
    )
    ax.hist(predictions, bins=50, alpha=0.6, label='Predictions', density=True)
    ax.legend()
    save_plot(fig, f'{output_dir}/{task_name}_pred_dist.png')


# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------

def plot_training_loss(training_json_path, output_dir):
    """Plot training and validation loss from a JSON history file."""
    training_path = Path(training_json_path)
    if not training_path.exists():
        print(f"\nTraining history file not found: {training_json_path}")
        return

    with open(training_path, 'r', encoding='utf-8') as f:
        history = json.load(f)

    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])

    if not train_loss and not val_loss:
        print(f"\nNo train_loss/val_loss arrays found in: {training_json_path}")
        return

    print("Plotting training history (train/val loss)...")
    fig, ax = setup_plot(xlabel='Epoch', ylabel='Loss', xlim=(1, len(train_loss)))

    if train_loss:
        ax.plot(np.arange(1, len(train_loss) + 1), train_loss,
                marker='o', markersize=4, lw=2, label='Train Loss')
    if val_loss:
        ax.plot(np.arange(1, len(val_loss) + 1), val_loss,
                marker='s', markersize=4, lw=2, label='Val Loss')

    ax.legend()
    save_plot(fig, f'{output_dir}/training_loss.png')


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(accuracy, tau_vs_qcd_auc, dm_accuracy=None, evq_auc=None, regression_results=None):
    """Print final summary of all tasks."""
    print("\n" + "=" * 60)
    print("MULTI-TASK LEARNING SUMMARY")
    print("=" * 60)

    print("\n[PRIMARY TASK] Tau vs QCD vs Electron (3-class)")
    print(f"  Overall Accuracy: {accuracy:.4f}")
    print(f"  Tau vs QCD AUC: {tau_vs_qcd_auc:.4f}")

    if dm_accuracy is not None:
        print("\n[AUXILIARY TASK 1] Decay Mode (5-class)")
        print(f"  Accuracy: {dm_accuracy:.4f}")

    if evq_auc is not None:
        print("\n[AUXILIARY TASK 2] Electron vs QCD")
        print(f"  AUC: {evq_auc:.4f}")

    if regression_results:
        for task_name, metrics in regression_results.items():
            print(f"\n[AUXILIARY TASK] {task_name.upper()} (Regression)")
            print(f"  MAE:  {metrics['mae']:.6f}")
            print(f"  RMSE: {metrics['rmse']:.6f}")
            print(f"  R²:   {metrics['r_squared']:.4f}")

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(results_path="results/outputs__tau_0.npz", output_dir="results",
         training_json_path="checkpoints/training_.json"):
    """Main analysis pipeline."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    data, predictions, true_labels = load_results(results_path)

    class_names = ['QCD', 'Tau', 'Electron']

    pred_labels, accuracy = analyze_primary_task(predictions, true_labels, class_names)

    plot_confusion_matrix(true_labels, pred_labels, class_names, output_dir)
    plot_roc_curves(predictions, true_labels, class_names, output_dir)
    plot_score_distributions(predictions, true_labels, class_names, output_dir)

    tau_qcd_mask = (true_labels == 0) | (true_labels == 1)
    tau_score = predictions[tau_qcd_mask, 1]
    is_tau = (true_labels[tau_qcd_mask] == 1).astype(int)
    fpr, tpr, _ = roc_curve(is_tau, tau_score)
    tau_vs_qcd_auc = auc(fpr, tpr)

    plot_tau_vs_qcd(predictions, true_labels, output_dir)

    dm_accuracy = analyze_decay_mode(data, true_labels, output_dir)
    evq_auc = analyze_electron_vs_qcd(data, true_labels, output_dir)
    regression_results = analyze_regression_task(data, output_dir)

    print_summary(accuracy, tau_vs_qcd_auc, dm_accuracy, evq_auc, regression_results)

    plot_training_loss(training_json_path, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze OmniLearned Tau Classification Results")
    parser.add_argument('--results_path', type=str, default="results/outputs__tau_0.npz")
    parser.add_argument('--output_dir', type=str, default="results")
    parser.add_argument('--training_json_path', type=str, default="checkpoints/training_.json")
    args = parser.parse_args()

    main(results_path=args.results_path, output_dir=args.output_dir,
         training_json_path=args.training_json_path)