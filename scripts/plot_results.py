#!/usr/bin/env python3
"""
Tau Classification Results Analysis

Analyze the output from OmniLearned evaluation.

Primary task (2-class): Tau vs QCD only. Class 0=QCD, 1=Tau. 
Electron jets are excluded from primary metrics.

Auxiliary tasks: 
- decay_mode (5-class: 1p0n, 1p1n, 1pXn, 3p0n, 3pXn)
- electron_vs_qcd (QCD vs Electron)
"""

import numpy as np
import matplotlib.pyplot as plt
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

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 12


def load_results(results_path="results/outputs__tau_0.npz"):
    """Load evaluation results from NPZ file."""
    data = np.load(results_path)
    print("Available arrays:", list(data.keys()))

    predictions = data['prediction']  # Shape: [N_jets, 3] - softmax probabilities
    true_labels = data['pid']  # Shape: [N_jets] - true class labels

    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Labels shape: {true_labels.shape}")
    print(f"Total jets: {len(true_labels)}")

    return data, predictions, true_labels


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

    # Overall accuracy
    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Per-class accuracy
    print("\nPer-class Accuracy:")
    for i, name in enumerate(class_names):
        mask = true_labels == i
        if mask.sum() > 0:
            class_acc = (pred_labels[mask] == i).mean()
            print(f"  {name}: {class_acc:.4f} ({class_acc*100:.2f}%)")

    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(true_labels, pred_labels, target_names=class_names, digits=4))

    return pred_labels, accuracy


def plot_confusion_matrix(true_labels, pred_labels, class_names, output_dir):
    """Plot and save confusion matrices."""
    print("\nPlotting confusion matrices...")
    cm = confusion_matrix(true_labels, pred_labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    disp1 = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp1.plot(ax=axes[0], cmap='Blues', values_format='d')
    axes[0].set_title('Confusion Matrix (Counts)')

    # Normalized (row-wise = recall)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    disp2 = ConfusionMatrixDisplay(cm_norm, display_labels=class_names)
    disp2.plot(ax=axes[1], cmap='Blues', values_format='.3f')
    axes[1].set_title('Confusion Matrix (Normalized by True Label)')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curves(predictions, true_labels, class_names, output_dir):
    """Plot ROC curves for each class (one-vs-rest)."""
    print("Plotting ROC curves...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = ['#e41a1c', '#377eb8', '#4daf4a']  # Red, Blue, Green

    for i, (name, color) in enumerate(zip(class_names, colors)):
        y_true_binary = (true_labels == i).astype(int)
        y_score = predictions[:, i]

        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)

        axes[i].plot(fpr, tpr, color=color, lw=2, label=f'AUC = {roc_auc:.4f}')
        axes[i].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[i].set_xlim([0.0, 1.0])
        axes[i].set_ylim([0.0, 1.05])
        axes[i].set_xlabel('False Positive Rate')
        axes[i].set_ylabel('True Positive Rate')
        axes[i].set_title(f'{name} vs Rest')
        axes[i].legend(loc='lower right')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_tau_vs_qcd(predictions, true_labels, output_dir):
    """Plot Tau vs QCD ROC curve and background rejection."""
    print("Plotting Tau vs QCD analysis...")
    tau_qcd_mask = (true_labels == 0) | (true_labels == 1)
    tau_qcd_labels = true_labels[tau_qcd_mask]
    tau_qcd_probs = predictions[tau_qcd_mask]

    tau_score = tau_qcd_probs[:, 1]
    is_tau = (tau_qcd_labels == 1).astype(int)

    fpr, tpr, _ = roc_curve(is_tau, tau_score)
    roc_auc = auc(fpr, tpr)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC curve
    axes[0].plot(fpr, tpr, 'b-', lw=2, label=f'Tau vs QCD (AUC = {roc_auc:.4f})')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[0].set_xlabel('False Positive Rate (QCD misID)')
    axes[0].set_ylabel('True Positive Rate (Tau efficiency)')
    axes[0].set_title('Tau vs QCD ROC Curve')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)

    # Background rejection vs signal efficiency
    rejection = np.where(fpr > 0, 1.0 / fpr, np.inf)
    valid = rejection < 1e6

    axes[1].semilogy(tpr[valid], rejection[valid], 'b-', lw=2)
    axes[1].set_xlabel('Tau Efficiency (Signal)')
    axes[1].set_ylabel('QCD Rejection (1/FPR)')
    axes[1].set_title('Background Rejection vs Signal Efficiency')
    axes[1].grid(True, alpha=0.3, which='both')
    axes[1].set_xlim([0, 1])

    plt.tight_layout()
    plt.savefig(f'{output_dir}/tau_vs_qcd.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Print working points
    print("\nTau vs QCD Working Points:")
    print(f"  AUC: {roc_auc:.4f}")
    print("\n  Tau efficiency -> QCD rejection:")
    for target_eff in [0.5, 0.6, 0.7, 0.8, 0.9]:
        idx = np.argmin(np.abs(tpr - target_eff))
        if fpr[idx] > 0:
            print(f"    {target_eff*100:.0f}% -> {1/fpr[idx]:.1f}x rejection")


def plot_score_distributions(predictions, true_labels, class_names, output_dir):
    """Plot score distributions for each class."""
    print("Plotting score distributions...")
    colors = ['#e41a1c', '#377eb8', '#4daf4a']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, name in enumerate(class_names):
        ax = axes[i]
        score = predictions[:, i]

        for j, (true_name, color) in enumerate(zip(class_names, colors)):
            mask = true_labels == j
            ax.hist(score[mask], bins=50, alpha=0.5, label=f'True {true_name}',
                    color=color, density=True)

        ax.set_xlabel(f'{name} Score')
        ax.set_ylabel('Density')
        ax.set_title(f'{name} Score Distribution')
        ax.legend()
        ax.set_xlim([0, 1])

    plt.tight_layout()
    plt.savefig(f'{output_dir}/score_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_decay_mode_score_distributions(dm_pred_valid, dm_true_valid, prong_names, output_dir):
    """Plot score distributions for each decay mode class."""
    print("Plotting decay mode score distributions...")
    colors_dm = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, (name, color) in enumerate(zip(prong_names, colors_dm)):
        ax = axes[i]
        score = dm_pred_valid[:, i]

        for j, (true_name, true_color) in enumerate(zip(prong_names, colors_dm)):
            mask = dm_true_valid == j
            ax.hist(score[mask], bins=50, alpha=0.5, label=f'True {true_name}',
                    color=true_color, density=True)

        ax.set_xlabel(f'{name} Score')
        ax.set_ylabel('Density')
        ax.set_title(f'{name} Score Distribution')
        ax.legend(fontsize=8)
        ax.set_xlim([0, 1])

    fig.delaxes(axes[5])
    plt.tight_layout()
    plt.savefig(f'{output_dir}/decay_mode_score_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()


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
    valid_mask = tau_mask & (decay_mode_true >= 0)

    print(f"\nTotal tau jets: {tau_mask.sum()}")
    print(f"Tau jets with valid decay mode: {valid_mask.sum()}")

    dm_pred_valid = decay_mode_pred[valid_mask]
    dm_true_valid = decay_mode_true[valid_mask]
    dm_pred_labels = np.argmax(dm_pred_valid, axis=1)

    prong_names = ['1p0n', '1p1n', '1pXn', '3p0n', '3pXn']

    print("\nDecay mode distribution:")
    for i, name in enumerate(prong_names):
        count = (dm_true_valid == i).sum()
        print(f"  {name}: {count}")

    # Accuracy
    dm_accuracy = accuracy_score(dm_true_valid, dm_pred_labels)
    print(f"\nDecay Mode Accuracy: {dm_accuracy:.4f} ({dm_accuracy*100:.2f}%)")

    # Classification report
    print("\nDecay Mode Classification Report:")
    print(classification_report(dm_true_valid, dm_pred_labels, target_names=prong_names, digits=4))

    # Score distributions
    plot_decay_mode_score_distributions(dm_pred_valid, dm_true_valid, prong_names, output_dir)

    # Confusion matrix
    print("Plotting decay mode confusion matrices...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cm_dm = confusion_matrix(dm_true_valid, dm_pred_labels)
    disp1 = ConfusionMatrixDisplay(cm_dm, display_labels=prong_names)
    disp1.plot(ax=axes[0], cmap='Oranges', values_format='d')
    axes[0].set_title('Decay Mode Confusion Matrix (Counts)')

    cm_dm_norm = cm_dm.astype('float') / cm_dm.sum(axis=1)[:, np.newaxis]
    disp2 = ConfusionMatrixDisplay(cm_dm_norm, display_labels=prong_names)
    disp2.plot(ax=axes[1], cmap='Oranges', values_format='.3f')
    axes[1].set_title('Decay Mode Confusion Matrix (Normalized)')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/decay_mode_confusion.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ROC curves for each decay mode
    print("Plotting decay mode ROC curves...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    colors_dm = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    print("\nDecay Mode AUC (one-vs-rest):")
    for i, (name, color) in enumerate(zip(prong_names, colors_dm)):
        y_true_binary = (dm_true_valid == i).astype(int)
        y_score = dm_pred_valid[:, i]

        fpr_dm, tpr_dm, _ = roc_curve(y_true_binary, y_score)
        auc_dm = auc(fpr_dm, tpr_dm)

        axes[i].plot(fpr_dm, tpr_dm, color=color, lw=2,
                    label=f'{name} vs Rest (AUC = {auc_dm:.4f})')
        axes[i].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[i].set_xlabel('False Positive Rate')
        axes[i].set_ylabel('True Positive Rate')
        axes[i].set_title(f'Decay Mode: {name}')
        axes[i].legend(loc='lower right')
        axes[i].grid(True, alpha=0.3)

        print(f"  {name}: {auc_dm:.4f}")

    fig.delaxes(axes[5])
    plt.tight_layout()
    plt.savefig(f'{output_dir}/decay_mode_roc.png', dpi=150, bbox_inches='tight')
    plt.close()

    return dm_accuracy


def analyze_electron_vs_qcd(data, true_labels, output_dir):
    """Analyze electron vs QCD auxiliary task."""
    if 'aux_electron_vs_qcd_pred' not in data.keys():
        print("\nNo electron_vs_qcd auxiliary predictions found.")
        return None

    print("\n" + "=" * 60)
    print("AUXILIARY TASK 2: Electron vs QCD")
    print("=" * 60)

    evq_pred = data['aux_electron_vs_qcd_pred']

    evq_mask = (true_labels == 0) | (true_labels == 2)  # QCD=0, Electron=2
    evq_pred_valid = evq_pred[evq_mask]
    evq_true = (true_labels[evq_mask] == 2).astype(int)

    print(f"\nElectron vs QCD samples: {evq_mask.sum()}")
    print(f"  QCD jets: {(true_labels[evq_mask] == 0).sum()}")
    print(f"  Electron jets: {(true_labels[evq_mask] == 2).sum()}")

    evq_score = evq_pred_valid[:, 1]
    fpr_evq, tpr_evq, _ = roc_curve(evq_true, evq_score)
    auc_evq = auc(fpr_evq, tpr_evq)

    print("Plotting electron vs QCD analysis...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(fpr_evq, tpr_evq, 'green', lw=2, label=f'Electron vs QCD (AUC = {auc_evq:.4f})')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[0].set_xlabel('False Positive Rate (QCD misID)')
    axes[0].set_ylabel('True Positive Rate (Electron efficiency)')
    axes[0].set_title('Electron vs QCD ROC Curve')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)

    # Score distribution
    axes[1].hist(evq_score[evq_true == 0], bins=50, alpha=0.6,
                 label='True QCD', color='red', density=True)
    axes[1].hist(evq_score[evq_true == 1], bins=50, alpha=0.6,
                 label='True Electron', color='green', density=True)
    axes[1].set_xlabel('Electron Score')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Electron vs QCD Score Distribution')
    axes[1].legend()
    axes[1].set_xlim([0, 1])

    plt.tight_layout()
    plt.savefig(f'{output_dir}/electron_vs_qcd.png', dpi=150, bbox_inches='tight')
    plt.close()

    evq_acc = accuracy_score(evq_true, (evq_score > 0.5).astype(int))
    print(f"\nElectron vs QCD Accuracy: {evq_acc:.4f}")
    print(f"Electron vs QCD AUC: {auc_evq:.4f}")

    return auc_evq


def analyze_regression_task(data, output_dir):
    """Analyze regression auxiliary tasks."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # Find all regression task predictions
    regression_tasks = {}
    for key in data.keys():
        print(key)
        if key.startswith('aux_') and key.endswith('_pred'):
            task_name = key.replace('aux_', '').replace('_pred', '')
            # Check if it's likely a regression task (1D output)
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
        # Look for corresponding ground truth
        truth_key = f"aux_{task_name}_true"
        if truth_key not in data.keys():
            print(f"\nWarning: No ground truth found for {task_name}, skipping.")
            continue

        ground_truth = data[truth_key].flatten()

        if len(predictions) != len(ground_truth):
            print(f"\nWarning: Shape mismatch for {task_name}, skipping.")
            continue

        # Calculate metrics
        mae = mean_absolute_error(ground_truth, predictions)
        mse = mean_squared_error(ground_truth, predictions)
        rmse = np.sqrt(mse)
        correlation = np.corrcoef(ground_truth, predictions)[0, 1]

        # R-squared
        ss_res = np.sum((ground_truth - predictions) ** 2)
        ss_tot = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        print(f"\n{task_name.upper()}:")
        print(f"  MAE:           {mae:.6f}")
        print(f"  RMSE:          {rmse:.6f}")
        print(f"  Correlation:   {correlation:.4f}")
        print(f"  R²:            {r_squared:.4f}")

        results[task_name] = {
            'mae': mae,
            'rmse': rmse,
            'correlation': correlation,
            'r_squared': r_squared,
        }

        # Create plots
        _plot_regression_task(ground_truth, predictions, task_name, output_dir)

    return results


def _plot_regression_task(ground_truth, predictions, task_name, output_dir):
    """Plot regression task predictions vs ground truth."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Prediction vs Ground Truth
    axes[0, 0].scatter(ground_truth, predictions, alpha=0.5, s=10)
    min_val = min(ground_truth.min(), predictions.min())
    max_val = max(ground_truth.max(), predictions.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    axes[0, 0].set_xlabel('Ground Truth')
    axes[0, 0].set_ylabel('Prediction')
    axes[0, 0].set_title(f'{task_name}: Prediction vs Ground Truth')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Residuals
    residuals = predictions - ground_truth
    axes[0, 1].scatter(ground_truth, residuals, alpha=0.5, s=10)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Ground Truth')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title(f'{task_name}: Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)

    # Residual Distribution
    axes[1, 0].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'{task_name}: Residual Distribution')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 0].grid(True, alpha=0.3)

    # Distribution comparison
    # axes[1, 1].hist(ground_truth, bins=50, alpha=0.6, label='Ground Truth', density=True)
    axes[1, 1].hist(predictions, bins=50, alpha=0.6, label='Predictions', density=True)
    axes[1, 1].set_xlabel(f'{task_name} Value')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title(f'{task_name}: Distribution Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{task_name}_regression.png', dpi=150, bbox_inches='tight')
    plt.close()


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


def main(results_path="results/outputs__tau_0.npz", output_dir="results"):
    """Main analysis pipeline."""
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading results...")
    data, predictions, true_labels = load_results(results_path)

    class_names = ['QCD', 'Tau', 'Electron']

    # Primary task analysis
    pred_labels, accuracy = analyze_primary_task(predictions, true_labels, class_names)

    # Plotting
    plot_confusion_matrix(true_labels, pred_labels, class_names, output_dir)
    plot_roc_curves(predictions, true_labels, class_names, output_dir)
    plot_score_distributions(predictions, true_labels, class_names, output_dir)

    # Tau vs QCD analysis
    tau_qcd_mask = (true_labels == 0) | (true_labels == 1)
    tau_score = predictions[tau_qcd_mask, 1]
    is_tau = (true_labels[tau_qcd_mask] == 1).astype(int)
    fpr, tpr, _ = roc_curve(is_tau, tau_score)
    tau_vs_qcd_auc = auc(fpr, tpr)

    plot_tau_vs_qcd(predictions, true_labels, output_dir)

    # Auxiliary tasks
    dm_accuracy = analyze_decay_mode(data, true_labels, output_dir)
    evq_auc = analyze_electron_vs_qcd(data, true_labels, output_dir)
    regression_results = analyze_regression_task(data, output_dir)

    # Summary
    print_summary(accuracy, tau_vs_qcd_auc, dm_accuracy, evq_auc, regression_results)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Analyze OmniLearned Tau Classification Results")
    parser.add_argument('--results_path', type=str, default="results/outputs__tau_0.npz",
                        help="Path to the NPZ results file")
    parser.add_argument('--output_dir', type=str, default="results",
                        help="Directory to save output plots")
    args = parser.parse_args()
    
    main(results_path=args.results_path, output_dir=args.output_dir)