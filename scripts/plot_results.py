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
# Confusion matrices
# ---------------------------------------------------------------------------

def plot_confusion_matrix(true_labels, pred_labels, class_names, output_dir):
    """Save raw-count and normalised confusion matrices as separate files."""
    print("\nPlotting confusion matrices...")
    cm = confusion_matrix(true_labels, pred_labels)

    # Raw counts
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm.T, display_labels=class_names).plot(
        ax=ax, cmap='Blues', values_format='d')
    ax.set_xlabel('True label')
    ax.set_ylabel('Predicted label')
    ax.invert_yaxis()
    ax.grid(False)
    ax.set_title('Counts')
    save_plot(fig, f'{output_dir}/confusion_matrix_counts.png')

    # Normalised
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm_norm.T, display_labels=class_names).plot(
        ax=ax, cmap='Blues', values_format='.3f')
    ax.set_xlabel('True label')
    ax.set_ylabel('Predicted label')
    ax.invert_yaxis()
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

def plot_tau_vs_qcd(predictions, true_labels, output_dir, reco_id=None):
    """Save Tau vs QCD ROC curve and background-rejection curve."""
    print("Plotting Tau vs QCD analysis...")
    tau_qcd_mask = (true_labels == 0) | (true_labels == 1)
    tau_qcd_labels = true_labels[tau_qcd_mask]
    tau_qcd_probs = predictions[tau_qcd_mask]

    tau_score = tau_qcd_probs[:, 1]
    is_tau = (tau_qcd_labels == 1).astype(int)
    fpr, tpr, _ = roc_curve(is_tau, tau_score)
    roc_auc = auc(fpr, tpr)

    # Build reco baselines
    reco_baselines = {}
    if reco_id is not None:
        reco_id_masked = reco_id[tau_qcd_mask]
        reco_baselines["RNN"] = reco_id_masked[:, 2]    # TauRNNJetScore_Raw
        reco_baselines["GNTau"] = reco_id_masked[:, 5]  # TauGNNJetScore_SigTrans

    reco_colors = ["darkorange", "green"]

    # ROC curve
    fig, ax = setup_plot(
        xlabel='False Positive Rate',
        ylabel='True Positive Rate',
        title='Tau vs QCD',
        xlim=[0.0, 1.0],
        ylim=[0.0, 1.0],
    )
    ax.plot(fpr, tpr, 'b-', lw=2, label=f'OmniTau AUC = {roc_auc:.4f}')
    for (name, score), color in zip(reco_baselines.items(), reco_colors):
        fpr_r, tpr_r, _ = roc_curve(is_tau, score)
        auc_r = auc(fpr_r, tpr_r)
        ax.plot(fpr_r, tpr_r, '-', color=color, lw=2, label=f'{name} AUC = {auc_r:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.legend(loc='lower right')
    save_plot(fig, f'{output_dir}/tau_vs_qcd_roc.png')

    # Background rejection
    rejection = np.divide(1.0, fpr, out=np.full_like(fpr, np.inf), where=fpr > 0)
    valid = rejection < 1e6

    fig, ax = setup_plot(
        xlabel='Tau Efficiency',
        ylabel='QCD Rejection',
        title='Tau vs QCD',
        xlim=[0, 1],
    )
    ax.semilogy(tpr[valid], rejection[valid], 'b-', lw=2, label='OmniTau')
    for (name, score), color in zip(reco_baselines.items(), reco_colors):
        fpr_r, tpr_r, _ = roc_curve(is_tau, score)
        rej_r = np.divide(1.0, fpr_r, out=np.full_like(fpr_r, np.inf), where=fpr_r > 0)
        valid_r = rej_r < 1e6
        ax.semilogy(tpr_r[valid_r], rej_r[valid_r], '-', color=color, lw=2, label=name)
    ax.legend(loc='upper right')
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
# Tau vs Electron
# ---------------------------------------------------------------------------

def plot_tau_vs_election(predictions, true_labels, output_dir, reco_id=None):
    """Save Tau vs Electron ROC curve and background-rejection curve."""
    print("Plotting Tau vs Electron analysis...")
    tau_ele_mask = (true_labels == 1) | (true_labels == 2)
    tau_ele_labels = true_labels[tau_ele_mask]
    tau_ele_probs = predictions[tau_ele_mask]

    tau_score = tau_ele_probs[:, 1]
    is_tau = (tau_ele_labels == 1).astype(int)
    fpr, tpr, _ = roc_curve(is_tau, tau_score)
    roc_auc = auc(fpr, tpr)

    # Build reco baselines
    reco_baselines = {}
    if reco_id is not None:
        reco_id_masked = reco_id[tau_ele_mask]
        reco_baselines["RNN"] = reco_id_masked[:, 0]  # TauRNNEleScore_Raw

    reco_colors = ["darkorange"]

    # ROC curve
    fig, ax = setup_plot(
        xlabel='False Positive Rate',
        ylabel='True Positive Rate',
        title='Tau vs Electron',
        xlim=[0.0, 1.0],
        ylim=[0.0, 1.0],
    )
    ax.plot(fpr, tpr, 'b-', lw=2, label=f'OmniTau AUC = {roc_auc:.4f}')
    for (name, score), color in zip(reco_baselines.items(), reco_colors):
        fpr_r, tpr_r, _ = roc_curve(is_tau, score)
        auc_r = auc(fpr_r, tpr_r)
        ax.plot(fpr_r, tpr_r, '-', color=color, lw=2, label=f'{name} AUC = {auc_r:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.legend(loc='lower right')
    save_plot(fig, f'{output_dir}/tau_vs_ele_roc.png')

    # Background rejection
    rejection = np.divide(1.0, fpr, out=np.full_like(fpr, np.inf), where=fpr > 0)
    valid = rejection < 1e6

    fig, ax = setup_plot(
        xlabel='Tau Efficiency',
        ylabel='Electron Rejection',
        title='Tau vs Electron',
        xlim=[0, 1],
    )
    ax.semilogy(tpr[valid], rejection[valid], 'b-', lw=2, label='OmniTau')
    for (name, score), color in zip(reco_baselines.items(), reco_colors):
        fpr_r, tpr_r, _ = roc_curve(is_tau, score)
        rej_r = np.divide(1.0, fpr_r, out=np.full_like(fpr_r, np.inf), where=fpr_r > 0)
        valid_r = rej_r < 1e6
        ax.semilogy(tpr_r[valid_r], rej_r[valid_r], '-', color=color, lw=2, label=name)
    ax.legend(loc='upper right')
    save_plot(fig, f'{output_dir}/tau_vs_ele_rejection.png')

    # Print working points
    print("\nTau vs Electron Working Points:")
    print(f"  AUC: {roc_auc:.4f}")
    print("\n  Tau efficiency -> Electron rejection:")
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
        
def _save_dm_confusion(true, pred, tag, title_suffix, n_classes_plot, prong_names, output_dir):
    pred = np.asarray(pred)
    # Filter to valid true labels AND pred labels in 0-4
    valid = (true >= 0) & (true < n_classes_plot) & (pred >= 0) & (pred < n_classes_plot)
    t, p = true[valid], pred[valid]

    cm = confusion_matrix(t, p, labels=list(range(n_classes_plot)))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm.T, display_labels=prong_names).plot(
        ax=ax, cmap='Oranges', values_format='d')
    ax.set_xlabel('True label')
    ax.set_ylabel('Predicted label')
    ax.invert_yaxis()
    ax.grid(False)
    ax.set_title(f'{title_suffix}')
    save_plot(fig, f'{output_dir}/decay_mode_confusion_counts_{tag}.png')

    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.divide(cm.astype('float'), row_sums, where=row_sums != 0,
                        out=np.zeros_like(cm, dtype=float))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm_norm.T, display_labels=prong_names).plot(
        ax=ax, cmap='Oranges', values_format='.3f')
    ax.set_xlabel('True label')
    ax.set_ylabel('Predicted label')
    ax.invert_yaxis()
    ax.grid(False)
    ax.set_title(f'{title_suffix}')
    save_plot(fig, f'{output_dir}/decay_mode_confusion_normalised_{tag}.png')

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

    # Confusion matrices — model + two reco baselines
    print("Plotting decay mode confusion matrices...")

    _save_dm_confusion(dm_true_valid, dm_pred_labels, 'model', 'Model', n_classes_plot, prong_names, output_dir)

    reco_decay_mode = data.get('reco_decay_mode')
    if reco_decay_mode is not None:
        reco_dm_valid = reco_decay_mode[valid_mask]
        _save_dm_confusion(dm_true_valid, reco_dm_valid[:, 0], 'rnn', 'TauNNDecayMode', n_classes_plot, prong_names, output_dir)
        _save_dm_confusion(dm_true_valid, reco_dm_valid[:, 1], 'pantau', 'TauPanTauBDTDecayMode', n_classes_plot, prong_names, output_dir)

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
# Regression tasks
# ---------------------------------------------------------------------------

def analyze_regression_task(data, true_labels, output_dir):
    """Analyze regression auxiliary tasks."""
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    tau_mask = true_labels == 1

    regression_tasks = {}
    for key in data.keys():
        if key.startswith('aux_') and key.endswith('_pred'):
            task_name = key.replace('aux_', '').replace('_pred', '')
            arr = data[key][tau_mask]
            if arr.ndim == 1 or arr.shape[1] == 1:
                regression_tasks[task_name] = arr.flatten()

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

        ground_truth = data[truth_key][tau_mask].flatten()

        if len(predictions) != len(ground_truth):
            print(f"\nWarning: Shape mismatch for {task_name}, skipping.")
            continue

        valid = np.isfinite(ground_truth) & np.isfinite(predictions)
        # only test on actual pions
        if 'pion' in task_name:
            pion_valid = (ground_truth != -999.0) & (predictions != -999.0)
            valid = valid & pion_valid
        ground_truth = ground_truth[valid]
        predictions = predictions[valid]

        if len(ground_truth) == 0:
            print(f"\nWarning: No valid samples for {task_name}, skipping.")
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

        is_log = task_name in ("tes", "charged_pion_pt", "neutral_pion_pt")

        # Determine which particle's kinematic variables to plot against.
        # Keys match the aux_*_true arrays saved by evaluate.py.
        if task_name.startswith('neutral_pion'):
            kin_keys = [('neutral_pion_pt', True), ('neutral_pion_eta', False), 
                        ('neutral_pion_phi', False)]
            particle_label = 'neutral π'
        elif task_name.startswith('charged_pion'):
            kin_keys = [('charged_pion_pt', True), ('charged_pion_eta', False), 
                        ('charged_pion_phi', False)]
            particle_label = 'charged π'
        else:
            kin_keys = [('tes', True), ('tau_eta', False), ('tau_phi', False)]
            particle_label = 'tau'

        kin_vars = {}
        for kin_key, kin_log in kin_keys:
            kin_true_key = f'aux_{kin_key}_true'
            if kin_true_key not in data.keys():
                continue
            kin_arr = data[kin_true_key][tau_mask].flatten()[valid]
            if kin_log:
                kin_arr = np.exp(kin_arr) / 1000  # log-MeV → GeV
            if 'pt' in kin_key or kin_key == 'tes':
                x_label = f'True {particle_label} $p_T$ [GeV]'
                x_suffix = 'pt'
            elif 'eta' in kin_key:
                x_label = f'True {particle_label} $\\eta$'
                x_suffix = 'eta'
            else:
                x_label = f'True {particle_label} $\\phi$'
                x_suffix = 'phi'
            kin_vars[x_suffix] = (kin_arr, x_label)

        # Build baseline_responses: reco values in physical units, same valid mask applied.
        # reco_tau_4mom cols: [PanTau_pt, PanTau_eta, PanTau_phi, TauFinalCalib_pt, 
        #                                           TauFinalCalib_eta, TauFinalCalib_phi]
        # reco_charged_pions cols: [pt, eta, phi]   (summed tracks, in MeV)
        # reco_neutral_pions cols: [pt, eta, phi]   (PanTauPi0, in MeV)
        baseline_responses = None
        if task_name.startswith('neutral_pion') and 'reco_neutral_pions' in data:
            reco_np = data['reco_neutral_pions'][tau_mask][valid]
            if task_name.endswith('_pt'):
                baseline_responses = {'PanTau Pi0': reco_np[:, 0] / 1000}   # MeV → GeV
            elif task_name.endswith('_eta'):
                baseline_responses = {'PanTau Pi0': reco_np[:, 1]}
            elif task_name.endswith('_phi'):
                baseline_responses = {'PanTau Pi0': reco_np[:, 2]}
        elif task_name.startswith('charged_pion') and 'reco_charged_pions' in data:
            reco_cp = data['reco_charged_pions'][tau_mask][valid]
            if task_name.endswith('_pt'):
                baseline_responses = {'Reco': reco_cp[:, 0] / 1000}          # MeV → GeV
            elif task_name.endswith('_eta'):
                baseline_responses = {'Reco': reco_cp[:, 1]}
            elif task_name.endswith('_phi'):
                baseline_responses = {'Reco': reco_cp[:, 2]}
        elif 'reco_tau_4mom' in data:
            # tau tasks: tes / tau_eta / tau_phi
            reco_t4 = data['reco_tau_4mom'][tau_mask][valid]
            if task_name == 'tes':
                baseline_responses = {
                    'PanTau':   reco_t4[:, 0] / 1000,   # MeV → GeV
                    'Combined': reco_t4[:, 3] / 1000,
                }
            elif task_name == 'tau_eta':
                baseline_responses = {
                    'PanTau':   reco_t4[:, 1],
                    'Combined': reco_t4[:, 4],
                }
            elif task_name == 'tau_phi':
                baseline_responses = {
                    'PanTau':   reco_t4[:, 2],
                    'Combined': reco_t4[:, 5],
                }

        _plot_regression_task(ground_truth, predictions, task_name, output_dir,
                              log_scale=is_log, kin_vars=kin_vars or None,
                              baseline_responses=baseline_responses)

    return results


_BASELINE_COLORS = ['steelblue', 'darkorange']


def _plot_response_vs_variable(response, x_var, x_label, x_suffix, task_name, output_dir,
                               log_scale, baselines=None):
    """Plot response and resolution curves vs a single kinematic variable.

    Parameters
    ----------
    response : array
        pred/truth (log_scale=True) or pred-truth (log_scale=False) for each event.
    x_var : array
        Kinematic variable to bin on (already in physical units, same length as response).
    x_label : str
        Axis label for x_var.
    x_suffix : str
        Short string used in the output filename, e.g. 'pt', 'eta', 'phi'.
    log_scale : bool
        Controls y-axis labelling and whether resolution is shown as a percentage.
    baselines : dict or None
        Mapping of {label: response_array} for established reco methods to overlay.
    """
    from plotting.utils import response_curve, make_bins

    if log_scale:
        response_ylabel = f'Predicted / True {task_name}'
        resol_ylabel = f'{task_name} response at 68% CL [%]'
    else:
        response_ylabel = f'Predicted - True {task_name}'
        resol_ylabel = f'{task_name} residual at 68% CL'

    bins_def = make_bins(x_var.min(), x_var.max(), 25)

    result = response_curve(response, x_var, bins_def, cl=0.68)
    if len(result[0]) == 0:
        return
    bins, bin_errors, means, errs, resol = result

    fig, ax = setup_plot(xlabel=x_label, ylabel=response_ylabel, title="")
    plt.errorbar(bins, means, errs, bin_errors, fmt='o', color='purple', label='Prediction')
    if baselines:
        for (bl_label, bl_response), color in zip(baselines.items(), _BASELINE_COLORS):
            bl_result = response_curve(bl_response, x_var, bins_def, cl=0.68)
            if len(bl_result[0]) == 0:
                continue
            bl_bins, bl_bin_errs, bl_means, bl_errs, _ = bl_result
            plt.errorbar(bl_bins, bl_means, bl_errs, bl_bin_errs, fmt='s', 
                         color=color, label=bl_label)
    ax.legend()
    save_plot(fig, f'{output_dir}/{task_name}_response_vs_{x_suffix}.png')

    fig, ax = setup_plot(xlabel=x_label, ylabel=resol_ylabel, title="")
    plt.plot(bins, 100 * resol if log_scale else resol, color='purple', label='Prediction')
    if baselines:
        for (bl_label, bl_response), color in zip(baselines.items(), _BASELINE_COLORS):
            bl_result = response_curve(bl_response, x_var, bins_def, cl=0.68)
            if len(bl_result[0]) == 0:
                continue
            bl_bins, _, _, _, bl_resol = bl_result
            plt.plot(bl_bins, 100 * bl_resol if log_scale else bl_resol, 
                     color=color, label=bl_label)
    ax.legend()
    save_plot(fig, f'{output_dir}/{task_name}_resolution_vs_{x_suffix}.png')


def _plot_regression_task(ground_truth, predictions, task_name, output_dir,
                          log_scale=False, kin_vars=None, baseline_responses=None):
    """Save one set of plots per regression task.

    Parameters
    ----------
    log_scale : bool
        If True, apply exp()/1000 before plotting (targets stored as log-MeV).
        Use for pt-like targets (tes, charged_pion_pt, neutral_pion_pt).
    kin_vars : dict or None
        Mapping of {x_suffix: (x_array, x_label)} for kinematic variables to
        plot response/resolution against.  Each array must already be filtered
        to the same valid-event mask as ground_truth/predictions and must be
        in physical units (GeV for pt, radians for eta/phi).
        If None, falls back to plotting vs the task's own truth value.
    baseline_responses : dict or None
        Mapping of {label: array} where each array is the reco baseline value
        in the same physical units as ground_truth (GeV for pt, rad for eta/phi),
        filtered to the same valid-event mask.
    """

    # Convert log-space pt targets to physical GeV
    if log_scale:
        ground_truth = np.exp(ground_truth) / 1000
        predictions  = np.exp(predictions)  / 1000

    # Overlaid truth vs prediction distributions
    all_vals = [ground_truth, predictions]
    if baseline_responses:
        all_vals.extend(baseline_responses.values())
    bins = np.linspace(min(ground_truth), max(ground_truth), 51)

    fig, ax = setup_plot(
        xlabel=f'{task_name}{" [GeV]" if log_scale else ""}',
        ylabel='Events',
        title='',
    )
    ax.hist(ground_truth, bins=bins, label='Truth', color='green', histtype="step")
    ax.hist(predictions,  bins=bins, label='Prediction', color='purple', histtype="step")
    if baseline_responses:
        for (bl_label, bl_arr), color in zip(baseline_responses.items(), _BASELINE_COLORS):
            ax.hist(bl_arr, bins=bins, label=bl_label, color=color, 
                    histtype="step")
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("Events (logged)")
    ax.legend()
    save_plot(fig, f'{output_dir}/{task_name}_pred_dist.png')

    # Response lineshape histogram (ratio, only meaningful for pt tasks)
    if log_scale:
        response_ratio = predictions / ground_truth
        fig, ax = setup_plot(
            xlabel=f'{task_name} response (pred/truth)',
            ylabel='Events',
            title='',
        )
        ax.hist(response_ratio, bins=np.linspace(0, 2, 75), label="Prediction", histtype="step", 
                color="purple")
        if baseline_responses:
            for (bl_label, bl_arr), color in zip(baseline_responses.items(), _BASELINE_COLORS):
                bl_ratio = bl_arr / ground_truth
                ax.hist(bl_ratio, bins=np.linspace(0, 2, 75), label=bl_label,
                        color=color, histtype="step")
        ax.set_yscale("log")
        ax.legend()
        save_plot(fig, f'{output_dir}/{task_name}_response.png')

    # Compute per-event response: ratio for pt, additive residual for eta/phi
    if log_scale:
        response = predictions / ground_truth
    else:
        response = predictions - ground_truth

    # Baseline per-event responses for the kinematic curves
    bl_kinematic = None
    if baseline_responses:
        bl_kinematic = {}
        for bl_label, bl_arr in baseline_responses.items():
            bl_kinematic[bl_label] = bl_arr / ground_truth if log_scale else bl_arr - ground_truth

    # Fall back to plotting vs the task's own truth if no kin_vars supplied
    if kin_vars is None:
        kin_vars = {
            'truth': (ground_truth, f'True {task_name}{" [GeV]" if log_scale else ""}')
        }

    for x_suffix, (x_var, x_label) in kin_vars.items():
        _plot_response_vs_variable(response, x_var, x_label, x_suffix, task_name,
                                   output_dir, log_scale, baselines=bl_kinematic)


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

def print_summary(accuracy, tau_vs_qcd_auc, dm_accuracy=None, regression_results=None):
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

    reco_id = data.get('reco_id')
    plot_tau_vs_qcd(predictions, true_labels, output_dir, reco_id=reco_id)
    plot_tau_vs_election(predictions, true_labels, output_dir, reco_id=reco_id)

    dm_accuracy = analyze_decay_mode(data, true_labels, output_dir)
    regression_results = analyze_regression_task(data, true_labels, output_dir)

    print_summary(accuracy, tau_vs_qcd_auc, dm_accuracy, regression_results)

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