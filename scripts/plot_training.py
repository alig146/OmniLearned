#!/usr/bin/env python3
"""
Training Log Analysis

Parse a training log file and plot:
  - Training and validation loss vs epoch
  - Time per epoch
  - Learning rate schedule
"""

import re
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use(hep.style.ATLAS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_plot(xlabel="", ylabel="", title="", xlim=None, ylim=None):
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
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_log(log_path):
    """Extract epoch metrics from a training log file.

    Returns a dict with lists: epoch, train_loss, val_loss, lr, time_sec.
    """
    epoch_re  = re.compile(
        r"Epoch \[(\d+)/\d+\] Loss: ([\d.]+), Val Loss: ([\d.eE+\-]+)\s*, lr: ([\d.eE+\-]+)"
    )
    time_re   = re.compile(r"Time taken for epoch (\d+) is ([\d.]+) sec")

    data = {}  # keyed by 1-based epoch number

    with open(log_path) as f:
        for line in f:
            # Strip optional rank prefix "  0: "
            line = re.sub(r"^\s*\d+:\s*", "", line)

            m = epoch_re.search(line)
            if m:
                ep = int(m.group(1))
                data.setdefault(ep, {})
                data[ep]["train_loss"] = float(m.group(2))
                data[ep]["val_loss"]   = float(m.group(3))
                data[ep]["lr"]         = float(m.group(4))
                continue

            m = time_re.search(line)
            if m:
                # "Time taken for epoch N" uses 0-based N → epoch N+1
                ep = int(m.group(1)) + 1
                data.setdefault(ep, {})
                data[ep]["time_sec"] = float(m.group(2))

    epochs      = sorted(data.keys())
    train_loss  = [data[e]["train_loss"]           for e in epochs]
    val_loss    = [data[e]["val_loss"]             for e in epochs]
    lr          = [data[e]["lr"]                   for e in epochs]
    time_sec    = [data[e].get("time_sec", None)   for e in epochs]

    return dict(epochs=epochs, train_loss=train_loss, val_loss=val_loss,
                lr=lr, time_sec=time_sec)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_loss(metrics, output_dir):
    epochs     = metrics["epochs"]
    train_loss = metrics["train_loss"]
    val_loss   = metrics["val_loss"]

    fig, ax = setup_plot(
        xlabel="Epoch",
        ylabel="Loss",
    )
    ax.plot(epochs, train_loss, color='#e41a1c', lw=2, label="Train")
    ax.plot(epochs, val_loss,   color='#377eb8', lw=2, label="Validation")
    ax.legend()
    save_plot(fig, f"{output_dir}/training_loss.png")


def plot_time_per_epoch(metrics, output_dir):
    epochs   = metrics["epochs"]
    times    = metrics["time_sec"]

    # Drop epochs where timing wasn't recorded
    pairs = [(e, t) for e, t in zip(epochs, times) if t is not None]
    if not pairs:
        print("No timing data found — skipping time-per-epoch plot.")
        return

    ep_vals, t_vals = zip(*pairs)
    t_minutes = [t / 60.0 for t in t_vals]

    fig, ax = setup_plot(
        xlabel="Epoch",
        ylabel="Time per epoch [min]",
    )
    ax.plot(ep_vals, t_minutes, color='#4daf4a', lw=2, marker='o', ms=4)
    save_plot(fig, f"{output_dir}/time_per_epoch.png")


def plot_learning_rate(metrics, output_dir):
    epochs = metrics["epochs"]
    lr     = metrics["lr"]

    fig, ax = setup_plot(
        xlabel="Epoch",
        ylabel="Learning rate",
    )
    ax.plot(epochs, lr, color='#984ea3', lw=2)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    save_plot(fig, f"{output_dir}/learning_rate.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot training metrics from log file.")
    parser.add_argument("log", help="Path to training log file")
    parser.add_argument("-o", "--output-dir", default="plots/results",
                        help="Directory to save plots (default: plots/results)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = parse_log(args.log)
    print(f"Parsed {len(metrics['epochs'])} epochs from {args.log}")

    plot_loss(metrics, output_dir)
    plot_time_per_epoch(metrics, output_dir)
    plot_learning_rate(metrics, output_dir)


if __name__ == "__main__":
    main()
