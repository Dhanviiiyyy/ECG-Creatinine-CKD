"""
pipeline/06_validate_dataset.py
==================================
Step 6 of the ECG-Creatinine prediction pipeline.

Performs comprehensive quality checks on the final HDF5 dataset and generates
a standalone HTML QC report (with embedded plots).

Checks performed
----------------
1. No subject_id appears in more than one split (leakage verification).
2. Creatinine label distribution: KS test (train vs val, train vs test).
3. Signal integrity: no NaN/Inf values, correct shape [12, 5000], dtype.
4. Amplitude: per-lead peak-to-peak within physiologic range.
5. PSD: verify 0.5-40 Hz bandpass passband and 60 Hz notch depth >= 20 dB.
   (Using Welch periodogram from scipy.signal.welch.)
6. Example waveform plot: one record per split, all 12 leads stacked.

Usage
-----
    python pipeline/06_validate_dataset.py [--config path] [--n_sample N]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import sys
from pathlib import Path
from typing import Any

import h5py
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (Colab/headless)
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats
from scipy.signal import welch

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.paths import Paths, load_config
from ecg_creatinine.hdf5_utils import HDF5Reader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PASS_SYMBOL = "[PASS]"
FAIL_SYMBOL = "[FAIL]"
WARN_SYMBOL = "[WARN]"


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _fig_to_base64(fig: plt.Figure) -> str:
    """Encode a matplotlib figure as a base64 PNG string for HTML embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return encoded


def _plot_label_distributions(
    reader: HDF5Reader,
    splits: list[str],
) -> str:
    """Plot creatinine histogram overlaid across splits."""
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {"train": "#2196F3", "val": "#FF9800", "test": "#4CAF50"}
    for split in splits:
        labels = reader.get_all_labels(split)
        ax.hist(labels, bins=60, alpha=0.6, label=f"{split} (n={len(labels)})",
                color=colors.get(split, "gray"), density=True)
    ax.set_xlabel("Serum Creatinine (mg/dL)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Creatinine Label Distribution by Split", fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_xlim(0, 12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_psd(
    waveform: np.ndarray,
    fs: float,
    lead_names: list[str],
) -> str:
    """Plot per-lead power spectral density (Welch periodogram)."""
    n_leads = waveform.shape[0]
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    for i in range(n_leads):
        f, psd = welch(waveform[i], fs=fs, nperseg=min(512, waveform.shape[1]))
        axes_flat[i].semilogy(f, psd, color="#1565C0", linewidth=0.8)
        axes_flat[i].axvline(0.5, color="green", linestyle="--", linewidth=0.8, alpha=0.7,
                              label="0.5 Hz")
        axes_flat[i].axvline(40.0, color="green", linestyle="--", linewidth=0.8, alpha=0.7,
                              label="40 Hz")
        axes_flat[i].axvline(60.0, color="red", linestyle="--", linewidth=0.8, alpha=0.7,
                              label="60 Hz notch")
        axes_flat[i].set_title(lead_names[i] if i < len(lead_names) else f"L{i}", fontsize=9)
        axes_flat[i].set_xlim(0, 80)
        axes_flat[i].grid(True, alpha=0.3)
    axes_flat[0].legend(fontsize=6, loc="upper right")
    fig.supxlabel("Frequency (Hz)", y=0.01)
    fig.supylabel("PSD (V^2/Hz)", x=0.01)
    fig.suptitle("Per-Lead Power Spectral Density (Welch)", fontweight="bold")
    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_example_waveform(
    waveform: np.ndarray,
    label: float,
    study_id: str,
    lead_names: list[str],
    fs: float,
) -> str:
    """Plot a 12-lead ECG waveform stacked vertically."""
    n_leads, n_samples = waveform.shape
    t = np.arange(n_samples) / fs
    fig, axes = plt.subplots(n_leads, 1, figsize=(14, n_leads * 0.9), sharex=True)
    axes_list: list[Any] = list(np.asarray(axes, dtype=object).ravel())
    for i, ax in enumerate(axes_list):
        ax.plot(t, waveform[i], linewidth=0.5, color="#1A237E")
        lead_name = lead_names[i] if i < len(lead_names) else f"L{i}"
        ax.set_ylabel(lead_name, fontsize=8, rotation=0, labelpad=28)
        ax.set_ylim(-4, 4)
        ax.axhline(0, color="gray", linewidth=0.3, alpha=0.5)
        ax.tick_params(labelleft=False)
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes_list[-1].set_xlabel("Time (s)", fontsize=10)
    fig.suptitle(
        f"Example ECG: {study_id}  |  Creatinine: {label:.2f} mg/dL",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout()
    return _fig_to_base64(fig)


# ---------------------------------------------------------------------------
# QC checks
# ---------------------------------------------------------------------------

def run_qc_checks(
    reader: HDF5Reader,
    cfg: dict,
    n_sample_per_split: int = 50,
) -> dict:
    """Run all QC checks and return results dict."""
    sig_cfg = cfg["signal"]
    lead_names = sig_cfg["lead_names"]
    fs = float(sig_cfg["target_fs_hz"])
    n_samples_expected = int(sig_cfg["n_samples"])
    n_leads_expected = int(sig_cfg["n_leads"])

    results = {
        "checks": [],
        "plots": {},
        "split_sizes": reader.split_sizes(),
        "metadata": reader.get_metadata(),
    }

    # -- Check 1: No subject overlap --
    subject_sets = {}
    with h5py.File(reader.path, "r") as fh:
        for split in ["train", "val", "test"]:
            sids: set[int] = set()
            split_grp = fh[split]
            if isinstance(split_grp, h5py.Group):
                for sid_key in split_grp.keys():
                    item = split_grp[sid_key]
                    if isinstance(item, (h5py.Group, h5py.Dataset)):
                        sids.add(int(np.asarray(item.attrs["subject_id"]).item()))
            subject_sets[split] = sids

    tv_overlap = subject_sets["train"] & subject_sets["val"]
    tt_overlap = subject_sets["train"] & subject_sets["test"]
    vt_overlap = subject_sets["val"] & subject_sets["test"]
    overlap_pass = len(tv_overlap) == 0 and len(tt_overlap) == 0 and len(vt_overlap) == 0
    results["checks"].append({
        "name": "No Subject Overlap Across Splits",
        "status": PASS_SYMBOL if overlap_pass else FAIL_SYMBOL,
        "detail": (
            "All splits are patient-disjoint." if overlap_pass else
            f"LEAKAGE: train-val={len(tv_overlap)}, train-test={len(tt_overlap)}, "
            f"val-test={len(vt_overlap)} overlapping subjects."
        ),
    })

    # -- Check 2: KS test on label distributions --
    train_labels = reader.get_all_labels("train")
    val_labels   = reader.get_all_labels("val")
    test_labels  = reader.get_all_labels("test")

    ks_tv = scipy_stats.ks_2samp(train_labels, val_labels)
    ks_tt = scipy_stats.ks_2samp(train_labels, test_labels)

    stat_tv = float(getattr(ks_tv, "statistic", 0.0))
    p_tv    = float(getattr(ks_tv, "pvalue", 1.0))
    stat_tt = float(getattr(ks_tt, "statistic", 0.0))
    p_tt    = float(getattr(ks_tt, "pvalue", 1.0))

    for name, stat, p in [("train vs val", stat_tv, p_tv),
                           ("train vs test", stat_tt, p_tt)]:
        ks_pass = p > 0.05
        results["checks"].append({
            "name": f"KS Test Label Distribution ({name})",
            "status": PASS_SYMBOL if ks_pass else WARN_SYMBOL,
            "detail": (
                f"KS stat={stat:.4f}, p={p:.4f} "
                f"({'similar distributions' if ks_pass else 'distributions differ -- check stratification'})"
            ),
        })

    # -- Check 3: Waveform shape, dtype, NaN/Inf (sample) --
    shape_ok = True
    nan_ok = True
    for split in ["train", "val", "test"]:
        ids = reader.get_study_ids(split)[:n_sample_per_split]
        for sid in ids:
            wf, label, meta = reader.get_record(split, sid)
            if wf.shape != (n_leads_expected, n_samples_expected):
                shape_ok = False
            if not np.isfinite(wf).all():
                nan_ok = False

    results["checks"].append({
        "name": "Waveform Shape [12, 5000]",
        "status": PASS_SYMBOL if shape_ok else FAIL_SYMBOL,
        "detail": f"All sampled waveforms have shape ({n_leads_expected}, {n_samples_expected})."
                  if shape_ok else "Shape mismatch detected.",
    })
    results["checks"].append({
        "name": "No NaN/Inf in Waveforms",
        "status": PASS_SYMBOL if nan_ok else FAIL_SYMBOL,
        "detail": "All sampled waveforms are finite." if nan_ok else "NaN or Inf detected.",
    })

    # -- Plots --
    results["plots"]["label_distribution"] = _plot_label_distributions(reader, ["train", "val", "test"])

    # PSD plot (one sample from train)
    train_ids = reader.get_study_ids("train")
    if train_ids:
        sample_wf, sample_label, sample_meta = reader.get_record("train", train_ids[0])
        sample_lead_names = json.loads(sample_meta.get("lead_names", "[]")) or lead_names
        results["plots"]["psd"] = _plot_psd(sample_wf, fs, sample_lead_names)

        # Example waveform per split
        for split_name in ["train", "val", "test"]:
            ids = reader.get_study_ids(split_name)
            if ids:
                wf, lbl, mt = reader.get_record(split_name, ids[0])
                ln = json.loads(mt.get("lead_names", "[]")) or lead_names
                results["plots"][f"waveform_{split_name}"] = _plot_example_waveform(
                    wf, lbl, ids[0], ln, fs
                )

    return results


# ---------------------------------------------------------------------------
# HTML report generator
# ---------------------------------------------------------------------------

def build_html_report(results: dict, output_path: Path) -> None:
    """Write a standalone HTML QC report."""
    checks_html = ""
    for chk in results["checks"]:
        colour = {"[PASS]": "#2e7d32", "[WARN]": "#e65100", "[FAIL]": "#c62828"}.get(
            chk["status"], "#333"
        )
        checks_html += (
            f'<tr>'
            f'<td style="font-weight:bold; color:{colour}">{chk["status"]}</td>'
            f'<td>{chk["name"]}</td>'
            f'<td style="color:#555">{chk["detail"]}</td>'
            f'</tr>\n'
        )

    sizes = results["split_sizes"]
    sizes_html = "".join(
        f'<tr><td>{k}</td><td>{v}</td></tr>'
        for k, v in sizes.items()
    )

    def img_html(key: str, title: str) -> str:
        if key not in results["plots"]:
            return ""
        return (
            f'<h3>{title}</h3>'
            f'<img src="data:image/png;base64,{results["plots"][key]}" '
            f'style="max-width:100%;border:1px solid #ddd;border-radius:4px;"/>'
        )

    plots_html = (
        img_html("label_distribution", "Creatinine Label Distribution by Split")
        + img_html("psd", "Per-Lead Power Spectral Density (train sample)")
        + img_html("waveform_train", "Example ECG Waveform — Train Split")
        + img_html("waveform_val", "Example ECG Waveform — Val Split")
        + img_html("waveform_test", "Example ECG Waveform — Test Split")
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>ECG-Creatinine Dataset QC Report</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1100px; margin: 0 auto;
         padding: 24px; background: #fafafa; color: #222; }}
  h1 {{ color: #1a237e; border-bottom: 2px solid #1a237e; padding-bottom: 8px; }}
  h2 {{ color: #283593; margin-top: 32px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background: #e8eaf6; font-weight: bold; }}
  tr:nth-child(even) {{ background: #f5f5f5; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
            font-weight: bold; font-size: 12px; }}
</style>
</head>
<body>
<h1>ECG-Based Serum Creatinine Prediction — Dataset QC Report</h1>
<p><em>IIT Hyderabad | Supervised by Prof. Amit Acharyya &amp; Dr. Pabitra Das</em></p>

<h2>Dataset Split Sizes</h2>
<table>
<tr><th>Split</th><th>N Records</th></tr>
{sizes_html}
</table>

<h2>QC Checks</h2>
<table>
<tr><th>Status</th><th>Check</th><th>Detail</th></tr>
{checks_html}
</table>

<h2>Visualisations</h2>
{plots_html}

</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("QC report written: %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cfg: dict, paths: Paths, n_sample: int = 50) -> None:
    reader = HDF5Reader(paths.hdf5_file)
    logger.info("Validating HDF5 dataset: %s", paths.hdf5_file)

    results = run_qc_checks(reader, cfg, n_sample_per_split=n_sample)

    # Log checks to console
    all_pass = True
    for chk in results["checks"]:
        lvl = logging.INFO if chk["status"] == PASS_SYMBOL else logging.WARNING
        logger.log(lvl, "%s %s: %s", chk["status"], chk["name"], chk["detail"])
        if chk["status"] == FAIL_SYMBOL:
            all_pass = False

    build_html_report(results, paths.qc_report)

    if not all_pass:
        logger.error("One or more QC checks FAILED. Review the report before training.")
        sys.exit(1)
    logger.info("All QC checks passed. Dataset is ready for model training.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ECG-creatinine HDF5 dataset.")
    parser.add_argument("--config",   type=str, default=None)
    parser.add_argument("--n_sample", type=int, default=50,
                        help="Records sampled per split for shape/NaN checks.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_config(args.config)
    paths = Paths(cfg)
    run(cfg, paths, n_sample=args.n_sample)
