# ECG-Based Serum Creatinine Prediction from 12-Lead ECG

**IIT Hyderabad Semester Project**  
Supervised by Prof. Amit Acharyya and Dr. Pabitra Das

## Overview

This repository is a research prototype for developing an ECG-based serum-creatinine prediction workflow.

The intended long-term task is to estimate continuous serum creatinine from a standard 10-second, 12-lead ECG waveform sampled at 500 Hz (`12 × 5000`), then derive eGFR for exploratory renal-risk analysis.

The current repository supports a complete **synthetic-data pipeline**. Its purpose is to verify the software, preprocessing, patient-level split logic, storage schema, model-training workflow, and evaluation code before use with credentialed clinical data.

Synthetic results are not clinical validation and must not be interpreted as evidence of real-world diagnostic performance.

## Current status

Implemented:

- Synthetic paired ECG and serum-creatinine data generation
- Temporal ECG–lab matching with a configurable nearest-lab window
- Patient-level train/validation/test splitting
- ECG preprocessing, quality control, and HDF5 dataset construction
- 1D-CNN and 1D ResNet-34 training workflows
- Demographic baselines, regression metrics, derived eGFR calculations, and uncertainty utilities
- Unit tests and generated dataset-quality reports

Planned:

- MIMIC-IV-ECG ingestion and linkage to MIMIC-IV laboratory measurements
- Real-world cohort construction and quality control
- Patient-level internal validation
- Evaluation on a separate cross-dataset cohort
- Integration of a real pretrained ECG foundation-model backbone

## Important limitations

- The repository currently runs in `synthetic` mode by default.
- MIMIC ingestion is not yet implemented. Changing the configuration to `mode: "mimic"` will not create a real-data cohort yet.
- The ECG foundation-model options currently use a placeholder backbone. They are interface prototypes, not pretrained foundation-model experiments.
- eGFR is derived from predicted creatinine using demographic inputs; it is not an ECG-only measurement.
- A single estimated eGFR must not be interpreted as a clinical diagnosis of chronic kidney disease.

## Repository structure

```text
.
├── config/
│   ├── pipeline_config.yaml      # Reproducibility and pipeline settings
│   └── paths.py                  # Centralised path resolution
├── ecg_creatinine/               # Core signal-processing and data utilities
├── pipeline/
│   ├── generate_synthetic_dataset.py
│   ├── 03_match_ecg_creatinine.py
│   ├── 04_preprocess_signals.py
│   ├── 05_build_hdf5.py
│   └── 06_validate_dataset.py
├── datasets/                     # HDF5 dataset and DataLoader code
├── models/                       # CNN, ResNet, and foundation-model interfaces
├── training/                     # Losses and training loop
├── evaluation/                   # Metrics, eGFR, baselines, uncertainty
├── notebooks/                    # Exploratory and training notebooks
├── tests/                        # Unit tests
├── docs/                         # Methodology and references
├── run_pipeline.py               # Synthetic data-pipeline entry point
├── run_training.py               # Model-training entry point
└── requirements.txt
```

Generated data, model checkpoints, local environments, and experiment outputs are intentionally excluded from version control.

## Setup

Use a project-specific environment. Python 3.10 is recommended because the dependencies are pinned and tested for that version.

```bash
conda create --prefix .conda python=3.10 -y
conda activate .conda
pip install -r requirements.txt
```

## Run the synthetic pipeline

The default configuration generates 5,000 synthetic records, matches ECGs to synthetic creatinine values, preprocesses the waveforms, constructs an HDF5 dataset, and produces a quality-control report.

```bash
python run_pipeline.py
```

To skip report generation:

```bash
python run_pipeline.py --skip-validate
```

Generated outputs remain local under `data/` and are not uploaded to GitHub.

## Train a model

After building the synthetic HDF5 dataset:

```bash
python run_training.py --model cnn
```

Available model options:

```bash
python run_training.py --model cnn
python run_training.py --model resnet34
python run_training.py --model ecgfm_probe
python run_training.py --model ecgfm_lora
```

The ECG foundation-model commands currently use a mock backbone and are included to validate the training interfaces. They are not equivalent to training or fine-tuning a released pretrained ECG foundation model.

## Signal-processing workflow

Each waveform is processed as follows:

1. Resample to 500 Hz when required.
2. Apply a zero-phase 0.5–40 Hz Butterworth bandpass filter.
3. Apply waveform quality-control checks, including lead amplitude limits.
4. Standardise each lead with z-score normalisation.
5. Store both filtered raw-scale and normalised waveform representations in HDF5.

The target input representation is 12 leads by 5,000 samples.

## Data-splitting and reproducibility safeguards

- All records from the same `subject_id` are assigned to one split only.
- The code checks for patient overlap across train, validation, and test splits.
- ECG–lab pairs use the nearest creatinine result within the configured time window.
- Generated outputs record relevant configuration values, random seed, pipeline version, and Git revision where available.

## Tests

```bash
pytest tests/ -v --tb=short
```

For coverage reporting:

```bash
pytest tests/ -v --tb=short --cov=ecg_creatinine --cov-report=term-missing
```

## Roadmap

1. Complete large-scale synthetic pipeline stress testing.
2. Implement MIMIC-IV-ECG cohort ingestion and laboratory matching.
3. Train and validate models using patient-level held-out test data.
4. Evaluate robustness across clinically relevant subgroups.
5. Perform cross-dataset evaluation with carefully documented waveform and cohort differences.

## Academic integrity and attribution

This repository is original project code. Scientific methods, datasets, and external literature are documented in `docs/METHODOLOGY.md` and `docs/references.bib`.

No raw MIMIC data, patient-level data, credentials, or access tokens are stored in this repository.
