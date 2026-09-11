# Methodology -- ECG-Based Serum Creatinine Prediction
**IIT Hyderabad Semester Project**
Supervised by Prof. Amit Acharyya and Dr. Pabitra Das

---

## 1. Research Hypothesis

Renal function biomarkers (serum creatinine, eGFR) leave measurable electrophysiological
fingerprints on the 12-lead ECG. Specifically:

- **Hyperkalaemia** (K+ retention in CKD) produces peaked T-waves, prolonged PR,
  and widened QRS -- all directly measurable waveform features.
- **Volume overload** (reduced GFR -> fluid retention) causes left ventricular hypertrophy
  patterns (tall R-waves in V5-V6, deep S in V1).
- **Uraemic pericarditis** (late-stage CKD) creates diffuse ST-elevation patterns.
- **Metabolic acidosis** (chronic in CKD) affects repolarisation and QTc.

The hypothesis is that a deep neural network with sufficient representational capacity
can learn these multi-lead morphological signatures jointly from raw waveforms,
avoiding hand-crafted feature engineering.

**Target output**: Continuous serum creatinine (mg/dL), from which eGFR is
deterministically derived via the race-free 2021 CKD-EPI equation (Inker et al., 2021).

---

## 2. Dataset

| Property | Value |
|----------|-------|
| Source | MIMIC-IV-ECG v1.0 (Johnson et al., 2023) |
| ECG shape | [12 leads x 5000 samples] at 500 Hz (10 s) |
| Lab linkage | labevents.itemid = 50912 (serum creatinine, mg/dL) |
| Temporal window (Delta-t) | +/-6 hours (ECG acquisition <-> blood draw) |
| Pilot size | 5,000 matched (ECG, creatinine) pairs |
| Split | 70/15/15 train/val/test, **patient-level** (subject_id) |
| Stratification | Creatinine quartile (ensures CKD severity balance per split) |

**Delta-t = +/-6 h justification**: Serum creatinine is a relatively stable biomarker
(half-life ~4 h in stable patients). A 6-hour window balances sample yield with
physiologic fidelity. Literature precedent: Lopez Alcaraz and Strodthoff (2025)
use a <=24 h window; we adopt the stricter 6 h to minimise state change confounding.

---

## 3. Signal Processing

| Step | Method | Parameters | Justification |
|------|--------|-----------|---------------|
| Resample | scipy.signal.resample_poly | -> 500 Hz | Preserves morphology; GCD-rational ratio avoids aliasing |
| Bandpass | 4th-order zero-phase Butterworth | 0.5-40 Hz | AHA/ANSI EC11 clinical ECG standard (Kligfield et al., 2007) |
| Notch | IIR notch, Q=30 | 60 Hz | US powerline; Q=30 gives 2 Hz bandwidth |
| QC | Peak-to-peak amplitude | 0.05-20 mV | Rejects disconnected (<0.05) and saturated (>20) leads |
| Normalise | Per-lead z-score | mu=0, sigma=1 | Standard for neural ECG models (Ribeiro et al., 2020) |

**Filter order 4 justification**: Matches the AHA recommendation for ECG processing
(Kligfield et al., 2007). Higher orders introduce more group delay; order 4 with
zero-phase application (sosfiltfilt) eliminates all phase distortion.

**Huber loss delta=1.0 justification**: Serum creatinine has a right-skewed distribution
(healthy peak ~0.9 mg/dL, CKD tail to ~15 mg/dL). MSE over-penalises rare high values;
MAE under-trains on them. Huber loss with delta=1.0 mg/dL (~1 SD of the healthy
distribution) provides a principled interpolation. Lopez-Martinez et al. (2023) use
Huber loss for similar continuous biomarker regression tasks.

---

## 4. Model Architectures

### 4.1 Baseline: 1D-CNN

**Reference architecture**: Adapted from Ribeiro et al. (2020) *Nature Communications*
"Automatic diagnosis of the 12-lead ECG using a deep neural network" -- we adapt the
convolutional feature extraction blocks for regression rather than 6-label classification.
Also informed by Holmstrom et al. (2023) *Communications Medicine* who specifically
apply 12-lead CNN to CKD screening from ECG.

**Modifications from cited work**:

- Output head changed from multi-class softmax -> single linear neuron (regression)

- L2 output activation removed (unbounded creatinine prediction)

- Batch normalisation added to all convolutional blocks for training stability

- Dropout p=0.5 on FC layers (Srivastava et al., 2014)

**Architecture**:
`
Input [B, 12, 5000]
-> ConvBlock(12->32,  k=7) -> MaxPool(4) -> [B, 32, 1250]
-> ConvBlock(32->64,  k=5) -> MaxPool(4) -> [B, 64, 312]
-> ConvBlock(64->128, k=5) -> MaxPool(4) -> [B, 128, 78]
-> ConvBlock(128->256,k=3) -> MaxPool(2) -> [B, 256, 39]
-> GlobalAveragePool                     -> [B, 256]
-> FC(256->64) -> ReLU -> Dropout(0.5)
-> FC(64->1)  [creatinine mg/dL]
`
Parameters: ~1.1M

### 4.2 1D ResNet-34

**Reference architecture**: He et al. (2016) *CVPR* "Deep Residual Learning for Image
Recognition" -- adapted from 2D to 1D convolutions, from ImageNet classification to
12-lead ECG regression. The 1D ResNet for ECG was validated by Hannun et al. (2019)
*Nature Medicine* for arrhythmia detection.

**Modifications from cited work**:
- All Conv2d -> Conv1d; all BN2d -> BN1d
- Input channels: 3 (RGB) -> 12 (ECG leads)
- Stem kernel: 7x7 -> 15 (wider receptive field for ECG temporal patterns)
- Output: 1000-class -> 1 continuous value
- Head: GAP -> Dropout(0.5) -> FC(512->256) -> FC(256->1)

**Architecture (layer counts = ResNet-34)**:
`
Stem: Conv1d(12->64, k=15, s=2) -> BN -> ReLU -> MaxPool(k=3, s=2)
Layer1: 3 x BasicBlock1d(64->64)
Layer2: 4 x BasicBlock1d(64->128, s=2)
Layer3: 6 x BasicBlock1d(128->256, s=2)
Layer4: 3 x BasicBlock1d(256->512, s=2)
AdaptiveAvgPool1d(1) -> FC(512->256) -> ReLU -> Dropout(0.5) -> FC(256->1)
`
Parameters: ~21.3M

### 4.3 ECG Foundation Model -- ST-MEM / ECG-FM

**Reference**: Lopez Alcaraz and Strodthoff (2025) *Scientific Reports* benchmark
ECG-FM and MERL on MIMIC-IV-ECG for 30+ lab value prediction including creatinine.
ST-MEM (Na et al., 2024) is a self-supervised ViT pretrained via masked autoencoding.

**Two fine-tuning strategies**:

1. **Linear Probe** (frozen backbone): Freeze all transformer weights. Train only a
   single linear head on top of the [CLS] token. Tests whether pretrained representations
   are linearly predictive of creatinine.

2. **LoRA Adaptation** (Hu et al., 2022 *ICLR*): Apply Low-Rank Adaptation to the
   Query and Value projections of every attention layer. Rank r=8, alpha=16.
   Only LoRA parameters + head are trained (~590K trainable parameters).

---

## 5. Training Protocol

| Setting | Value | Justification |
|---------|-------|---------------|
| Optimiser | AdamW | Adam + weight decay (Loshchilov & Hutter, 2019) |
| Learning rate | 3e-4 (CNN/ResNet), 1e-4 (ECG-FM) | Lower LR for pretrained models |
| LR schedule | CosineAnnealingLR (T_max=n_epochs) | Smooth decay (Loshchilov & Hutter, 2017) |
| Batch size | 32 | GPU memory; effective for ECG regression (Ribeiro et al., 2020) |
| Epochs | 100 max | Early stopping prevents overfitting |
| Early stopping | Patience=15, monitor val_mae | Prevents training set overfitting |
| Gradient clip | max_norm=1.0 | Prevents exploding gradients |
| Loss | Huber (delta=1.0) | See Section 3 justification |
| Mixed precision | float16 (AMP) | 2x speedup on A100/T4 |

---

## 6. Evaluation Metrics

| Metric | Clinical Relevance |
|--------|--------------------|
| MAE (creatinine, mg/dL) | Primary: clinical decisions use ~0.2 mg/dL thresholds |
| RMSE | Penalises large errors (dangerous mis-staging) |
| R-squared | Explained variance |
| Pearson r | Linear correlation strength |
| eGFR-MAE (mL/min/1.73m2) | Direct clinical impact on treatment decisions |
| CKD Stage Accuracy | Staging: G1>=90, G2=60-89, G3a=45-59, G3b=30-44, G4=15-29, G5<15 |

**eGFR derivation**: Race-free 2021 CKD-EPI equation (Inker et al., 2021, NEJM).
Requires: predicted creatinine + patient age + sex.

---

## 7. Comparisons to Prior Work

| Paper | Task | Model | Data | Our Difference |
|-------|------|-------|------|----------------|
| Holmstrom et al. (2023) *Commun. Med.* | CKD screening (binary) | 12-lead CNN | UK Biobank | We do continuous regression, not binary |
| Lopez Alcaraz & Strodthoff (2025) *Sci. Rep.* | Lab prediction (30+ tests) | LSTM/Transformer | MIMIC-IV-ECG | We focus exclusively on creatinine+eGFR |
| Ribeiro et al. (2020) *Nat. Commun.* | Arrhythmia (6-class) | 1D ResNet | CODE-15% | Backbone adapted for regression |
| Hannun et al. (2019) *Nat. Med.* | Arrhythmia (12-class) | 1D ResNet-34 | iRhythm | Architecture reference |
| Attia et al. (2019) *Nat. Med.* | EF screening from ECG | CNN | Mayo Clinic | Demonstrates continuous cardiac function from ECG |

---

## 8. Reproducibility Checklist

- [x] Global seed fixed (42) in config and logged in all HDF5 records
- [x] Patient-level split (no subject in multiple splits, leakage assertion)
- [x] Delta-t window reproducibly stored per record in HDF5
- [x] Git hash embedded in every HDF5 record
- [x] Full YAML config snapshot stored in HDF5 /metadata
- [x] All filter parameters in config.yaml (not hardcoded)
- [ ] Model checkpoints to be versioned with DVC / Git LFS
- [ ] MIMIC-IV data access via credentialed PhysioNet (pending)
