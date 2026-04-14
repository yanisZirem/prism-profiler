<div align="center">

<img src="Profiler_Desktop1.2/app/assets/profiler_logo.png" alt="Profiler Logo" width="180"/>

# Profiler Desktop v1.2 & MSI2Profiler — Offline Multi-Omics Analysis

**No upload. No limits. 100% local.**

[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](License%20%26%20Intellectual%20Property.txt)
[![Python](https://img.shields.io/badge/Python-3.8.20-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%2010%2F11-lightgrey.svg)](https://github.com/yanisZirem/prism-profiler/releases/latest)
[![DOI](https://img.shields.io/badge/DOI-10.1093%2Fbioinformatics%2Fbtaf644-green.svg)](https://doi.org/10.1093/bioinformatics/btaf644)
[![Release](https://img.shields.io/github/v/release/yanisZirem/prism-profiler?color=orange)](https://github.com/yanisZirem/prism-profiler/releases/latest)

Developed by [PRISM U1192 Laboratory](https://www.inserm.fr/en/research-inserm/prism-u1192/) — INSERM / CHU de Lille / Université de Lille

[⬇️ Download Latest Release](https://github.com/yanisZirem/prism-profiler/releases/latest) · [🏠 Profiler Homepage](https://prism-profiler.univ-lille.fr/desktop/) · [🌐 Web Version](https://prism-profiler.univ-lille.fr) · [📦 Test Datasets](https://github.com/yanisZirem/Profiler_v1_requests_datatests) · [📄 Paper](https://doi.org/10.1093/bioinformatics/btaf644)

</div>

---

## Overview

**Profiler Desktop v1.2** is the standalone, fully offline version of [Profiler](https://prism-profiler.univ-lille.fr) — an interactive platform for multi-omics data analysis. It runs entirely on your local machine: no internet connection, no data upload, no account required.

| | Web version | Desktop v1.2 |
|---|---|---|
| Installation | None | Automated (one double-click) |
| Internet required | Yes | **No** |
| Upload size limit | Restricted | **Unlimited** |
| Data privacy | In-session | **100% local** |
| All features | ✓ | ✓ |
| HTML report export | ✓ | ✓ |
| Raw data conversion | ✓ | ✓ (optional) |

---

## What's New in v1.2

- **GSEA** — Gene Set Enrichment Analysis from any output (volcano, heatmap, Venn/UpSet), joining ORA across 100+ databases
- **Regression modeling** — ML and MLP for continuous targets; R², RMSE, residual plots, cross-validation
- **Longitudinal analysis** — mixed-effects models, trajectory visualisation, repeated-measures statistics
- **HTML report generator** — one-click self-contained export of all session plots, tables, and metrics
- **Clinical metadata support** — any column ending with `_meta` used as clinical covariate or classification target
- **Extended format support** — Spectronaut, FragPipe, DESeq2/edgeR, Salmon, kallisto, MetaboAnalyst, XCMS, MZmine and more
- **Refactored architecture** — clean `app/` package structure, robust installer, `protobuf` conflict resolved

---

## Installation — Windows 10 / 11

### Requirements

- Windows 10 or 11 (64-bit)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download) — **must be installed first**
- ~3 GB disk space

> 💡 Administrator rights are **not** required.
⚠️ **Important:** During the installation of Miniconda or Anaconda, make sure to check the option **"Add to my PATH environment variable"** (or similar).  
> Otherwise, the `conda` command will not be recognized in your terminal.
---

### Step 1 — Download

Download the latest ZIP from either:

- 👉 [**GitHub Releases**](https://github.com/yanisZirem/prism-profiler/releases/latest)
- 👉 [**Profiler Homepage**](https://prism-profiler.univ-lille.fr/desktop/)

Extract it anywhere on your machine (e.g. `C:\Users\YourName\Desktop\Profiler_Desktop1.2`).

> ⚠️ Do not run anything before extracting the ZIP.

---

### Step 2 — Run the Installer

Open the extracted folder and **double-click `install_profiler.bat`**.

A terminal window will open and run automatically. Installation takes a few minutes.

**What the installer does:**

1. Verifies Conda is available
2. Accepts Conda Terms of Service automatically (conda ≥ 24.x)
3. Creates a `profiler` conda environment (Python 3.8.20)
4. Installs all Python dependencies from `requirements.txt`
5. Attempts to install **ProteoWizard / msconvert** (optional — for RAW file conversion only)
6. Creates a **"Profiler Desktop"** shortcut on your Desktop

> If ProteoWizard cannot be downloaded automatically, Profiler still works fully — only RAW file conversion is affected. Manual install instructions are displayed.

---

### Step 3 — Launch

Double-click the **"Profiler Desktop"** shortcut on your Desktop, or run `run_profiler.bat` directly.

Profiler will open in your default browser. No terminal, no commands, fully local.

> ⏱️ First launch may take 30–60 seconds while Streamlit initialises.

---

### Manual Installation (Linux / macOS / advanced)
### Requirements

- Windows 10 or 11 (64-bit)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download) — **must be installed first**
- ~3 GB disk space

> 💡 Administrator rights are **not** required.
⚠️ **Important:** During the installation of Miniconda or Anaconda, make sure to check the option **"Add to my PATH environment variable"** (or similar).  
> Otherwise, the `conda` command will not be recognized in your terminal.
---
```bash
# 1. Create environment
conda create -n profiler python=3.8.20
conda activate profiler

# 2. Install dependencies
cd /path/to/Profiler_Desktop1.2
pip install -r requirements.txt
pip install tensorflow==2.10.1 --no-deps

# 3. Set protobuf compatibility flag (required for tensorflow 2.10.1)
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python   # Linux/macOS
# or on Windows:
# set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# 4. Launch
python app/main.py
```

---

## Project Structure

```
Profiler_Desktop1.2/
│
├── install_profiler.bat       ← Windows installer (double-click to install)
├── setup_environment.ps1      ← PowerShell setup script (called by installer)
├── run_profiler.bat           ← Windows launcher (double-click to run)
├── requirements.txt           ← Python dependencies
│
└── app/
    ├── main.py                ← Entry point
    ├── gui/
    │   └── Profiler_Desktop_Gui.py   ← Main Streamlit interface
    ├── core/
    │   ├── profiler_preprocessing.py ← Normalisation & binning
    │   ├── profiler_DL.py            ← MLP / CNN / RNN
    │   └── profiler_training.py      ← Classical ML
    ├── analysis/
    │   ├── profiler_features_importance.py
    │   ├── profiler_genes_enrichment.py
    │   ├── profiler_survival.py
    │   ├── profiler_unsupervised.py
    │   ├── profiler_visualization.py
    │   ├── profiler_normality.py
    │   ├── profiler_longitudinal.py
    │   ├── profiler_rt.py
    │   └── profiler_sampling.py
    ├── data/
    │   ├── profiler_data_loading.py
    │   ├── profiler_data_exploration.py
    │   ├── profiler_conversion.py
    │   └── profiler_structured_data_file.py
    ├── utils/
    │   ├── profiler_imports.py
    │   ├── session_store.py
    │   └── reset_data_session.py
    └── assets/
        ├── profiler_logo.png
        └── profiler_icons.ttf
```

---

## Additional Tool: MSI2Profiler

Located in `Additional_tools/MSI2Profiler/` — a companion tool for **Mass Spectrometry Imaging (MSI) preprocessing**.

Supports `.imzML` files from MALDI-MSI and DESI-MSI. Outputs a Profiler-ready CSV matrix.

### ⬇️ Download for Windows (standalone executable)

**MSI2Profiler is available as a Windows executable — no installation, no Python required.**

👉 [**Download MSI2Profiler for Windows (.exe)**](https://github.com/yanisZirem/prism-profiler/releases/latest)

Just double-click and run. No Conda, no terminal, no setup.

> Also available on the [Profiler homepage](https://prism-profiler.univ-lille.fr/desktop/).

### Run from source (all platforms)

```bash
pip install pandas numpy plotly pyimzml
python MSI2profiler.py
```

Full documentation: [`Additional_tools/MSI2Profiler/`](Additional_tools/MSI2Profiler/)

---

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| OS | Windows 10 64-bit | Windows 11 |
| RAM | 16 GB | 32 GB |
| CPU | 4 cores | 8 cores |
| Storage | 3 GB | 10 GB+ |

---

## Citation

If you use Profiler or Profiler Desktop in your research, please cite:

> **Zirem, Y., Ledoux, L., Fournier, I., & Salzet, M.**
> *Profiler: an open web platform for multi-omics analysis.*
> **Bioinformatics**, Oxford University Press, 2025.
> DOI: [10.1093/bioinformatics/btaf644](https://doi.org/10.1093/bioinformatics/btaf644)
> PMID: [41324558](https://pubmed.ncbi.nlm.nih.gov/41324558/)

---

## Test Data

Example datasets available at:
👉 https://github.com/yanisZirem/Profiler_v1_requests_datatests

Includes: MaxQuant / DIA-NN outputs, Bruker & Waters RAW files, multi-omics tabular data, survival data, longitudinal data, and peer-review paper datasets.

---

## License & Intellectual Property

Profiler Desktop is proprietary software registered with the **Agence pour la Protection des Programmes (APP)**.

**IDDN Certificate:** `IDDN.FR.001.300044.000.S6.C7.2025.0009.3123010`

All rights reserved. See [`License & Intellectual Property.txt`](License%20%26%20Intellectual%20Property.txt) for full terms.

For licensing or collaboration: **yanis.zirem@univ-lille.fr**

---

## Authors

**Yanis Zirem** — PhD Candidate, PRISM U1192
📧 yanis.zirem@univ-lille.fr

Supervised by **Prof. Michel Salzet** and **Prof. Isabelle Fournier**
PRISM U1192 — INSERM / CHU de Lille / Université de Lille
