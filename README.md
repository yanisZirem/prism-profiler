# Profiler Desktop — v1.2

> **Offline multi-omics analysis. No upload. No limits.**  
> Developed by [PRISM U1192 Laboratory](https://www.inserm.fr/en/research-inserm/prism-u1192/), Université de Lille — Protected by INSERM Transfert

---

## Overview

**Profiler Desktop v1.2** is the standalone, fully offline version of [Profiler](https://prism-profiler.univ-lille.fr) — an interactive platform for multi-omics data analysis. It runs entirely on your local machine: no internet connection, no data upload, no account required.

**v1.2 adds:** GSEA enrichment, continuous regression (ML + MLP), longitudinal analysis module, one-click HTML report generation, and improved SHAP/LIME visualisations.

### Why Desktop?

| | Web version | Desktop v1.2 |
|---|---|---|
| Installation | None | Required (automated) |
| Internet required | Yes | **No** |
| Upload size limit | Restricted | **Unlimited** |
| Data privacy | In-session | **100% local** |
| All features | ✓ | ✓ |
| HTML report export | ✓ | ✓ |
| Raw data conversion | ✓ (sidebar) | ✓ (sidebar) |

---

## What's New in v1.2

- **GSEA** — Gene Set Enrichment Analysis from any analysis output (volcano, heatmap, Venn/UpSet), joining ORA across 100+ databases
- **Regression modeling** — ML and MLP for continuous targets; R², RMSE, residual plots, cross-validation
- **Longitudinal analysis** — mixed-effects models, trajectory visualisation, repeated-measures statistics (`Subject_ID` + `Time` columns)
- **HTML report generator** — one-click self-contained export of all session plots, tables, and metrics
- **Clinical metadata support** — any column ending with `_meta` is treated as clinical metadata; available as alternative classification/regression targets, and used to colour heatmap annotations, PCA/UMAP points, and model training
- **Extended format support** — auto-detection extended to Spectronaut, FragPipe, DESeq2/edgeR, Salmon, kallisto, MetaboAnalyst, XCMS, MZmine...
- **Bug fixes & stability**

---

## Citation

If you use **Profiler** or **Profiler Desktop** in your research, please cite:

> **Zirem, Y., Ledoux, L., Fournier, I., & Salzet, M.**  
> *Profiler: an open web platform for multi-omics analysis.*  
> **Bioinformatics**, Oxford University Press, 2025.  
> DOI: [10.1093/bioinformatics/btaf644](https://doi.org/10.1093/bioinformatics/btaf644)  
> PMID: [41324558](https://pubmed.ncbi.nlm.nih.gov/41324558/)

---

## Test Data

Example datasets for Profiler Desktop are available at:  
👉 https://github.com/yanisZirem/Profiler_v1_requests_datatests

Includes: MaxQuant / DIA-NN outputs, raw Bruker & Waters files, multi-omics tabular data (binary & multi-class), survival data, and the peer-review paper datasets and longitudinal data.

---

## Installation Guide

Profiler Desktop v1.2 is designed primarily for **Windows** with a fully automated installer.  
Manual installation is also available for Linux and macOS.

---

### ✅ Recommended — Windows 10 / 11 (Automatic Installer)

#### Step 1 — Download

- Go to the Profiler Desktop GitHub repository
- Click **Code → Download ZIP**
- **Extract the ZIP** anywhere on your computer (e.g. `C:\Users\YourName\Downloads\prism-profiler-desktop-v1.2`)

> ⚠️ Do not run anything before extracting the ZIP.

#### Step 2 — Run the Installer

1. Open the extracted folder
2. **Double-click:** `install_profiler.cmd`

> ✅ This is the only file you need to run.  
> ❌ Do **not** run `install_profiler.ps1` manually.

The installer opens a command window and runs automatically. Installation may take a few minutes.

> **Conda Terms of Service:** Since 2024, Conda requires acceptance of its Terms of Service. The installer handles this automatically — no action required from the user.

#### What the Installer Does

The installer automatically:

- Detects whether **Conda** is already installed
- Installs **Miniconda (user-only)** if missing
- Creates a Conda environment named **`profiler`**
- Installs all Python dependencies
- Sets up the **raw data conversion** module (Bruker / Waters / Thermo Fisher → mzML)
- Creates a **Desktop shortcut** named *Profiler Desktop*

> 💡 Administrator rights are **not** required.

#### Step 3 — Launch

Double-click the **"Profiler Desktop"** shortcut on your Desktop.

Profiler Desktop will start automatically and open in your default web browser. Fully local — no terminal, no Conda commands, no configuration.

---

### 🛠️ Manual Installation (All Platforms)

#### 1. Install Conda

Download **Miniconda** (recommended) or Anaconda:
- https://docs.conda.io/en/latest/miniconda.html
- https://www.anaconda.com/download

Restart your terminal after installation.

#### 2. Create the Environment

```bash
conda create -n profiler python=3.8.20
```

When prompted **Proceed ([y]/n)?** → type `y` and press Enter.

```bash
conda activate profiler
```

#### 3. Install Dependencies

Navigate to the Profiler Desktop folder and run:

```bash
cd /path/to/prism-profiler-desktop-v1.2
pip install -r requirements.txt
```

#### 4. Launch Profiler Desktop

Each time you want to use Profiler Desktop:

```bash
conda activate profiler
cd /path/to/prism-profiler-desktop-v1.2
python profiler_desktop.py
```

Profiler Desktop will open automatically in your browser.

#### 5. (Optional) Automate Launch

**Windows — `run_profiler.bat`**

```batch
@echo off
call conda activate profiler
python profiler_desktop.py
pause
```

**Linux / macOS — `run_profiler.sh`**

```bash
#!/bin/bash
conda activate profiler
python profiler_desktop.py
```

```bash
chmod +x run_profiler.sh
./run_profiler.sh
```

---

### ❌ If the Automatic Installer Fails

In rare cases (corporate firewall, offline machines), the automatic installer may not complete.  
→ Follow the **Manual Installation** steps above.

---

## Additional Tool: MSI2Profiler

Profiler Desktop includes **MSI2Profiler**, a companion tool for **Mass Spectrometry Imaging (MSI) preprocessing**.

Located in: `Additional_tools/MSI2Profiler/`

### Features

MSI2Profiler allows you to:

- Load `.imzML` files from **MALDI-MSI** and **DESI-MSI** experiments
- Normalise spectra (**TIC**, **Median**, **RMS**)
- Bin **m/z** features
- Concatenate **ROIs (Regions of Interest)**
- Export a **Profiler-ready CSV matrix**

> Download **MSI2Profiler** directly from the  
> [Profiler homepage](https://prism-profiler.univ-lille.fr)  
> or get the Windows executable here:  
> https://prism-profiler.univ-lille.fr/desktop/

### Run MSI2Profiler

```bash
python MSI2profiler.py
```

### Dependencies

```bash
pip install pandas numpy plotly pyimzml
```

### Documentation

For full instructions:

https://github.com/yanisZirem/prism-profiler/blob/main/Additional_tools/MSI2Profiler/MSI2Profiler%20README.md

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|--------|-------------|
| OS | Windows 10 / macOS 11 / Ubuntu 20.04 | Windows 11 / macOS 13 / Ubuntu 22.04 |
| RAM | 16 GB | 32 GB |
| CPU | 4 cores | 8 cores |
| Storage | 2 GB (app) + data | 10 GB+ |

---

## Authors & Contact

Developed by **Yanis Zirem**  
📧 yanis.zirem@univ-lille.fr  

Supervised by **Prof. Michel Salzet** and **Prof. Isabelle Fournier**
