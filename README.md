# Profiler Desktop

---

## Overview
**Profiler Desktop** is the standalone, offline version of the Profiler platform ([prism-profiler.univ-lille.fr](https://prism-profiler.univ-lille.fr)), an interactive application for **multi-omics data analysis**.
Designed for users who prefer **local execution**, Profiler Desktop delivers the same analytical power as the web version, directly on your computer.

## Citation
If you use **Profiler** or **Profiler Desktop** in your research, please cite our peer-reviewed publication:

> **Zirem, Y., Ledoux, L., Fournier, I., & Salzet, M.**
> *Profiler: an open web platform for multi-omics analysis.*
> **Bioinformatics**, 2025.
> DOI: [10.1093/bioinformatics/btaf644](https://doi.org/10.1093/bioinformatics/btaf644)
> PMID: [41324558](https://pubmed.ncbi.nlm.nih.gov/41324558/)

## Test Data
Example and test datasets for Profiler Desktop can be found here:  
https://github.com/yanisZirem/Profiler_v1_requests_datatests

---

## 🖥️ Installation Guide
Follow the steps below to install and run **Profiler Desktop**.

---
### 1️⃣ Clone or Download the Repository
You can either **clone** or **download** the Profiler Desktop folder.

#### Option A – Download ZIP
1. Go to the GitHub repository page.
2. Click **Code** → **Download ZIP**.
3. Extract the folder (e.g., `prism_profiler-desktop-v1.0`) on your computer.



#### Option B – Clone via Git
```bash
git clone https://github.com/yanisZirem/prism-profiler
```
---

### 2️⃣ Install Anaconda
Download and install Anaconda from:
👉 [https://www.anaconda.com/download](https://www.anaconda.com/download)

During installation, check the box:
✅ **"Add Anaconda3 to my PATH environment variable"**

---

### 3️⃣ Automatic Installation Script (Windows Only)
For Windows users, Profiler Desktop can be installed automatically using the provided PowerShell script `install_profiler.ps1`.
This script will handle everything for you (environment setup, dependency installation and ProteoWizard configuration). Supported System : Windows 10 or 11

####  Automatic Installation)
1. Open **Windows PowerShell as Administrator**.
   navigate to prism-profiler-desktop-v1.0 folder :
   ```
   cd /yourpath/prism-profiler-desktop-v1.0
   ```
3. Run the installer (bypass PowerShell restrictions if needed):
   ```powershell
   Set-ExecutionPolicy Bypass -Scope Process -Force
   .\install_profiler.ps1
   ```
   The script will:
   - Verify your Anaconda installation
   - Create a Conda environment named `profiler` with Python 3.8.20
   - Install all dependencies from `requirements.txt`
   - Download and install ProteoWizard (msconvert)
   - Add ProteoWizard to your system PATH

####  After Installation
Once the setup is complete:
1. Open **Anaconda Prompt**.
2. Activate the environment:
   ```bash
   conda activate profiler
   ```
3. Navigate to your Profiler Desktop folder:
   ```bash
   cd path\to\prism-profiler-desktop-v1.0
   ```
4. Run Profiler Desktop:
   ```bash
   python profiler_desktop.py
   ```

####  Notes
- If `msconvert` is not recognized after installation, simply restart your computer to refresh the PATH.

---

### 4️⃣ Manual Installation (All Platforms)
If you are not using the automatic script (or are on macOS/Linux), follow these steps:

#### Create a Python Environment
Open **Anaconda Prompt** (or terminal on Linux/macOS) and type:
```bash
conda create -n profiler python=3.8.20
```
When asked:
**Proceed ([y]/n)?**
Type `y` and press **Enter**.

#### Activate the Environment
```bash
conda activate profiler
```

#### Install Dependencies
Go to your **Profiler Desktop** folder, for example:
```bash
cd /path/to/prism-profiler-desktop-v1.0
```
Then install the required Python packages:
```bash
pip install -r requirements.txt
```

---

### 5️⃣ Install ProteoWizard (msconvert) – Optional
**Note:** This step is only required if you plan to convert raw mass spectrometry files (e.g., `.raw`, `.wiff`, `.d` to `.mzML` or `.mzXML`). If you do not need this functionality, you can skip to the next step.

#### 🔧 Windows Installation
1. Download ProteoWizard:
   👉 [https://proteowizard.sourceforge.io/download.html](https://proteowizard.sourceforge.io/download.html)
2. Install it (default path: `C:\Program Files\ProteoWizard\ProteoWizard 3.0.24143`).
3. Add this path to your system **PATH** variable:
   ```
   C:\Program Files\ProteoWizard\ProteoWizard 3.0.24143
   ```
4. Verify installation:
   ```bash
   msconvert --help
   ```
   If you see the list of `msconvert` options, it’s installed correctly 

####  Linux Installation
1. Download the latest Linux binary from:
   👉 [https://proteowizard.sourceforge.io/download.html](https://proteowizard.sourceforge.io/download.html)
2. Extract the downloaded file and move the `msconvert` binary to `/usr/local/bin`:
   ```bash
   sudo mv msconvert /usr/local/bin/
   ```
3. Verify installation:
   ```bash
   msconvert --help
   ```

####  macOS Installation
1. Download the latest macOS binary from:
   👉 [https://proteowizard.sourceforge.io/download.html](https://proteowizard.sourceforge.io/download.html)
2. Open the downloaded `.dmg` file and drag `msconvert` to your Applications folder or `/usr/local/bin`.
3. Verify installation:
   ```bash
   msconvert --help
   ```

---

### 6️⃣ Run Profiler Desktop
Each time you want to use **Profiler Desktop**:
1. Open **Anaconda Prompt** (or terminal on Linux/macOS).
2. Activate your environment:
   ```bash
   conda activate profiler
   ```
3. Go to the **Profiler Desktop** directory:
   ```bash
   cd /path/to/prism-profiler-desktop-v1.0
   ```
4. Launch the application:
   ```bash
   python profiler_desktop.py
   ```
Profiler Desktop will open automatically in your default web browser.

---

### 7️⃣ (Optional) Automate Launch with Scripts
To simplify the launch process, you can create a script to activate the environment and run Profiler Desktop automatically.

#### 🪟 Windows (`.bat` file)
1. Create a new file named `run_profiler.bat` in your **Profiler Desktop** folder.
2. Add the following content:
   ```batch
   @echo off
   call conda activate profiler
   python profiler_desktop.py
   pause
   ```
3. Double-click the file to run Profiler Desktop.

#### 🐧 Linux/macOS (`.sh` file)
1. Create a new file named `run_profiler.sh` in your **Profiler Desktop** folder.
2. Add the following content:
   ```bash
   #!/bin/bash
   conda activate profiler
   python profiler_desktop.py
   ```
3. Make the script executable:
   ```bash
   chmod +x run_profiler.sh
   ```
4. Run the script:
   ```bash
   ./run_profiler.sh
   ```
## Additional Tool: MSI2Profiler

Profiler Desktop includes an additional utility, **MSI2Profiler**, located in the `MSI2Profiler/` folder.  
This standalone desktop app allows you to extract and preprocess MSI (Mass Spectrometry Imaging) data from `.imzML` files for use in Profiler.

➡️ **For detailed instructions**, see the **[MSI2Profiler README](https://github.com/yanisZirem/prism-profiler/blob/main/Additional_tools/MSI2Profiler/MSI2Profiler%20README.md)**.
---

Developed by **PRISM U1192 Laboratory, Université de Lille**
Protected by **INSERM Transfert**
Built by **Yanis Zirem (Third year PhD Candidate, 2025)** under the supervision of **Prof. Michel Salzet** and **Prof. Isabelle Fournier**

