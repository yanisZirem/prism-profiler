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

## 🖥️ Installation Guide (Windows – Recommended)

Profiler Desktop is designed to be installed **automatically on Windows** with a single click.  
⚠️ **Manual installation is only required if the automatic setup fails.**

---

## ✅ Recommended Installation (Automatic – Windows 10 / 11)

### 1️⃣ Download Profiler Desktop

You can either **download** or **clone** the repository.

#### Option A – Download ZIP (recommended for most users)
1. Go to the GitHub repository page.
2. Click **Code → Download ZIP**.
3. Extract the folder on your computer  
   (example: `prism-profiler-desktop-v1.0`).

---

### 2️⃣ Install Anaconda (Required)

Download and install **Anaconda** from:  
👉 https://www.anaconda.com/download

During installation, **make sure to check**:  
✅ **“Add Anaconda3 to my PATH environment variable”**

Then **restart your computer**.

---

### 3️⃣ Automatic Installation (One Click)

Profiler Desktop includes an automatic installer for Windows.

#### How to run the installer
1. Open the extracted folder `prism-profiler-desktop-v1.0`
2. **Right-click** on `install_profiler.ps1`
3. Click **“Run with PowerShell”**
4. If a security message appears, click **Yes / Allow**

⏳ The installation may take several minutes.

#### What the installer does automatically
- Verifies your Anaconda installation
- Creates a Conda environment named **`profiler`** (Python 3.8.20)
- Installs all required Python dependencies
- Downloads and installs **ProteoWizard (msconvert)**
- Adds ProteoWizard to the system PATH
- Creates a **desktop shortcut** named **Profiler Desktop**

---

### ✅ After Installation

Once the installation is finished:

👉 **Double-click the “Profiler Desktop” shortcut on your Desktop**

Profiler Desktop will start automatically and open in your default web browser.

💡 **You do NOT need to open Anaconda or a terminal anymore.**


---

## ❌ If the Automatic Installation Fails

If the PowerShell installer does not work on your system, follow the **manual installation** below.

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

