# Profiler Desktop
---

## Overview
**Profiler Desktop** is the standalone, offline version of the Profiler platform ([prism-profiler.univ-lille.fr](https://prism-profiler.univ-lille.fr)), an interactive  interactive application dedicated to **multi-omics data analysis**, designed for users who want to run analyses **locally on their own computer**, without uploading data to a remote server.

Profiler Desktop provides:
- full offline execution
- local data privacy
- the same analytical features as the web version




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

Profiler Desktop is primarily designed for **Windows users** with a **fully automated installer**.  
A manual installation is also available for advanced users or non-Windows systems.

---

## ✅ Recommended Installation (Windows 10 / 11 – Automatic)

### 1️⃣ Download Profiler Desktop

#### Option A – Download ZIP (recommended)
1. Go to the Profiler Desktop GitHub repository.
2. Click **Code → Download ZIP**.
3. **Extract the ZIP archive** anywhere on your computer  
   (example: `C:\Users\YourName\Downloads\prism-profiler-desktop-v1.0`).

⚠️ **Do not run anything before extracting the ZIP file.**

---

### 2️⃣ Start the Automatic Installer (VERY IMPORTANT)

After extracting the folder:

1. Open the extracted folder `prism-profiler-desktop-v1.0`
2. **Double-click on:**
   install_profiler.cmd file
✅ **This is the only file you need to run.**  
❌ Do **NOT** run `install_profiler.ps1` manually.

The installer will open in a command window and guide you automatically.

⏳ Installation may take several minutes — this is normal.

---

### 🔧 What the Automatic Installer Does

The installer automatically:
- Detects whether **Conda** is already installed
- Installs **Miniconda (user-only)** if Conda is missing
- Creates a Conda environment named **`profiler`**
- Installs all required Python dependencies
- Downloads and installs **ProteoWizard (msconvert)** locally
- Adds ProteoWizard to the **user PATH**
- Creates a **Desktop shortcut** named **Profiler Desktop**

💡 **Administrator rights are NOT required.**

---

### 🚀 Launch Profiler Desktop

Once installation is complete:

👉 **Double-click the “Profiler Desktop” shortcut on your Desktop**

Profiler Desktop will:
- start automatically
- open in your default web browser
- run fully locally on your machine

✅ No terminal  
✅ No Conda commands  
✅ No configuration required  

---


✅ **This is the only file you need to run.**  
❌ Do **NOT** run `install_profiler.ps1` manually.

The installer will open in a command window and guide you automatically.

⏳ Installation may take several minutes — this is normal.

---

### 🔧 What the Automatic Installer Does

The installer automatically:
- Detects whether **Conda** is already installed
- Installs **Miniconda (user-only)** if Conda is missing
- Creates a Conda environment named **`profiler`**
- Installs all required Python dependencies
- Downloads and installs **ProteoWizard (msconvert)** locally
- Adds ProteoWizard to the **user PATH**
- Creates a **Desktop shortcut** named **Profiler Desktop**

💡 **Administrator rights are NOT required.**

---

### 🚀 Launch Profiler Desktop

Once installation is complete:

👉 **Double-click the “Profiler Desktop” shortcut on your Desktop**

Profiler Desktop will:
- start automatically
- open in your default web browser
- run fully locally on your machine

✅ No terminal  
✅ No Conda commands  
✅ No configuration required  

---

## ❌ If the Automatic Installation Fails

In rare cases (corporate firewall, offline machines), the automatic installer may fail.

➡️ In that case, follow the **manual installation** below.

---

## 🛠️ Manual Installation (All Platforms)

### 1️⃣ Install Conda
Install **Anaconda** or **Miniconda**:
- https://www.anaconda.com/download  
- https://docs.conda.io/en/latest/miniconda.html  

Restart your terminal after installation.

---

### 2️⃣ Create the Conda Environment
Open **Anaconda Prompt** (Windows) or a terminal (Linux/macOS):

```bash
conda create -n profiler python=3.8.20

When asked:
**Proceed ([y]/n)?**
Type `y` and press **Enter**.

#### Activate the Environment
```bash
conda activate profiler
```

### 3️⃣Install Dependencies
Go to your **Profiler Desktop** folder, for example:
```bash
cd /path/to/prism-profiler-desktop-v1.0
```
Then install the required Python packages:
```bash
pip install -r requirements.txt
```

---

### 4️⃣Install ProteoWizard (msconvert) – Optional
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

### 5️⃣ Run Profiler Desktop (Manual)
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

### 6️⃣ (Optional) Automate Launch with Scripts
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

