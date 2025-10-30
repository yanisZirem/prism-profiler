# MSI2Profiler

**MSI2Profiler** is an additional desktop tool for **Profiler**, designed to extract and preprocess **MSI (Mass Spectrometry Imaging)** data from `.imzML` files.
It allows users to **bin**, **normalize**, **visualize**, and **export** MSI spectra for direct import into **Profiler** for downstream analysis.

---

## Overview

`MSI2profiler.py` is a **standalone desktop application** developed in **Python with Tkinter**.
It provides a simple graphical interface to process MSI data and export them as labeled CSV or Excel tables, ready for analysis in **Profiler**.

---

## Features

| Feature | Description |
|---------|-------------|
| **Load `.imzML` files** | Import MSI data from full tissue sections or defined regions of interest (ROIs). |
| **Bin spectral intensities** | Bin intensities over a user-defined m/z range with configurable bin size. |
| **Normalize and transform** | Normalize and optionally log-transform spectral intensities. |
| **Export processed data** | Export as labeled `.csv` or `.xlsx` files, ready for import into **Profiler**. |
| **Interactive visualization** | Visualize average spectra (using Plotly) to check data consistency and detect anomalies. |
| **Reuse workflows** | Reuse parameters for multiple ROIs or replicate experiments. |
| **Concatenate files** | Merge multiple CSV/Excel files into a single dataset for batch analysis in Profiler. |

---

## 🖥️ Usage

1. **Open a terminal** (CMD, PowerShell, or macOS/Linux terminal).
2. Navigate to the folder containing `MSI2profiler.py`.
3. Run the following command:
   ```bash
   python MSI2profiler.py
   ```
4. The GUI will open.
5. Load your `.imzML` file.
6. Set parameters (m/z min, m/z max, bin size, normalization, etc.).
7. Generate and export your data for Profiler.

---

## Requirements

Before running, ensure all dependencies are installed:
```bash
pip install pandas numpy plotly pyimzml
```

---

## Technical Notes

- **Built with**: Tkinter, pandas, numpy, plotly, and pyimzml.
- **Main class**: `MSIExtractApp`, which manages the GUI and data workflow.
- **Binning function**: `matrix_class_binned()`.

---

## Preview

For a visual reference, the Profiler **User Manual** includes a preview of the MSI2Profiler GUI. 

---

## License

This tool is distributed under the same license as **Profiler**.
Please refer to the main repository’s license file for details.
