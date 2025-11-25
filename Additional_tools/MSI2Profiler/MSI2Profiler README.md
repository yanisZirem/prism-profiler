# MSI2Profiler

**MSI2Profiler** is an additional desktop tool for **Profiler**, designed to extract and preprocess **MSI (Mass Spectrometry Imaging)** data from `.imzML` files.  

It allows users to **bin**, **normalize**, **visualize**, and **export** MSI spectra for direct import into **Profiler** for downstream analysis.

---

## 📥 Download

The MSI2Profiler executable is now **available directly from the Profiler web page** for easy installation.  

**🔗 Collaboration:** The source code (`MSI2profiler.py`) remains available for developers and bioinformaticians to explore, customize, or contribute to the tool.

---

## 🛠️ Features

| Feature | Description |
|---------|-------------|
| **Load `.imzML` files** | Import MSI data from full tissue sections or regions of interest (ROIs). |
| **Bin spectral intensities** | Bin intensities over a user-defined m/z range with configurable bin size. |
| **Normalize & transform** | Normalize and optionally log-transform spectral intensities. |
| **Export processed data** | Export as labeled `.csv` or `.xlsx` files, ready for import into **Profiler**. |
| **Interactive visualization** | Visualize average spectra to check data consistency and detect anomalies. |
| **Reuse workflows** | Save and reuse processing parameters for multiple ROIs or experiments. |
| **Concatenate files** | Merge multiple CSV/Excel files into a single dataset for batch analysis. |

---

## 🖥️ Open-Source Usage (for collaboration)

For developers or bioinformaticians who want to run the source code:

1. Open a terminal (CMD, PowerShell, or macOS/Linux terminal).  
2. Navigate to the folder containing `MSI2profiler.py`.  
3. Run the following command:
```bash
python MSI2profiler.py


## Requirements

Before running, ensure all dependencies are installed:
```bash
pip install pandas numpy plotly pyimzml ttkthemes tkinter
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
