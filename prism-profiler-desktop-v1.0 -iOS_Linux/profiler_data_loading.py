"""
Software Name: Profiler
Module name : loading
Author: Yanis Zirem
Email : yanis.zirem@yahoo.com / yanis.zirem@univ-lille.fr
Creation Date: 15/01/2025
Last Updated: 23/10/2025
Version: 1.0.0

Context:
This module is part of the "Profiler" project, originally developed for a web version (https://prism-profiler.univ-lille.fr) and now adapted for a desktop version (profiler_desktop_GUI).
It is designed for archiving on Zenodo and integration into GitHub releases.

License: l’Agence pour la Protection des Programmes IDDN (InterDeposit Digital Number) : FR2 .0013 .0300044 .0005 .S6 .C7 .20258 .0009 .312301
Citation:
If Profiler or this module (a part of Profiler) is used in a publication, please cite:
Zirem, Y. (2025). Profiler: an open web platform for multi-omics analysis. Journal of Bioinformatics. [DOI or Zenodo/GitHub link available in the article].

Links:
- GitHub temporary Repository: https://github.com/yanisZirem/Profiler_v1_requests_datatests
"""


import os
import numpy as np
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from pyopenms import MSExperiment, MzMLFile, MzXMLFile
from scipy.signal import detrend, find_peaks
import plotly.express as px
from io import BytesIO
import tempfile
import gc

def save_uploaded_file(uploaded_file):
    """Temporarily saves an uploaded file and returns its path."""
    temp_path = os.path.join("temp_files", uploaded_file.name)
    os.makedirs("temp_files", exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path

def process_file(uploaded_file, class_name, progress_bar, current_file, total_files, all_mz_values):
    """Process a single uploaded file and return its data."""
    file_path = save_uploaded_file(uploaded_file)
    exp = MSExperiment()
    try:
        if file_path.endswith(".mzML"):
            MzMLFile().load(file_path, exp)
        elif file_path.endswith(".mzXML"):
            MzXMLFile().load(file_path, exp)
        else:
            st.warning(f"Unsupported format: {uploaded_file.name}")
            return pd.DataFrame()

        chromatograms = exp.getChromatograms()
        spectra = exp.getSpectra()

        if not chromatograms and not spectra:
            st.warning(f"No chromatograms or spectra found in {uploaded_file.name}")
            return pd.DataFrame()

        peak_spectra = []

        if chromatograms:
            chromatogram = chromatograms[0]
            times, intensities = chromatogram.get_peaks()

            if len(times) > 0 and len(intensities) > 0:
                intensities = detrend(intensities)
                peak_height_threshold = np.max(intensities) * 0.5
                peaks, _ = find_peaks(intensities, height=peak_height_threshold)
                peak_rt_values = [times[i] for i in peaks]

                for spectrum in spectra:
                    if any(abs(spectrum.getRT() - rt) < 1 for rt in peak_rt_values):
                        peak_spectra.append(spectrum)
        else:
            for spectrum in spectra:
                mz_values, intensities = spectrum.get_peaks()
                if len(mz_values) > 0 and len(intensities) > 0:
                    if any(intensity > np.mean(intensities) * 1000 for intensity in intensities):
                        peak_spectra.append(spectrum)

        peak_spectra = [spectrum for spectrum in peak_spectra if len(spectrum.get_peaks()[0]) > 0]

        if not peak_spectra:
            st.warning(f"No valid peak spectra found in {uploaded_file.name}")
            return pd.DataFrame()

        file_mz_values = sorted(set(mz for spectrum in peak_spectra for mz, _ in zip(*spectrum.get_peaks())))
        all_mz_values.update(file_mz_values)

        if not file_mz_values:
            st.warning(f"No m/z values extracted from {uploaded_file.name}")
            return pd.DataFrame()

        data = []
        for spectrum in peak_spectra:
            mz_values, intensities = spectrum.get_peaks()
            rt_value = spectrum.getRT()
            intensity_dict = dict(zip(mz_values, intensities))
            row_data = {mz: intensity_dict.get(mz, np.nan) for mz in file_mz_values}
            row_data.update({'Class': class_name, 'File': uploaded_file.name, 'RT': rt_value, 'Sum': sum(intensities)})
            data.append(row_data)

        file_data = pd.DataFrame(data).astype({col: np.float32 for col in data[0] if col not in ['Class', 'File']})

        # Puis on force seulement les deux colonnes Class et File en str
        file_data['Class'] = file_data['Class'].astype(str)
        file_data['File'] = file_data['File'].astype(str)

        return file_data
    except Exception as e:
        st.error(f"Error loading {uploaded_file.name}: {str(e)}")
        return pd.DataFrame()

    finally:
        os.remove(file_path)  # Remove the temporary file
        progress_bar.progress((current_file + 1) / total_files)


def add_group():
    st.session_state["file_groups"].append({"class_name": f"Class {len(st.session_state['file_groups']) + 1}", "files": []})
    st.rerun()  # Forcer le rafraîchissement immédiat


def load_uploaded_files(grouped_files, progress_bar, peak_height_threshold=50):
    """Loads and structures mzML/mzXML files for each group with threshold filtering."""
    all_data = pd.DataFrame()
    progress_placeholder = st.empty()
    total_files = sum(len(group["files"]) for group in grouped_files)
    current_file = 0
    all_mz_values = set()

    for group in grouped_files:
        class_name = group["class_name"]
        for uploaded_file in group["files"]:
            current_file += 1
            tmp_file_path = None
            try:
                suffix = ".mzML" if uploaded_file.name.endswith(".mzML") else ".mzXML"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name

                exp = MSExperiment()
                if suffix == ".mzML":
                    MzMLFile().load(tmp_file_path, exp)
                elif suffix == ".mzXML":
                    MzXMLFile().load(tmp_file_path, exp)
                else:
                    st.warning(f"Unsupported format: {uploaded_file.name}")
                    continue

                chromatograms = exp.getChromatograms()
                spectra = exp.getSpectra()
                peak_spectra = []

                if not chromatograms and not spectra:
                    st.warning(f"No chromatograms or spectra found in {uploaded_file.name}")
                    continue

                if chromatograms:
                    chromatogram = chromatograms[0]
                    times, intensities = chromatogram.get_peaks()
                    if len(times) > 0 and len(intensities) > 0:
                        intensities = detrend(intensities)
                        actual_threshold = np.max(intensities) * (peak_height_threshold / 100.0)
                        peaks, _ = find_peaks(intensities, height=actual_threshold)
                        peak_rt_values = [times[i] for i in peaks]

                        for spectrum in spectra:
                            if any(abs(spectrum.getRT() - rt) < 1 for rt in peak_rt_values):
                                peak_spectra.append(spectrum)

                # En l'absence de chromatogrammes ou pics détectés, fallback sur spectres
                if not peak_spectra and spectra:
                    for spectrum in spectra:
                        mz_values, intensities = spectrum.get_peaks()
                        if len(mz_values) == 0 or len(intensities) == 0:
                            continue
                        # Appliquer un filtrage par seuil même ici
                        if np.max(intensities) >= np.mean(intensities) * (peak_height_threshold / 100.0):
                            peak_spectra.append(spectrum)

                peak_spectra = [s for s in peak_spectra if len(s.get_peaks()[0]) > 0]

                if not peak_spectra:
                    st.warning(f"No valid peak spectra found in {uploaded_file.name} (threshold too high or no signal).")
                    empty_row = {'Class': class_name, 'File': uploaded_file.name, 'RT': np.nan, 'Sum': 0.0}
                    all_data = pd.concat([all_data, pd.DataFrame([empty_row])], ignore_index=True)
                    continue

                file_mz_values = sorted(set(
                    mz for spectrum in peak_spectra for mz, _ in zip(*spectrum.get_peaks())
                ))
                all_mz_values.update(file_mz_values)

                if not file_mz_values:
                    st.warning(f"No m/z values extracted from {uploaded_file.name}")
                    empty_row = {'Class': class_name, 'File': uploaded_file.name, 'RT': np.nan, 'Sum': 0.0}
                    all_data = pd.concat([all_data, pd.DataFrame([empty_row])], ignore_index=True)
                    continue

                data = []
                for spectrum in peak_spectra:
                    mz_values, intensities = spectrum.get_peaks()
                    rt_value = spectrum.getRT()
                    intensity_dict = dict(zip(mz_values, intensities))
                    row_data = {mz: intensity_dict.get(mz, np.nan) for mz in file_mz_values}
                    row_data.update({
                        'Class': class_name,
                        'File': uploaded_file.name,
                        'RT': rt_value,
                        'Sum': sum(intensities)
                    })
                    data.append(row_data)

                file_data = pd.DataFrame(data)
                if not file_data.empty:
                    all_data = pd.concat([all_data, file_data], ignore_index=True)

            except Exception as e:
                st.error(f"Error loading {uploaded_file.name}: {str(e)}")

            finally:
                if tmp_file_path and os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)

            progress_bar.progress(current_file / total_files)
            progress_placeholder.write(f"Progress: {current_file / total_files:.0%}")

    if not all_data.empty:
        desired_columns = ['Class', 'File', 'RT', 'Sum'] + sorted(all_mz_values)
        all_data = all_data.reindex(columns=desired_columns)
        st.success("Data loaded and structured successfully.")

        st.session_state['class_colors'] = {
            class_name: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
            for i, class_name in enumerate(all_data['Class'].dropna().unique())
        }
    else:
        st.error("No valid files were loaded. Check formats, signal quality, or threshold.")

    gc.collect()
    return all_data

# --- Fonction utilitaire pour récupérer les données selon la source ---
def get_data(source_name):
    mapping = {
        "Raw Data": st.session_state.get('final_data', st.session_state.get('data')),
        "Preprocessed": st.session_state.get('preprocessed_data'),
        "Oversampled": st.session_state.get('oversampled_data'),
        "Undersampled": st.session_state.get('undersampled_data')
    }
    return mapping.get(source_name, None)

# --- Lazy import helper ---
def lazy_import(module_name: str):
    import importlib
    return importlib.import_module(module_name)


def safe_load_data(data_source):
    if data_source == 'None':
        return None

    session_key = f"cached_data_{data_source}"
    if session_key not in st.session_state:
        with st.spinner("Loading selected data source..."):
            df = get_data(data_source)
            if df is not None:
                st.session_state[session_key] = df
            else:
                st.session_state[session_key] = None
    return st.session_state[session_key]