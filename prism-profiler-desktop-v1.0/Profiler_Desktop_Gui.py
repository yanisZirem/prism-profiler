"""
Software Name: Profiler
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
import tempfile
import shutil
import uuid
import functools
from typing import Callable, Any
import streamlit as st
import re
import gseapy as gp
from neurocombat_sklearn import CombatModel
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import joblib
#import yaml
import base64
import os
import bcrypt
import shutil
import zipfile
import threading
from pathlib import Path
from itertools import combinations
#from yaml.loader import SafeLoader
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from passlib.context import CryptContext
from neurocombat_sklearn import CombatModel
from reset_data_session import reset_data_session_keys
import pandas as pd
import multiprocessing
from profiler_imports import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from scipy.stats import f_oneway
import io
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import traceback
from streamlit_autorefresh import st_autorefresh
import uuid
import time, datetime
import gc
import plotly.express as px
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns



if "workflow_step" not in st.session_state:
    st.session_state.workflow_step = 0
# Liste des étapes
workflow_steps = [
    "🧭 <strong>Data Conversion (if needed)</strong><br><small>Convert raw files (e.g., .raw) to mzML using MSConvert.</small>",
    "📥 <strong>Data Import</strong><br><small>Upload tabular omics datasets (e.g., CSV, mzML, MaxQuant).</small>",
    "🧹 <strong>Preprocessing</strong><br><small>Clean, normalize,correct batches, impute, and transform your data.</small>",
    "🔍 <strong>Exploratory Analysis</strong><br><small>Check distributions, correlations, and class balance.</small>",
    "🌀 <strong>Clustering & Heterogeneity</strong><br><small>Discover natural groups or study tumor heterogeneity.</small>",
    "📊 <strong>Visualization</strong><br><small>Generate PCA, t-SNE, heatmaps, volcano plots, etc.</small>",
    "📈 <strong>Statistical Modeling</strong><br><small>Differential analysis and hypothesis testing.</small>",
    "🤖 <strong>Machine Learning</strong><br><small>Train, evaluate, and interpret predictive models.</small>",
    "🕸️ <strong>Pathway & Enrichment</strong><br><small>Biological interpretation via enrichment analysis.</small>",
    "⏳ <strong>Survival Analysis</strong><br><small>Model survival outcomes and stratify risk groups.</small>",
    "⚡ <strong>Real-Time Prediction</strong><br><small>Predict instantly from new unseen data.</small>",
]
st.markdown(
    """
    <style>
        .sidebar .block-container {
            padding-top: 20px;
            padding-bottom: 20px;
        }
        .sidebar img {
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s;
        }
        .sidebar img:hover {
            transform: scale(1.05);
        }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <style>
        .sidebar .block-container {
            padding-top: 20px;
            padding-bottom: 20px;
        }
        .sidebar img {
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s;
        }
        .sidebar img:hover {
            transform: scale(1.05);
        }
    </style>
    """,
    unsafe_allow_html=True
)
def initialize_session_state():
    session_vars = [
        'data', 'preprocessed_data', 'oversampled_data', 'undersampled_data',
        'compressed_data', 'reduced_data', 'models', 'dl_models', 'class_colors',
        'oversampling_technique', 'undersampling_technique', 'feature_distribution_plot',
        'mean_spectrum_plot', 'individual_spectra_plot', 'pc_index', 'file_groups',
        'class_renaming', 'rename_pending', 'feature_data_source_1', 'feature_data_source_2',
        'feature_data_source_3', 'show_shap', 'show_lime', 'show_boxplots', 'show_heatmap',
        'show_volcano', 'class_column', 'is_maxquant', 'selected_feature_row',
        'selected_data_source', 'latest_result', 'processed_files', 'monitoring',
        'normalization', 'numeric_features', 'label_encoder', 'svd_model', 'label_colors',
        'selected_feature', 'survival_data', 'expand_load_data', 'workflow_step', 'workflow_active'
    ]

    for var in session_vars:
        if var not in st.session_state:
            if var in ['models', 'dl_models', 'class_colors', 'label_colors', 'class_renaming', 'rename_pending', 'processed_files']:
                st.session_state[var] = {}
            elif var in ['oversampling_technique', 'undersampling_technique', 'feature_data_source_1', 'feature_data_source_2', 'feature_data_source_3', 'selected_feature_row', 'selected_data_source', 'normalization']:
                st.session_state[var] = 'None'
            elif var in ['numeric_features']:
                st.session_state[var] = []
            elif var in ['pc_index']:
                st.session_state[var] = 0
            elif var in ['is_maxquant', 'monitoring', 'show_shap', 'show_lime', 'show_boxplots', 'show_heatmap', 'show_volcano', 'expand_load_data']:
                st.session_state[var] = False  
            elif var == 'class_column':
                st.session_state[var] = 'Class'
            elif var == 'file_groups':
                st.session_state[var] = []  
            else:
                st.session_state[var] = None


def main():
    initialize_session_state()

    # working directory to the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    #logo
    st.sidebar.image("profiler_logo.png", use_column_width=True)

    # Initialisation des variables de session 
    keys_with_defaults = {
        'data': None,
        'preprocessed_data': None,
        'oversampled_data': None,
        'undersampled_data': None,
        'compressed_data': None,
        'reduced_data': None,
        'models': {},
        'dl_models': {},
        'class_colors': {},
        'oversampling_technique': 'None',
        'undersampling_technique': 'None',
        'feature_distribution_plot': None,
        'mean_spectrum_plot': None,
        'individual_spectra_plot': None,
        'pc_index': 0,
        'file_groups': [],
        'class_renaming': {},
        'rename_pending': {},
        'feature_data_source_1': 'None',
        'feature_data_source_2': 'None',
        'feature_data_source_3': 'None',
        'show_shap': False,
        'show_lime': False,
        'show_boxplots': False,
        'show_heatmap': False,
        'show_volcano': False,
        'class_column': 'Class',
        'is_maxquant': False,
        'selected_feature_row': "Choose an option",
        'selected_data_source': 'None',
        'latest_result': None,
        'processed_files': set(),
        'monitoring': False,
        'normalization': "None",
        'numeric_features': [],
        'label_encoder': None,
        'svd_model': None,
        'label_colors': {},
        'selected_feature': None,
        'survival_data': None,
        'expand_load_data': True,
        'expander_open_anomaly': True,
        'workflow_step': 0,
        'workflow_active': False
    }

    for key, default in keys_with_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

    with st.sidebar.expander("🔄 Session Management"):
        if st.sidebar.button("Reset session", help="Clear all analysis-related data"):
            keys_to_keep = {"auth_status", "username", "password", "name", "last_active"}
            keys_to_remove = [key for key in st.session_state.keys() if key not in keys_to_keep]

            for key in keys_to_remove:
                del st.session_state[key]
            import gc
            gc.collect()
            st.rerun()

    # Sidebar Inputs for Conversion
    with st.sidebar.expander("🧭 Data Conversion"):
        raw_dir = st.text_input("RAW Files Directory", key="raw_dir")
        output_dir = st.text_input("Output Directory", key="output_dir")
        file_type = st.selectbox("File Type", ["waters", "thermo", "bruker"], key="file_type")
        mass_range = st.text_input("Mass Range (e.g., [600,1000])", key="mass_range")
        peak_picking = st.checkbox("Enable Peak Picking", key="peak_picking")
        lock_mass = st.text_input("Lock Mass Value (e.g., 554)", key="lock_mass") if file_type == "waters" else None
        output_format = st.selectbox("Output Format", ['mzML', 'mzXML', 'mz5', 'mzDB'], key="output_format")

        if st.button("Convert RAW", key="convert_raw"):
            try:
                if os.path.exists(raw_dir) and os.path.exists(output_dir):
                    convert_raw_to_mzml(
                        raw_dir, output_dir, file_type,
                        eval(mass_range) if mass_range else None,
                        peak_picking,
                        float(lock_mass) if lock_mass else None,
                        output_format
                    )
                    st.success("Conversion successful!")
                else:
                    st.error("Specified directories do not exist.")
            except Exception as e:
                st.error(f"Error during conversion: {e}")


    with st.sidebar.expander("📂 Load MS1 spectra standard format"):
        for i, group in enumerate(st.session_state["file_groups"]):
            st.session_state["file_groups"][i]["class_name"] = st.text_input(
                f"Class name",
                value=group["class_name"],
                key=f"class_name_{i}",
                help="Enter a name for this class of samples (e.g., Tumor, Control)."
            )

            uploaded = st.file_uploader(
                f"Select files (mzML...) for {group['class_name'] or f'Class {i}'}",
                accept_multiple_files=True,
                type=["mzML", "mzXML"],
                key=f"files_{i}",
                help="Upload one or multiple .mzML files corresponding to this class."
            )

            # Affectation directe, sans conserver `uploaded`
            st.session_state["file_groups"][i]["files"] = uploaded
            del uploaded

        peak_height_threshold = st.number_input(
            "Peak Height Threshold (% of peaks above baseline)",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            key="peak_height_threshold",
            help="ℹ️ Adjust this threshold to filter out noise and focus on relevant spectra."
        )

        if st.button("Initiate a class"):
            add_group()

        if st.button("Load Data"):
            if any(group.get("files") for group in st.session_state["file_groups"]):
                with st.spinner("🔄 Loading and processing MS files..."):
                    progress_bar = st.progress(0)

                    st.session_state["data"] = load_uploaded_files(
                        st.session_state["file_groups"],
                        progress_bar,
                        peak_height_threshold
                    )

                    # Nettoyage mémoire
                    del progress_bar
                    for group in st.session_state["file_groups"]:
                        group.pop("files", None)
                    gc.collect()
            else:
                st.error("Please select at least one file.")


    # Initialisation de l'état de session pour l'expander si non défini
    if 'expand_load_data' not in st.session_state:
        st.session_state.expand_load_data = False


    with st.sidebar.expander("🗂️ Load Tabular Data", expanded=st.session_state.expand_load_data):
        st.markdown("Supports Protein Group files directly from DIA-NN or MaxQuant.")
        uploaded_file = st.file_uploader(
            "Upload a tabular dataset: Proteomic, Metabolomic, RNAseq...",
            type=["csv", "xlsx", "txt", "tsv"],
            key="uploaded_file",
            help="Upload a CSV, Excel, or text file containing your structured omics data. Make sure it includes a 'Class' column if possible."
        )

        def finalize_data_load(df, source_label):
            """Add required columns and store dataframe in session state."""
            if df is not None:
                for col in ['File', 'RT', 'Sum']:
                    if col not in df.columns:
                        df[col] = "Unknown" if col == 'File' else 0
                st.session_state['data'] = df
                if 'Class' in df.columns:
                    st.session_state['class_renaming'] = {cls: cls for cls in df['Class'].unique()}
                    import plotly.express as px
                    st.session_state['class_colors'] = {
                        class_name: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                        for i, class_name in enumerate(df['Class'].unique())
                    }
                st.success(f"{source_label} data processed successfully!")

        if uploaded_file is not None:
            df = load_structured_data(uploaded_file)
            st.session_state.expand_load_data = True

            if df is not None and 'Class' in df.columns:
                finalize_data_load(df, "Generic")

            else:
                file_type = st.selectbox("Select file type", ["Choose an option", "DIA-NN", "Perseus", "MaxQuant"])

                if file_type == "Perseus":
                    feature_options = ["Choose an option", "T: Gene names", "T: Protein names"]
                    selected = st.selectbox("Select the row to use for feature names:", feature_options,
                                            index=feature_options.index(st.session_state.get('selected_feature_row', "Choose an option")))
                    st.session_state['selected_feature_row'] = selected

                    if selected != "Choose an option":
                        with st.spinner('Processing Perseus file...'):
                            feature_row_index = -2 if selected == "T: Gene names" else -1
                            df = perseus_data(uploaded_file, feature_row_index=feature_row_index)
                            finalize_data_load(df, "Perseus")

                elif file_type == "DIA-NN":
                    feature_options = ["Choose an option", "Genes", "Protein.Names"]
                    selected = st.selectbox("Select the row to use for feature names:", feature_options)
                    if selected != "Choose an option":
                        feature_row_index = 1 if selected == "Genes" else 0
                        with st.spinner('Processing DIA-NN file...'):
                            df = diann_data(uploaded_file, feature_row_index=feature_row_index )
                            finalize_data_load(df, "DIA-NN")

                elif file_type == "MaxQuant":
                    feature_options = ["Choose an option", "Gene names", "Protein names"]
                    selected = st.selectbox("Select the row to use for feature names:", feature_options)
                    if selected != "Choose an option":
                        feature_row_index = -1 if selected == "Gene names" else -2
                        with st.spinner('Processing MaxQuant file...'):
                            df = maxquant_data(uploaded_file, feature_row_index=feature_row_index)
                            finalize_data_load(df, "MaxQuant")


        st.markdown("### 📄 Expected Format Example")
        st.info(
            """
            | Class  | F1   | F2   | ... |
            |--------|------|------|-----|
            | A      | 1257 | 1.0  | ... |
            | B      | 7521 | 443  | ... |

            - **Class** = target labels (e.g., Control, Condition1)  
            - **F1, F2** = any features (e.g., proteins, genes, ions)
            """
        )

    with st.sidebar.expander("🕰️ Load Survival Data"):
        import pandas as pd 
        uploaded_file = st.file_uploader(
            "Upload a survival dataset (CSV, XLSX, TXT):",
            type=["csv", "xlsx", "txt"],
            key="uploaded_file_survival_side",
            help="Upload a file containing survival data. Accepted formats: CSV, Excel, or TXT."
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith('.txt'):
                    file_content = uploaded_file.getvalue().decode('utf-8')  # Decode text file
                    buffer = io.StringIO(file_content)
                    delimiter = detect_delimiter(file_content)
                    df = pd.read_csv(buffer, delimiter=delimiter)

                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)  # Directly read Excel file, no decoding

                else:
                    st.error("Unsupported file format. Please upload a CSV, XLSX, or TXT file.")
                    df = None
            except Exception as e:
                st.error(f"Error loading file: {e}")
                df = None

            if df is not None:


                analysis_type = st.selectbox(
                    "Choose Analysis Type",
                    ["Choose an option", "Kaplan-Meier", "Cox Model"],
                    help="Select the type of survival analysis you want to perform: Kaplan-Meier for group comparison, or Cox Model for multivariate regression."
                )
                st.session_state['analysis_type'] = analysis_type

                if analysis_type == "Kaplan-Meier":
                    required_columns = ['Overall survival', 'State', 'Class']
                    if all(col in df.columns for col in required_columns):
                        if 'survival_data' not in st.session_state:
                            st.session_state['survival_data'] = None
                        st.session_state['survival_data'] = df
                        st.success("Survival Data Loaded Successfully!")
                    else:
                        st.error("The uploaded file must contain the columns: 'Overall survival', 'State', and 'Class'.")

                elif analysis_type == "Cox Model":
                    required_columns = ['Overall survival', 'State']
                    if all(col in df.columns for col in required_columns):
                        st.session_state['survival_data'] = df
                        st.success("Survival Data Loaded Successfully!")
                    else:
                        st.error("The uploaded file must contain the columns: 'Overall survival' and 'State'")
            del df
            del uploaded_file
            import gc
            gc.collect()
        st.markdown("### 📄 Expected Format Example")
        st.info(
            """
        **Kaplan-Meier Format:**

        | Overall survival | State | Class |
        |------------------|-------|--------|
        | 12               | 1     | A      |
        | 8                | 0     | B      |

        - **Overall survival**: time (e.g., months/days)  
        - **State**: Event indicator (0 = censored, 1 = death/event, relapse/event...)
        - **Class**: group/condition to compare
        """
        )
        st.info(
            """
        **Cox Model Format:**

        |O.. S..|State|Age|lipidX|
        |----------------|-----|---|------|
        |10              |1    |67 | 2.3  |
        |15              |0    |59 | 1.8  |

        - **Remaining columns** = covariates (e.g., age, gene expression, protein expression...)
        """
        )

        st.markdown(
    """
    <style>
        div[data-baseweb="tab-list"] {
            display: flex;
            justify-content: space-evenly;
            width: 100%;
        }
        button[data-baseweb="tab"] {
            flex-grow: 1;
            text-align: center;
            font-size: 24px;
            padding: 15px 0;
        }
        /* Style amélioré pour TOUS les boutons, y compris ceux avec help= */
        .stButton button {
            background-color: #318CE7;
            color: white;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 12px;
            transition: all 0.2s ease;
        }
        .stButton button:hover {
            background-color: #4682B4;
        }
        .stButton button:active {
            background-color: #4682B4;
            box-shadow: 0 5px #666;
            transform: translateY(4px);
        }
    </style>
    """,
    unsafe_allow_html=True
)



    hide_decoration_bar_style = '''
        <style>
            header {visibility: hidden;}
            div[data-baseweb="tab-list"] {
                display: flex;
                justify-content: space-evenly;
                width: 100%;
            }
            button[data-baseweb="tab"] {
                flex-grow: 1;
                text-align: center;
                font-size: 24px;
                padding: 15px 0;
            }
            .stButton button {
                background-color: #318CE7;
                color: white;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 12px;
                transition: all 0.2s ease;
            }
            .stButton button:hover {
                background-color: #4682B4;
            }
            .stButton button:active {
                background-color: #4682B4;
                box-shadow: 0 5px #666;
                transform: translateY(4px);
            }
        </style>
    '''
    st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

    tabs = st.tabs(["**Home**", "**Data Exploration**", "**AI Modeling**", "**Biomarker Discovery**", "**Enrichment**", "**Survival Analysis**", "**Wizard**"])

    with tabs[0]:
        st.markdown("""
            <h3 style="text-align: center; color: #318CE7; padding: 12px; background-color: #f0f8ff; border-radius: 8px;">
                Welcome to <strong>Profiler</strong>
            </h3>
            <p><strong>Profiler</strong> is an innovative omics data analysis platform developed by the
            <a href="https://www.laboratoire-prism.fr/" target="_blank"><strong>PRISM U1192 laboratory</strong></a> and protected by <strong>INSERM Transfer</strong>.
            The platform was designed as part of ongoing academic research to advance automated, AI-driven analysis in the field of omics.</p>
            <p>Profiler brings the power of artificial intelligence, statistical modeling, and automation to the analysis of complex biological data.</p>
            <h4 style="color: #318CE7;"> Key features</h4>
            <ul>
                <li><strong>Multi-Omics Support:</strong> Proteomics, metabolomics, lipidomics, genomics, transcriptomics, and more.</li>
                <li><strong>Simple Data Preprocessing:</strong> Convert, clean, and explore your data effortlessly.</li>
                <li><strong>AI & Statistical Integration:</strong> Combine classical stats, ML, and DL in one platform.</li>
                <li><strong>Explainable Results:</strong> LIME, SHAP, volcano plots, and clustering heatmaps.</li>
                <li><strong>Smart Suggestions:</strong> Recommended tests and imputations based on your data.</li>
                <li><strong>High-Speed Processing:</strong> Optimized for performance with powerful back-end.</li>
                <li><strong>End-to-End Workflow:</strong> From raw input to biological insights.</li>
            </ul>
        """, unsafe_allow_html=True)

        # Add download buttons for documentation and IDDN certificate
        with st.container():
            st.markdown("""
                
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                with open("documentation.pdf", "rb") as doc_file:
                    st.download_button(
                        label="📘 Download Profiler Documentation",
                        data=doc_file,
                        file_name="documentation.pdf",
                        mime="application/pdf"
                    )
            with col2:
                with open("IDDN Certificate.pdf", "rb") as iddn_file:
                    st.download_button(
                        label="⚖️  Download INSERM Protection Certificate",
                        data=iddn_file,
                        file_name="IDDN Certificate.pdf",
                        mime="application/pdf"
                    )

        st.markdown("""
            <div style='
                text-align: center;
                margin-top: 40px;
                margin-bottom: 40px;'>
                <a href='https://github.com/yanisZirem/Profiler_v1_requests_datatests.git' target='_blank' style='
                    display: inline-block;
                    font-size: 22px;
                    font-weight: bold;
                    padding: 15px 30px;
                    background-color: #318CE7;
                    color: white;
                    text-decoration: none;
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);'>
                    ➡️ Access Profiler's GitHub 🌐
                </a>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div style='
                text-align: center;
                margin-top: 40px;
                margin-bottom: 40px;'>
                <a href='https://github.com/yanisZirem/prism-profiler.git' target='_blank' style='
                    display: inline-block;
                    font-size: 22px;
                    font-weight: bold;
                    padding: 15px 30px;
                    background-color: #318CE7;
                    color: white;
                    text-decoration: none;
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);'>
                    ➡️ Profiler Desktop Version  🌐
                </a>
            </div>
        """, unsafe_allow_html=True)


        st.markdown("""
             <h4 style="color: #318CE7;">📚 Citation & Publications</h4>
             <p><strong>Until the peer-reviewed publication is released, please cite our pre-print:</strong></p>
             <ul>
                 <li><strong>Zirem, Y.</strong>, <strong>Ledoux, L.</strong>, <strong>Fournier, I.</strong>, <strong>Salzet, M.</strong> "Profiler: an open web platform for multi-omics analysis" <em>Université de Lille</em>, 2025. DOI: <a href="https://doi.org/10.21203/rs.3.rs-7058776/v1">10.21203/rs.3.rs-7058776/v1</a></li>
             </ul>
             <p><strong>Related Papers:</strong></p>
             <ul>
                 <li><strong>Zirem, Y.</strong>, <strong>Ledoux, L.</strong>, et al. "Real-time glioblastoma tumor microenvironment assessment by SpiderMass..." <em>Cell Reports Medicine</em>, 2024.</li>
                 <li><strong>Zirem, Y.</strong>, et al. "Protocol to analyze 1D and 2D mass spectrometry data..." <em>STAR Protocols</em>, 2024.</li>
                 <li><strong>Zirem, Y.</strong>, <strong>Lagache, L.</strong>, et al. "Predicting Protein Pathways Associated to Tumor Heterogeneity by Correlating Spatial Lipidomics : The Dry Proteomic Concept...<em>Molecular & Cellular Proteomics</em>, 2025"</li>
            </ul>
        """, unsafe_allow_html=True)


        st.markdown("""
            <div style='text-align: center; margin-top: 40px;'>
                <h4 style='color: #318CE7;'>🔁 Profiler Workflow</h4>
                <p style='font-size: 1.05rem;'>Follow an automatic tour of the full analysis pipeline</p>
            </div>
        """, unsafe_allow_html=True)

        if "workflow_step" not in st.session_state:
            st.session_state.workflow_step = 0
        if "workflow_active" not in st.session_state:
            st.session_state.workflow_active = False

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.markdown("""
                <style>
                div.stButton > button {
                    font-size: 22px;
                    font-weight: bold;
                    padding: 15px 30px;
                    background-color: #318CE7;
                    color: white;
                    border: none;
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                }
                </style>
            """, unsafe_allow_html=True)

            if st.button("▶️ **Workflow Tour**"):
                st.session_state.workflow_active = True
                st.session_state.workflow_step = 0
                st.rerun()


            placeholder = st.empty()
            progress_placeholder = st.empty()

            if st.session_state.workflow_active:
                i = st.session_state.workflow_step
                with placeholder.container():
                    st.markdown(f"""
                        <div style='
                            text-align: center;
                            font-size: 1.2rem;
                            padding: 25px;
                            border-radius: 12px;
                            margin-top: 10px;
                            background-color: #f0f8ff;
                            border: 2px solid #e0eaff;
                            box-shadow: 0 0 10px rgba(0,0,0,0.05);'>
                            {workflow_steps[i]}
                        </div>
                    """, unsafe_allow_html=True)

                progress_html = "<div style='text-align: center; padding: 10px;'>"
                for j in range(len(workflow_steps)):
                    color = "#318CE7" if j == i else "#d0d8e8"
                    progress_html += f"<span style='display: inline-block; width: 14px; height: 14px; margin: 3px; border-radius: 50%; background-color: {color};'></span>"
                progress_html += "</div>"
                progress_placeholder.markdown(progress_html, unsafe_allow_html=True)

                time.sleep(3)
                if i < len(workflow_steps) - 1:
                    st.session_state.workflow_step += 1
                    st.rerun()
                else:
                    st.session_state.workflow_active = False




    with tabs[1]:
        st.markdown(
            """
            <h3 style="
                font-size: 1.2rem;
                border-bottom: 2px solid #318CE7;
                text-align: center;
                background-color: #f0f8ff;
                padding: 10px;
                border-radius: 5px;">
                Data Preparation
            </h3>
            """,
            unsafe_allow_html=True
        )

        
        # -------------------- Data Overview (avec Load Features) --------------------
        if 'show_info' not in st.session_state:
            st.session_state.show_info = {
                "dataset_info": False,
                "missing_values": False,
                "shapiro_wilk_test": False,
                "feature_distributions": False
            }

        # clefs pour l'overview
        if 'overview_df' not in st.session_state:
            st.session_state['overview_df'] = None
        if 'overview_source' not in st.session_state:
            st.session_state['overview_source'] = None

        with st.expander("**📑 Data Overview**", expanded=True):
            st.markdown(
                '<p style="color: gray; font-size: 14px">Comprehensive exploration of dataset structure, missingness, and feature distributions (omics-ready: proteomics, metabolomics, transcriptomics)</p>',
                unsafe_allow_html=True
            )
            with st.form("load_features_form"):
                source_choice = st.selectbox(
                    "Select dataset source:",
                    ['None', 'Raw', 'Edited/Renamed', 'Preprocessed', 'Oversampled', 'Undersampled'],
                    index=0,
                    key="overview_source_select"
                )
                load_btn = st.form_submit_button("📥 Load dataset features")
            if load_btn:
                # Map "Raw" to final_data (renamed classes) instead of data (raw data)
                ds_map = {
                    "Raw": st.session_state.get("final_data", st.session_state.get("data")),  # Prioritize final_data
                    "Edited/Renamed": st.session_state.get("final_data"),
                    "Preprocessed": st.session_state.get("preprocessed_data"),
                    "Oversampled": st.session_state.get("oversampled_data"),
                    "Undersampled": st.session_state.get("undersampled_data")
                }
                if source_choice == "None":
                    st.warning("Select a real source before loading features.")
                    st.session_state['overview_df'] = None
                    st.session_state['overview_source'] = None
                else:
                    sel_df = ds_map.get(source_choice)
                    if sel_df is None:
                        st.error(f"Source '{source_choice}' is not available in session_state.")
                        st.session_state['overview_df'] = None
                        st.session_state['overview_source'] = None
                    else:
                        st.session_state['overview_df'] = sel_df.copy()
                        st.session_state['overview_source'] = source_choice
                        if 'class_colors' not in st.session_state:
                            st.session_state['class_colors'] = {}
                        if 'Class' in sel_df.columns:
                            for cls in sel_df['Class'].unique():
                                if cls not in st.session_state['class_colors']:
                                    st.session_state['class_colors'][cls] = "#000000"
                        st.success(f"Loaded '{source_choice}' — {sel_df.shape[0]:,} samples × {sel_df.shape[1]:,} features")



        # Récupère le DF chargé (ou None)
        df = st.session_state.get('overview_df')

        if df is None:
            st.info("No dataset loaded. Choose a source above and click ▶️ Load Features to start analysing.")
            # on n'affiche pas la suite si rien n'est chargé
        else:

            with st.expander("**ℹ️ Dataset Info and suggestions**", expanded=st.session_state.show_info["dataset_info"]):

                with st.form("dataset_info_form"):
                    toggle = st.form_submit_button("Show Info")

                if toggle:
                    st.session_state.show_info["dataset_info"] = not st.session_state.show_info["dataset_info"]

                if st.session_state.show_info["dataset_info"]:
                    st.markdown(f"""
                        <div style='background-color: #f0f8ff; padding: 10px; border-radius: 10px; font-size: 15px;'>
                            <strong>Dataset source:</strong> {st.session_state.get('overview_source')}
                            <br><strong>Dataset dimensions:</strong> {df.shape[0]:,} samples × {df.shape[1]:,} features
                            <br><strong>Classes:</strong> {df['Class'].nunique() if 'Class' in df.columns else 'N/A'}
                        </div>
                    """, unsafe_allow_html=True)

                    display_class_info(df)
                    st.info("Summary: balanced classes → unbiased models.")

            # ---------------- Missing Values ----------------

            with st.expander("**❓ Missing Values and suggestions**", expanded=st.session_state.show_info["missing_values"]):

                with st.form("missing_values_form"):
                    toggle = st.form_submit_button("Show Missing Values Info")

                if toggle:
                    st.session_state.show_info["missing_values"] = not st.session_state.show_info["missing_values"]

                if st.session_state.show_info["missing_values"]:

                    relevant_cols, missing_df = calculate_missing_values(df)

                    if missing_df.empty:
                        st.success("No missing values detected.")
                    else:            
                        # Global % missingness (sur tout le dataset)
                        total_missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
                        st.success(f"Overall missingness: **{total_missing_pct:.2f}%** of all values.")
                        st.markdown("**Per-feature summary:**")
                        st.dataframe(missing_df, use_container_width=True)

                        # Normality check via skewness
                        skewness = df[relevant_cols].skew().dropna()
                        skew_mean = skewness.abs().mean()
                        is_normal = skew_mean < 0.5

                        # 💡 Global omics-aware recommendation
                        st.markdown("**💡 Global Imputation Strategy Suggestion:**")
                        if total_missing_pct < 5:
                            st.info("🔹 Low missingness (<5%) → Delete missing value or fill missing value with 0 if considered as exclusive features, or use simple Mean/Median/Modal imputation.")
                        elif total_missing_pct < 20:
                            if is_normal:
                                st.info("🔹 Moderate missingness (5–20%) & data ≈ normal → Mean or Regression-based imputation.")
                            else:
                                st.info("🔹 Moderate missingness (5–20%) & non-normal (skewed) → Median, Shifted Gaussian, or KNN imputation.")
                        else:
                            st.info("🔹 High missingness (>20%) → KNN or Regression-based imputation. "
                                    "Consider removing features with excessive missing values using Minimum features detection rate per class (%) in preprocessing expander.")

                        # Summary skewness
                        if is_normal:
                            st.success(f"Features are mostly normal (mean |skew| = {skew_mean:.2f}). Mean imputation valid.")
                        else:
                            st.warning(f"Features are skewed (mean |skew| = {skew_mean:.2f}). Prefer Median/Shifted Gaussian/KNN imputation.")


                        if 'Class' in df.columns:

                            # Count missing values per class (robust method)
                            missing_by_class = df.groupby('Class').apply(lambda g: g.isnull().sum().sum()).astype(float)
                            total_missing = missing_by_class.sum()

                            if total_missing > 0:

                                missing_pct_class = (missing_by_class / total_missing) * 100
                                import plotly.express as px

                                # st.markdown("**Class-specific missingness:**")

                                # Ensure class_colors exists
                                if 'class_colors' not in st.session_state:
                                    st.session_state['class_colors'] = {}

                                # Assign default colors if missing
                                palette = px.colors.qualitative.Plotly
                                for i, cls in enumerate(missing_pct_class.index):
                                    if cls not in st.session_state['class_colors']:
                                        st.session_state['class_colors'][cls] = palette[i % len(palette)]

                                # Build map
                                color_map = {cls: st.session_state['class_colors'][cls] for cls in missing_pct_class.index}

                                # Pie chart
                                fig = px.pie(
                                    names=missing_pct_class.index,
                                    values=missing_pct_class.values,
                                    color=missing_pct_class.index,
                                    color_discrete_map=color_map,
                                    title="Missing Data by Class"
                                )

                                fig.update_traces(
                                    textposition='inside',
                                    textinfo='percent+label',
                                    textfont_size=18
                                )

                                fig.update_layout(
                                    legend_title_text='Class',
                                    legend=dict(font=dict(size=16)),
                                    title=dict(font=dict(size=18))
                                )

                                st.plotly_chart(fig, use_container_width=True)




            # ---------------- Normality & Statistical Overview ----------------

            with st.expander("**📊 Normality/Statistical Overview and suggestions**", 
                            expanded=st.session_state.show_info["shapiro_wilk_test"]):

                with st.form("normality_form"):
                    toggle = st.form_submit_button("Show normality info")

                if toggle:
                    st.session_state.show_info["shapiro_wilk_test"] = (
                        not st.session_state.show_info["shapiro_wilk_test"]
                    )



                if st.session_state.show_info.get("shapiro_wilk_test", False):
                    relevant_cols, _ = calculate_missing_values(df)

                    # Shapiro-Wilk / normality
                    p_val, norm_ratio = perform_shapiro_wilk_test(df, relevant_cols)
                    normal_count = int(norm_ratio * len(relevant_cols))
                    non_normal_count = len(relevant_cols) - normal_count

                    import plotly.express as px
                    fig = px.pie(
                        names=['Normal', 'Non-Normal'],
                        values=[normal_count, non_normal_count],
                        title="Normal vs Non-Normal Features",
                        color_discrete_sequence=['#4CAF50', '#F44336']
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)

                    # Skewness plot
                    skewness = df[relevant_cols].skew().dropna()
                    skew_mean = skewness.abs().mean()
                    fig_density = px.histogram(
                        skewness,
                        nbins=30,
                        marginal="violin",
                        title="Density of Feature Skewness",
                        labels={'value': 'Skewness', 'count': 'Count'},
                        color_discrete_sequence=['#FF9800']
                    )
                    st.plotly_chart(fig_density, use_container_width=True)

                    # recommendations...
                    st.markdown("**⚖️ Normalization Recommendation:**")
                    if skew_mean < 0.5:
                        st.info("✅ Low skewness → TIC/RMS/BasePeak normalization suitable (common in omics).")
                    elif skew_mean > 1:
                        st.info("❗ High skewness → Log/Log10/Log2 normalization recommended for omics.")
                    else:
                        st.info("⚠️ Moderate skewness (0.5–1) →\n"
                                "- Proteomics/Metabolomics: prefer RMS normalization.\n"
                                "- Transcriptomics: Quantile normalization is often more appropriate.")

                    # statistical test suggestions (identique à ton code)
                    n_classes = df['Class'].nunique() if 'Class' in df.columns else 0
                    if skew_mean < 0.5:
                        normality_status = "mostly normal"
                        prefer_parametric = True
                    elif skew_mean > 1:
                        normality_status = "highly skewed"
                        prefer_parametric = False
                    else:
                        normality_status = "moderately skewed"
                        prefer_parametric = False

                    if n_classes == 2:
                        test_suggestion = "t-test (parametric)" if prefer_parametric else "Mann-Whitney U (non-parametric)"
                    elif n_classes > 2:
                        test_suggestion = "ANOVA (parametric)" if prefer_parametric else "Kruskal-Wallis (non-parametric)"
                    else:
                        test_suggestion = "N/A (no classes detected)"

                    st.markdown("**📊 Recommended Statistical Test:**")
                    st.info(
                        f"💡 Data appear **{normality_status}** (mean |skew| = {skew_mean:.2f}).\n\n"
                        f"Suggested test: **{test_suggestion}**"
                    )

                    st.info(
                        "⚠️ Note: Central Limit Theorem: For n ≥ 30, sampling distribution of the mean approximates normality."
                    )


                    # -------------------- Survival Data --------------------
                    if 'Overall survival' in df.columns and 'State' in df.columns:
                        st.subheader("Survival Data Preview")
                        st.dataframe(df[['Overall survival', 'State']])
                        gc.collect()

                elif st.session_state.get('survival_data') is not None:
                    surv_df = st.session_state['survival_data']
                    if 'Overall survival' in surv_df.columns and 'State' in surv_df.columns:
                        st.markdown("*Survival Data*")
                        st.dataframe(surv_df)
                        del surv_df
                        gc.collect()


        with st.expander("**📝 Class Renaming Options**", expanded=False):
            st.markdown(
                '<p style="color: gray; font-size: 14px">Standardize class labels: unify replicates, merge groups, or relabel unknowns.</p>',
                unsafe_allow_html=True
            )
            if "data" not in st.session_state or st.session_state["data"] is None:
                st.warning("⚠️ No data available. Please upload or load a dataset first.")
            else:
                if "final_data" not in st.session_state:
                    st.session_state["final_data"] = st.session_state["data"].copy()
                class_names = list(st.session_state["final_data"]["Class"].unique())
                st.session_state.setdefault("class_renaming", {cls: cls for cls in class_names})
                st.session_state.setdefault("rename_pending", {cls: cls for cls in class_names})
                mode = st.radio(
                    "🔧 Select renaming mode:",
                    [
                        "🔁 Apply same name to all classes",
                        "✏️ Rename each class individually",
                        "🧩 Group classes under a common name"
                    ],
                    key="rename_mode",
                    help="Choose whether to apply a global name, rename individually, or group multiple classes."
                )
                n_groups = 1
                if mode.startswith("🧩"):
                    n_groups = st.number_input("Number of class groups:", 1, 100, 1, key="num_groups")
                with st.form(key="class_renaming_form"):
                    temp_mapping = st.session_state["rename_pending"].copy()
                    if mode.startswith("🔁"):
                        global_name = st.text_input("🆕 New name for all classes:", "UnifiedClass", key="global_class_name")
                        if global_name:
                            temp_mapping = {cls: global_name for cls in class_names}
                    elif mode.startswith("✏️"):
                        for cls in class_names:
                            new_name = st.text_input(
                                f"Rename '**{cls}**' to:",
                                st.session_state["rename_pending"].get(cls, cls),
                                key=f"rename_{cls}"
                            )
                            temp_mapping[cls] = new_name
                    elif mode.startswith("🧩"):
                        for i in range(n_groups):
                            gname = st.text_input(f"Group {i+1} name:", key=f"group_name_{i}")
                            selected = st.multiselect(
                                f"Select classes to include in Group {i+1}:", class_names, key=f"selected_classes_{i}"
                            )
                            if gname and selected:
                                for cls in selected:
                                    temp_mapping[cls] = gname
                    col1, col2 = st.columns(2)
                    with col1:
                        apply_changes = st.form_submit_button("Apply Renaming")
                    with col2:
                        reset_changes = st.form_submit_button("🔄 Reset Changes")
                if apply_changes:
                    with st.spinner("Applying class name changes..."):
                        st.session_state["rename_pending"] = temp_mapping.copy()
                        st.session_state["class_renaming"] = st.session_state["rename_pending"].copy()
                        st.session_state["final_data"]["Class"] = st.session_state["final_data"]["Class"].replace(
                            st.session_state["class_renaming"]
                        )
                        st.session_state["data"] = st.session_state["final_data"]
                        st.success("Class names updated successfully!")
                if reset_changes:
                    st.session_state["rename_pending"] = {cls: cls for cls in class_names}
                    st.session_state["class_renaming"] = {cls: cls for cls in class_names}
                    st.session_state["final_data"]["Class"] = st.session_state["final_data"]["Class"].map(
                        lambda x: x if x in class_names else x
                    )
                    st.session_state["data"] = st.session_state["final_data"]
                    st.success("🗑️ All renaming changes have been reset.")

                    gc.collect()


        with st.expander("**🧹 Edit Dataset Options**", expanded=False):
            st.markdown(
                '<p style="color: gray; font-size: 14px">Remove or keep specific rows/columns from the dataset as needed.</p>',
                unsafe_allow_html=True
            )
            if "final_data" in st.session_state:

                df = st.session_state["final_data"].reset_index()
                label_df = df[["index", "Class"]]

                row_options = list(label_df.apply(lambda row: f"Index {row['index']} → {row['Class']}", axis=1))

                with st.form(key="edit_dataset_form"):

                    # ---------------------- ROWS TO REMOVE ----------------------
                    selected_rows = st.multiselect("Rows to remove:", row_options, key="selected_rows_to_remove")
                    selected_indexes = [int(row.split()[1]) for row in selected_rows] if selected_rows else []

                    # ---------------------- COLUMNS TO REMOVE (EXCLUDE CLASS) ----------------------
                    columns_available_to_remove = [col for col in df.columns if col not in ["Class", "index"]]

                    selected_columns = st.multiselect(
                        "Columns to remove:",
                        columns_available_to_remove,
                        key="selected_columns_to_remove"
                    )

                    # ---------------------- REMOVE CLASS VALUES ----------------------
                    unique_classes = sorted(df["Class"].dropna().unique().tolist())
                    selected_classes_to_remove = st.multiselect(
                        "Remove all samples belonging to the following Class(es):",
                        unique_classes,
                        key="classes_to_remove"
                    )

                    # ---------------------- ROWS TO KEEP ----------------------
                    rows_to_keep = st.multiselect("Rows to keep:", row_options, key="rows_to_keep")
                    rows_keep_indexes = [int(row.split()[1]) for row in rows_to_keep] if rows_to_keep else []


                    # ---------------------- COLUMNS TO KEEP (Class optional in UI) ----------------------
                    all_columns = [c for c in df.columns if c != "index"]

                    columns_to_keep = st.multiselect(
                        "Columns to keep:",
                        all_columns,
                        key="columns_to_keep"
                    )
                    # Force Class retention (protection)
                    if "Class" not in columns_to_keep:
                        columns_to_keep.append("Class")

                    col1, col2 = st.columns(2)
                    with col1:
                        apply_changes = st.form_submit_button("Apply Changes")
                    with col2:
                        reset_changes = st.form_submit_button("🔄 Reset All Changes")


                # ------------------------- APPLY CHANGES -------------------------
                if apply_changes:
                    with st.spinner("Applying modifications..."):

                        # KEEP ROWS first (if selected)
                        if rows_keep_indexes:
                            st.session_state["final_data"] = st.session_state["final_data"].loc[rows_keep_indexes]

                        # KEEP COLUMNS (force Class retention)
                        if columns_to_keep:
                            if "Class" not in columns_to_keep:
                                columns_to_keep.append("Class")
                            st.session_state["final_data"] = st.session_state["final_data"][columns_to_keep]

                        # REMOVE ROWS
                        if selected_indexes:
                            st.session_state["final_data"].drop(index=selected_indexes, inplace=True)

                        # REMOVE COLUMNS (Class excluded automatically)
                        if selected_columns:
                            st.session_state["final_data"].drop(columns=selected_columns, inplace=True)

                        # REMOVE CLASS SAMPLES
                        if selected_classes_to_remove:
                            st.session_state["final_data"] = st.session_state["final_data"][
                                ~st.session_state["final_data"]["Class"].isin(selected_classes_to_remove)
                            ]

                        st.session_state["final_data"].reset_index(drop=True, inplace=True)
                        st.session_state["data"] = st.session_state["final_data"]
                        st.success("✅ Modifications applied successfully.")
                # ------------------------- RESET -------------------------
                if reset_changes:
                    st.session_state["final_data"] = st.session_state["data"].copy()
                    st.success("🔁 Dataset has been restored to its original state.")

                # ------------------------- PREVIEW -------------------------
                st.markdown("**Preview Updated Dataset**")
                preview_df = st.session_state["final_data"]

                if preview_df.shape[1] > 100:
                    st.dataframe(pd.concat([preview_df.iloc[:, :50], preview_df.iloc[:, -50:]], axis=1))
                else:
                    st.dataframe(preview_df)

                # Ask user for filename
                custom_filename = st.text_input(
                    "📄 Filename for the cleaned dataset:",
                    value="Cleaned_Data.csv",
                    help="Enter a filename (must end with .csv)."
                )

                # Safety: enforce .csv extension
                if not custom_filename.lower().endswith(".csv"):
                    custom_filename = custom_filename + ".csv"

                csv = preview_df.to_csv(index=False).encode('utf-8')

                st.download_button(
                    label="📥 Download Cleaned Dataset (CSV)",
                    data=csv,
                    file_name=custom_filename,
                    mime='text/csv'
                )




        st.markdown(
            """
            <h3 style="
                font-size: 1.2rem;
                border-bottom: 2px solid #318CE7;
                text-align: center;
                background-color: #f0f8ff;
                padding: 10px;
                border-radius: 5px;">
                Data Wrangling
            </h3>
            """,
            unsafe_allow_html=True
        )

        with st.expander("**⚙️ Preprocessing**", expanded=False):
            st.markdown(
                '<p style="color: gray; font-size: 14px">Filtering, Imputation, Binning, Normalization, Batch Effect Correction and sparse matrix handling</p>',
                unsafe_allow_html=True
            )

            df_overview = st.session_state.get('overview_df')
            submitted = False
            if df_overview is None or df_overview.empty:
                st.warning("⚠️ You must load a dataset in **Data Overview** before running preprocessing steps.")
            else:
                # base copy
                data_to_preprocess = df_overview.copy()

                apply_binning_option = st.checkbox(
                    "Shrink mass range or apply Binning?",
                    key="apply_binning_option",
                    help="Only relevant for ion spectra (MS1 data)."
                )

                # helper utilities local to this block
                cols_exclude = ['Class', 'File', 'RT', 'Sum']

                def get_numeric_features(df):
                    return [c for c in df.columns if c not in cols_exclude and pd.api.types.is_numeric_dtype(df[c])]

                def shifted_gaussian_fill(series, shift=1.8, width=0.3, rng=None):
                    vals = series.dropna()
                    if vals.empty:
                        return series
                    mu, sigma = vals.mean(), vals.std(ddof=0)
                    n_missing = series.isna().sum()
                    if n_missing == 0:
                        return series
                    if rng is None:
                        rng = np.random.default_rng()
                    filled = rng.normal(loc=mu - shift * sigma, scale=width * sigma, size=n_missing)
                    out = series.copy()
                    out.loc[out.isna()] = filled
                    return out

                with st.form("preprocessing_form"):

                    # ------------------ Filters ------------------
                    min_detection_threshold = st.number_input(
                        "Minimum features detection rate per class (%)",
                        min_value=0,
                        max_value=100,
                        value=0,
                        step=1,
                        help="Keep features detected in at least X% of samples per class. "
                            "If zero-inflated filter is applied, zeros are considered as missing values."
                    )

                    apply_zero_inflated_filter = st.checkbox(
                        "Apply zero-inflated feature filter (sparse matrix)?",
                        value=False,
                        help="Treat zeros as missing values for filtering features that are mostly zeros and choose Detele Missing Values method"
                            "(useful for RNA-seq / transcriptomics and MS1 data)."
                    )

                    # ------------------ Imputation ------------------
                    imputation_method = st.selectbox(
                        "Select Missing Value Imputation Method",
                        [
                            'None', 'Mean Imputation', 'Median Imputation', 'Mode Imputation',
                            'Delete Missing Values', 'KNN Imputation', 'Fillna with 0', 'Shifted Gaussian'
                        ],
                        key="imputation_method"
                    )

                    impute_by_class = st.checkbox(
                        "🔹 Impute missing values per class?",
                        value=False,
                        help="Perform imputation separately within each Class."
                    )

                    remove_exclusive_missing = st.checkbox(
                        "🔹Remove features that remain entirely missing in any class after class-wise imputation?",
                        value=False,
                        help="Removes features that are still all NaN in at least one class after imputation per Class."
                    )

                    # ------------------ Binning ------------------
                    bin_width = mass_range_min = mass_range_max = None
                    if apply_binning_option:
                        mz_cols = [col for col in data_to_preprocess.columns if col not in cols_exclude]
                        mz_values = []
                        for col in mz_cols:
                            try:
                                mz_values.append(float(str(col).replace("mz_", "").strip()))
                            except:
                                continue
                        if mz_values:
                            min_detected, max_detected = min(mz_values), max(mz_values)
                            st.info(f"Detected m/z range: {min_detected:.4f} - {max_detected:.4f} Da")
                            bin_width = st.number_input("Bin Width (Da)", 0.0001, 100.0, 0.1, 0.1)
                            mass_range_min = st.number_input("Min Mass Range", value=min_detected, format="%.4f")
                            mass_range_max = st.number_input("Max Mass Range", value=max_detected, format="%.4f")
                        else:
                            st.warning("No valid m/z columns found for binning.")
                            bin_width, mass_range_min, mass_range_max = 0.1, 0.0, 0.0

                    # ------------------ Normalization ------------------
                    normalization_type = st.selectbox(
                        "Normalization Type",
                        ['None', 'TIC', 'RMS', 'BasePeak', 'QNorm', 'Log Normalization', 'Log10', 'Log2'],
                        key="normalization_type"
                    )

                    apply_combat = st.checkbox(
                        "Apply batch effect correction (Combat)?",
                        help=(
                            "Apply Combat batch effect correction using the column 'Class' as batch labels.\n"
                            "⚠️ If you want to correct for other factors (e.g., time points, tissue type, biological conditions), "
                            "rename the relevant column to 'Class' before running this step.\n"
                            "After correction, download the dataset and optionally rename columns according to your experimental design."
                        )
                    )

                    # debug / performance options
                    # debug_mode = st.checkbox("Show debug logs?", value=False)
                    knn_k = st.number_input("K for KNN (if selected)", min_value=1, max_value=50, value=5, step=1,help="K defines how many nearest neighbours are used to generate each new imputed value. By default, K=5 means the missing value is estimated from the 5 closest samples. Because this method relies on neighbour structure, results may slightly vary and will never be strictly identical. Increase K for a more smoother imputation, or decrease it for more local sensitivity.")

                    submitted = st.form_submit_button("Preprocess Data")

                if submitted:
                    try:
                        data = data_to_preprocess.copy()
                        progress = st.progress(0)

                        # Stats for summary
                        removed_by_detection = 0
                        removed_by_exclusive_missing = 0

                        # ------------------ Detection Filter ------------------
                        try:
                            numeric_cols = get_numeric_features(data)
                            if min_detection_threshold > 0 and 'Class' in data.columns and numeric_cols:
                                keep_features = []
                                for col in numeric_cols:
                                    keep_col = True
                                    for _, group in data.groupby('Class'):
                                        values = group[col]
                                        # If zero-inflated => zero counts as missing
                                        if apply_zero_inflated_filter:
                                            present = ((values.notna()) & (values != 0)).sum()
                                        else:
                                            present = values.notna().sum()
                                        present_pct = present / len(values) * 100
                                        if present_pct < min_detection_threshold:
                                            keep_col = False
                                            # if debug_mode:
                                            #     st.write(f"Drop {col}: {present_pct:.1f}% < {min_detection_threshold}% in a class")
                                            # break
                                    if keep_col:
                                        keep_features.append(col)

                                removed_by_detection = len(numeric_cols) - len(keep_features)
                                if removed_by_detection > 0:
                                    st.warning(f"{removed_by_detection} features removed due to threshold per class.")
                                # keep at least cols_exclude; if no feature remains, keep only metadata
                                kept_cols = cols_exclude + keep_features
                                kept_cols = [c for c in kept_cols if c in data.columns]
                                if keep_features:
                                    data = data[kept_cols].copy()
                                else:
                                    # keep only metadata to avoid empty dataframes
                                    data = data[[c for c in cols_exclude if c in data.columns]].copy()
                        except Exception as e:
                            st.error(f"Error in detection filter: {e}")
                            st.stop()
                        else:
                            progress.progress(20)

                        # ------------------ Imputation ------------------
                        try:
                            numeric_cols = get_numeric_features(data)  # recalc after filtering
                            if imputation_method != 'None' and numeric_cols:
                                st.info(f"Applying {imputation_method}...")
                                rng = np.random.default_rng()

                                if imputation_method in ['Mean Imputation', 'Median Imputation']:
                                    func = np.mean if imputation_method == 'Mean Imputation' else np.median
                                    if impute_by_class and 'Class' in data.columns:
                                        # fill per class using group agg
                                        for cls, grp in data.groupby('Class'):
                                            idx = grp.index
                                            fill_vals = grp[numeric_cols].aggregate(func)
                                            data.loc[idx, numeric_cols] = grp[numeric_cols].fillna(fill_vals).values
                                    else:
                                        fill_vals = data[numeric_cols].aggregate(func)
                                        data[numeric_cols] = data[numeric_cols].fillna(fill_vals)

                                elif imputation_method == 'Mode Imputation':
                                    for col in numeric_cols:
                                        if impute_by_class and 'Class' in data.columns:
                                            for cls, grp in data.groupby('Class'):
                                                mode = grp[col].mode()
                                                if not mode.empty:
                                                    data.loc[grp.index, col] = grp[col].fillna(mode.iloc[0])
                                        else:
                                            mode = data[col].mode()
                                            if not mode.empty:
                                                data[col] = data[col].fillna(mode.iloc[0])

                                elif imputation_method == 'Delete Missing Values':
                                    # delete columns that contain ANY NaN
                                    data = data.dropna(axis=1)
                                elif imputation_method == 'KNN Imputation':
                                    from sklearn.impute import KNNImputer
                                    numeric_cols = get_numeric_features(data)
                                    if not numeric_cols:
                                        # nothing to impute
                                        pass
                                    else:
                                        if impute_by_class and 'Class' in data.columns:
                                            frames = []
                                            for cls, grp in data.groupby('Class'):
                                                grp_numeric = grp[numeric_cols].copy()
                                                if len(grp_numeric) < max(1, int(knn_k)):
                                                    st.error(f"Class '{cls}' has fewer than {knn_k} samples for KNN. Aborting.")
                                                    st.stop()
                                                # keep columns that have at least one non-NaN otherwise KNNImputer can't work on a constant-empty column
                                                valid_cols = grp_numeric.columns[grp_numeric.notna().any()]
                                                if len(valid_cols) == 0:
                                                    # nothing to impute in this class, keep as-is
                                                    frames.append(grp)
                                                    continue
                                                imputer = KNNImputer(n_neighbors=min(int(knn_k), len(grp_numeric) - 1))
                                                transformed = imputer.fit_transform(grp_numeric[valid_cols])
                                                grp.loc[:, valid_cols] = pd.DataFrame(transformed, columns=valid_cols, index=grp.index)
                                                frames.append(grp)
                                            data = pd.concat(frames)
                                        else:
                                            if len(data) < max(1, int(knn_k)):
                                                st.error(f"Dataset has fewer than {knn_k} samples for KNN. Aborting.")
                                                st.stop()
                                            grp_numeric = data[numeric_cols].copy()
                                            valid_cols = grp_numeric.columns[grp_numeric.notna().any()]
                                            if len(valid_cols) > 0:
                                                imputer = KNNImputer(n_neighbors=min(int(knn_k), len(data) - 1))
                                                data.loc[:, valid_cols] = pd.DataFrame(imputer.fit_transform(grp_numeric[valid_cols]), columns=valid_cols, index=data.index)

                                elif imputation_method == 'Fillna with 0':
                                    data[numeric_cols] = data[numeric_cols].fillna(0)

                                elif imputation_method == 'Shifted Gaussian':
                                    numeric_cols = get_numeric_features(data)
                                    if impute_by_class and 'Class' in data.columns:
                                        parts = []
                                        for _, grp in data.groupby('Class'):
                                            grp2 = grp.copy()
                                            for c in numeric_cols:
                                                grp2[c] = shifted_gaussian_fill(grp2[c], rng=rng)
                                            parts.append(grp2)
                                        data = pd.concat(parts)
                                    else:
                                        for c in numeric_cols:
                                            data[c] = shifted_gaussian_fill(data[c], rng=rng)

                                else:
                                    st.error(f"Unknown imputation method: {imputation_method}")
                            # if imputation method is None or no numeric cols, nothing to do
                        except Exception as e:
                            st.error(f"Error during imputation: {e}")
                            st.stop()
                        else:
                            progress.progress(40)
                            st.success("✅ Imputation completed")

                        # ------------------ Remove features still entirely missing in some class ------------------
                        try:
                            # recalc numeric columns after imputation / drops
                            numeric_cols = get_numeric_features(data)
                            if remove_exclusive_missing and numeric_cols:
                                to_drop = []
                                if impute_by_class and 'Class' in data.columns:
                                    for col in numeric_cols:
                                        # if any class has all NaN for this column -> drop it
                                        if any(group[col].isna().all() for _, group in data.groupby('Class')):
                                            to_drop.append(col)
                                else:
                                    to_drop = [col for col in numeric_cols if data[col].isna().all()]

                                if to_drop:
                                    data.drop(columns=to_drop, inplace=True)
                                    removed_by_exclusive_missing = len(to_drop)
                                    if debug_mode:
                                        st.write(f"Removed exclusive-missing features: {to_drop}")
                        except Exception as e:
                            st.error(f"Error removing exclusive missing features: {e}")
                            st.stop()
                        else:
                            progress.progress(60)

                        # Final strict validation: no NaN allowed in numeric features
                        try:
                            numeric_cols_final = get_numeric_features(data)
                            total_remaining_nans = int(data[numeric_cols_final].isna().sum().sum()) if numeric_cols_final else 0
                            if total_remaining_nans > 0:
                                st.error(f"❌ {total_remaining_nans} missing values remain after preprocessing. Please adjust imputation settings or enable removal of exclusive missing features.")
                                st.stop()
                        except Exception as e:
                            st.error(f"Validation error after imputation: {e}")
                            st.stop()

                        # ------------------ Binning ------------------
                        try:
                            if apply_binning_option and bin_width is not None and mass_range_min is not None and mass_range_max is not None:
                                # call domain-specific binning function (must exist in your codebase)
                                data = apply_binning_to_mass_range(data, bin_width, (mass_range_min, mass_range_max))
                        except Exception as e:
                            st.error(f"Binning error: {e}")
                        else:
                            progress.progress(70)
                            if apply_binning_option:
                                st.success(f"Binning applied: {mass_range_min:.2f}-{mass_range_max:.2f} Da, bin width {bin_width:.2f}")

                        # ------------------ Normalization ------------------
                        try:
                            if normalization_type != 'None':
                                data = preprocess_data(data, normalization_type, progress)
                        except Exception as e:
                            st.error(f"Normalization error: {e}")
                        else:
                            progress.progress(85)
                            if normalization_type != 'None':
                                st.success(f"{normalization_type} normalization applied")
                            else:
                                st.info("No normalization applied")

                        # ------------------ Combat Batch Correction ------------------
                        try:
                            if apply_combat:
                                if 'Class' not in data.columns or len(data['Class'].unique()) < 2:
                                    st.warning("Combat skipped: need at least 2 classes with 'Class' column.")
                                else:
                                    from neurocombat_sklearn import CombatModel
                                    from sklearn.preprocessing import LabelEncoder
                                    le = LabelEncoder()
                                    batch_labels = le.fit_transform(data['Class']).reshape(-1, 1)
                                    feat_cols = [c for c in data.columns if c not in cols_exclude]
                                    # ensure numeric and drop any col that still contains NaN (shouldn't happen)
                                    features = data[feat_cols].select_dtypes(include=[np.number]).copy()
                                    features = features.dropna(axis=1)
                                    if features.shape[1] == 0:
                                        st.warning("No numeric features available for Combat after dropping NaNs.")
                                    else:
                                        combat = CombatModel()
                                        corrected = combat.fit_transform(features, batch_labels)
                                        meta = data[[c for c in cols_exclude if c in data.columns]].reset_index(drop=True)
                                        data = pd.concat([meta, pd.DataFrame(corrected, columns=features.columns)], axis=1)
                                        st.success("✅ Combat correction applied")
                        except Exception as e:
                            st.error(f"❌ Combat correction failed: {e}")

                        progress.progress(100)

                        # ------------------ Save & Summary ------------------
                        st.session_state['preprocessed_data'] = data

                        st.markdown("**Preprocessing Summary**")
                        st.write(f"- Detection filter: {min_detection_threshold}% per class → {removed_by_detection} features removed")
                        st.write(f"- Imputation: {imputation_method} {'(per class)' if impute_by_class else ''} → {removed_by_exclusive_missing} exclusive missing features removed")
                        st.write(f"- Binning: {'Yes' if apply_binning_option else 'No'}")
                        st.write(f"- Normalization: {normalization_type}")
                        st.write(f"- Batch correction: {'Yes' if apply_combat else 'No'}")
                        # count features excluding metadata
                        feature_count = len([c for c in data.columns if c not in cols_exclude])
                        st.write(f"- Total features after preprocessing: {feature_count}")

                        # ------------------ Preview & Download ------------------
                        df_preview = st.session_state['preprocessed_data']
                        st.markdown("**Preprocessed Data Preview**")
                        total_cols = df_preview.shape[1]

                        if total_cols > 100:
                            st.info(f"Too many features ({total_cols}). Showing first 50 & last 50 columns. You can download the full preprocessed dataset below.")
                            preview_df = pd.concat([df_preview.iloc[:, :50], df_preview.iloc[:, -50:]], axis=1)
                            st.dataframe(preview_df, hide_index=True)
                        else:
                            st.dataframe(df_preview, hide_index=True)

                        # Text input pour nom du fichier
                        file_name_input = st.text_input(
                            "Enter a name for your CSV file:",
                            value="preprocessed_data",
                            help="Provide a custom name for the preprocessed dataset CSV file (without extension)."
                        )

                        # Bouton de téléchargement
                        st.download_button(
                            label="📥 Download full preprocessed data (CSV)",
                            data=df_preview.to_csv(index=False).encode('utf-8'),
                            file_name=f"{file_name_input}.csv",
                            mime='text/csv'
                        )
                    except Exception as e:
                        st.error(f"Preprocessing failed: {e}")


        # ------------------ Oversampling ------------------
        with st.expander("**🔼 Oversampling**", expanded=False):
            st.markdown('<p style="color: gray; font-size: 14px">Class balancing strategies</p>', unsafe_allow_html=True)
            with st.form("oversampling_form"):
                source = st.selectbox("Select Data Source", ['Raw Data', 'Preprocessed'])
                technique = st.selectbox("Oversampling Technique", ['None', 'SMOTE', 'ADASYN'])
                apply_btn = st.form_submit_button("✅ Apply Oversampling")
            if apply_btn and technique != 'None':
                try:
                    data_use = st.session_state.get('preprocessed_data') if source=='Preprocessed' else st.session_state.get('final_data', st.session_state.get('data'))
                    if data_use is None: st.error("No valid data"); st.stop()
                    progress = st.progress(0)
                    st.session_state['oversampled_data'] = apply_sampling(data_use, technique.lower(), _progress_bar=progress)
                    st.success("✅ Oversampling successful")
                except Exception as e: st.error(f"Oversampling error: {e}")

        # ------------------ Undersampling ------------------
        with st.expander("**🔽 Undersampling**", expanded=False):
            st.markdown('<p style="color: gray; font-size: 14px">Class balancing strategies</p>', unsafe_allow_html=True)
            with st.form("undersampling_form"):
                source = st.selectbox("Select Data Source", ['Raw Data', 'Preprocessed'])
                technique = st.selectbox("Undersampling Technique", ['None', 'RandomUnderSampler', 'NearMiss'])
                apply_btn = st.form_submit_button("✅ Apply Undersampling")
            if apply_btn and technique != 'None':
                try:
                    data_use = st.session_state.get('preprocessed_data') if source=='Preprocessed' else st.session_state.get('final_data', st.session_state.get('data'))
                    if data_use is None: st.error("No valid data"); st.stop()
                    X = data_use.drop(['Class', 'File', 'RT', 'Sum'], axis=1, errors='ignore'); y = data_use['Class']
                    if technique=='RandomUnderSampler': from imblearn.under_sampling import RandomUnderSampler; X_res, y_res = RandomUnderSampler(random_state=1).fit_resample(X,y)
                    elif technique=='NearMiss': from imblearn.under_sampling import NearMiss; X_res, y_res = NearMiss(version=1).fit_resample(X,y)
                    st.session_state['undersampled_data'] = pd.concat([X_res, y_res], axis=1)
                    st.write(st.session_state['undersampled_data']['Class'].value_counts())
                    st.success("✅ Undersampling successful")
                except Exception as e: st.error(f"Undersampling error: {e}")

        st.markdown("""
        <h3 style="
            font-size: 1.2rem;
            border-bottom: 2px solid #318CE7;
            text-align: center;
            background-color: #f0f8ff;
            padding: 10px;
            border-radius: 5px;">
            Data Visualization
        </h3>
        """, unsafe_allow_html=True)


        with st.expander("🎨 **Customize Class Colors**", expanded=False):
            st.markdown(
                '<p style="color: gray; font-size: 14px">Assign custom colors to each class to personalize your visualizations.</p>',
                unsafe_allow_html=True
            )
            apply_colors = None
            with st.form("class_colors_form"):
                # --- Choix de la source ---
                color_data_source = st.selectbox(
                    "Select Data Source for Class Customization",
                    ['Raw Data', 'Preprocessed', 'Oversampled', 'Undersampled'],
                    key="color_data_source_form",
                    help="Choose the dataset to retrieve classes for color customization."
                )

                # --- Récupération des données selon la source ---
                if color_data_source == "Preprocessed":
                    color_data = st.session_state.get('preprocessed_data')
                elif color_data_source == "Raw Data":
                    color_data = st.session_state.get('final_data', st.session_state.get('data'))
                elif color_data_source == "Oversampled":
                    color_data = st.session_state.get('oversampled_data')
                elif color_data_source == "Undersampled":
                    color_data = st.session_state.get('undersampled_data')
                else:
                    color_data = None

                if color_data is None:
                    st.warning("⚠️ No valid data available for class color customization.")
                else:
                    unique_classes = list(color_data['Class'].unique())

                    # --- Initialisation de class_colors ---
                    if 'class_colors' not in st.session_state:
                        st.session_state['class_colors'] = {}

                    # --- Ajout des classes manquantes avec une couleur par défaut ---
                    import plotly.express as px
                    for cls in unique_classes:
                        if cls not in st.session_state['class_colors']:
                            st.session_state['class_colors'][cls] = px.colors.qualitative.Plotly[unique_classes.index(cls) % len(px.colors.qualitative.Plotly)]

                    # --- Sélecteurs de couleur ---
                    st.write("Select a color for each class:")
                    for class_name in unique_classes:
                        st.session_state['class_colors'][class_name] = st.color_picker(
                            f"Color for {class_name}",
                            st.session_state['class_colors'][class_name],
                            key=f"color_{class_name}_custom"
                        )

                    # --- Bouton de soumission (obligatoire dans un formulaire) ---
                    apply_colors = st.form_submit_button("✅ Apply Colors")

            # --- Action après soumission ---
            if apply_colors and color_data is not None:
                st.success("Class colors updated successfully!")



        with st.expander("📊 **Feature Distribution by Class**", expanded=False):
            st.markdown(
                '<p style="color: gray; font-size: 14px">Explore the distribution of a single feature across classes for detailed insights.</p>',
                unsafe_allow_html=True
            )
            with st.form("feature_distribution_form"):
                feature_data_source = st.selectbox(
                    "Select Data Source for Feature Visualization",
                    ['None', 'Raw Data', 'Preprocessed', 'Oversampled', 'Undersampled'],
                    key="feature_data_source_select",
                    help="Choose the dataset to explore a specific feature."
                )
                load_features = st.form_submit_button("📥 Load Features")
                if 'available_features' not in st.session_state:
                    st.session_state['available_features'] = []
                if load_features:
                    if feature_data_source == "Raw Data":
                        data_viz_single = st.session_state.get('final_data', st.session_state.get('data'))  # Prioritize final_data
                    elif feature_data_source != "None":
                        data_viz_single = get_data(feature_data_source)
                    else:
                        data_viz_single = None
                    if data_viz_single is not None:
                        st.session_state['available_features'] = data_viz_single.columns.tolist()
                    else:
                        st.warning("⚠️ No valid data available to load features.")
                feature_to_explore = st.selectbox(
                    "Select Feature for Exploration",
                    st.session_state['available_features'],
                    key="feature_to_explore_single"
                )
                histfunc_single = st.selectbox(
                    "Select Aggregation Function",
                    ['sum', 'count', 'avg', 'min', 'max'],
                    key="histfunc_select_single"
                )
                apply_viz = st.form_submit_button("✅ Show Feature Distribution")
            if 'feature_distribution_plot' not in st.session_state:
                st.session_state['feature_distribution_plot'] = go.Figure()
            if apply_viz:
                if feature_data_source == "Raw Data":
                    data_viz_single = st.session_state.get('final_data', st.session_state.get('data'))  # Prioritize final_data
                elif feature_data_source != "None":
                    data_viz_single = get_data(feature_data_source)
                else:
                    data_viz_single = None
                if data_viz_single is not None and feature_to_explore:
                    fig = plot_feature_distribution(
                        data_viz_single,
                        feature_to_explore,
                        st.session_state['class_colors'],
                        histfunc_single
                    )
                    st.session_state['feature_distribution_plot'] = fig
                else:
                    st.warning("⚠️ No valid data available for plotting.")
            if isinstance(st.session_state.get('feature_distribution_plot', None), go.Figure):
                st.plotly_chart(st.session_state['feature_distribution_plot'])
            import gc
            gc.collect()


        with st.expander("📈 **Multi-Feature Comparison: Radar, Line & Bar Charts**", expanded=False):
            st.markdown(
                '<p style="color: gray; font-size: 14px">Visualize and compare multiple features across different classes using dynamic chart types such as radar, line, and bar plots. Ideal for uncovering patterns and class-specific trends.</p>',
                unsafe_allow_html=True
            )
            with st.form("multi_feature_form"):
                multi_feature_data_source = st.selectbox(
                    "Select Data Source for Multi-Feature Comparison",
                    ['None', 'Raw Data', 'Preprocessed', 'Oversampled', 'Undersampled'],
                    key="multi_feature_data_source_select",
                    help="Choose the dataset to compare multiple features."
                )
                load_features = st.form_submit_button("📥 Load Features")
                if 'available_multi_features' not in st.session_state:
                    st.session_state['available_multi_features'] = []
                if load_features:
                    if multi_feature_data_source == "Raw Data":
                        data_viz_multi = st.session_state.get('final_data', st.session_state.get('data'))  # Prioritize final_data
                    elif multi_feature_data_source != "None":
                        data_viz_multi = get_data(multi_feature_data_source)
                    else:
                        data_viz_multi = None
                    if data_viz_multi is not None:
                        st.session_state['available_multi_features'] = [col for col in data_viz_multi.columns if col != 'Class']
                    else:
                        st.warning("⚠️ No valid data available to load features.")
                features_to_explore = st.multiselect(
                    "Select Features for Comparison",
                    st.session_state['available_multi_features'],
                    help="Choose two or more features to compare."
                )
                plot_type = st.selectbox(
                    "Select Visualization Type",
                    ['Radar Chart', 'Line Plot', 'Bar Chart'],
                    key="plot_type_select_multi",
                    help="Choose the type of plot to visualize multiple features."
                )
                if plot_type == 'Bar Chart':
                    available_histfuncs = ['sum', 'count', 'mean', 'min', 'max', 'percentage']
                else:
                    available_histfuncs = ['sum', 'mean', 'min', 'max', 'percentage']
                histfunc_multi = st.selectbox(
                    "Select Aggregation Function",
                    available_histfuncs,
                    key="histfunc_multi_select"
                )
                if plot_type == 'Line Plot':
                    error_type = st.selectbox(
                        "Select Error Bar Type (only for line plot)",
                        ['None', 'SEM', 'STD'],
                        key="error_bar_type_select"
                    )
                else:
                    error_type = 'None'
                if plot_type == 'Bar Chart':
                    if 'feature_colors' not in st.session_state:
                        st.session_state['feature_colors'] = {}
                    for feature in features_to_explore:
                        if feature not in st.session_state['feature_colors']:
                            st.session_state['feature_colors'][feature] = '#636EFA'
                        st.session_state['feature_colors'][feature] = st.color_picker(
                            f"Color for {feature}",
                            st.session_state['feature_colors'][feature],
                            key=f"color_feature_{feature}"
                        )
                apply_viz = st.form_submit_button("✅ Show Multi-Feature Comparison")
            if 'multi_feature_plot' not in st.session_state:
                st.session_state['multi_feature_plot'] = go.Figure()
            if apply_viz:
                if multi_feature_data_source == "Raw Data":
                    data_viz_multi = st.session_state.get('final_data', st.session_state.get('data'))  # Prioritize final_data
                elif multi_feature_data_source != "None":
                    data_viz_multi = get_data(multi_feature_data_source)
                else:
                    data_viz_multi = None
                if data_viz_multi is not None and len(features_to_explore) >= 2:
                    if plot_type == 'Bar Chart':
                        fig = plot_multiple_features_distribution(
                            data_viz_multi,
                            features_to_explore,
                            st.session_state['feature_colors'],
                            histfunc_multi
                        )
                    elif plot_type == 'Line Plot':
                        fig = plot_multiple_features_line(
                            data_viz_multi,
                            features_to_explore,
                            st.session_state['class_colors'],
                            histfunc=histfunc_multi,
                            error_type=error_type.lower() if error_type != 'None' else None
                        )
                    elif plot_type == 'Radar Chart':
                        fig = plot_multiple_features_radar(
                            data_viz_multi,
                            features_to_explore,
                            st.session_state['class_colors'],
                            histfunc_multi
                        )
                    st.session_state['multi_feature_plot'] = fig
                else:
                    st.warning("⚠️ Please select at least two features and a valid data source.")
            if isinstance(st.session_state.get('multi_feature_plot', None), go.Figure):
                st.plotly_chart(st.session_state['multi_feature_plot'])

            gc.collect()



        with st.expander("**🔍 Signal & Molecular Profile Visualization**", expanded=False):
            st.markdown(
                '<p style="color: gray; font-size: 14px">Visualize individual and average signal profiles, grouped by class for in-depth exploration.</p>',
                unsafe_allow_html=True
            )
            with st.form("signal_profile_form"):
                data_source = st.selectbox(
                    "Select Data Source",
                    ['None', 'Raw Data', 'Preprocessed', 'Oversampled', 'Undersampled'],
                    key="signal_data_source",
                    help="Choose the dataset version from which the spectra will be plotted."
                )
                apply_viz = st.form_submit_button("📥 Load Features")
                if apply_viz:
                    if data_source == "Raw Data":
                        data = st.session_state.get('final_data', st.session_state.get('data'))  # Prioritize final_data
                    elif data_source != "None":
                        data = safe_load_data(data_source)
                    else:
                        data = None
                    if data is None:
                        st.warning("⚠️ No valid data available.")
                        data = None
                    st.session_state['signal_data'] = data
                class_options = st.session_state['signal_data']['Class'].unique().tolist() if st.session_state.get('signal_data') is not None else []
                class_to_plot = st.multiselect(
                    "Select Class for Mean Profile",
                    class_options,
                    key="class_to_plot_signal"
                )
                apply_mean_profile = st.form_submit_button("✅ Show Average Profile")
                index_labels = {}
                if st.session_state.get('signal_data') is not None:
                    index_labels = {idx: f"Index {idx} (Class {row['Class']})" for idx, row in st.session_state['signal_data'].iterrows()}
                selected_indices = st.multiselect(
                    "Select Individual Profiles to Display",
                    options=list(index_labels.keys()),
                    format_func=lambda x: index_labels[x] if x in index_labels else str(x),
                    key="selected_indices_signal"
                )
                apply_individual = st.form_submit_button("✅ Show Individual Profiles")
            if apply_mean_profile and st.session_state.get('signal_data') is not None and class_to_plot:
                fig_mean = plot_mean_spectrum(
                    st.session_state['signal_data'],
                    class_to_plot,
                    st.session_state['class_colors']
                )
                st.session_state['mean_spectrum_plot'] = fig_mean
            if 'mean_spectrum_plot' in st.session_state and isinstance(st.session_state['mean_spectrum_plot'], go.Figure):
                st.plotly_chart(st.session_state['mean_spectrum_plot'])
            if apply_individual and st.session_state.get('signal_data') is not None and selected_indices:
                fig_ind = plot_individual_spectra(
                    st.session_state['signal_data'],
                    st.session_state['class_colors'],
                    selected_indices
                )
                st.session_state['individual_spectra_plot'] = fig_ind
            if 'individual_spectra_plot' in st.session_state and isinstance(st.session_state['individual_spectra_plot'], go.Figure):
                st.plotly_chart(st.session_state['individual_spectra_plot'])

            gc.collect()
        
        import gc
        from itertools import combinations
        from collections import defaultdict
        import matplotlib.pyplot as plt

        # --- Fonction optimisée pour les intersections ---
        def calculate_maximal_intersections(venn_data, classes):
            class_features = {}
            for cls in classes:
                cls_features = set(venn_data[venn_data['Class'] == cls].drop(columns=['Class']).columns)
                class_features[cls] = cls_features

            feature_counts = defaultdict(int)
            feature_to_classes = defaultdict(set)

            for cls, features in class_features.items():
                for feature in features:
                    feature_counts[feature] += 1
                    feature_to_classes[feature].add(cls)

            shared_features = {feature: classes for feature, classes in feature_to_classes.items() if len(classes) >= 2}

            maximal_intersections = defaultdict(set)
            for feature, cls_set in shared_features.items():
                for r in range(2, len(cls_set) + 1):
                    for combo in combinations(cls_set, r):
                        maximal_intersections[combo].add(feature)

            maximal_intersections_filtered = {}
            for combo, features in maximal_intersections.items():
                is_maximal = True
                for other_combo, other_features in maximal_intersections.items():
                    if set(combo).issubset(other_combo) and features <= other_features and combo != other_combo:
                        is_maximal = False
                        break
                if is_maximal:
                    maximal_intersections_filtered[combo] = features

            return maximal_intersections_filtered




        def calculate_all_intersections(venn_data, classes):
            # Dictionnaire {classe: set(features non-nulles)}
            class_features = {}
            relevant_cols = [col for col in venn_data.columns if col != 'Class']
            
            for cls in classes:
                mask = venn_data['Class'] == cls
                cls_features = set(venn_data.loc[mask, relevant_cols].columns[venn_data.loc[mask, relevant_cols].notnull().any()])
                class_features[cls] = cls_features

            all_intersections = {}
            used_features = set()  # pour éviter les duplications

            # On parcourt d'abord les combinaisons les plus larges (N classes), puis descend
            for r in range(len(classes), 1, -1):
                for combo in combinations(classes, r):
                    intersect = set.intersection(*(class_features[cls] for cls in combo))
                    # On enlève ce qui a déjà été attribué à une intersection plus grande
                    intersect -= used_features
                    if intersect:
                        all_intersections[combo] = intersect
                        used_features.update(intersect)  # marquer ces features comme utilisées

            return all_intersections
        


        with st.expander("**🧮 Venn / UpSet Analysis**", expanded=False):
            st.markdown(
                '<p style="color: gray; font-size: 14px">'
                'Visualize class relationships and feature overlaps using Venn diagrams (≤6 classes) or UpSet plots (>6 classes).'
                '</p>',
                unsafe_allow_html=True
            )

            with st.form("venn_upset_form"):
                venn_data_source = st.selectbox(
                    "Select Data Source",
                    ['Raw Data', 'Preprocessed'],
                    key="venn_data_source_select",
                    help="Choose the dataset to analyze class intersections and exclusive features."
                )

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    show_venn = st.form_submit_button("Show Venn Diagram")
                with col2:
                    show_upset = st.form_submit_button("Show UpSet Plot")
                with col3:
                    show_features = st.form_submit_button("Show Exclusive Features")
                with col4:
                    show_all_intersections = st.form_submit_button("Show Intersection Features")

            # --- Traitement après clic sur un bouton ---
            if show_venn or show_upset or show_features or show_all_intersections:
                # Récupérer les données selon la source
                if venn_data_source == "Preprocessed":
                    venn_data = st.session_state.get('preprocessed_data')
                else:
                    venn_data = st.session_state.get('final_data', st.session_state.get('data'))

                if venn_data is not None:
                    # Supprimer colonnes non pertinentes
                    venn_data = venn_data.drop(columns=['File', 'RT', 'Sum'], errors='ignore')
                    classes = venn_data['Class'].unique()
                    num_classes = len(classes)
                    st.write(f"Detected {num_classes} unique classes.")

                    # --- Calcul intersections et features exclusives ---
                    st.session_state['all_intersections'] = calculate_all_intersections(venn_data, classes)
                    st.session_state['exclusive_features'] = {}

                    for cls in classes:
                        cls_data = venn_data[venn_data['Class'] == cls].drop(columns=['Class'])
                        other_data = venn_data[venn_data['Class'] != cls].drop(columns=['Class'])

                        cls_features = set(cls_data.columns[cls_data.notna().any()])
                        other_features = set(other_data.columns[other_data.notna().any()])

                        st.session_state['exclusive_features'][cls] = list(cls_features - other_features)

                        #Nettoyage intermédiaire par classe
                        del cls_data, other_data, cls_features, other_features
                        gc.collect()

                    # --- Cas 1 : Venn diagram (≤ 6 classes) ---
                    if show_venn:
                        if num_classes <= 6:
                            fig = plot_venn_diagram(venn_data, 'Class', st.session_state['class_colors'], venn_data_source)
                            st.session_state['venn_diagram_plot'] = fig
                            if isinstance(fig, plt.Figure):
                                st.pyplot(fig)
                            del fig
                        else:
                            st.error("⚠️ Too many classes (more than 6). Only UpSet plot is available.")

                    # --- Cas 2 : UpSet plot ---
                    if show_upset:
                        if num_classes > 1:
                            fig = plot_upset(venn_data, 'Class', venn_data_source)
                            st.session_state['upset_plot'] = fig
                            if isinstance(fig, plt.Figure):
                                st.pyplot(fig)
                            # Suppression pour libérer la RAM (souvent lourd)
                            plt.close(fig)
                            del fig
                        else:
                            st.warning("⚠️ At least 2 classes are required for an UpSet plot.")

                    # --- Cas 3 : Features exclusives ---
                    if show_features:
                        any_exclusive = any(st.session_state['exclusive_features'].values())
                        if not any_exclusive:
                            st.warning("⚠️ No exclusive features available for the selected classes.")
                        else:
                            for cls, features in st.session_state['exclusive_features'].items():
                                if features:
                                    st.info(f"**Exclusive to {cls}**: {', '.join(map(str, features))}")

                    # --- Cas 4 : Intersections ---
                    if show_all_intersections:
                        if not st.session_state['all_intersections']:
                            st.warning("⚠️ No intersection features available.")
                        else:
                            for combo, features in st.session_state['all_intersections'].items():
                                st.info(f"**Common to {', '.join(combo)}**: {', '.join(features)}")

                    # --- Nettoyage mémoire global ---
                    del venn_data, classes, num_classes
                    gc.collect()

                else:
                    st.warning("⚠️ No valid data available for Venn/UpSet.")


        st.markdown("""
        <h3 style="
            font-size: 1.2rem;
            border-bottom: 2px solid #318CE7;
            text-align: center;
            background-color: #f0f8ff;
            padding: 10px;
            border-radius: 5px;">
            Correlations and Similarities
        </h3>
        """, unsafe_allow_html=True)



        # ======================= CORRELATION ======================= #
        with st.expander("**⚖️ Correlation**", expanded=False):
            st.markdown(
                '<p style="color: gray; font-size: 14px">Compute correlations between the average feature vectors of each class using Pearson or Spearman methods.</p>',
                unsafe_allow_html=True
            )
            with st.form("correlation_form"):
                data_source_corr = st.selectbox(
                    "Select Data Source for Correlation",
                    ['Raw Data', 'Preprocessed', 'Oversampled', 'Undersampled'],
                    key="corr_data_source",
                )
                corr_method = st.selectbox(
                    "Correlation Method",
                    ['None', 'Pearson', 'Spearman'],
                    index=0,
                    key="corr_method"
                )
                apply_corr = st.form_submit_button("✅ Apply Correlation")
            
            if apply_corr:
                data_corr = get_data(data_source_corr)
                if data_corr is None or 'Class' not in data_corr.columns:
                    st.warning("No valid data available for correlation.")
                elif corr_method == 'None':
                    st.warning("Please select a correlation method.")
                else:
                    numeric_data = data_corr.drop(columns=['Class'], errors='ignore').select_dtypes(include='number')
                    if numeric_data.empty:
                        st.warning("No numeric features found for correlation.")
                    else:
                        grouped = data_corr.groupby('Class')[numeric_data.columns].mean()
                        corr_matrix = grouped.T.corr(method=corr_method.lower())
                        plot_heatmap(
                            corr_matrix,
                            f"{corr_method} Correlation",
                            "Each cell shows the correlation coefficient between classes (averaged features)."
                        )
                del data_corr
                gc.collect()


        with st.expander("**📐 Similarity**", expanded=False):
            st.markdown(
                '<p style="color: gray; font-size: 14px">Compare class profiles using Cosine Similarity (continuous) or Cohen\'s Kappa (categorical after discretization).</p>',
                unsafe_allow_html=True
            )
            with st.form("similarity_form"):
                data_source_sim = st.selectbox(
                    "Select Data Source for Similarity",
                    ['Raw Data', 'Preprocessed', 'Oversampled', 'Undersampled'],
                    key="sim_data_source",
                )
                similarity_method = st.selectbox(
                    "Similarity Method",
                    ['None', 'Cosine Similarity', "Cohen's Kappa"],
                    index=0,
                    key="sim_method"
                )
                apply_sim = st.form_submit_button("✅ Apply Similarity")
            
            if apply_sim:
                data_sim = get_data(data_source_sim)
                
                # Check if data is valid
                if data_sim is None or 'Class' not in data_sim.columns:
                    st.warning("No valid data available for similarity.")
                elif similarity_method == 'None':
                    st.warning("Please select a similarity method.")
                else:
                    # Check for missing values
                    if data_sim.drop(columns=['Class'], errors='ignore').isnull().values.any():
                        st.error(
                            "⚠️ Missing values detected in the dataset. "
                            "Please go to the Preprocessing step to either remove or impute missing values before running similarity analysis."
                        )
                    else:
                        numeric_data = data_sim.drop(columns=['Class'], errors='ignore').select_dtypes(include='number')
                        if numeric_data.empty:
                            st.warning("No numeric features found for similarity analysis.")
                        else:
                            grouped = data_sim.groupby('Class')[numeric_data.columns].mean()
                            classes = grouped.index

                            if similarity_method == "Cosine Similarity":
                                sim_matrix = cosine_similarity(grouped)
                                sim_df = pd.DataFrame(sim_matrix, index=classes, columns=classes)
                                plot_heatmap(
                                    sim_df,
                                    "Cosine Similarity",
                                    "Cosine similarity measures the angle between feature vectors of each class (1=identical, 0=orthogonal)."
                                )

                            elif similarity_method == "Cohen's Kappa":
                                nb_bins = st.slider(
                                    "Discretization Levels (Bins)", 2, 6, 3, 1,
                                    help="Number of bins to discretize features for Cohen's Kappa."
                                )
                                kappa_matrix = pd.DataFrame(index=classes, columns=classes, dtype=float)
                                for i, class_i in enumerate(classes):
                                    for j, class_j in enumerate(classes):
                                        vec1_cat = pd.qcut(grouped.loc[class_i].rank(method="first"), q=nb_bins, labels=False)
                                        vec2_cat = pd.qcut(grouped.loc[class_j].rank(method="first"), q=nb_bins, labels=False)
                                        kappa_matrix.iloc[i, j] = cohen_kappa_score(vec1_cat, vec2_cat)
                                plot_heatmap(
                                    kappa_matrix,
                                    "Cohen's Kappa Similarity",
                                    "Cohen’s Kappa evaluates the agreement in categorized feature profiles (1=perfect, 0=random, <0=disagreement)."
                                )
                del data_sim
                gc.collect()


    with tabs[2]:
    
        st.markdown(
            """
            <h3 style="
                font-size: 1.2rem;
                border-bottom: 2px solid #318CE7;
                text-align: center;
                background-color: #f0f8ff;
                padding: 10px;
                border-radius: 5px;">
                Unsupervised Learning
            </h3>
            """,
            unsafe_allow_html=True
        )




        # --- Initialisation des variables de session ---
        for key, default in [
            ("X_reduction", None),
            ("y_reduction", None),
            ("compressed_data", None),
            ("reduction_method", None),
            ("svd_available", False),
            ("fig_initial", None),
            ("fig_feature", None),
            ("n_components_reduction", 2)
        ]:
            if key not in st.session_state:
                st.session_state[key] = default

        with st.expander("**🗺️ Dimensionality Reduction**"):
            st.markdown(
                '<p style="color: gray; font-size: 14px;">Reduce Dimensionality and Visualize Clusters Using PCA, UMAP or t-SNE</p>',
                unsafe_allow_html=True
            )

            # --- FORM 1 : Apply Reduction ---
            with st.form("dim_reduction_form"):
                method = st.selectbox("Visualization by Data Reduction", ['None', 'PCA', 'UMAP', 't-SNE'], key="reduction_choice")
                data_source = st.selectbox(
                    "Data Source for Reduction",
                    ['Raw Data', 'Preprocessed', 'Oversampled', 'Undersampled'],
                    key="reduction_data_source"
                )
                n_components = st.number_input("Number of Components", min_value=2, max_value=200,
                                            value=st.session_state["n_components_reduction"], step=1, key="n_components")
                apply_reduction = st.form_submit_button("✅ Apply Reduction")

            if apply_reduction:
                # --- Récupération des données ---
                if data_source == 'Raw Data':
                    df = st.session_state.get('final_data', st.session_state.get('data'))
                elif data_source == 'Preprocessed':
                    df = st.session_state.get('preprocessed_data')
                elif data_source == 'Oversampled':
                    df = st.session_state.get('oversampled_data')
                elif data_source == 'Undersampled':
                    df = st.session_state.get('undersampled_data')
                else:
                    df = None

                if df is None:
                    st.warning("Selected data source not available.")
                else:
                    drop_cols = ['Class', 'File', 'RT', 'Sum'] if data_source in ['Raw Data', 'Preprocessed'] else ['Class']
                    try:
                        X = df.drop(columns=drop_cols, errors='ignore')
                        if data_source == 'Raw Data' and X.isna().any().any():
                            from sklearn.impute import SimpleImputer
                            X = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=X.columns)
                        y = df['Class']
                    except Exception as e:
                        st.error(f"Error preparing data: {e}")
                        st.stop()

                    if n_components > X.shape[1]:
                        st.error(f"Number of components ({n_components}) exceeds number of features ({X.shape[1]}).")
                        st.stop()

                    st.session_state["X_reduction"] = X
                    st.session_state["y_reduction"] = y
                    st.session_state["n_components_reduction"] = n_components

                    feature_col = "None"

                    # --- PCA / UMAP / t-SNE ---
                    try:
                        if method == "PCA":
                            svd_data, (loadings, explained_variance) = apply_pca(X, n_components, random_state=1)
                            st.session_state.update({
                                'compressed_data': svd_data,
                                'svd_model': (loadings, explained_variance),
                                'svd_available': True,
                                'reduction_method': "PCA"
                            })
                            st.session_state["fig_initial"] = plot_pca(svd_data, y, st.session_state['class_colors'], feature_col, X)

                        elif method == "UMAP":
                            df_umap = X.assign(Class=y)
                            st.session_state.update({'compressed_data': df_umap, 'reduction_method': "UMAP"})
                            st.session_state["fig_initial"] = plot_umap(df_umap, num_components=n_components,
                                                                    custom_colors=st.session_state['class_colors'],
                                                                    feature_intensity=feature_col)

                        elif method == "t-SNE":
                            df_tsne = X.assign(Class=y)
                            st.session_state.update({'compressed_data': df_tsne, 'reduction_method': "t-SNE"})
                            st.session_state["fig_initial"] = plot_tsne(df_tsne, num_components=n_components,
                                                                        custom_colors=st.session_state['class_colors'],
                                                                        feature_intensity=feature_col)
                    except Exception as e:
                        st.error(f"Error during dimensionality reduction: {e}")

            # --- Affichage du premier plot ---
            if st.session_state["fig_initial"] is not None:
                st.plotly_chart(st.session_state["fig_initial"], use_container_width=True, key="fig_initial")

            # --- FORM 2 : Feature Intensity + Regenerate Plot ---
            if st.session_state["compressed_data"] is not None:
                X = st.session_state["X_reduction"]
                y = st.session_state["y_reduction"]

                with st.form("feature_intensity_form"):
                    feature_col = st.selectbox("Feature Intensity", ['None'] + list(X.columns), key="feature_intensity")
                    refresh_plot = st.form_submit_button("✅  Show Feature")

                if refresh_plot:
                    method = st.session_state.get("reduction_method")
                    try:
                        if method == "PCA":
                            st.session_state["fig_feature"] = plot_pca(st.session_state["compressed_data"], y,
                                                                    st.session_state['class_colors'], feature_col, X)
                        elif method == "UMAP":
                            st.session_state["fig_feature"] = plot_umap(st.session_state["compressed_data"],
                                                                        num_components=st.session_state["n_components_reduction"],
                                                                        custom_colors=st.session_state['class_colors'],
                                                                        feature_intensity=feature_col)
                        elif method == "t-SNE":
                            st.session_state["fig_feature"] = plot_tsne(st.session_state["compressed_data"],
                                                                        num_components=st.session_state["n_components_reduction"],
                                                                        custom_colors=st.session_state['class_colors'],
                                                                        feature_intensity=feature_col)
                    except Exception as e:
                        st.error(f"Error regenerating plot: {e}")

            # --- Affichage du plot Feature Intensity ---
            if st.session_state.get("fig_feature") is not None:
                st.plotly_chart(st.session_state["fig_feature"], use_container_width=True, key="fig_feature")




            # --- FORM 3 : PCA Details ---
            if st.session_state.get("reduction_method") == "PCA" and st.session_state.get("svd_available", False):
                st.markdown("**PCA Details**")
                with st.form("pca_details_form"):
                    loadings, explained_variance = st.session_state['svd_model']
                    X = st.session_state["X_reduction"]

                    pc_index = st.selectbox("Select Principal Component",
                                            [f"PC{i+1}" for i in range(len(explained_variance))],
                                            index=0, key="pc_index_select")
                    top_n = st.number_input("Top Features to Display", min_value=5, max_value=50,
                                            value=10, step=1, key="top_loading_features_input")
                    show_contrib = st.form_submit_button("✅ Show PCA Contributions")

                if show_contrib:
                    try:
                        pc_idx = int(pc_index.replace("PC", "")) - 1

                        st.write("**Explained Variances**")
                        for i, var in enumerate(explained_variance):
                            st.write(f"PC{i + 1}: {var:.4f}")

                        contribs = pd.DataFrame({
                            'Feature': X.columns,
                            'Contribution': loadings[:, pc_idx],
                        }).assign(Abs_Contribution=lambda df: df['Contribution'].abs()) \
                            .sort_values(by='Abs_Contribution', ascending=False)

                        st.dataframe(contribs, hide_index=True)

                        top_contribs = contribs.head(top_n).sort_values(by="Contribution")

                        # --- Plotly Bar Chart ---
                        fig = px.bar(top_contribs, x="Contribution", y="Feature", orientation='h',
                                    text="Contribution", color="Contribution", color_continuous_scale='Blues')
                        fig.update_layout(title=f"Top {top_n} Feature Contributions to {pc_index}",
                                        xaxis_title="Loading (Contribution)",
                                        yaxis_title="Feature",
                                        yaxis=dict(autorange="reversed"),  # inverser l’ordre pour barh
                                        plot_bgcolor='white')
                        st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error displaying PCA information: {e}")




        with st.expander("**🧩 k-means Clustering and Silhouette Analysis**"):
            st.markdown(
                '<p style="color: gray; font-size: 14px">'
                'Assessing Group Formation and Heterogeneity'
                '</p>', unsafe_allow_html=True
            )

            # ----- Formulaire Silhouette -----
            with st.form("silhouette_form"):
                # --- Shared Inputs ---
                data_sources_list = ['None', 'Raw data', 'Preprocessed', 'Preprocessed + Oversampled', 'Preprocessed + Undersampled']
                data_source = st.selectbox(
                    "Select Data Source",
                    data_sources_list,
                    help="Choose which version of your dataset to use for clustering.",
                    key="form_data_source"
                )

                scaler_option = st.selectbox(
                    "Select Scaler",
                    ['StandardScaler', 'RobustScaler', 'MinMaxScaler'],
                    help="Choose a method to normalize your features before clustering.",
                    key="form_scaler"
                )

                cluster_range_input = st.text_input("Enter Range of Clusters for Silhouette (e.g., 2-10)", "2-10")
                run_silhouette = st.form_submit_button("Run Silhouette Analysis")

                # --- Préparation des données ---
                df, X_scaled, n_samples = None, None, 0
                if data_source != 'None':
                    source_map = {
                        'Raw data': 'data',
                        'Preprocessed': 'preprocessed_data',
                        'Preprocessed + Oversampled': 'oversampled_data',
                        'Preprocessed + Undersampled': 'undersampled_data'
                    }
                    data_kmeans = st.session_state.get(source_map.get(data_source))

                    # Si Raw data mais final_data existe, utiliser final_data à la place
                    if data_source == 'Raw data' and st.session_state.get('final_data') is not None:
                        data_kmeans = st.session_state['final_data']

                    if data_kmeans is not None:
                        df = data_kmeans.copy()
                        drop_cols = ['Class', 'File', 'RT', 'Sum']
                        X = df.drop(columns=[col for col in drop_cols if col in df.columns]).fillna(df.median())
                        scaler_cls = {'StandardScaler': StandardScaler, 'RobustScaler': RobustScaler, 'MinMaxScaler': MinMaxScaler}[scaler_option]
                        X_scaled = scaler_cls().fit_transform(X)
                        n_samples = X_scaled.shape[0]

                # --- Exécution silhouette ---
                if run_silhouette:
                    if X_scaled is None:
                        st.warning("No data available for the selected source.")
                    else:
                        with st.spinner("Running Silhouette Analysis..."):
                            try:
                                start, end = map(int, cluster_range_input.split('-'))
                                scores = []
                                progress = st.progress(0)
                                total = end - start + 1
                                
                                for i, n in enumerate(range(start, end + 1)):
                                    km = KMeans(n_clusters=n, n_init="auto", max_iter=1000, tol=0.01, random_state=1)
                                    labels = km.fit_predict(X_scaled)
                                    score = silhouette_score(X_scaled, labels)
                                    scores.append(score)
                                    progress.progress((i + 1) / total)
                                    time.sleep(0.05)
                                import plotly.express as px
                                fig = px.line(
                                    x=range(start, end + 1),
                                    y=scores,
                                    markers=True,
                                    title='Silhouette Score vs. Number of Clusters'
                                )
                                fig.update_layout(xaxis_title='Number of Clusters', yaxis_title='Silhouette Score')
                                st.plotly_chart(fig)

                            except ValueError:
                                st.error("Please enter a valid range (e.g., 2-10).")
                            finally:
                                del scores, km, labels, score
                                gc.collect()

            # ----- Formulaire Clustering Spécifique -----
            with st.form("specific_clustering_form"):
                # Même data_source et scaler que le formulaire silhouette
                k = st.number_input("Enter Specific Number of Clusters", min_value=2, max_value=10, value=4)
                method = st.selectbox("Select Dimensionality Reduction Method for vizualisation", ['PCA', 't-SNE', 'UMAP'])
                dims = st.selectbox("Select Number of Dimensions for vizualisation", [2, 3])
                apply_clustering = st.form_submit_button("Apply Specific Clustering")

                if apply_clustering:
                    if X_scaled is None:
                        st.warning("No data available for the selected source.")
                    else:
                        with st.spinner("Applying Clustering..."):
                            progress = st.progress(0.3)
                            time.sleep(0.05)

                            km = KMeans(n_clusters=k, n_init=1000, max_iter=1000, tol=0.01, random_state=1)
                            labels = km.fit_predict(X_scaled)
                            label_chars = [chr(ord('A') + i) for i in labels]
                            df['Class'] = label_chars

                            if method == 'PCA':
                                reducer = PCA(n_components=dims)
                            elif method == 't-SNE':
                                perplexity = max(5, min(int(np.sqrt(n_samples)), 50))
                                reducer = TSNE(n_components=dims, perplexity=perplexity)
                            elif method == 'UMAP':
                                n_neighbors = max(2, min(int(np.log2(n_samples)), 100))
                                reducer = umap.UMAP(n_components=dims, n_neighbors=n_neighbors)

                            progress.progress(0.6)
                            time.sleep(0.05)

                            reduced = reducer.fit_transform(X_scaled)

                            if dims == 2:
                                import plotly.express as px
                                fig = px.scatter(
                                    x=reduced[:, 0],
                                    y=reduced[:, 1],
                                    color=label_chars,
                                    title=f'{method} - Clustering (k={k})'
                                )
                                fig.update_layout(xaxis_title='Component 1', yaxis_title='Component 2')
                            else:
                                fig = px.scatter_3d(
                                    x=reduced[:, 0],
                                    y=reduced[:, 1],
                                    z=reduced[:, 2],
                                    color=label_chars,
                                    title=f'{method} - Clustering (k={k})'
                                )
                                fig.update_layout(
                                    scene=dict(
                                        xaxis_title='Component 1',
                                        yaxis_title='Component 2',
                                        zaxis_title='Component 3'
                                    )
                                )

                            st.plotly_chart(fig)
                            progress.progress(1.0)

                            st.write("###### Updated DataFrame with Cluster Labels")
                            st.dataframe(df, hide_index=True)
                            st.write("###### Cluster Distribution")
                            st.write(df['Class'].value_counts())

                            del reduced, reducer, km, labels, label_chars, fig
                            gc.collect()

            if df is not None:
                del X, X_scaled, df
                gc.collect()




        st.markdown(
            """
            <h3 style="
                font-size: 1.2rem;
                border-bottom: 2px solid #318CE7;
                text-align: center;
                background-color: #f0f8ff;
                padding: 10px;
                border-radius: 5px;">
                Supervised Learning
            </h3>
            """,
            unsafe_allow_html=True
        )



        # Expander pour Machine Learning
        with st.expander("**🤖 Train Machine Learning Models**"):
            st.markdown(
                '<p style="color: gray; font-size: 14px">'
                'Over 20 Machine Learning Models Available for Exploration'
                '</p>', unsafe_allow_html=True
            )

            # ----- Formulaire ML Training -----
            with st.form("ml_training_form"):
                # --- Inputs communs ---
                data_sources_list = ['None','Raw data','Preprocessed','Preprocessed + Oversampled', 'Preprocessed + Undersampled']
                data_source = st.selectbox(
                    "Data Source for Training",
                    data_sources_list,
                    key="form_train_data_source",
                    help="Choose the dataset on which to train the model."
                )

                apply_reduction = st.checkbox(
                    "Apply Dimensionality Reduction",
                    key="form_apply_reduction",
                    help="Reduce feature dimensions using PCA, UMAP, or t-SNE."
                )

                reduction_choice, n_components = None, None
                if apply_reduction:
                    reduction_choice = st.selectbox(
                        "Reduction Technique",
                        ['PCA', 'UMAP', 't-SNE'],
                        key="form_reduction_choice"
                    )
                    n_components = st.number_input(
                        "Number of Components",
                        min_value=2, max_value=200, value=2, step=1,
                        key="form_n_components"
                    )

                n_splits = st.number_input(
                    "Number of Splits for Cross-Validation",
                    min_value=2, max_value=50, value=5, step=1,
                    key="form_n_splits",
                    help="Number of folds for cross-validation."
                )

                train_models_btn = st.form_submit_button("Train Machine Learning Models")

            # ----- Préparer les données selon la source -----
            X, y = None, None
            if data_source != 'None':
                source_map = {
                    'Raw data': 'data',
                    'Preprocessed': 'preprocessed_data',
                    'Preprocessed + Oversampled': 'oversampled_data',
                    'Preprocessed + Undersampled': 'undersampled_data'
                }
                data_train = st.session_state.get(source_map.get(data_source))
                # Raw data → remplacer par final_data si existant
                if data_source == 'Raw data' and st.session_state.get('final_data') is not None:
                    data_train = st.session_state['final_data']

                if data_train is not None:
                    drop_cols = ['Class', 'File', 'RT', 'Sum']
                    X = data_train.drop(columns=[col for col in drop_cols if col in data_train.columns], errors='ignore')
                    y = data_train['Class']

            # ----- Exécution du training -----
            if train_models_btn:
                if X is None or y is None:
                    st.warning("Please select a valid data source.")
                else:
                    feature_names = X.columns.tolist()
                    # Appliquer la réduction si demandé
                    if apply_reduction and reduction_choice:
                        n_samples = X.shape[0]
                        if reduction_choice == 'PCA':
                            reducer = PCA(n_components)
                            X = reducer.fit_transform(X)
                        elif reduction_choice == 'UMAP':
                            reducer = umap.UMAP(n_components=n_components,
                                                n_neighbors=max(2, min(int(np.log2(n_samples)), 100)),
                                                random_state=1)
                            X = reducer.fit_transform(X)
                        elif reduction_choice == 't-SNE':
                            tsne = TSNE(n_components=n_components,
                                        perplexity=max(5, min(int(np.sqrt(n_samples)), 50)),
                                        random_state=1)
                            X = tsne.fit_transform(X)
                        feature_names = [f"Component {i+1}" for i in range(n_components)]

                    st.session_state['reduced_data'] = pd.DataFrame(X, columns=feature_names)
                    X = st.session_state['reduced_data']

                    try:
                        class_counts = Counter(y)
                        too_few_classes = [cls for cls, count in class_counts.items() if count < n_splits]

                        if too_few_classes:
                            st.error(f"The following class(es) have fewer samples than the number of CV splits ({n_splits}): {too_few_classes}")
                        elif len(class_counts) < 2:
                            st.error("At least two classes are required for training.")
                        else:
                            progress_bar = st.progress(0)
                            model_results = train_models(X, y, n_splits=n_splits, progress_bar=progress_bar)
                            st.session_state['models'] = model_results
                            st.success("Models trained successfully!")

                    except RuntimeError as e:
                        st.error(f"Runtime error: {e}")
                    except MemoryError:
                        st.error("MemoryError: Reduce dataset size or number of models.")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")
                        with st.expander("Show full traceback"):
                            st.code(traceback.format_exc(), language="python")

            # ----- Formulaire pour analyse des modèles -----
            # ----- Formulaire pour analyse des modèles -----
            if 'models' in st.session_state and st.session_state['models']:
                with st.form("ml_report_form"):
                    show_comparison_btn = st.form_submit_button("Show Model Comparison")
                    model_options = ['None'] + list(st.session_state['models'].keys()) if isinstance(st.session_state['models'], dict) else ['None']
                    
                    selected_model = st.selectbox(
                        "Select ML Model for Report",
                        model_options,
                        key="form_selected_model"
                    )
                    
                    # 🔑 Synchroniser avec la clé utilisée par SHAP/LIME
                    st.session_state["selected_model"] = selected_model  

                    show_report_btn = st.form_submit_button("Show Model Report")

                if show_report_btn and selected_model != 'None':
                    model_data = st.session_state['models'][selected_model]

                    # Classification Report
                    st.write("**Classification Report**")
                    report_data = model_data['classification_report']
                    if isinstance(report_data, dict):
                        st.dataframe(pd.DataFrame(report_data).T)
                    else:
                        st.text(report_data)

                    # Confusion Matrix
                    st.write("**Confusion Matrix**")
                    fig, ax = plt.subplots()
                    labels = model_data['label_encoder'].classes_
                    sns.heatmap(model_data['confusion_matrix'], annot=True, fmt='d',
                                cmap='viridis', xticklabels=labels, yticklabels=labels, ax=ax)
                    st.pyplot(fig)

                    # Learning Curves
                    try:
                        learning_curve_fig = plot_learning_curve(model_data['model'], X, y, n_splits=n_splits)
                        st.plotly_chart(learning_curve_fig)
                    except Exception as e:
                        st.error(f"Error plotting learning curve: {e}")

                # Affichage de la comparaison de modèles
                if show_comparison_btn:
                    fig = compare_models(st.session_state['models'])
                    st.plotly_chart(fig)

            del X, y
            gc.collect()





        with st.expander("**🧠 Train Deep Learning Models**"):
            st.markdown('<p style="color: gray; font-size: 14px ">CNN, RNN, and MLP Deep Learning Models for Advanced Tasks</p>', unsafe_allow_html=True)
            n_splits = st.number_input("Number of Splits", min_value=2, max_value=10, value=2, step=1, key="dl_n_splits", help="Higher values provide better generalization estimates but increase training time.")
            #epochs = st.number_input("Epochs", min_value=1, max_value=100, value=10, step=1, key="dl_epochs")
            epochs = st.number_input(
                "Epochs",
                min_value=1,
                max_value=100,
                value=10,
                step=1,
                key="dl_epochs",
                help=(
                    "Number of complete passes through the training dataset. "
                    "Too few may lead to underfitting, too many may cause overfitting. "
                    "Start with 10–20 and adjust based on performance."
                )
            )                
            # batch_size = st.number_input("Batch Size", min_value=1, max_value=128, value=32, step=1, key="dl_batch_size")
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=128,
                value=32,
                step=1,
                key="dl_batch_size",
                help=(
                    "Number of samples used per gradient update. "
                    "Smaller batches give noisier but more frequent updates. "
                    "Larger batches are more stable but require more memory. "
                    "A batch size of 32 is a common starting point."
                )
            )                
            learning_rate = st.number_input("Learning Rate", min_value=0.00001, max_value=0.9, value=0.001, step=0.0001, key="dl_learning_rate", format="%.3f", help="Lower values make learning slower but more stable. Try 0.001 as a starting point.")

            data_source = st.selectbox("Data Source for Training", [
                'None',
                'Raw data',
                'Preprocessed',
                'Preprocessed + Oversampled',
                'Preprocessed + Undersampled',
            ], key="dl_train_data_source")

            X, y = None, None
            if data_source == 'Raw data' and 'data' in st.session_state and st.session_state['data'] is not None:
                X = st.session_state['data'].drop(['Class', 'File', 'RT', 'Sum'], axis=1, errors='ignore')

                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='mean')  
                X = imputer.fit_transform(X)  
                y = st.session_state['data']['Class']
            if data_source == 'Preprocessed' and 'preprocessed_data' in st.session_state and st.session_state['preprocessed_data'] is not None:
                X = st.session_state['preprocessed_data'].drop(['Class', 'File', 'RT', 'Sum'], axis=1, errors='ignore')
                y = st.session_state['preprocessed_data']['Class']
            elif data_source == 'Preprocessed + Oversampled' and 'oversampled_data' in st.session_state and st.session_state['oversampled_data'] is not None:
                X = st.session_state['oversampled_data'].drop(['Class'], axis=1, errors='ignore')
                y = st.session_state['oversampled_data']['Class']
            elif data_source == 'Preprocessed + Undersampled' and 'undersampled_data' in st.session_state and st.session_state['undersampled_data'] is not None:
                X = st.session_state['undersampled_data'].drop(['Class'], axis=1, errors='ignore')
                y = st.session_state['undersampled_data']['Class']
            elif data_source == 'None':
                st.warning("Please select a valid data source.")
            if X is not None:
                if st.button("Train Deep Learning Models", key="train_dl_models"):
                    try:
                        progress_bar = st.progress(0)
                        model_results = train_DL(X, y, n_splits=n_splits, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
                        st.session_state['dl_models'] = model_results
                        st.success("Deep Learning models trained successfully!")
                    except Exception as e:
                        st.error(f"Error during DL model training: {e}")

            if st.button("Deep Learning Model Comparison", key="show_dl_model_comparison") and 'dl_models' in st.session_state:
                st.plotly_chart(compare_DL(st.session_state['dl_models']))

            selected_dl_model = st.selectbox("Select DL Model for Report", ['None'] + list(st.session_state['dl_models'].keys()), key="selected_dl_model")
            if selected_dl_model != 'None':
                dl_model_data = st.session_state['dl_models'][selected_dl_model]
                if st.button("DL Model Results", key="show_dl_model_results"):
                    X_test = dl_model_data['X_test']
                    y_test = dl_model_data['y_test']
                    history = dl_model_data['history']
                    label_encoder = dl_model_data['label_encoder']  # Récupérez le label_encoder à partir des résultats
                    display_model_results(dl_model_data['model'], X_test, y_test, history, label_encoder)

                if st.button("Show Cross-validated Results", key="show_global_results"):
                    display_global_results(dl_model_data)
            del X
            del y
            gc.collect()



        with st.expander("**💾 Save Model**"):
            st.markdown('<p style="color: gray; font-size: 14px;">Storing Your Trained Models for Future Use.</p>', unsafe_allow_html=True)

            model_type_save = st.selectbox("Select Model Type", ['Machine Learning', 'Deep Learning'], key="model_type_save", help="Select which type of model you want to save and export.")

            selected_model = None
            selected_dl_model = None

            if model_type_save == 'Machine Learning':
                available_models = list(st.session_state.get('models', {}).keys()) if isinstance(st.session_state.get('models', {}), dict) else []
                selected_model = st.selectbox("Select Model", available_models, key="selected_ml_model") if available_models else None
            else:
                available_dl_models = list(st.session_state.get('dl_models', {}).keys()) if isinstance(st.session_state.get('dl_models', {}), dict) else []
                selected_dl_model = st.selectbox("Select Deep Learning Model", available_dl_models, key="selected_dl_model_save") if available_dl_models else None

            save_directory = "saved_models"
            os.makedirs(save_directory, exist_ok=True)
            auto_save =True

            if auto_save and (selected_model or selected_dl_model):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = selected_model if selected_model else selected_dl_model
                model_path = os.path.join(save_directory, f"{model_name}_{timestamp}_model.pkl")
                features_path = os.path.join(save_directory, f"{model_name}_{timestamp}_features.pkl")
                label_encoder_path = os.path.join(save_directory, f"{model_name}_{timestamp}_label_encoder.pkl")
            else:
                model_path = os.path.join(save_directory, "model.pkl")
                features_path = os.path.join(save_directory, "features.pkl")
                label_encoder_path = os.path.join(save_directory, "label_encoder.pkl")




            if st.button("Save Model", key="save_model"):
                try:
                    if model_type_save == 'Machine Learning' and selected_model:
                        model_data = st.session_state['models'][selected_model]
                        model_name = selected_model
                    elif model_type_save == 'Deep Learning' and selected_dl_model:
                        model_data = st.session_state['dl_models'][selected_dl_model]
                        model_name = selected_dl_model
                    else:
                        st.error("No model selected!")
                        st.stop()

                    model_accuracy = model_data.get('accuracy', None)


                    reduced_data = st.session_state.get('reduced_data', None)
                    preprocessed_data = st.session_state.get('preprocessed_data', None)

                    if reduced_data is not None and hasattr(reduced_data, 'empty') and not reduced_data.empty:
                        feature_names = reduced_data.columns.tolist()
                    elif preprocessed_data is not None and hasattr(preprocessed_data, 'empty') and not preprocessed_data.empty:
                        feature_names = preprocessed_data.drop(['Class', 'File', 'RT', 'Sum'], axis=1, errors='ignore').columns.tolist()
                    elif model_data.get('features', None) is not None:
                        feature_names = model_data['features']

                    else:
                        st.error("No feature data available for saving.")
                        st.stop()

                    model_buffer = io.BytesIO()
                    joblib.dump({
                        'model': model_data['model'],
                        'accuracy': model_accuracy 
                    }, model_buffer)
                    model_buffer.seek(0)

                    features_buffer = io.BytesIO()
                    joblib.dump(feature_names, features_buffer)
                    features_buffer.seek(0)

                    label_encoder_buffer = io.BytesIO()
                    joblib.dump(model_data.get('label_encoder', None), label_encoder_buffer)
                    label_encoder_buffer.seek(0)

                    # Session state pour download
                    st.session_state.model_buffer = model_buffer
                    st.session_state.features_buffer = features_buffer
                    st.session_state.label_encoder_buffer = label_encoder_buffer
                    st.session_state.model_name = model_name
                    st.session_state.timestamp = timestamp
                    st.session_state.show_model_downloads = True

                    joblib.dump({'model': model_data['model'], 'accuracy': model_accuracy}, model_path)
                    joblib.dump(feature_names, features_path)
                    joblib.dump(model_data.get('label_encoder', None), label_encoder_path)

                except Exception as e:
                    st.error(f"Error saving model: {e}")


            # Show download buttons if available
            if st.session_state.get("show_model_downloads", False):
                st.download_button(
                    label="Download Model",
                    data=st.session_state.model_buffer,
                    file_name=f"{st.session_state.model_name}_{st.session_state.timestamp}_model.pkl",
                    mime="application/octet-stream"
                )
                st.download_button(
                    label="Download Features",
                    data=st.session_state.features_buffer,
                    file_name=f"{st.session_state.model_name}_{st.session_state.timestamp}_features.pkl",
                    mime="application/octet-stream"
                )
                st.download_button(
                    label="Download Label Encoder",
                    data=st.session_state.label_encoder_buffer,
                    file_name=f"{st.session_state.model_name}_{st.session_state.timestamp}_label_encoder.pkl",
                    mime="application/octet-stream"
                )
                st.success(f"Model '{st.session_state.model_name}' is available for download!")



        with st.expander("**🔎 Load & Verify Model**"):
            st.markdown(
                '<p style="color: gray; font-size: 14px;">Load a previously saved Machine Learning or Deep Learning model and inspect its features, classes, and accuracy (if available).</p>',
                unsafe_allow_html=True
            )

            model_type_load = st.selectbox("Select Model Type", ['Machine Learning', 'Deep Learning'], key="load_model_type")
            model_file = st.file_uploader("Upload Model File (.pkl)", type=["pkl"], key="load_model_file")
            features_file = st.file_uploader("Upload Features File (.pkl)", type=["pkl"], key="load_features_file")
            label_encoder_file = st.file_uploader("Upload Label Encoder File (.pkl, optional)", type=["pkl"], key="load_label_encoder_file")

            if st.button("Load & Inspect Model", key="load_verify_btn"):
                if model_file is None or features_file is None:
                    st.error("Please upload both the model and features files.")
                else:
                    try:
                        # Chargement du modèle (dict ou direct)
                        loaded_obj = joblib.load(model_file)

                        if isinstance(loaded_obj, dict):
                            model = loaded_obj.get('model', None)
                            accuracy = loaded_obj.get('accuracy', None)
                        else:
                            model = loaded_obj
                            accuracy = None

                        # Chargement des features et du label_encoder
                        features = joblib.load(features_file)
                        label_encoder = joblib.load(label_encoder_file) if label_encoder_file else None

                        st.success("✅ Model loaded successfully!")

                        # Affichage des features
                        st.markdown("**🔹 Features used in training**")
                        st.dataframe(pd.DataFrame(features, columns=["Feature"]))

                        # Affichage modèle ML ou DL
                        if model_type_load == 'Machine Learning':
                            st.markdown("**🔹 Model Details**")
                            st.text(str(model))
                        elif model_type_load == 'Deep Learning':
                            st.markdown("**🔹 Model Architecture**")
                            try:
                                model_summary = []
                                model.summary(print_fn=lambda x: model_summary.append(x))
                                st.text("\n".join(model_summary))
                            except Exception as e:
                                st.warning(f"Cannot display architecture: {e}")

                        # Labels / classes
                        st.markdown("**🔹 Classes / Labels**")
                        if label_encoder:
                            st.write(label_encoder.classes_)
                        elif hasattr(model, "classes_"):
                            st.write(model.classes_)
                        else:
                            st.info("No label encoder or embedded class list found.")

                        # Accuracy si disponible
                        if accuracy is not None:
                            st.markdown(f"**Model Accuracy** {accuracy:.4f}")
                        else:
                            st.info("No accuracy stored in this model.")

                    except Exception as e:
                        st.error(f"Error loading model or metadata: {e}")

                    # Cleanup
                    try:
                        del model, features
                        gc.collect()
                    except:
                        pass





    with tabs[3]:

        st.markdown(
            """
            <h3 style="
                font-size: 1.2rem;
                border-bottom: 2px solid #318CE7;
                text-align: center;
                background-color: #f0f8ff;
                padding: 10px;
                border-radius: 5px;">
                Differential Analysis
            </h3>
            """,
            unsafe_allow_html=True
        )


        

        # ---------------- Volcano Plot Expander ----------------
        with st.expander("**🌋 Volcano Plot**", expanded=st.session_state.get("show_volcano", False)):
            st.markdown(
                '<p style="color: gray; font-size: 14px;">'
                'Significant features between conditions using p-value and fold change thresholds '
                'for both binary and multi-class.'
                '</p>',
                unsafe_allow_html=True
            )

            selected_features = []

            # ---------------- FORM ----------------
            with st.form("volcano_form"):
                # Data source selection
                data_source = st.selectbox(
                    "Select Data Source for Volcano Plot",
                    ['None', 'Preprocessed', "Raw data", 'Oversampled', 'Undersampled'],
                    key="volcano_data_source",
                    help="Choose the data source for generating the Volcano Plot."
                )

                # Feature selection
                select_all_features_volcano = st.checkbox(
                    "Select All Features for Volcano Plot",
                    key="select_all_features_volcano",
                    help="Check this box to select all features automatically."
                )

                features_inputvp = st.text_input(
                    "Features",
                    key="features_inputvp",
                    help="Enter a comma-separated list of features to include in the Volcano Plot."
                )

                # Peak picking
                use_peak_picking = st.checkbox(
                    "Use Feature Detection",
                    key="use_peak_picking",
                    help="Enable feature detection to automatically identify peaks."
                )

                intensity_threshold = st.number_input(
                    "Peak Intensity Threshold",
                    min_value=0.0, max_value=100000.0, value=0.01, step=1.0, format="%.5f",
                    help="Set the intensity threshold for peak detection."
                )

                # Thresholds
                p_value_threshold = st.number_input(
                    "Select P-Value Threshold",
                    min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.3f",
                    help="Set the p-value threshold to filter significant features."
                )
                fold_change_threshold = st.number_input(
                    "Select Fold Change Threshold",
                    min_value=0.0, max_value=10.0, value=0.0, step=0.01, format="%.2f",
                    help="Set the fold change threshold to filter significant features."
                )

                correction_method = st.selectbox(
                    "Multiple-Testing Correction Method",
                    ["FDR (Benjamini–Hochberg)", "Bonferroni", "None"],
                    index=0,
                    help=(
                        "Choose the method for correcting p-values for multiple comparisons:\n"
                        "- **FDR (Benjamini–Hochberg)**: less strict, recommended for omics data.\n"
                        "- **Bonferroni**: more conservative.\n"
                        "- **None**: no correction applied."
                    )
                )

                # Highlight feature names
                highlight_features = st.checkbox(
                    "Highlight Feature Names",
                    value=False,
                    key="highlight_features",
                    help="Check this box to highlight feature names in the Volcano Plot."
                )
                # custom_highlight_input = st.text_area(
                #     "Features to highlight (optional)",
                #     placeholder="mz_123.456, ProtX, Feature_X...",
                #     help=(
                #         "Enter specific feature names (comma-separated) that you want to highlight on the Volcano plot. "
                #         "Only their names will be displayed, regardless of statistical significance."
                #     )
                # )
                # Submit button
                submitted = st.form_submit_button("Display Volcano Plot")

            # ---------------- ACTION ----------------
            if submitted:
                # Load selected data source
                data_vol = None
                if data_source == 'Raw data':
                    data_vol = st.session_state.get('data')
                elif data_source == 'Preprocessed':
                    data_vol = st.session_state.get('preprocessed_data')
                elif data_source == 'Oversampled':
                    data_vol = st.session_state.get('oversampled_data')
                elif data_source == 'Undersampled':
                    data_vol = st.session_state.get('undersampled_data')

                # Stocker la source choisie pour l'utiliser dans RESULTS
                st.session_state["volcano_data_source_df"] = data_vol

                class_column = st.session_state.get('class_column', 'Class')

                if data_vol is None:
                    st.warning("Please select a valid data source.")
                else:
                    # --- Feature selection ---
                    if select_all_features_volcano:
                        selected_features = [
                            col for col in data_vol.columns if col not in ['Class', 'File', 'RT', 'Sum']
                        ]
                    else:
                        selected_features = [f.strip() for f in features_inputvp.split(',')] if features_inputvp else []

                    # Peak picking
                    peak_features = detect_peaks(data_vol, intensity_threshold) if use_peak_picking else []
                    selected_features = list(set(selected_features + peak_features))

                    if not selected_features:
                        st.warning("Please select features to display the Volcano Plot.")
                    else:
                        try:
                            with st.spinner("Generating Volcano Plot..."):

                                method_map = {
                                    "FDR (Benjamini–Hochberg)": "fdr_bh",
                                    "Bonferroni": "bonferroni",
                                    "None": "none"
                                }

                                volcano_data = calculate_volcano_data(
                                    data_vol, class_column, selected_features, p_value_threshold, correction_method=method_map[correction_method]
                                )

                                # Apply fold change filter
                                filtered_volcano_data = volcano_data[
                                    (volcano_data['Log2 Fold Change'] >= fold_change_threshold) |
                                    (volcano_data['Log2 Fold Change'] <= -fold_change_threshold)
                                ]

                                # Plot
                                volcano_plot = plot_volcano(
                                    filtered_volcano_data,
                                    highlight_features,
                                    p_value_threshold,
                                    fold_change_threshold
                                )

                                st.plotly_chart(volcano_plot)

                                # Store results
                                st.session_state["volcano_data"] = filtered_volcano_data
                                st.session_state["show_volcano"] = True

                        except Exception as e:
                            st.error(f"Error generating Volcano Plot: {e}")
                        finally:
                            del data_vol
                            gc.collect()

            # ---------------- RESULTS ----------------
            if "volcano_data" in st.session_state:
                try:
                    volcano_data = st.session_state["volcano_data"]
                    data_vol = st.session_state.get("volcano_data_source_df")  
                    comparisons = volcano_data["Comparison"].unique()
                    significant_features = set()

                    for comparison in comparisons:
                        comparison_data = volcano_data[volcano_data["Comparison"] == comparison]
                        upregulated = comparison_data[comparison_data["Regulation Type"] == "Upregulated"]["Feature"].tolist()
                        downregulated = comparison_data[comparison_data["Regulation Type"] == "Downregulated"]["Feature"].tolist()

                        st.write(f"**Comparison: {comparison}**")

                        if upregulated:
                            st.info("**Upregulated Features:**")
                            st.success(", ".join(upregulated))
                            significant_features.update(upregulated)
                        else:
                            st.warning("No specifically upregulated features found.")

                        if downregulated:
                            st.info("**Downregulated Features:**")
                            st.success(", ".join(downregulated))
                            significant_features.update(downregulated)
                        else:
                            st.warning("No specifically downregulated features found.")

                    # Display dataframe of significant features from the selected data source
                    if significant_features and data_vol is not None:
                        significant_features = list(significant_features)
                        class_column = st.session_state.get("class_column", "Class")

                        missing_cols = [f for f in significant_features if f not in data_vol.columns]
                        if missing_cols:
                            st.warning(f"Missing columns in selected data source: {', '.join(missing_cols)}")
                            significant_features = [f for f in significant_features if f not in missing_cols]

                        if class_column in data_vol.columns:
                            significant_data = data_vol[[class_column] + significant_features]
                            st.write("**DataFrame with Significant Features from Selected Source:**")
                            st.dataframe(significant_data, hide_index=True)
                        else:
                            st.error("Class column missing from the selected data source.")
                    elif not significant_features:
                        st.warning("No significant features found.")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")



        # Expander Heatmap
        with st.expander("**🔥 Heatmap Clustering features and samples**", expanded=st.session_state.get("show_heatmap", False)):
            st.markdown(
                '<p style="color: gray; font-size: 14px;">'
                'Feature and sample clustering-heatmap can be performed on all or selected features, '
                'with statistical significance tested on original or log2 intensities.'
                '</p>',
                unsafe_allow_html=True
            )

            # 🔥 Formulaire pour configurer le heatmap
            with st.form("heatmap_form"):
                select_all_features = st.checkbox(
                    "Select All Features",
                    key="select_all_features",
                    help="Check this box to select all features automatically."
                )

                data_source = st.selectbox(
                    "Select Data Source for Heatmap",
                    ['None', 'Preprocessed', 'Raw data', 'Oversampled', 'Undersampled'],
                    key="heatmap_data_source",
                    help="Choose the data source for generating the Heatmap."
                )

                # Couleurs
                default_colors = ["#00FF00", "#000000", "#FF0000"]  # lime, black, red
                color_labels = ["Underexpression", "Neutral", "Overexpression"]
                custom_colors = [
                    st.color_picker(f"🎨 Color of {label}", default_colors[i], help=f"Select a custom color for {label}.")
                    for i, label in enumerate(color_labels)
                ]

                average_by_class = st.checkbox(
                    "Average by Class",
                    key="average_by_class",
                    help="Check this box to average the feature values by class."
                )

                perform_stat_test = st.checkbox(
                    "Perform Statistical Test",
                    key="perform_stat_test",
                    help="Check this box to perform a statistical test on the selected features."
                )

                if perform_stat_test:
                    p_value_threshold = st.number_input(
                        "P-value",
                        min_value=0.0000001,
                        max_value=1.0,
                        value=0.01,
                        step=0.0001,
                        key="p_value_threshold",
                        help="Set the p-value threshold for the statistical test."
                    )
                    data_type = st.selectbox(
                        "Select Data Type for Statistical Test",
                        ['Original Intensity', 'Log2'],
                        key="data_type",
                        help="Choose the data type for the statistical test (original or transformed log2)."
                    )
                else:
                    p_value_threshold = 1
                    data_type = 'Original Intensity'

                # Sélection manuelle des features
                features_input = ""
                if not select_all_features:
                    features_input = st.text_input(
                        "Features (comma-separated)",
                        key="features_input",
                        help="Enter a comma-separated list of features to include in the Heatmap."
                    )

                show_sample_names = st.checkbox(
                    "Show sample names on heatmap",
                    value=True,
                    help="Display sample names on the heatmap if dataset is small."
                )

                # Bouton de validation
                submitted = st.form_submit_button("Show Heatmap")



            # ----------------- Exécution quand le formulaire est soumis -----------------
            if submitted:
                try:
                    st.session_state["show_heatmap"] = True

                    # Charger la source des données
                    if data_source == 'Raw data':
                        data_heat = st.session_state.get('data')
                    elif data_source == 'Preprocessed':
                        data_heat = st.session_state.get('preprocessed_data')
                    elif data_source == 'Oversampled':
                        data_heat = st.session_state.get('oversampled_data')
                    elif data_source == 'Undersampled':
                        data_heat = st.session_state.get('undersampled_data')
                    else:
                        data_heat = None

                    if data_heat is None:
                        st.warning("Please select a valid data source.")
                        st.stop()

                    # Déterminer les features sélectionnées
                    if select_all_features:
                        selected_features = [col for col in data_heat.columns if col not in ['Class', 'File', 'RT', 'Sum']]
                    else:
                        selected_features = [f.strip() for f in features_input.split(',')] if features_input else []
                        if not selected_features:
                            st.warning("Please select features for the heatmap.")
                            st.stop()

                    # Vérifier que les features existent vraiment
                    selected_features = [f for f in selected_features if f in data_heat.columns]
                    if not selected_features:
                        st.error("No valid features found in the dataset. Please check your selection.")
                        st.stop()

                    # Appliquer log2 sur tout le DataFrame si demandé (évite NaN/-inf)
                    if perform_stat_test and data_type == 'Log2':
                        data_heat[selected_features] = np.log2(data_heat[selected_features].clip(lower=1e-6))

                    # Test statistique si demandé
                    if select_all_features and perform_stat_test:
                        significant_features = []
                        classes = data_heat['Class'].unique()

                        if len(classes) < 2:
                            st.error("The 'Class' column must contain at least two unique classes.")
                            st.stop()

                        for feature in selected_features:
                            groups = [data_heat[data_heat['Class'] == c][feature] for c in classes]

                            # Vérifier qu'il y a assez de données
                            if all(len(g) > 1 for g in groups):
                                if len(classes) > 2:
                                    _, p_value = f_oneway(*groups)
                                else:
                                    _, p_value = ttest_ind(groups[0], groups[1])
                                if p_value < p_value_threshold:
                                    significant_features.append(feature)
                            else:
                                st.warning(f"Not enough data for feature '{feature}' in one of the classes.")

                        if not significant_features:
                            st.warning("No significant features found. Please adjust the p-value threshold.")
                            st.stop()

                        selected_features = significant_features

                    # Moyenne par classe si demandé
                    if average_by_class:
                        data_heat = data_heat.groupby("Class").mean().reset_index()

                    st.markdown(f"**{len(selected_features)} feature(s)** selected for clustering.")
                    if perform_stat_test:
                        st.markdown(f"P-value threshold: `{p_value_threshold}` on `{data_type}` data")

                    # Affichage heatmap
                    with st.spinner("Generating heatmap..."):
                        plot_heatmap_samples(
                            data_heat,
                            st.session_state['class_colors'],
                            selected_features,
                            custom_colors,
                            show_sample_names=show_sample_names
                        )

                    st.markdown("**Overexpressed Features by Class**")
                    if average_by_class:
                        df_for_overexpr = data_heat.set_index("Class")
                    else:
                        df_for_overexpr = data_heat.copy()

                    classes = data_heat['Class'].unique()
                    overexpressed_features = {}

                    for cls in classes:
                        cls_mask = df_for_overexpr.index == cls if average_by_class else data_heat['Class'] == cls
                        cls_values = df_for_overexpr.loc[cls_mask, selected_features].mean() if average_by_class else df_for_overexpr.loc[cls_mask, selected_features].mean()
                        other_values = df_for_overexpr.loc[~cls_mask, selected_features].mean() if average_by_class else df_for_overexpr.loc[~cls_mask, selected_features].mean()
                        
                        diff = cls_values - other_values
                        # Features plus hautes que les autres classes
                        overexpressed_features[cls] = list(diff[diff > 0].sort_values(ascending=False).index)

                    # Affichage
                    for cls, feats in overexpressed_features.items():
                        if feats:
                            st.info(f"**Class {cls}**: {', '.join(feats)}")
                        else:
                            st.info(f"**Class {cls}**: No overexpressed features detected")

                    del data_heat, selected_features, custom_colors
                    gc.collect()

                except Exception as e:
                    import traceback
                    st.error(f"An error occurred while generating the Heatmap: {e}")
                    st.text(traceback.format_exc())



        if "show_boxplots" not in st.session_state:
            st.session_state["show_boxplots"] = False

        with st.expander("📊 **Statistical Visualization of Selected Features**",
                        expanded=st.session_state["show_boxplots"]):

            st.markdown(
                '<p style="color: gray; font-size: 14px;">Visualize and statistically compare the distribution '
                'of selected features across sample classes. Includes p-value correction and filtering.</p>',
                unsafe_allow_html=True
            )

            # -------------------------------------
            # Full form in st.form()
            # -------------------------------------
            with st.form("boxplot_stat_form", clear_on_submit=False):

                # Dataset selection
                data_source = st.selectbox(
                    "Select Data Source",
                    ['Raw data', 'Preprocessed', 'Oversampled', 'Undersampled'],
                    key="boxplot_data_source",
                    help="Choose the dataset used for visualization and statistical comparison."
                )

                # Feature input
                mz_values_input = st.text_input(
                    "Enter Feature Names",
                    help="Enter one or more feature names (m/z, prot, transcrit, gene...), separated by commas, space, tab.. or mix of theme"
                )

                # Statistical test
                test = st.selectbox(
                    "Select Statistical Test",
                    ['Kruskal', 'Mann-Whitney', 't-test_ind', 'ANOVA'],
                    index=0,
                    key="statistical_test",
                    help=(
                        "Choose a statistical test to compare groups:\n"
                        "- **Kruskal**: non-parametric, for 2+ groups\n"
                        "- **Mann-Whitney**: non-parametric, only for 2 groups\n"
                        "- **t-test_ind**: parametric, only for 2 groups\n"
                        "- **ANOVA**: parametric, for 2+ groups"
                    )
                )

                # P-value correction
                pval_correction = st.selectbox(
                    "Multiple Testing Correction",
                    ['None', 'Bonferroni', 'FDR (Benjamini-Hochberg)'],
                    key="pval_correction_method",
                    help=(
                        "Correction for multiple statistical tests:\n"
                        "- **None**: no correction\n"
                        "- **Bonferroni**: very strict, reduces false positives\n"
                        "- **FDR (Benjamini-Hochberg)**: controls false discovery rate (recommended)"
                    )
                )

                # Plot type
                plot_type = st.selectbox(
                    "Choose Plot Type",
                    ['Box Plot', 'Violin Plot', 'Bar Plot'],
                    key="plot_type",
                    help=(
                        "Select the type of visualization:\n"
                        "- **Box Plot**: distribution and quartiles\n"
                        "- **Violin Plot**: smoothed distribution shape\n"
                        "- **Bar Plot**: group means with error bars"
                    )
                )

                col1, col2 = st.columns(2)

                with col1:
                    show_scatter = st.checkbox(
                        "🔹 Show Individual Points",
                        key="show_scatter",
                        help="Overlay individual sample points on the plot."
                    )

                with col2:
                    use_log2 = st.checkbox(
                        "🔹 Apply log2 Transformation",
                        key="use_log2",
                        help="Apply a log2 transformation to reduce skew and stabilize variance."
                    )

                # Submit
                submitted = st.form_submit_button(
                    "Run",
                    help="Run statistical analysis and generate visualizations."
                )

            # --------------------------------------------------------
            # PROCESSING (EXECUTES ONLY ON SUBMIT)
            # --------------------------------------------------------
            if submitted:

                data_sig_key = {
                    'Raw data': 'data',
                    'Preprocessed': 'preprocessed_data',
                    'Oversampled': 'oversampled_data',
                    'Undersampled': 'undersampled_data'
                }.get(data_source)

                data_sig = st.session_state.get(data_sig_key, None)

                if data_sig is None:
                    st.error("❌ Dataset not loaded.")

                else:
                    import pandas as pd
                    import scipy.stats as stats
                    from statsmodels.stats.multitest import multipletests

                    data_sig = data_sig.copy()
                    class_col = st.session_state.get("label_column", "Class")

                    # Parsing features
                    # mz_values = [
                    #     mz.strip() for mz in mz_values_input.split(',')
                    #     if mz.strip() in data_sig.columns
                    # ]

                    raw_features = re.split(r"[,\s;]+", mz_values_input.strip())

                    mz_values = [
                        mz.strip() for mz in raw_features
                        if mz.strip() in data_sig.columns
                    ]

                    if not mz_values:
                        st.warning("⚠️ None of the entered features were found in the dataset.")
                    else:
                        # Compute stats
                        results = []
                        for mz in mz_values:
                            groups = [
                                data_sig[data_sig[class_col] == g][mz].dropna()
                                for g in data_sig[class_col].unique()
                            ]
                            p = None
                            try:
                                if test == 'Kruskal':
                                    p = stats.kruskal(*groups).pvalue
                                elif test == 'Mann-Whitney' and len(groups) == 2:
                                    p = stats.mannwhitneyu(groups[0], groups[1]).pvalue
                                elif test == 't-test_ind' and len(groups) == 2:
                                    p = stats.ttest_ind(groups[0], groups[1]).pvalue
                                elif test == 'ANOVA':
                                    p = stats.f_oneway(*groups).pvalue
                            except Exception:
                                p = None

                            results.append({"feature": mz, "pvalue": p})

                        result_df = pd.DataFrame(results).dropna()

                        # P-value correction
                        if pval_correction == 'Bonferroni':
                            result_df["adj_pvalue"] = multipletests(result_df["pvalue"], method="bonferroni")[1]
                        elif pval_correction == 'FDR (Benjamini-Hochberg)':
                            result_df["adj_pvalue"] = multipletests(result_df["pvalue"], method="fdr_bh")[1]
                        else:
                            result_df["adj_pvalue"] = result_df["pvalue"]

                        sig_df = result_df[result_df["adj_pvalue"] < 0.05]
                        nonsig_df = result_df[result_df["adj_pvalue"] >= 0.05]

                        sig_features = sig_df["feature"].tolist()
                        nonsig_features = nonsig_df["feature"].tolist()

                        st.info(f"**Significant features (p < 0.05)**: {len(sig_features)} / {len(mz_values)}")
                        st.info(f"**Non-significant features**: {len(nonsig_features)} / {len(mz_values)}")
                        st.write("✔️ All features will be displayed (significant + non-significant).")

                        # Features to plot
                        features_to_plot = sig_features + nonsig_features

                        # Mapping feature → adjusted p-value
                        significance_dict = dict(zip(result_df.feature, result_df.adj_pvalue))

                        class_colors = st.session_state.get("class_colors", None)

                        # Call plotting function
                        plot_significant_features(
                            data=data_sig,
                            mz_values=features_to_plot,
                            class_colors=class_colors,
                            test=test,
                            plot_type=plot_type.lower().split()[0],
                            show_scatter=show_scatter,
                            use_log2=use_log2,
                            pval_correction=pval_correction,
                            significance_dict=significance_dict
                        )


        st.markdown(
            """
            <h3 style="
                font-size: 1.2rem;
                border-bottom: 2px solid #318CE7;
                text-align: center;
                background-color: #f0f8ff;
                padding: 10px;
                border-radius: 5px;">
                Explainable AI: SHAP & LIME Visualizations
            </h3>
            """,
            unsafe_allow_html=True
        )



        if "show_shap" not in st.session_state:
            st.session_state["show_shap"] = False

        with st.expander("**💡 SHAP Values (Model Explainability)**", expanded=st.session_state["show_shap"]):
            st.markdown('<p style="color: gray; font-size: 14px;">Visualize how each feature contributes to model predictions using SHAP.</p>', unsafe_allow_html=True)

            model_type_for_interpretation = st.selectbox(
                "Select Model Type for Interpretation",
                ['None', 'Machine Learning', 'Deep Learning'],
                key="model_type_for_interpretation_shap",
                help="Choose the type of model you want to interpret using SHAP."
            )

            data_source = st.selectbox(
                "Select Data Source",
                ['Preprocessed', 'Oversampled', 'Undersampled', 'Raw data'],
                key="shap_data_source",
                help="Choose the dataset to use for SHAP value computation."
            )

            # Get appropriate dataset
            data_map = {
                'Preprocessed': st.session_state.get('preprocessed_data'),
                'Oversampled': st.session_state.get('oversampled_data'),
                'Undersampled': st.session_state.get('undersampled_data'),
                'Raw data': st.session_state.get('data'),
            }
            data_sh = data_map.get(data_source)

            if st.button("Show SHAP Values Importance"):
                if data_sh is None:
                    st.warning("Selected data source is not available.")
                    st.stop()

                st.session_state["show_shap"] = True  # Keep expander open
                st.write(f"Selected Model Type: {model_type_for_interpretation}")

                # Load appropriate model
                if model_type_for_interpretation == 'Machine Learning':
                    model_name = st.session_state.get('selected_model')
                    if model_name == 'None' or model_name not in st.session_state.get('models', {}):
                        st.warning("Please select a valid machine learning model.")
                        st.stop()
                    model = st.session_state['models'][model_name]['model']

                elif model_type_for_interpretation == 'Deep Learning':
                    model_name = st.session_state.get('selected_dl_model')
                    if model_name == 'None' or model_name not in st.session_state.get('dl_models', {}):
                        st.warning("Please select a valid deep learning model.")
                        st.stop()
                    model = st.session_state['dl_models'][model_name]['model']

                else:
                    st.warning("Please select a model type.")
                    st.stop()

                # Prepare features and labels
                y = data_sh["Class"]
                X = data_sh.drop(['Class', 'File', 'RT', 'Sum'], axis=1, errors='ignore')

                # Clear cached SHAP values if data/model changed
                if "cached_X" in st.session_state and not X.equals(st.session_state["cached_X"]):
                    st.session_state.pop("cached_shap_values", None)
                    st.session_state.pop("cached_X", None)
                    st.session_state.pop("cached_feature_names", None)

                # Run the full SHAP plotting function
                with st.spinner("Computing SHAP values..."):
                    plot_shap_values(model, X, st.session_state['class_colors'], sorted(y.unique()))

                # Free memory
                del data_sh, X, y
                gc.collect()

        # Expander LIME
        with st.expander("**🕵️ LIME Feature Importance (Model Explainability)**", expanded=st.session_state.get("show_lime", False)):
            st.markdown('<p style="color: gray; font-size: 14px;">🔍 Model-based interpretation using LIME. For binary classification, note that class orientation and top features may vary per sample.</p>', unsafe_allow_html=True)

            model_type_for_interpretation = st.selectbox(
                "Select Model Type for LIME Interpretation",
                ['None', 'Machine Learning', 'Deep Learning'],
                key="model_type_for_interpretation_lime"
            )

            data_source = st.selectbox(
                "Select Data Source for LIME",
                ['Preprocessed', 'Oversampled', 'Undersampled', 'Raw data'],
                key="lime_data_source"
            )

            data_li = None
            if data_source == 'Preprocessed':
                data_li = st.session_state.get('preprocessed_data')
            elif data_source == 'Oversampled':
                data_li = st.session_state.get('oversampled_data')
            elif data_source == 'Undersampled':
                data_li = st.session_state.get('undersampled_data')
            elif data_source == 'Raw data':
                data_li = st.session_state.get('data')

            if data_li is None:
                st.warning("Selected data source is not available.")
            elif st.button("Show LIME Feature Importance"):
                st.session_state["show_lime"] = True

                st.write(f"Selected Model: {st.session_state.get('selected_model')}")
                st.write(f"Selected DL Model: {st.session_state.get('selected_dl_model')}")

                if model_type_for_interpretation == 'None':
                    st.warning("Please select a model type.")
                    st.stop()

                model = None
                if model_type_for_interpretation == 'Machine Learning':
                    model_data = st.session_state['models'].get(st.session_state.get('selected_model'))
                    if model_data:
                        model = model_data['model']
                        label_encoder = model_data['label_encoder']
                    else:
                        st.warning("Selected machine learning model is not available.")
                        st.stop()
                elif model_type_for_interpretation == 'Deep Learning':
                    model_data = st.session_state['dl_models'].get(st.session_state.get('selected_dl_model'))
                    if model_data:
                        model = model_data['model']
                        label_encoder = model_data['label_encoder']
                    else:
                        st.warning("Selected deep learning model is not available.")
                        st.stop()

                with st.spinner("Computing LIME Explanations..."):
                    try:
                        html_contribution, lime_df = eli5_feature_importance(model, label_encoder, data_li)
                        st.components.v1.html(html_contribution, height=600, scrolling=True)
                        import plotly.express as px

                        fig = px.bar(
                            lime_df,
                            x='weight',
                            y='feature',
                            orientation='h',
                            color='weight',
                            color_continuous_scale='RdYlGn',
                            title="Top LIME Features Contributions"
                        )

                        fig.update_layout(
                            yaxis=dict(
                                autorange="reversed",
                                title="Features",
                                titlefont=dict(size=16, color='black'),
                                tickfont=dict(size=14, color='black')
                            ),
                            xaxis=dict(
                                title="Contribution (Weight)",
                                titlefont=dict(size=16, color='black'),
                                tickfont=dict(size=14, color='black')
                            ),
                            title=dict(
                                font=dict(size=20, color='black')
                            ),
                            coloraxis_colorbar=dict(
                                title="Weight",
                                titlefont=dict(size=14, color='black'),
                                tickfont=dict(size=12, color='black')
                            )
                        )

                        st.plotly_chart(fig, use_container_width=True)
                        # st.info("This plot is interpretable only for binary classification")
                    except Exception as e:
                        st.error(f"LIME analysis failed: {e}")

                # Free memory
                del data_li
                gc.collect()


    with tabs[4]:
        st.markdown("""
            <h3 style="font-size: 1.2rem; border-bottom: 2px solid #318CE7; text-align: center;
            background-color: #f0f8ff; padding: 10px; border-radius: 5px;">
            Biological and Molecular Pathway Enrichment
            </h3>""", unsafe_allow_html=True)


        with st.expander("🕸️ **Enrichment Analysis**", expanded=True):
            st.markdown(
                """
                <div style="color: #4A4A4A; font-size: 14px; margin-bottom: 10px;">
                    🧬 <strong>Analyze biological pathways</strong> to identify enriched molecular processes
                    across different gene/protein classes.
                </div>
                """,
                unsafe_allow_html=True
            )

            # ✅ Initialisation
            if 'gene_set_categories' not in st.session_state:
                st.session_state.gene_set_categories = {}
            if 'categories_loaded' not in st.session_state:
                st.session_state.categories_loaded = False
            if 'selected_category' not in st.session_state:
                st.session_state.selected_category = "None"

            # --- Bouton pour charger les catégories et selectbox hors formulaire ---
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("📥 Load databases categories"):
                    with st.spinner("Loading available gene databases categories..."):
                        try:
                            tmp_sets = load_gene_sets()
                            if isinstance(tmp_sets, dict):
                                st.session_state.gene_set_categories = tmp_sets
                                st.session_state.categories_loaded = True
                                st.success("✅ Categories loaded!")
                            else:
                                st.warning("⚠️ Invalid gene set format. Expected a dictionary.")
                        except Exception as e:
                            st.error(f"❌ Failed to load gene sets: {e}")
                            st.session_state.gene_set_categories = {}
                            st.session_state.categories_loaded = False
            with col2:
                if st.session_state.categories_loaded and st.session_state.gene_set_categories:
                    categories = ["None"] + list(st.session_state.gene_set_categories.keys())
                else:
                    categories = ["None"]
                st.session_state.selected_category = st.selectbox("Select a gene/protein specific database in a category", options=categories, index=0)

            # --- FORMULAIRE PRINCIPAL ---
            with st.form("enrichment_form", clear_on_submit=False):
                st.markdown("**⚙️ Configuration**", unsafe_allow_html=True)

                # --- Selectbox pour la database selon la catégorie choisie ---
                if st.session_state.selected_category != "None" and st.session_state.selected_category in st.session_state.gene_set_categories:
                    db_options = ["None"] + st.session_state.gene_set_categories[st.session_state.selected_category]
                else:
                    db_options = ["None"]
                selected_gene_set = st.selectbox("Select a database", options=db_options)

                # --- Autres paramètres ---
                selected_organism = st.selectbox("Select an organism", ["Human", "Mouse", "Rat", "Yeast", "Fly", "Worm", "Fish"])
                num_pathways = st.slider("Number of pathways to display", min_value=1, max_value=100, value=10)

                st.markdown("---")
                st.markdown("**Gene/proteins Classes**", unsafe_allow_html=True)

                if 'num_classes_enrich' not in st.session_state:
                    st.session_state.num_classes_enrich = 1

                gene_lists, class_names = [], []
                for i in range(st.session_state.num_classes_enrich):
                    with st.container():
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            class_name = st.text_input(f"Class name {i + 1}", key=f"class_name_enrich{i}", value=f"Class_{i+1}")
                        with col2:
                            class_genes_input = st.text_area(
                                f"Genes list {i + 1}",
                                placeholder="Enter genes separated by commas, spaces, or new lines",
                                key=f"class_genes_enrich{i}"
                            )
                            class_genes = [g.strip() for g in re.split(r'[,\s]+', class_genes_input) if g.strip()]
                        class_names.append(class_name)
                        gene_lists.append(class_genes)

                # --- Boutons d’ajout/suppression de classes ---
                col_add, col_remove = st.columns(2)
                with col_add:
                    if st.form_submit_button("➕ Add Class"):
                        st.session_state.num_classes_enrich += 1
                        st.rerun()
                with col_remove:
                    if st.form_submit_button("➖ Remove Class"):
                        st.session_state.num_classes_enrich = max(1, st.session_state.num_classes_enrich - 1)
                        st.rerun()

                st.markdown("---")
                run_enrichment = st.form_submit_button("✅ Perform Enrichment")

            # --- Logique après soumission ---
            if run_enrichment:
                if not selected_gene_set or selected_gene_set == "None":
                    st.error("Please select a gene set database before running enrichment.")
                elif any(len(g) == 0 for g in gene_lists):
                    st.error("Each class must contain at least one gene.")
                elif any(not name.strip() for name in class_names):
                    st.error("Each class must have a name.")
                else:
                    with st.spinner("Running GSEA and enrichment analysis..."):
                        # perform_gsea(
                        #     gene_lists=gene_lists,
                        #     class_names=class_names,
                        #     gene_set_db=selected_gene_set,
                        #     organism=selected_organism,
                        #     num_pathways=num_pathways
                        # )
                        perform_gsea(
                            gene_lists,
                            class_names,
                            selected_gene_set,
                            selected_organism,
                            num_pathways
                        )
                    st.success("✅ Enrichment analysis completed successfully!")




    with tabs[5]:

        # Vérifier si les données de survie sont disponibles
        survival_data_available = 'survival_data' in st.session_state
        if survival_data_available:
            survie = st.session_state['survival_data']
        else:
            survie = None
        st.markdown("""
        <h3 style="
            font-size: 1.2rem;
            border-bottom: 2px solid #318CE7;
            text-align: center;
            background-color: #f0f8ff;
            padding: 10px;
            border-radius: 5px;">
            Group comparison
        </h3>
        """, unsafe_allow_html=True)



        # Expander pour l'analyse Kaplan-Meier 
        with st.expander("**⏳ Kaplan-Meier Analysis**", expanded=False):
            st.markdown(
                '<p style="color: gray; font-size: 14px;">Kaplan-Meier survival analysis to assess time to a specific event (death/relapse...).</p>',
                unsafe_allow_html=True,
            )
            st.caption("ℹ️ Requires: 'Overall survival', 'State', and 'Class' columns.")

            if survie is not None:
                if {'Overall survival', 'State', 'Class'}.issubset(survie.columns):
                    if survie['Overall survival'].dtype == 'O':
                        survie['Overall survival'] = survie['Overall survival'].str.replace(',', '.').astype(float)

                    kmf = KaplanMeierFitter()
                    classes = survie['Class'].unique()
                    cmap = plt.cm.get_cmap('tab10', len(classes))

                    # Couleurs
                    if 'class_colors' in st.session_state and st.session_state['class_colors']:
                        class_colors = st.session_state['class_colors']
                    else:
                        class_colors = {cls: cmap(i) for i, cls in enumerate(classes)}

                    if st.button("Run Kaplan-Meier Analysis", help="Generate survival curves and perform log-rank tests."):
                        fig, ax = plt.subplots(figsize=(12, 8))

                        for i, cls in enumerate(classes):
                            group = survie['Class'] == cls
                            kmf.fit(
                                survie['Overall survival'][group],
                                survie['State'][group],
                                label=f'Group {cls}'
                            )
                            color = class_colors.get(cls, cmap(i))
                            kmf.plot_survival_function(
                                ax=ax, show_censors=True, color=color, ci_show=False, linewidth=2
                            )

                        # Log-rank test
                        p_values = {}
                        for i in range(len(classes)):
                            for j in range(i + 1, len(classes)):
                                group_A = survie['Class'] == classes[i]
                                group_B = survie['Class'] == classes[j]
                                results = logrank_test(
                                    durations_A=survie['Overall survival'][group_A],
                                    durations_B=survie['Overall survival'][group_B],
                                    event_observed_A=survie['State'][group_A],
                                    event_observed_B=survie['State'][group_B]
                                )
                                p_values[f'{classes[i]} vs {classes[j]}'] = results.p_value

                        # Titres et axes stylisés
                        ax.set_title("Kaplan-Meier Survival Curves", fontsize=20, weight='bold', color='black')
                        ax.set_xlabel("Survival Time", fontsize=16, weight='bold', color='black')
                        ax.set_ylabel("Survival Probability", fontsize=16, weight='bold', color='black')

                        # Légende
                        ax.legend(title='Group', fontsize=14, title_fontsize=16)
                        ax.tick_params(axis='both', which='major', labelsize=14)
                        ax.set_xticklabels(ax.get_xticks(), fontsize=14, weight='bold', color='black')
                        ax.set_yticklabels(ax.get_yticks(), fontsize=14, weight='bold', color='black')

                        # Format propre des échelles (ticks)
                        import matplotlib.ticker as ticker
                        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.1f}'))
                        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:.2f}'))


                        st.pyplot(fig)

                        # P-values
                        st.subheader("Log-rank Test P-values")
                        for key, value in p_values.items():
                            st.markdown(f"**{key}**: {value:.4f}")
                else:
                    st.error("The survival data must contain 'Overall survival', 'State', and 'Class' columns.")
            else:
                st.warning("Please upload survival data to perform Kaplan-Meier analysis.")





        # Expander for Cox Model Analysis
        st.markdown("""
        <h3 style="
            font-size: 1.2rem;
            border-bottom: 2px solid #318CE7;
            text-align: center;
            background-color: #f0f8ff;
            padding: 10px;
            border-radius: 5px;">
            Multivariate Regression
        </h3>
        """, unsafe_allow_html=True)
        # Expander for Cox Model Analysis
        with st.expander("**🔢 Cox Model Analysis**", expanded=False):
            st.markdown('<p style="color: gray; font-size: 14px;">Cox model to analyze the impact of covariates on survival.</p>', unsafe_allow_html=True)
            st.caption("ℹ️ Requires: 'Overall survival', 'State', and covariates such as age, BMI, markers... (numeric or categorical).")

            if survie is not None:
                if 'Class' not in survie.columns:
                    st.error("The uploaded file must contain the column 'Class' for Cox Model analysis.")
                else:
                    pipeline = create_cox_pipeline(survie)
                    cph = None  # Initialize cph to None

                    if st.button("Run Cox Model Analysis", help="Fit a Cox model with regularization and display results"):
                        # Fit the pipeline to the data
                        pipeline.fit(survie)

                        # Retrieve the transformed columns
                        num_columns = pipeline.named_steps['preprocessor'].transformers_[0][2]  # Unchanged numeric columns

                        # Get the column names after OneHotEncoding
                        ohe = pipeline.named_steps['preprocessor'].transformers_[1][1]  # Instance of the OneHotEncoder
                        cat_columns = ohe.get_feature_names_out(input_features=pipeline.named_steps['preprocessor'].transformers_[1][2])

                        # Create the final list of columns
                        columns_after_transformation = list(num_columns) + list(cat_columns)

                        # Transform the data
                        X_transformed = pipeline.transform(survie)
                        X_transformed_df = pd.DataFrame(X_transformed, columns=columns_after_transformation)

                        X_transformed_df.fillna(X_transformed_df.median(), inplace=True)  # Handle NaN values

                        # Add 'Overall survival' and 'State' columns
                        X_transformed_df[['Overall survival', 'State']] = survie[['Overall survival', 'State']]

                        X_numeric = X_transformed_df.drop(columns=['Overall survival', 'State'])  # Exclude survival columns

                        # Skip collinearity detection if only one covariate is present
                        if X_numeric.shape[1] > 1:
                            collinearity_detected = detect_collinearity(X_numeric)
                        else:
                            collinearity_detected = False  # Cannot have collinearity with one covariate

                        # Apply penalty only if collinearity is detected
                        penalizer_value = 0.1 if collinearity_detected else 0.0
                        # Create and train the Cox model
                        cph = CoxPHFitter(penalizer=penalizer_value)
                        cph.fit(X_transformed_df, duration_col='Overall survival', event_col='State')
                        st.session_state.cph = cph

                        st.markdown("**Cox Model Summary**")
                        st.write(cph.summary)

                        st.write(f"Concordance = {cph.concordance_index_}")

                        # Adjust figure size dynamically based on the number of variables
                        num_vars = len(X_numeric.columns)
                        fig_height = max(6, num_vars * 0.3)  # Adjust height based on the number of variables

                        plt.figure(figsize=(10, fig_height))
                        ax = cph.plot()

                        # Improve visibility of labels and title
                        plt.title("Forest Plot of Cox Model Coefficients", fontsize=14, fontweight='bold')
                        plt.xticks(fontsize=12)
                        plt.yticks(fontsize=12)
                        plt.grid(True, linestyle='--', alpha=0.6)

                        st.pyplot(plt)
                        plt.close()

                    model_name = st.text_input("Enter model name:", value="cox_model", help="Custom name for saving your Cox model")

                    if st.button("Save Cox Model", help="Save the trained Cox model and its preprocessing pipeline for blind predictions"):
                        if 'cph' in st.session_state:
                            pipeline.fit(survie)

                            # Save the model to a bytes buffer
                            model_buffer = io.BytesIO()
                            joblib.dump(st.session_state.cph, model_buffer)
                            model_buffer.seek(0)
                            st.session_state.model_buffer = model_buffer

                            # Save the pipeline to a bytes buffer
                            pipeline_buffer = io.BytesIO()
                            joblib.dump(pipeline, pipeline_buffer)
                            pipeline_buffer.seek(0)
                            st.session_state.pipeline_buffer = pipeline_buffer

                            # Set a flag to show the download buttons
                            st.session_state.show_downloads = True
                        else:
                            st.warning("Please run the Cox Model analysis before saving.")

                    # Show download buttons if ready
                    if st.session_state.get("show_downloads", False):
                        st.download_button(
                            label="Download Cox Model",
                            data=st.session_state.model_buffer,
                            file_name=f"{model_name}.pkl",
                            mime="application/octet-stream"
                        )
                        st.download_button(
                            label="Download Preprocessor Pipeline",
                            data=st.session_state.pipeline_buffer,
                            file_name="preprocessor_pipeline.pkl",
                            mime="application/octet-stream"
                        )                        
            else:
                st.warning("Please upload survival data to perform Cox Model analysis.")


        with st.expander("**⏱️ Import test data and Make Predictions**", expanded=False):
            st.markdown('<p style="color: gray; font-size: 14px;">Import a CSV or Excel file and use a Saved Cox model to make predictions.</p>', unsafe_allow_html=True)

            uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])
            model_file = st.file_uploader("Upload your pre-saved Cox model (.pkl)", type=["pkl"])
            pipeline_file = st.file_uploader("Upload your preprocessor pipeline (.pkl)", type=["pkl"])

            if uploaded_file and model_file and pipeline_file:
                # Load the CSV or Excel file
                if uploaded_file.name.endswith('.csv'):
                    data_pred = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    data_pred = pd.read_excel(uploaded_file)

                # Load the pre-saved Cox model and pipeline
                cph = joblib.load(model_file)
                pipeline = joblib.load(pipeline_file)

                try:
                    if not hasattr(pipeline, "transform"):
                        raise NotFittedError("The pipeline has not been fitted.")

                    # Transform the data using the pipeline
                    X_transformed = pipeline.transform(data_pred)
                    X_transformed_df = pd.DataFrame(
                        X_transformed,
                        columns=pipeline.named_steps['preprocessor'].transformers_[0][2] +
                                list(pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out())
                    )

                    if st.button("Make Predictions", key="make_predictions_test_data"):
                        # Predict key survival metrics
                        median_survival = cph.predict_median(X_transformed_df)
                        overall_survival = cph.predict_expectation(X_transformed_df)  # Expected overall survival
                        hazard_ratios = cph.predict_partial_hazard(X_transformed_df)

                        results_df = data_pred.copy()
                        results_df["Predicted Overall Survival"] = overall_survival
                        results_df["Median Survival Time"] = median_survival
                        results_df["Hazard Ratio"] = hazard_ratios
                        # results_df["P-Value"] = p_values
                        st.success('Survival prediction successful')
                        # Display results
                        st.write("#### Survival Predictions Summary")
                        st.dataframe(results_df)
                        st.write("#### Descriptive statistics")
                        st.write(results_df[["Predicted Overall Survival", "Hazard Ratio"]].describe())
                        # Visualizations
                        st.write("#### Visualizations")

                        plt.figure(figsize=(10, 6))
                        sns.histplot(median_survival, kde=True)
                        plt.title('Distribution of Median Survival Time', fontsize=16, fontweight='bold')
                        plt.xlabel('Median Survival Time', fontsize=14, fontweight='bold')
                        plt.ylabel('Frequency', fontsize=14, fontweight='bold')

                        # Show the plot
                        st.pyplot(plt)

                        # Plot for Hazard Ratios
                        plt.figure(figsize=(10, 6))
                        sns.histplot(hazard_ratios, kde=True)
                        plt.title('Distribution of Hazard Ratios', fontsize=16, fontweight='bold')
                        plt.xlabel('Hazard Ratio', fontsize=14, fontweight='bold')
                        plt.ylabel('Frequency', fontsize=14, fontweight='bold')

                        # Show the plot
                        st.pyplot(plt)                            
                        # st.write(results_df.describe())
                        


                except NotFittedError:
                    st.error("The uploaded pipeline has not been fitted. Please ensure it has been fitted before using it for predictions.")


    with tabs[6]:
  
        st.markdown(
            """
            <h3 style="
                font-size: 1.2rem;
                border-bottom: 2px solid #318CE7;
                text-align: center;
                background-color: #f0f8ff;
                padding: 10px;
                border-radius: 5px;">
                Real-Time Predictions
            </h3>
            """,
            unsafe_allow_html=True
        )


        with st.expander("**🔴 Real-Time and Post-Acquisition**", expanded=False):
            watch_directory = st.text_input("Directory to monitor:")
            output_directory = st.text_input("Output directory for mzML files:")

            model_files = st.file_uploader("Upload model files", accept_multiple_files=True, type=["pkl"])
            feature_file = st.file_uploader("Upload feature file", type=["pkl"])
            label_encoder_file = st.file_uploader("Upload label encoder file", type=["pkl"])

            if st.button("Load All"):
                load_all_rt(model_files, feature_file, label_encoder_file)

            if st.session_state['label_encoder']:
                labels = st.session_state['label_encoder'].classes_
                for label in labels:
                    st.session_state['label_colors'][label] = st.color_picker(f"Color for {label}", "#FFFFFF")

            if st.button("Start Monitoring Directory"):
                if not st.session_state['numeric_features']:
                    st.warning("Please load the numeric features before starting monitoring.")
                else:
                    st.session_state['monitoring'] = True
                    st.write("Started monitoring...")

            if st.button("Stop Monitoring"):
                st.session_state['monitoring'] = False
                st.write("Stopped monitoring.")

            if st.session_state['monitoring'] and watch_directory and output_directory:
                new_files = convert_raw_to_mzml_rt(watch_directory, output_directory)
                if new_files:
                    mzml_files = [f for f in os.listdir(output_directory) if f.endswith(".mzML")]
                    for mzml_file in mzml_files:
                        if mzml_file in st.session_state['processed_files']:
                            continue
                        data = load_data_single_file_rt(os.path.join(output_directory, mzml_file))
                        if not data.empty:
                            processed_data = preprocess_data_rt(data, normalization_type=normalization_type)
                            # 🔹 Removed SVD choice — process directly
                            st.session_state['latest_result'] = decision_rt(processed_data)
                        st.session_state['processed_files'].add(mzml_file)

                if st.session_state['latest_result'] is not None:
                    st.write("Latest Prediction Results:")
                    st.dataframe(st.session_state['latest_result'])
                    visualize_predictions_circles_rt(st.session_state['latest_result'])

                time.sleep(2)
                st.rerun()




        st.markdown(
            """
            <h3 style="
                font-size: 1.2rem;
                border-bottom: 2px solid #318CE7;
                text-align: center;
                background-color: #f0f8ff;
                padding: 10px;
                border-radius: 5px;">
                Post-hoc Predictions
            </h3>
            """,
            unsafe_allow_html=True
        )



        with st.expander("**🖥️ Using tabular data**", expanded=False):

            uploaded_file = st.file_uploader("Upload CSV/XLSX File", type=["csv", "xlsx"], help="Upload your tabular data file. It can be labeled (with a 'Class' column) or unlabeled.")
            model_file = st.file_uploader("Upload Trained Model (Pickle)", type=["pkl"], help="Upload the trained classification model saved as a .pkl file.")
            feature_file = st.file_uploader("Upload Feature Names (Pickle)", type=["pkl"], help="Upload the file containing the ordered list of features used to train the model as a .pkl files.")
            label_encoder_file = st.file_uploader("Upload Label Encoder (Pickle)", type=["pkl"],help="Upload the label encoder as .pkl format used during training to map class labels.")

            col1, col2 = st.columns(2)
            with col1:
                predict_with_gt = st.button("Predict with Ground Truth",help="Use this if your file includes a 'Class' column to evaluate predictions with true labels.")
            with col2:
                predict_without_gt = st.button("Predict without Ground Truth",help="Use this if your file does not have true labels and you only want predictions.")

            def run_prediction(has_ground_truth):
                import pandas as pd
                import numpy as np
                import joblib
                from sklearn.metrics import confusion_matrix, classification_report
                import matplotlib.pyplot as plt
                import seaborn as sns

                if uploaded_file and model_file and feature_file and label_encoder_file:
                    # Load file
                    file_extension = uploaded_file.name.split(".")[-1].lower()
                    df = pd.read_csv(uploaded_file) if file_extension == "csv" else pd.read_excel(uploaded_file)


                    # Load components
                    model_data = joblib.load(model_file)
                    if isinstance(model_data, dict) and 'model' in model_data:
                        model = model_data['model']  # ✅ extrait le vrai pipeline
                    else:
                        model = model_data  # ancien format (directement un pipeline)

                    feature_names = joblib.load(feature_file)
                    label_encoder = joblib.load(label_encoder_file)

                    if has_ground_truth and "Class" not in df.columns:
                        st.error("Missing 'Class' column in file.")
                        return

                    # Separate features
                    X = df.drop(columns=["Class"]) if has_ground_truth else df.copy()

                    # Check features
                    missing = set(feature_names) - set(X.columns)
                    extra = set(X.columns) - set(feature_names)

                    # Ajouter les features manquants avec valeur 0
                    if missing:
                        st.warning(f"Missing features detected and filled with 0: {missing}")
                        for col in missing:
                            X[col] = 0

                    # Supprimer les colonnes en trop
                    if extra:
                        st.warning(f"Extra features will be ignored: {extra}")
                        X = X.drop(columns=list(extra))

                    # Réordonner les colonnes pour correspondre au modèle
                    X = X[feature_names]

                    # Prédictions
                    predictions = model.predict(X)  # supposé aligné avec les lignes de df
                    predicted_labels = label_encoder.inverse_transform(predictions)
                    df.insert(0, "Predicted_Class", predicted_labels)

                    # Affichage général
                    st.write("#### Prediction Results")
                    st.dataframe(df)

                    if has_ground_truth:
                        # --- Gestion des labels ground-truth inconnus ---
                        gt_raw = df["Class"]
                        # Nettoyage simple (supprime espaces en tête/queue)
                        gt_clean_str = gt_raw.astype(str).str.strip()

                        # Map stringified known classes vers leur valeur d'origine
                        classes_map = {str(c): c for c in label_encoder.classes_}
                        unseen_str = set(gt_clean_str.unique()) - set(classes_map.keys())

                        if unseen_str:
                            st.warning(f"Found unseen ground-truth labels (these rows will be excluded from metrics): {unseen_str}")

                        # mask des échantillons utilisables pour l'évaluation
                        mask_eval = ~gt_clean_str.isin(unseen_str)

                        if mask_eval.sum() == 0:
                            st.error("None of the ground-truth labels match the label encoder classes. Cannot compute evaluation metrics.")
                        else:
                            # Remapper les valeurs nettoyées vers leurs valeurs d'origine (types originaux)
                            gt_mapped = gt_clean_str.map(classes_map)

                            # Encodage pour la métrique
                            true_encoded = label_encoder.transform(gt_mapped[mask_eval])

                            # Récupérer les prédictions correspondantes (predictions est un ndarray aligné avec df)
                            pred_for_eval = predictions[mask_eval.values]

                            # Matrice de confusion et rapport
                            cm = confusion_matrix(true_encoded, pred_for_eval)
                            report = classification_report(
                                true_encoded,
                                pred_for_eval,
                                labels=list(range(len(label_encoder.classes_))),
                                target_names=list(label_encoder.classes_),
                                output_dict=True
                            )

                            # Construire la colonne Correct et Eval_included
                            correct = np.zeros(len(df), dtype=bool)
                            correct_pos = (true_encoded == pred_for_eval)
                            correct[mask_eval.values] = correct_pos
                            df["Correct"] = correct
                            df["Eval_included"] = mask_eval.values

                            misclassified = df[mask_eval & (~df["Correct"])]

                            # Plot confusion matrix
                            fig, ax = plt.subplots(figsize=(6, 4))
                            sns.heatmap(cm, annot=True, fmt="d", cmap="jet",
                                        xticklabels=label_encoder.classes_,
                                        yticklabels=label_encoder.classes_, ax=ax)
                            plt.xlabel("Predicted")
                            plt.ylabel("Actual")
                            plt.title("Confusion Matrix")
                            st.pyplot(fig)

                            st.markdown("#### Classification Report")
                            st.dataframe(pd.DataFrame(report).T)

                            st.write("#### Misclassified Samples (only among evaluated rows)")
                            st.dataframe(misclassified)

                    # Export
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")
                else:
                    st.error("Please upload all required files (data, model, feature names, and label encoder).")

            # Trigger logic
            if predict_with_gt:
                run_prediction(has_ground_truth=True)

            if predict_without_gt:
                run_prediction(has_ground_truth=False)

if __name__ == "__main__":
    import matplotlib
    # matplotlib.use('TkAgg')
    matplotlib.use('Agg')
    main()

