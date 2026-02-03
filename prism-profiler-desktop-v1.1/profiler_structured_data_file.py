
"""
Software Name: Profiler
Module name : impots dependences
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
import re
import numpy as np
import pandas as pd
import streamlit as st
import chardet

def detect_encoding(uploaded_file, sample_size=5000):
    rawdata = uploaded_file.read(sample_size)
    uploaded_file.seek(0)
    result = chardet.detect(rawdata)
    return result['encoding'] if result['encoding'] else 'utf-8'

def infer_dtypes(sample_df):
    dtypes = {}
    for col in sample_df.columns:
        values = sample_df[col].dropna().astype(str).str.replace(',', '.', regex=False)
        numeric = pd.to_numeric(values, errors='coerce')
        dtypes[col] = 'float32' if numeric.notna().mean() > 0.8 else 'str'
    return dtypes

# def load_structured_data(uploaded_file):
#     try:
#         file_name = uploaded_file.name.lower()

#         if file_name.endswith(('.csv', '.tsv', '.txt')):
#             # Forcer le séparateur `;` et l'encodage `utf-8-sig`
#             df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8-sig', on_bad_lines='skip')

#             # Nettoyer les noms de colonnes (supprimer les caractères invisibles)
#             df.columns = df.columns.str.replace(r'[^\x00-\x7F]+', '', regex=True).str.strip('"')

#             return df if not df.empty else None

#         elif file_name.endswith(('.xls', '.xlsx')):
#             df = pd.read_excel(uploaded_file, engine='openpyxl')
#             return df if not df.empty else None

#         else:
#             st.error("file fomat not supported.")
#             return None

#     except Exception as e:
#         st.error(f"Erreur : {e}")
#         return None


def load_structured_data(uploaded_file):
    try:
        file_name = uploaded_file.name.lower()

        if file_name.endswith(('.csv', '.tsv', '.txt')):
            # Lire les premières lignes pour détecter le séparateur
            uploaded_file.seek(0)
            first_line = uploaded_file.readline().decode('utf-8-sig')
            uploaded_file.seek(0)

            # Détecter le séparateur (virgule, point-virgule ou tabulation)
            if ';' in first_line:
                sep = ';'
            elif ',' in first_line:
                sep = ','
            else:
                sep = '\t'

            # Lire le fichier avec le séparateur détecté
            df = pd.read_csv(uploaded_file, sep=sep, encoding='utf-8-sig', on_bad_lines='skip')

            # Nettoyer les noms de colonnes (supprimer caractères invisibles et guillemets)
            df.columns = df.columns.str.replace(r'[^\x20-\x7E]+', '', regex=True).str.strip().str.strip('"')

            return df if not df.empty else None

        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            return df if not df.empty else None

        else:
            st.error("Format de fichier non supporté.")
            return None

    except Exception as e:
        st.error(f"Erreur : {e}")
        return None



def update_class_names(data, class_renaming):
    data['Class'] = data['Class'].map(class_renaming)
    return data


def get_data(data_source):
    """Efficiently fetch data from session state without unnecessary copies."""
    data_key_map = {
        "Raw Data": "data",
        "Preprocessed": "preprocessed_data",
        "Oversampled": "oversampled_data",
        "Undersampled": "undersampled_data"
    }
    data_key = data_key_map.get(data_source)

    if not data_key or data_key not in st.session_state:
        return None

    df = st.session_state[data_key]
    if df is None or df.empty:
        return None

    # Ne pas copier inutilement
    if "class_renaming" in st.session_state and "Class" in df.columns:
        renaming_dict = st.session_state["class_renaming"]
        # Vérifie si le remplacement est vraiment nécessaire
        if any(c in renaming_dict for c in df["Class"].unique()):
            df = df.copy()
            df["Class"] = df["Class"].replace(renaming_dict)

    return df


def cluster_index_to_letter(index):
    return chr(ord('A') + index)





def load_data_safely(file, sep=';', encodings=['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']):
    name = file.name
    if name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file)
    
    for enc in encodings:
        for s in [sep, '\t']:
            file.seek(0)
            try:
                return pd.read_csv(file, sep=s, encoding=enc, engine='python')
            except Exception:
                continue
    raise ValueError(f"Unable to read file {name}. Tried encodings: {encodings}")

def perseus_data(file, rename_mapping=None, feature_row_index=-2):
    try:
        df = load_data_safely(file)

        df.drop(columns=[
            "C: Only identified by site", "N: Razor + unique peptides", "C: Reverse",
            "C: Potential contaminant", "N: Peptides", "N: Unique peptides",
            "N: Sequence coverage [%]", "N: Unique sequence coverage [%]",
            "N: Mol. weight [kDa]", "N: Q-value", "N: Score", "N: Intensity",
            "N: MS/MS count", "T: Majority protein IDs", "T: id",
            "N: Unique + razor sequence coverage [%]", "T: First.Protein.Description",
            "T: Protein.Ids", "T: Protein.Group", "T: Protein IDs"
        ], axis=1, errors='ignore', inplace=True)

        df = df.T
        df.columns = df.iloc[feature_row_index].values
        df = df.drop(df.index[-2:]).reset_index().rename(columns={"index": "Class"})
        df.iloc[:, 1:] = df.iloc[:, 1:].replace(',', '.', regex=True)

        if rename_mapping:
            df.replace(rename_mapping, inplace=True)

        df = df.loc[:, ~df.columns.duplicated()]
        return df

    except Exception as e:
        raise ValueError(f"Error reading file: {e}")


def diann_data(file, rename_mapping=None, feature_row_index=2):
    try:
        df = load_data_safely(file)

        df.drop(columns=[
            "Protein.Group", "Protein.Ids", "First.Protein.Description",
            "Accession", "Description", "T: Protein IDs", "N.Sequences", "N.Proteotypic.Sequences"
        ], axis=1, errors='ignore', inplace=True)

        df = df.T
        df.columns = df.iloc[feature_row_index].values
        df = df.drop(df.index[:2]).reset_index().rename(columns={"index": "Class"})
        df.iloc[:, 1:] = df.iloc[:, 1:].replace(',', '.', regex=True)

        if rename_mapping:
            df.replace(rename_mapping, inplace=True)

        df = df.loc[:, ~df.columns.duplicated()]
        return df

    except Exception as e:
        raise ValueError(f"Error reading file: {e}")


def maxquant_data(file, rename_mapping=None, feature_row_index=0):
    try:
        df = load_data_safely(file)


        keep_cols = [
            col for col in df.columns
            if (col.startswith("LFQ") and not re.search(r'_\d{12}$', col))
            or col in ["T: Gene names", "T: Protein names", "Gene names", "Protein names"]
        ]

        df = df[keep_cols].T

        df.columns = df.iloc[feature_row_index].values
        df = df.drop(df.index[:2]).reset_index().rename(columns={"index": "Class"})
        df.iloc[:, 1:] = df.iloc[:, 1:].replace(',', '.', regex=True)

        if rename_mapping:
            df.replace(rename_mapping, inplace=True)

        df = df.loc[:, ~df.columns.duplicated()]
        return df

    except Exception as e:
        raise ValueError(f"Error reading file: {e}")
