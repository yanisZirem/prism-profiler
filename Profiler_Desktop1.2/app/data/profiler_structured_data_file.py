"""
profiler_structured_data_file.py
Handles tabular data loading with:
 - Smart CSV delimiter detection (,  ;  tab  |) with user confirmation
 - Column alias normalisation: Class / class / target / condition / label → Class
 - Auto-creation of ID column if absent
 - Support for _meta columns (clinical variables)
 - DIA-NN, MaxQuant, Perseus parsers
"""

import io
import os
import ntpath
import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from typing import Optional

# ─── Column alias constants ────────────────────────────────────────────────────
CLASS_ALIASES = {"class", "target", "condition", "label", "group", "outcome",
                 "classe", "categorie", "category"}
ID_ALIASES    = {"id", "sample_id", "sampleid", "sample", "name",
                 "patient", "patient_id", "subject", "subject_id"}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename common class/ID alias columns to canonical names.
    Also keeps _meta columns intact.
    """
    renames = {}
    for col in df.columns:
        col_lower = col.strip().lower()
        # Class aliases
        if col_lower in CLASS_ALIASES and "Class" not in df.columns:
            renames[col] = "Class"
        # ID aliases
        elif col_lower in ID_ALIASES and "ID" not in df.columns:
            renames[col] = "ID"
    if renames:
        df = df.rename(columns=renames)
    return df


def _ensure_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add an ID column filled with 'Unknown' if none exists."""
    if "ID" not in df.columns:
        df.insert(0, "ID", [f"Sample_{i+1}" for i in range(len(df))])
    else:
        df["ID"] = df["ID"].fillna("Unknown").astype(str)
    return df


def _detect_delimiter(raw_bytes: bytes) -> str:
    """Detect the most likely CSV delimiter from raw bytes."""
    try:
        first_lines = raw_bytes.decode("utf-8-sig", errors="ignore").split("\n")[:5]
        first_block = "\n".join(first_lines)
    except Exception:
        return ","

    counts = {sep: first_block.count(sep) for sep in [",", ";", "\t", "|"]}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def _clean_col_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.astype(str)
        .str.replace(r"[^\x20-\x7E]+", "", regex=True)
        .str.strip()
        .str.strip('"')
    )
    return df


def load_structured_data(uploaded_file, *, _sep_override=None):
    """
    Smart tabular loader with:
      - auto separator detection for CSV/TSV/TXT
      - column alias normalisation (class, target, condition → Class; id → ID)
      - auto ID column creation
      - _meta column preservation

    Returns (df, detected_sep) — df may be None on failure.
    Returns just df when called from legacy code (backward-compat).
    """
    try:
        file_name = uploaded_file.name.lower()
        uploaded_file.seek(0)
        raw = uploaded_file.read()
        uploaded_file.seek(0)

        if file_name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file, engine="openpyxl")
            df = _clean_col_names(df)
            df = _normalise_columns(df)
            df = _ensure_id_column(df)
            return df if not df.empty else None

        # CSV / TSV / TXT
        sep = _sep_override or _detect_delimiter(raw)
        text = raw.decode("utf-8-sig", errors="ignore")

        # Strip comment / metadata lines
        lines = [l for l in text.splitlines()
                 if l.strip() and not l.startswith(("#", "##", "sep="))]

        df = pd.read_csv(
            io.StringIO("\n".join(lines)),
            sep=sep, engine="python", on_bad_lines="skip"
        )

        if df.shape[1] <= 1 and sep != "\t":
            # Fallback: try tab
            df = pd.read_csv(
                io.StringIO("\n".join(lines)),
                sep="\t", engine="python", on_bad_lines="skip"
            )

        df = _clean_col_names(df)
        df = df.dropna(axis=1, how="all")
        df = _normalise_columns(df)
        df = _ensure_id_column(df)

        return (df, sep) if not df.empty else (None, sep)

    except Exception as e:
        st.error(f"Loading error: {e}")
        return None, ","


def render_tabular_loader(uploaded_file, finalize_fn):
    """
    Interactive Streamlit UI for tabular loading — shows detected delimiter,
    lets user confirm or override, handles class/ID column mapping, _meta columns.
    Called from Profiler.py sidebar.
    """
    file_name = uploaded_file.name.lower()
    _file_id  = f"{uploaded_file.name}_{uploaded_file.size}"

    if st.session_state.get("_last_tabular_file_id") == _file_id:
        return  # already processed

    # ── Delimiter detection (CSV only) ────────────────────────────────────────
    if not file_name.endswith((".xls", ".xlsx")):
        uploaded_file.seek(0)
        raw = uploaded_file.read()
        uploaded_file.seek(0)
        auto_sep = _detect_delimiter(raw)

        sep_labels = {",": "Comma  (,)", ";": "Semicolon  (;)",
                      "\t": "Tab  (\\t)", "|": "Pipe  (|)"}
        sep_options = list(sep_labels.keys())

        st.info(f"🔍 Detected delimiter: **{sep_labels.get(auto_sep, auto_sep)}**")
        chosen_sep_label = st.selectbox(
            "Confirm or change delimiter:",
            options=[sep_labels[s] for s in sep_options],
            index=sep_options.index(auto_sep),
            key="tabular_sep_choice"
        )
        sep = sep_options[[sep_labels[s] for s in sep_options].index(chosen_sep_label)]
    else:
        sep = None  # Excel

    # ── Load ─────────────────────────────────────────────────────────────────
    result = load_structured_data(uploaded_file, _sep_override=sep)
    if isinstance(result, tuple):
        df, _ = result
    else:
        df = result

    if df is None:
        st.error("❌ Could not parse file. Please check the format and delimiter.")
        return

    # ── Check for Class column ────────────────────────────────────────────────
    if "Class" not in df.columns:
        st.warning("⚠️ No class/target column detected automatically.")

        # Try to let user map a column
        all_cols = df.columns.tolist()
        class_col = st.selectbox(
            "Which column contains your class / target labels?",
            options=["— none / not applicable —"] + all_cols,
            key="tabular_class_col"
        )
        if class_col != "— none / not applicable —":
            df = df.rename(columns={class_col: "Class"})
        else:
            # Propose specialised parsers
            ftype = st.selectbox(
                "Or select the file format to parse automatically:",
                ["Choose…", "DIA-NN", "Perseus", "MaxQuant"],
                key="tabular_filetype"
            )
            if ftype == "Perseus":
                feat_opts = ["T: Gene names", "T: Protein names"]
                sel = st.selectbox("Feature row:", feat_opts, key="tabular_feat_row")
                fri = -2 if sel == "T: Gene names" else -1
                with st.spinner("Processing Perseus…"):
                    df = perseus_data(uploaded_file, feature_row_index=fri)
                    finalize_fn(df, "Perseus")
                    st.session_state["_last_tabular_file_id"] = _file_id
                return
            elif ftype == "DIA-NN":
                feat_opts = ["Genes", "Protein.Names"]
                sel = st.selectbox("Feature row:", feat_opts, key="tabular_feat_row")
                fri = 1 if sel == "Genes" else 0
                with st.spinner("Processing DIA-NN…"):
                    df = diann_data(uploaded_file, feature_row_index=fri)
                    finalize_fn(df, "DIA-NN")
                    st.session_state["_last_tabular_file_id"] = _file_id
                return
            elif ftype == "MaxQuant":
                feat_opts = ["Gene names", "Protein names"]
                sel = st.selectbox("Feature row:", feat_opts, key="tabular_feat_row")
                fri = -1 if sel == "Gene names" else -2
                with st.spinner("Processing MaxQuant…"):
                    df = maxquant_data(uploaded_file, feature_row_index=fri)
                    finalize_fn(df, "MaxQuant")
                    st.session_state["_last_tabular_file_id"] = _file_id
                return
            else:
                return  # wait for user choice

    # ── _meta columns info ────────────────────────────────────────────────────
    meta_cols = [c for c in df.columns if str(c).endswith("_meta")]
    system_cols = {"Class", "ID", "File", "RT", "Sum"}
    feature_cols = [c for c in df.columns if c not in system_cols and not str(c).endswith("_meta")]

    if meta_cols:
        with st.expander(f"ℹ️ {len(meta_cols)} clinical/_meta column(s) detected", expanded=False):
            st.write(meta_cols)
            st.caption(
                "These columns (ending with `_meta`) are recognised as clinical metadata. "
                "They can be used as alternative targets for classification/regression, "
                "or as colour variables in PCA/UMAP/heatmap. They are preserved but not used as features by default."
            )

    # ── Preview + confirm ─────────────────────────────────────────────────────
    with st.expander("📋 Preview (first 5 rows)", expanded=False):
        st.dataframe(df.head(5), use_container_width=True)
        st.caption(
            f"Shape: **{df.shape[0]}** samples × **{len(feature_cols)}** features "
            f"| Class: **{'Class' in df.columns}** "
            f"| ID: **{'ID' in df.columns}** "
            f"| Meta columns: **{len(meta_cols)}**"
        )

    if st.button("✅ Confirm & Load", key="tabular_confirm"):
        finalize_fn(df, "Generic")
        st.session_state["_last_tabular_file_id"] = _file_id
        st.success("✅ Data loaded successfully!")


# ─── Legacy function kept for backward compat ──────────────────────────────────
def load_structured_data_simple(uploaded_file):
    """Old-style loader — returns df or None."""
    result = load_structured_data(uploaded_file)
    return result[0] if isinstance(result, tuple) else result


# ─── Helpers ───────────────────────────────────────────────────────────────────
def update_class_names(data, class_renaming):
    data["Class"] = data["Class"].map(class_renaming)
    return data


def get_data(data_source):
    """Fetch data from session state by source name."""
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
    if "class_renaming" in st.session_state and "Class" in df.columns:
        renaming_dict = st.session_state["class_renaming"]
        if any(c in renaming_dict for c in df["Class"].unique()):
            df = df.copy()
            df["Class"] = df["Class"].replace(renaming_dict)
    return df


def cluster_index_to_letter(index):
    return chr(ord("A") + index)


def load_data_safely(file, sep=None, encodings=None):
    """
    Robust file reader that auto-detects delimiter and encoding.
    Works with Streamlit UploadedFile, plain file objects, and BytesIO.
    sep=None  → auto-detect from first 8 KB.
    sep given → try that separator first, then fallback to auto-detect.
    """
    if encodings is None:
        encodings = ["utf-8", "utf-8-sig", "latin1", "ISO-8859-1", "cp1252"]
    name = getattr(file, "name", "")

    if name.endswith((".xls", ".xlsx")):
        file.seek(0)
        return pd.read_excel(file)

    # ── Read all bytes once → wrap in BytesIO for repeatable reads ──────────
    file.seek(0)
    raw_bytes = file.read()
    file.seek(0)

    # Auto-detect separator from first 8 KB
    auto_sep = _detect_delimiter(raw_bytes[:8192])
    seps_to_try = list(dict.fromkeys([s for s in [sep, auto_sep, "\t", ",", ";"] if s]))

    for enc in encodings:
        for s in seps_to_try:
            try:
                buf = io.BytesIO(raw_bytes)
                df = pd.read_csv(buf, sep=s, encoding=enc,
                                 engine="python", on_bad_lines="skip")
                if df.shape[1] > 1:   # meaningful parse
                    return df
            except Exception:
                continue
    raise ValueError(f"Unable to read file {name}.")


# ─── Shared helpers ────────────────────────────────────────────────────────────

def _make_unique_labels(series: pd.Series) -> list:
    """
    Given a Series of feature labels, return a list where duplicate values
    are made unique by appending __2, __3, … suffixes.
    Empty / NaN entries become 'unknown'.
    """
    seen: dict = {}
    result = []
    for v in series.fillna("").astype(str):
        v = v.strip() or "unknown"
        if v in seen:
            seen[v] += 1
            result.append(f"{v}__{seen[v]}")
        else:
            seen[v] = 0
            result.append(v)
    return result


def _shorten_sample_name(path_str: str) -> str:
    """
    Extract a short sample name from a full file path (Windows or Unix).
    Strips directory, extension, and common suffixes like '_run-NNNN'.
    """

    p = str(path_str).strip()
    # ntpath.basename handles both Windows and POSIX separators
    base = ntpath.basename(p) or os.path.basename(p)
    base = os.path.splitext(base)[0]      # remove .d / .raw / .mzML …
    base = re.sub(r"_run[-_]\d+.*$", "", base, flags=re.IGNORECASE)  # _run-1287…
    return base.strip() or p


def _pick_feature_label_col(df: pd.DataFrame, candidates: list) -> Optional[str]:
    """
    From a list of candidate column names (in priority order), return the first
    one that is present in df.  Falls back to None.
    """
    for c in candidates:
        if c in df.columns:
            return c
    # fuzzy: column whose name contains 'gene'
    for c in df.columns:
        if "gene" in str(c).lower():
            return c
    # fuzzy: column whose name contains 'protein' and 'name'
    for c in df.columns:
        cl = str(c).lower()
        if "protein" in cl and "name" in cl:
            return c
    return None


def _finalise_transposed(df_t: pd.DataFrame, rename_mapping=None) -> pd.DataFrame:
    """
    Common post-processing after transpose:
      - comma → dot for European locale files
      - convert feature columns to numeric
      - ensure all column names are strings (sklearn requirement)
      - drop all-NaN columns
      - de-duplicate columns
      - add ID column
    """
    # European decimal commas
    feat_cols = df_t.columns[1:]
    df_t[feat_cols] = (
        df_t[feat_cols]
        .apply(lambda s: s.astype(str).str.replace(",", ".", regex=False))
        .apply(pd.to_numeric, errors="coerce")
    )
    df_t = df_t.dropna(axis=1, how="all")
    # ── CRITICAL: all column names must be strings for sklearn ──────────────
    df_t.columns = df_t.columns.astype(str)
    df_t = df_t.loc[:, ~df_t.columns.duplicated()]
    if rename_mapping:
        df_t.replace(rename_mapping, inplace=True)
    df_t = _ensure_id_column(df_t)
    return df_t


# ─── Format-specific parsers ───────────────────────────────────────────────────

def perseus_data(file, rename_mapping=None, feature_row_index=-2):
    """
    Perseus matrix export (tab-separated).
    Rows = proteins, columns = samples + annotation columns prefixed T:, N:, C:
    After transpose: rows = samples, columns = proteins (labelled by gene / protein name).
    feature_row_index: which annotation row to use as feature labels
      -2 → T: Gene names  (default)
      -1 → T: Protein names
    """
    try:
        df = load_data_safely(file)

        META_PREFIXES = ("T:", "N:", "C:")
        annot_cols    = [c for c in df.columns if str(c).strip().startswith(META_PREFIXES)]
        sample_cols   = [c for c in df.columns if c not in annot_cols]

        if not sample_cols:
            raise ValueError("No intensity/sample columns found in Perseus file.")

        # Choose feature label column
        label_candidates = [
            "T: Gene names", "T: Protein names", "T: Majority protein IDs", "T: Protein IDs"
        ]
        feat_label_col = _pick_feature_label_col(
            pd.DataFrame(columns=annot_cols), label_candidates
        ) or (annot_cols[feature_row_index] if annot_cols else None)

        if feat_label_col and feat_label_col in df.columns:
            feature_labels = _make_unique_labels(df[feat_label_col])
        else:
            feature_labels = [str(i) for i in range(len(df))]

        # Transpose intensity block
        df_t = df[sample_cols].T.reset_index()
        df_t.columns = ["Class"] + feature_labels

        return _finalise_transposed(df_t, rename_mapping)

    except Exception as e:
        raise ValueError(f"Error reading Perseus file: {e}")


def diann_data(file, rename_mapping=None, feature_row_index=None):
    """
    DIA-NN protein-group matrix (TSV).
    Layout: rows = proteins, first columns = annotation (Protein.Group, Genes, …),
    remaining columns = samples (full file paths as column headers).

    After transpose: rows = samples (shortened paths), columns = proteins (gene names).
    feature_row_index kept for backward-compat but ignored — gene/protein name
    is detected automatically.
    """
    try:
        df = load_data_safely(file)

        # Known annotation column names (DIA-NN style)
        DIANN_ANNOT = {
            "Protein.Group", "Protein.Ids", "Protein.Names", "Genes",
            "First.Protein.Description", "Accession", "Description",
            "N.Sequences", "N.Proteotypic.Sequences",
            # also handle MaxQuant-style T: columns mixed in
            "T: Protein IDs", "T: Majority protein IDs", "T: Protein names",
            "T: Gene names", "T: id",
        }
        annot_cols  = [c for c in df.columns if c in DIANN_ANNOT or
                       str(c).strip().startswith(("T:", "N:", "C:"))]
        sample_cols = [c for c in df.columns if c not in annot_cols]

        if not sample_cols:
            raise ValueError("No sample/intensity columns found in DIA-NN file.")

        # Feature labels — prefer Genes, then Protein.Names
        label_candidates = [
            "Genes", "T: Gene names", "Protein.Names", "T: Protein names",
            "Protein.Group", "Protein.Ids",
        ]
        feat_label_col = _pick_feature_label_col(
            df[annot_cols] if annot_cols else df, label_candidates
        )

        if feat_label_col and feat_label_col in df.columns:
            feature_labels = _make_unique_labels(df[feat_label_col])
        else:
            feature_labels = [str(i) for i in range(len(df))]

        # Transpose
        df_t = df[sample_cols].T.reset_index()
        df_t.columns = ["Class"] + feature_labels

        # Shorten long file-path sample names
        df_t["Class"] = df_t["Class"].apply(_shorten_sample_name)

        return _finalise_transposed(df_t, rename_mapping)

    except Exception as e:
        raise ValueError(f"Error reading DIA-NN file: {e}")


def maxquant_data(file, rename_mapping=None, feature_row_index=None):
    """
    MaxQuant proteinGroups.txt (tab-separated).
    Intensity columns start with 'LFQ intensity '.
    Annotation columns are prefixed T:, N:, C: (Perseus export) or named
    'Gene names' / 'Protein names' (plain MaxQuant export).

    After transpose: rows = samples (LFQ suffix stripped), columns = proteins.
    feature_row_index kept for backward-compat but ignored — gene/protein name
    is detected automatically.
    """
    try:
        df = load_data_safely(file)

        META_PREFIXES = ("C:", "N:", "T:")
        annot_cols    = [c for c in df.columns if str(c).strip().startswith(META_PREFIXES)
                         or c in ("Gene names", "Protein names",
                                  "Majority protein IDs", "Protein IDs")]
        lfq_cols      = [c for c in df.columns if c.startswith("LFQ intensity")]

        if not lfq_cols:
            raise ValueError(
                "No 'LFQ intensity' columns found. "
                "Please verify this is a MaxQuant proteinGroups file."
            )

        # Feature labels — prefer gene names
        label_candidates = [
            "T: Gene names", "Gene names",
            "T: Protein names", "Protein names",
            "T: Majority protein IDs", "Majority protein IDs",
        ]
        feat_label_col = _pick_feature_label_col(df, label_candidates)

        if feat_label_col:
            feature_labels = _make_unique_labels(df[feat_label_col])
        else:
            feature_labels = [str(i) for i in range(len(df))]

        # Transpose LFQ intensity block
        df_t = df[lfq_cols].T.reset_index()
        df_t.columns = ["Class"] + feature_labels

        # Clean up sample names: strip 'LFQ intensity ' prefix
        df_t["Class"] = (
            df_t["Class"]
            .str.replace(r"^LFQ intensity\s*", "", regex=True)
            .str.strip()
        )

        return _finalise_transposed(df_t, rename_mapping)

    except Exception as e:
        raise ValueError(f"Error reading MaxQuant file: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities (used by Profiler.py + profiler_imports.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_meta_columns(df) -> list:
    """Return all columns whose name ends with '_meta'."""
    return [c for c in df.columns if str(c).endswith('_meta')]


def get_omics_columns(df) -> list:
    """Return feature (omics) columns — everything except Class, ID, File, RT, Sum and _meta cols."""
    exclude = {'Class', 'ID', 'File', 'RT', 'Sum'} | set(get_meta_columns(df))
    return [c for c in df.columns if c not in exclude]


def get_target_column_options(df) -> list:
    """
    Returns possible target columns: 'Class' (if present) + all _meta columns.
    Used to let the user pick what to predict/colour by.
    """
    opts = []
    if 'Class' in df.columns:
        opts.append('Class')
    opts += get_meta_columns(df)
    return opts


# =============================================================================
# ──────────────────────────────────────────────────────────────────────────────
#  EXTENDED FORMAT PARSERS
#  Proteomics  : Spectronaut, FragPipe/MSFragger, Proteome Discoverer,
#                Progenesis QI, PEAKS Studio
#  Peptide-level: MaxQuant peptides, DIA-NN precursors, Spectronaut peptides
#  Transcriptomics: DESeq2/edgeR counts, Salmon/kallisto, featureCounts,
#                   STAR, HTSeq
#  Metabolomics : MetaboAnalyst / XCMS / MZmine generic feature tables
#  Auto-detect  : detect_omics_format() + load_omics_auto()
# =============================================================================

# ─── Spectronaut protein report ───────────────────────────────────────────────

def spectronaut_protein_data(file, rename_mapping=None):
    """
    Spectronaut ProteinReport.tsv (long or wide format).

    Wide format  : rows = proteins, cols include PG.Genes / PG.ProteinGroups
                   + one column per sample named like 'SampleName [Intensity]'
                   or raw file path columns.
    Long format  : columns R.FileName, PG.Genes, PG.Quantity → pivoted here.

    Returns a DataFrame: rows = samples, cols = proteins (gene names).
    """
    try:
        df = load_data_safely(file, sep="\t")

        ANNOT_COLS = {
            "PG.ProteinGroups", "PG.Genes", "PG.ProteinNames", "PG.UniProtIds",
            "PG.FastaFiles", "PG.NrOfStrippedSequencesIdentified",
            "PG.NrOfPrecursorsIdentified", "EG.ModifiedSequence",
        }

        # ── Detect long vs wide ──────────────────────────────────────────────
        if "R.FileName" in df.columns and "PG.Quantity" in df.columns:
            # Long format → pivot
            feat_col = next(
                (c for c in ["PG.Genes", "PG.ProteinGroups", "PG.ProteinNames"] if c in df.columns),
                None
            )
            if feat_col is None:
                raise ValueError("Cannot find a protein/gene identifier column in long Spectronaut report.")
            df_wide = df.pivot_table(
                index="R.FileName", columns=feat_col, values="PG.Quantity", aggfunc="first"
            ).reset_index().rename(columns={"R.FileName": "Class"})
            df_wide["Class"] = df_wide["Class"].apply(_shorten_sample_name)
            df_wide.columns = df_wide.columns.astype(str)
            return _finalise_transposed(df_wide, rename_mapping)

        # Wide format
        annot_present = [c for c in df.columns if c in ANNOT_COLS]
        sample_cols   = [c for c in df.columns if c not in ANNOT_COLS]

        if not sample_cols:
            raise ValueError("No sample columns detected in Spectronaut wide report.")

        feat_col = _pick_feature_label_col(
            df[annot_present] if annot_present else df,
            ["PG.Genes", "PG.ProteinGroups", "PG.ProteinNames", "PG.UniProtIds"]
        )
        feature_labels = _make_unique_labels(df[feat_col]) if feat_col else [str(i) for i in range(len(df))]

        df_t = df[sample_cols].T.reset_index()
        df_t.columns = ["Class"] + feature_labels
        df_t["Class"] = df_t["Class"].apply(_shorten_sample_name)

        return _finalise_transposed(df_t, rename_mapping)

    except Exception as e:
        raise ValueError(f"Error reading Spectronaut file: {e}")


def spectronaut_peptide_data(file, rename_mapping=None):
    """
    Spectronaut PeptideReport.tsv.
    Cols: PEP.StrippedSequence, PG.Genes (annotation) + sample columns.
    """
    try:
        df = load_data_safely(file, sep="\t")

        ANNOT_COLS = {
            "PEP.StrippedSequence", "PEP.ModifiedSequence", "PG.Genes",
            "PG.ProteinGroups", "PG.ProteinNames", "EG.ModifiedSequence",
            "R.FileName",
        }

        if "R.FileName" in df.columns and "EG.Quantity" in df.columns:
            # Long format
            feat_col = next(
                (c for c in ["PEP.StrippedSequence", "EG.ModifiedSequence"] if c in df.columns), None
            )
            if feat_col is None:
                raise ValueError("No peptide sequence column found.")
            df_wide = df.pivot_table(
                index="R.FileName", columns=feat_col, values="EG.Quantity", aggfunc="first"
            ).reset_index().rename(columns={"R.FileName": "Class"})
            df_wide["Class"] = df_wide["Class"].apply(_shorten_sample_name)
            df_wide.columns = df_wide.columns.astype(str)
            return _finalise_transposed(df_wide, rename_mapping)

        annot_present = [c for c in df.columns if c in ANNOT_COLS]
        sample_cols   = [c for c in df.columns if c not in ANNOT_COLS]
        feat_col = _pick_feature_label_col(
            df, ["PEP.StrippedSequence", "EG.ModifiedSequence", "PEP.ModifiedSequence"]
        )
        feature_labels = _make_unique_labels(df[feat_col]) if feat_col else [str(i) for i in range(len(df))]
        df_t = df[sample_cols].T.reset_index()
        df_t.columns = ["Class"] + feature_labels
        df_t["Class"] = df_t["Class"].apply(_shorten_sample_name)
        return _finalise_transposed(df_t, rename_mapping)

    except Exception as e:
        raise ValueError(f"Error reading Spectronaut peptide file: {e}")


# ─── FragPipe / MSFragger combined_protein.tsv ────────────────────────────────

def fragpipe_data(file, rename_mapping=None):
    """
    FragPipe combined_protein.tsv or MSFragger protein output.
    Intensity columns end with ' MaxLFQ Intensity' or ' Intensity'.
    Annotation columns: Protein, Gene, Description, Organism, etc.
    """
    try:
        df = load_data_safely(file, sep="\t")

        ANNOT_PATTERNS = (
            "Protein", "Gene", "Description", "Organism", "Length",
            "Coverage", "Indistinguishable", "Protein ID", "Entry Name",
            "Protein Existence", "Gene Names",
        )
        annot_cols  = [c for c in df.columns if any(c.startswith(p) for p in ANNOT_PATTERNS)]
        # Intensity cols: end with 'Intensity' or 'MaxLFQ Intensity'
        intens_cols = [c for c in df.columns
                       if c.endswith((" MaxLFQ Intensity", " Intensity", " Spectral Count"))
                       and c not in annot_cols]

        if not intens_cols:
            # Fallback: anything not annotation
            intens_cols = [c for c in df.columns if c not in annot_cols]

        feat_col = _pick_feature_label_col(
            df, ["Gene", "Gene Names", "Protein", "Protein ID", "Entry Name"]
        )
        feature_labels = _make_unique_labels(df[feat_col]) if feat_col else [str(i) for i in range(len(df))]

        df_t = df[intens_cols].T.reset_index()
        df_t.columns = ["Class"] + feature_labels
        # Strip common suffixes from sample names
        df_t["Class"] = (
            df_t["Class"]
            .str.replace(r"\s*MaxLFQ Intensity$", "", regex=True)
            .str.replace(r"\s*Intensity$", "", regex=True)
            .str.strip()
        )
        return _finalise_transposed(df_t, rename_mapping)

    except Exception as e:
        raise ValueError(f"Error reading FragPipe file: {e}")


# ─── Proteome Discoverer ──────────────────────────────────────────────────────

def proteome_discoverer_data(file, rename_mapping=None):
    """
    Proteome Discoverer Proteins.txt or PSMs.txt export.
    Abundance columns are named 'Abundance: F1: …' or 'Abundance Ratio: …'
    Annotation: Accession, Gene Symbol, Description, etc.
    """
    try:
        df = load_data_safely(file, sep="\t")

        ANNOT_COLS = {
            "Accession", "Gene Symbol", "Gene ID", "Description",
            "# Peptides", "# PSMs", "# Unique Peptides",
            "MW [kDa]", "Sequence", "Modifications",
            "Protein FDR Confidence: Combined", "Master",
        }
        annot_present = [c for c in df.columns if c in ANNOT_COLS]
        intens_cols   = [c for c in df.columns
                         if c.startswith("Abundance:") and "Ratio" not in c]

        if not intens_cols:
            intens_cols = [c for c in df.columns
                           if c.startswith("Abundance") and c not in annot_present]

        if not intens_cols:
            raise ValueError("No 'Abundance' columns found. Check Proteome Discoverer export format.")

        feat_col = _pick_feature_label_col(df, ["Gene Symbol", "Accession", "Description"])
        feature_labels = _make_unique_labels(df[feat_col]) if feat_col else [str(i) for i in range(len(df))]

        df_t = df[intens_cols].T.reset_index()
        df_t.columns = ["Class"] + feature_labels
        # Clean: 'Abundance: F1: Sample_A' → 'Sample_A'
        df_t["Class"] = (
            df_t["Class"]
            .str.replace(r"^Abundance:\s*F\d+:\s*", "", regex=True)
            .str.strip()
        )
        return _finalise_transposed(df_t, rename_mapping)

    except Exception as e:
        raise ValueError(f"Error reading Proteome Discoverer file: {e}")


# ─── Progenesis QI ────────────────────────────────────────────────────────────

def progenesis_data(file, rename_mapping=None):
    """
    Progenesis QI normalised abundance export (CSV).
    Layout: multi-header rows; sample names in row 1, abundances from row 3+.
    Annotation cols: Accession, Description, #Peptides, etc.
    """
    try:
        df_raw = load_data_safely(file, sep=",")

        # Progenesis has 2 header rows — detect and handle
        # Row 0: group/condition names (may be blank for annotation cols)
        # Row 1: actual column names including 'Normalised abundance'

        # Heuristic: if first cell of row 0 looks like a group label, skip it
        # Simply try to detect: if >50% of first row values are numeric → no extra header
        first_row = df_raw.iloc[0]
        numeric_frac = pd.to_numeric(first_row, errors="coerce").notna().mean()

        if numeric_frac < 0.3:
            # First row is a header / group row — use it as group info then skip
            df_raw.columns = [
                f"{h}||{c}" if str(h).strip() and str(h) != "nan" else c
                for h, c in zip(first_row.values, df_raw.columns)
            ]
            df_raw = df_raw.iloc[1:].reset_index(drop=True)

        ANNOT_COLS = {
            "Accession", "Description", "Gene", "Gene name",
            "#Peptides", "#Unique peptides", "Protein FDR",
            "Mass", "Charge", "Retention time (min)", "Score",
        }
        annot_present = [c.split("||")[-1].strip() for c in df_raw.columns
                         if c.split("||")[-1].strip() in ANNOT_COLS]
        intens_cols   = [c for c in df_raw.columns
                         if c.split("||")[-1].strip() not in ANNOT_COLS]

        # Rebuild clean annot lookup
        col_map = {c: c.split("||")[-1].strip() for c in df_raw.columns}
        df_raw = df_raw.rename(columns=col_map)

        annot_present = [c for c in df_raw.columns if c in ANNOT_COLS]
        intens_cols   = [c for c in df_raw.columns if c not in ANNOT_COLS]

        feat_col = _pick_feature_label_col(df_raw, ["Gene", "Gene name", "Accession", "Description"])
        feature_labels = _make_unique_labels(df_raw[feat_col]) if feat_col else [str(i) for i in range(len(df_raw))]

        df_t = df_raw[intens_cols].T.reset_index()
        df_t.columns = ["Class"] + feature_labels
        return _finalise_transposed(df_t, rename_mapping)

    except Exception as e:
        raise ValueError(f"Error reading Progenesis file: {e}")


# ─── PEAKS Studio ─────────────────────────────────────────────────────────────

def _detect_peaks_version(df: pd.DataFrame) -> str:
    """
    Heuristic to distinguish PEAKS Studio format versions.
    Returns '13.1' for PEAKS 13.1+, 'legacy' otherwise.

    PEAKS 13.1 signature:
      - 'Protein Group' column (numeric group id)
      - 'Accession' column
      - Per-sample Area columns as 'Area <SampleName>' (no colon)
      - Per-sample #Spec columns as '#Spec <SampleName>'
      - Per-sample Coverage columns as 'Coverage(%) <SampleName>'
      - 'Average Mass' (vs legacy 'Avg. Mass')
      - '-10LgP' (capital L, vs legacy '-10lgP')
      - 'Top' boolean column
    """
    has_protein_group  = "Protein Group" in df.columns
    has_accession      = "Accession" in df.columns
    has_average_mass   = "Average Mass" in df.columns
    has_top            = "Top" in df.columns
    # Per-sample Area columns with a space (no colon) — e.g. 'Area BT-HLA-Chymo'
    area_space_cols    = [c for c in df.columns
                          if re.match(r"^Area\s+\S", c)]

    if (has_protein_group and has_accession
            and (has_average_mass or has_top or area_space_cols)):
        return "13.1"
    return "legacy"


def peaks_data(file, rename_mapping=None):
    """
    PEAKS Studio protein/peptide export (CSV).

    Supports:
      • Legacy PEAKS (≤12):
          - Annotation: 'Protein ID', 'Gene', 'Description', 'Coverage (%)', '#Peptides', …
          - Intensity columns: 'Area:SampleName' or 'SampleName Area'
      • PEAKS Studio 13.1+:
          - Annotation: 'Protein Group', 'Top', 'Accession', 'Gene', '-10LgP',
            'Coverage(%)', '#Peptides', '#Unique', 'PTM', 'Average Mass', 'Description'
          - Per-sample annotation: 'Coverage(%) <Sample>', '#Spec <Sample>'
          - Intensity columns: 'Area <SampleName>' (space-separated, no colon)
    """
    try:
        df = load_data_safely(file, sep=",")
        df = _clean_col_names(df)
        version = _detect_peaks_version(df)

        if version == "13.1":
            # ── PEAKS 13.1 ────────────────────────────────────────────────────
            # Fixed annotation columns (not per-sample)
            FIXED_ANNOT = {
                "Protein Group", "Top", "Accession", "Gene",
                "-10LgP", "-10lgP",          # handle both capitalisations
                "Coverage(%)", "Coverage (%)",
                "#Peptides", "#Unique", "PTM", "Average Mass",
                "Avg. Mass", "Description",
            }
            # Per-sample pattern columns (Coverage(%) X, #Spec X) — annotation, not intensity
            per_sample_annot_prefixes = ("Coverage(%) ", "Coverage (%) ", "#Spec ", "#De Novo ")

            area_cols = []
            for c in df.columns:
                # 'Area <SampleName>' — the defining intensity column of PEAKS 13.1
                if re.match(r"^Area\s+\S", c):
                    area_cols.append(c)

            if not area_cols:
                # Fallback: any column not in fixed annot and not a per-sample annot prefix
                area_cols = [
                    c for c in df.columns
                    if c not in FIXED_ANNOT
                    and not any(c.startswith(p) for p in per_sample_annot_prefixes)
                ]

            feat_col = _pick_feature_label_col(
                df, ["Gene", "Accession", "Description"]
            )
            feature_labels = (
                _make_unique_labels(df[feat_col]) if feat_col
                else [str(i) for i in range(len(df))]
            )

            df_t = df[area_cols].T.reset_index()
            df_t.columns = ["Class"] + feature_labels
            # Strip leading 'Area ' prefix to recover sample name
            df_t["Class"] = (
                df_t["Class"]
                .str.replace(r"^Area\s+", "", regex=True)
                .str.strip()
            )

        else:
            # ── Legacy PEAKS (≤12) ────────────────────────────────────────────
            ANNOT_COLS = {
                "Protein ID", "Protein Group", "Accession", "Gene", "Description",
                "Species", "Organism", "Coverage (%)", "#Peptides", "#Unique",
                "#Spectra", "#Spec", "#De Novo", "Avg. Mass", "Average Mass", "Score",
                "-10lgP", "-10LgP", "PTM", "Group Profile (Ratio)",
                "Peptide", "Sequence", "Modified Sequence", "Charge",
                "m/z", "RT", "Area", "ppm",
            }
            annot_present = [
                c for c in df.columns
                if c in ANNOT_COLS or c.split(":")[0].strip() in ANNOT_COLS
            ]
            area_cols = [
                c for c in df.columns
                if c.startswith("Area:") or c.endswith(" Area")
                or (c not in annot_present and "area" in c.lower())
            ]
            if not area_cols:
                area_cols = [c for c in df.columns if c not in annot_present]

            feat_col = _pick_feature_label_col(
                df, ["Gene", "Protein ID", "Description", "Peptide", "Sequence"]
            )
            feature_labels = (
                _make_unique_labels(df[feat_col]) if feat_col
                else [str(i) for i in range(len(df))]
            )

            df_t = df[area_cols].T.reset_index()
            df_t.columns = ["Class"] + feature_labels
            df_t["Class"] = (
                df_t["Class"]
                .str.replace(r"^Area:\s*", "", regex=True)   # 'Area:SampleName'
                .str.replace(r"\s*Area$",  "", regex=True)   # 'SampleName Area'
                .str.strip()
            )

        return _finalise_transposed(df_t, rename_mapping)

    except Exception as e:
        raise ValueError(f"Error reading PEAKS file: {e}")


# ─── MaxQuant peptides.txt ────────────────────────────────────────────────────

def maxquant_peptide_data(file, rename_mapping=None):
    """
    MaxQuant peptides.txt.
    Intensity columns start with 'Intensity ' (per sample).
    Annotation: Sequence, Proteins, Gene names, Missed cleavages, etc.
    """
    try:
        df = load_data_safely(file, sep="\t")

        intens_cols = [c for c in df.columns
                       if c.startswith("Intensity ") and not c.startswith("Intensity ")]
        # MaxQuant uses 'Intensity SampleName' (note the space)
        intens_cols = [c for c in df.columns if c.startswith("Intensity ")]

        if not intens_cols:
            raise ValueError("No 'Intensity ' columns found in MaxQuant peptides file.")

        feat_col = _pick_feature_label_col(df, ["Sequence", "Modified sequence", "Gene names", "Proteins"])
        feature_labels = _make_unique_labels(df[feat_col]) if feat_col else [str(i) for i in range(len(df))]

        df_t = df[intens_cols].T.reset_index()
        df_t.columns = ["Class"] + feature_labels
        df_t["Class"] = df_t["Class"].str.replace(r"^Intensity\s+", "", regex=True).str.strip()
        return _finalise_transposed(df_t, rename_mapping)

    except Exception as e:
        raise ValueError(f"Error reading MaxQuant peptides file: {e}")


# ─── DIA-NN precursors (peptide-level) ───────────────────────────────────────

def diann_peptide_data(file, rename_mapping=None):
    """
    DIA-NN precursors.tsv or peptide-level report.
    Annotation: Precursor.Id, Modified.Sequence, Stripped.Sequence, Protein.Names, Genes.
    Sample columns: full file paths as column headers (same pattern as protein report).
    """
    try:
        df = load_data_safely(file, sep="\t")

        ANNOT_COLS = {
            "Precursor.Id", "Modified.Sequence", "Stripped.Sequence",
            "Protein.Group", "Protein.Ids", "Protein.Names", "Genes",
            "First.Protein.Description", "Proteotypic", "Precursor.Charge",
            "Q.Value", "Global.Q.Value", "RT", "Predicted.RT",
        }
        annot_present = [c for c in df.columns if c in ANNOT_COLS]
        sample_cols   = [c for c in df.columns if c not in ANNOT_COLS]

        feat_col = _pick_feature_label_col(
            df, ["Stripped.Sequence", "Modified.Sequence", "Precursor.Id", "Genes"]
        )
        feature_labels = _make_unique_labels(df[feat_col]) if feat_col else [str(i) for i in range(len(df))]

        df_t = df[sample_cols].T.reset_index()
        df_t.columns = ["Class"] + feature_labels
        df_t["Class"] = df_t["Class"].apply(_shorten_sample_name)
        return _finalise_transposed(df_t, rename_mapping)

    except Exception as e:
        raise ValueError(f"Error reading DIA-NN peptide file: {e}")


# =============================================================================
#  TRANSCRIPTOMICS PARSERS
# =============================================================================

def _counts_wide_to_df(df, gene_col, sample_cols, rename_mapping=None):
    """
    Generic helper: given a counts DataFrame in wide format
    (rows = genes, cols = samples), return a Profiler-ready DataFrame
    (rows = samples, cols = genes).
    """
    feature_labels = _make_unique_labels(df[gene_col]) if gene_col else [str(i) for i in range(len(df))]
    df_t = df[sample_cols].T.reset_index()
    df_t.columns = ["Class"] + feature_labels
    df_t["Class"] = df_t["Class"].apply(_shorten_sample_name)
    return _finalise_transposed(df_t, rename_mapping)


def rnaseq_counts_data(file, rename_mapping=None):
    """
    Generic RNA-seq count matrix (DESeq2 / edgeR / raw counts).
    Expected: rows = genes, first column(s) = gene_id / gene_name,
    remaining columns = one per sample.
    Also handles DESeq2 results tables (baseMean, log2FC, padj …) —
    those are detected and rejected gracefully.
    """
    try:
        df = load_data_safely(file)

        ANNOT_COLS = {
            "gene_id", "gene_name", "Gene", "GeneID", "Geneid",
            "ensembl_gene_id", "hgnc_symbol", "external_gene_name",
            "gene", "Name", "target_id",
            # DESeq2 result columns → not a count matrix
            "baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj",
        }
        annot_present = [c for c in df.columns if c in ANNOT_COLS]
        sample_cols   = [c for c in df.columns if c not in ANNOT_COLS]

        # If DESeq2 result table detected → raise informative error
        if any(c in df.columns for c in ("log2FoldChange", "padj", "baseMean")):
            raise ValueError(
                "This looks like a DESeq2 *results* table (log2FoldChange, padj …), "
                "not a count matrix. Please provide the raw counts or normalised counts matrix."
            )

        if not sample_cols:
            raise ValueError("No sample count columns detected.")

        gene_col = _pick_feature_label_col(
            df, ["gene_name", "hgnc_symbol", "external_gene_name",
                 "Gene", "gene", "gene_id", "GeneID", "Geneid", "Name"]
        )
        return _counts_wide_to_df(df, gene_col, sample_cols, rename_mapping)

    except Exception as e:
        raise ValueError(f"Error reading RNA-seq count matrix: {e}")


def salmon_kallisto_data(file, rename_mapping=None):
    """
    Salmon quant.sf  → cols: Name, Length, EffectiveLength, TPM, NumReads
    Kallisto abundance.tsv → cols: target_id, length, eff_length, est_counts, tpm

    Since these are single-sample files, this function merges a *list* of files
    into one matrix. When a single file is passed, returns a single-sample df.
    """
    try:
        df = load_data_safely(file)

        # Salmon
        if "TPM" in df.columns and "Name" in df.columns:
            gene_col = "Name"
            value_col = "TPM"
        # Kallisto
        elif "tpm" in df.columns and "target_id" in df.columns:
            gene_col = "target_id"
            value_col = "tpm"
        elif "est_counts" in df.columns and "target_id" in df.columns:
            gene_col = "target_id"
            value_col = "est_counts"
        else:
            raise ValueError(
                "Cannot detect Salmon/kallisto format. "
                "Expected columns: Name+TPM (Salmon) or target_id+tpm (kallisto)."
            )

        sample_name = _shorten_sample_name(getattr(file, "name", "sample"))
        row = {gene: val for gene, val in zip(df[gene_col].astype(str), df[value_col])}
        row["Class"] = sample_name
        df_out = pd.DataFrame([row])
        df_out.columns = df_out.columns.astype(str)
        return _finalise_transposed(df_out, rename_mapping)

    except Exception as e:
        raise ValueError(f"Error reading Salmon/kallisto file: {e}")


def featurecounts_data(file, rename_mapping=None):
    """
    featureCounts output counts.txt.
    First line: '# Program:featureCounts …' (comment) — skipped.
    Second line: header with Geneid, Chr, Start, End, Strand, Length, then sample paths.
    """
    try:
        df = load_data_safely(file, sep="\t")

        # Drop comment lines starting with '#'
        df = df[~df.iloc[:, 0].astype(str).str.startswith("#")].reset_index(drop=True)

        ANNOT_COLS = {"Geneid", "Chr", "Start", "End", "Strand", "Length"}
        annot_present = [c for c in df.columns if c in ANNOT_COLS]
        sample_cols   = [c for c in df.columns if c not in ANNOT_COLS]

        if not sample_cols:
            raise ValueError("No sample columns found in featureCounts output.")

        gene_col = "Geneid" if "Geneid" in df.columns else (annot_present[0] if annot_present else None)
        df_t = df[sample_cols].T.reset_index()
        feature_labels = _make_unique_labels(df[gene_col]) if gene_col else [str(i) for i in range(len(df))]
        df_t.columns = ["Class"] + feature_labels
        df_t["Class"] = df_t["Class"].apply(_shorten_sample_name)
        return _finalise_transposed(df_t, rename_mapping)

    except Exception as e:
        raise ValueError(f"Error reading featureCounts file: {e}")


def star_counts_data(file, rename_mapping=None):
    """
    STAR ReadsPerGene.out.tab — single sample, 4 columns:
    gene_id | unstranded | strand1 | strand2
    Returns a single-row DataFrame (sample name from filename).
    """
    try:
        df = load_data_safely(file, sep="\t")
        df.columns = ["gene_id", "unstranded", "strand1", "strand2"]
        # Drop the first 4 summary rows (N_unmapped, N_multimapping, etc.)
        df = df[~df["gene_id"].str.startswith("N_")].reset_index(drop=True)

        sample_name = _shorten_sample_name(getattr(file, "name", "sample"))
        row = {gene: int(count) for gene, count in zip(df["gene_id"], df["unstranded"])}
        row["Class"] = sample_name
        df_out = pd.DataFrame([row])
        df_out.columns = df_out.columns.astype(str)
        return _finalise_transposed(df_out, rename_mapping)

    except Exception as e:
        raise ValueError(f"Error reading STAR counts file: {e}")


def htseq_counts_data(file, rename_mapping=None):
    """
    HTSeq-count output: two columns, no header.
    Col 0 = gene_id, Col 1 = count.
    Last rows are summary stats starting with '__' → removed.
    """
    try:
        df = load_data_safely(file, sep="\t")
        df.columns = ["gene_id", "count"]
        df = df[~df["gene_id"].str.startswith("__")].reset_index(drop=True)

        sample_name = _shorten_sample_name(getattr(file, "name", "sample"))
        row = {gene: int(count) for gene, count in zip(df["gene_id"], df["count"])}
        row["Class"] = sample_name
        df_out = pd.DataFrame([row])
        df_out.columns = df_out.columns.astype(str)
        return _finalise_transposed(df_out, rename_mapping)

    except Exception as e:
        raise ValueError(f"Error reading HTSeq counts file: {e}")


# =============================================================================
#  METABOLOMICS PARSERS
# =============================================================================

def metaboanalyst_data(file, rename_mapping=None):
    """
    MetaboAnalyst-style CSV: rows = samples, columns = metabolites.
    First column = Sample (sample ID), second column may be Class/Group label.
    This format is already in sample-major orientation → minimal transform needed.
    """
    try:
        df = load_data_safely(file, sep=",")
        df = _clean_col_names(df)

        # Detect sample ID column
        id_col = next((c for c in df.columns if c.lower() in ("sample", "sampleid", "sample_id", "name")), None)
        # Detect class/group column
        class_col = next((c for c in df.columns if c.lower() in CLASS_ALIASES and c != id_col), None)

        if class_col:
            df = df.rename(columns={class_col: "Class"})
        else:
            df["Class"] = "Unknown"

        if id_col:
            df = df.rename(columns={id_col: "ID"})

        # All remaining cols are features
        exclude = {"Class", "ID"}
        feat_cols = [c for c in df.columns if c not in exclude]
        df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce")
        df.columns = df.columns.astype(str)
        df = df.loc[:, ~df.columns.duplicated()]
        if rename_mapping:
            df.replace(rename_mapping, inplace=True)
        return _ensure_id_column(df)

    except Exception as e:
        raise ValueError(f"Error reading MetaboAnalyst file: {e}")


def xcms_mzmine_data(file, rename_mapping=None):
    """
    XCMS / MZmine feature table.
    Common formats:
      - XCMS: rows = features (mz_rt), columns = samples + annotation
      - MZmine: similar, may have 'row ID', 'row m/z', 'row retention time' annotation cols
    Returns: rows = samples, cols = features (labelled mz_rt or feature ID).
    """
    try:
        df = load_data_safely(file)
        df = _clean_col_names(df)

        ANNOT_PATTERNS = (
            "row id", "row m/z", "row retention time", "mz", "rt",
            "mzmed", "rtmed", "mzmin", "mzmax", "rtmin", "rtmax",
            "npeaks", "isotopes", "adduct", "molecular formula",
            "name", "feature id",
        )
        annot_cols  = [c for c in df.columns if c.lower().strip() in ANNOT_PATTERNS]
        sample_cols = [c for c in df.columns if c not in annot_cols]

        if not sample_cols:
            raise ValueError("No sample intensity columns found in XCMS/MZmine feature table.")

        # Feature label: prefer mz_rt combo
        mz_col = next((c for c in annot_cols if "mz" in c.lower() and "row" in c.lower()), None)
        rt_col = next((c for c in annot_cols if "rt" in c.lower() and "row" in c.lower()), None)
        id_col = next((c for c in annot_cols if "id" in c.lower()), None)

        if mz_col and rt_col:
            feature_labels = [
                f"mz{round(float(m),4)}_rt{round(float(r),2)}"
                for m, r in zip(df[mz_col], df[rt_col])
            ]
        elif id_col:
            feature_labels = _make_unique_labels(df[id_col])
        else:
            feature_labels = [str(i) for i in range(len(df))]

        df_t = df[sample_cols].T.reset_index()
        df_t.columns = ["Class"] + feature_labels
        df_t["Class"] = df_t["Class"].apply(_shorten_sample_name)
        return _finalise_transposed(df_t, rename_mapping)

    except Exception as e:
        raise ValueError(f"Error reading XCMS/MZmine file: {e}")


# =============================================================================
#  AUTO-DETECT ENGINE
# =============================================================================

# Signature rules: list of (detector_fn, parser_fn, label)
# detector_fn(df, filename) → bool

def _sig_maxquant_protein(df, fname):
    return any(c.startswith("LFQ intensity") for c in df.columns)

def _sig_maxquant_peptide(df, fname):
    return (any(c.startswith("Intensity ") for c in df.columns)
            and "Sequence" in df.columns and "LFQ intensity" not in " ".join(df.columns))

def _sig_diann_protein(df, fname):
    return "Protein.Group" in df.columns or "Protein.Ids" in df.columns

def _sig_diann_peptide(df, fname):
    return ("Precursor.Id" in df.columns or "Stripped.Sequence" in df.columns
            or "Modified.Sequence" in df.columns)

def _sig_spectronaut_protein(df, fname):
    return any(c.startswith("PG.") for c in df.columns)

def _sig_spectronaut_peptide(df, fname):
    return any(c.startswith("PEP.") or c.startswith("EG.") for c in df.columns)

def _sig_fragpipe(df, fname):
    return ("Gene" in df.columns and
            any(c.endswith((" MaxLFQ Intensity", " Intensity")) for c in df.columns))

def _sig_proteome_discoverer(df, fname):
    return ("Accession" in df.columns and
            any(c.startswith("Abundance:") for c in df.columns))

def _sig_progenesis(df, fname):
    return ("Accession" in df.columns and
            any("normalised" in c.lower() or "normalized" in c.lower() for c in df.columns))

def _sig_peaks(df, fname):
    has_area = any(c.startswith("Area") or c.endswith(" Area") or c == "Area" for c in df.columns)
    # Legacy PEAKS (≤12): 'Protein ID' or 'Peptide' + Area columns
    legacy = ("Protein ID" in df.columns or "Peptide" in df.columns) and has_area
    # PEAKS Studio 13.1: 'Protein Group' (numeric group id) + 'Accession' + per-sample Area columns
    peaks_131 = ("Protein Group" in df.columns and "Accession" in df.columns and has_area)
    return legacy or peaks_131

def _sig_featurecounts(df, fname):
    return "Geneid" in df.columns and "Chr" in df.columns

def _sig_star(df, fname):
    return (df.shape[1] == 4 and
            df.iloc[:, 0].astype(str).str.startswith("ENS").any())

def _sig_htseq(df, fname):
    return (df.shape[1] == 2 and
            df.iloc[:, 0].astype(str).str.startswith("__").any() is False and
            df.iloc[-5:, 0].astype(str).str.startswith("__").any())

def _sig_salmon(df, fname):
    return "TPM" in df.columns and "Name" in df.columns

def _sig_kallisto(df, fname):
    return "tpm" in df.columns and "target_id" in df.columns

def _sig_metaboanalyst(df, fname):
    return (any(c.lower() in ("sample", "sampleid") for c in df.columns) and
            not any(c.startswith("LFQ") or c.startswith("PG.") for c in df.columns))

def _sig_xcms_mzmine(df, fname):
    return any(c.lower() in ("row m/z", "row retention time", "mzmed", "rtmed") for c in df.columns)


_FORMAT_REGISTRY = [
    # (detector,                    parser,                      label)
    (_sig_maxquant_protein,        maxquant_data,               "MaxQuant proteins"),
    (_sig_maxquant_peptide,        maxquant_peptide_data,       "MaxQuant peptides"),
    (_sig_diann_protein,           diann_data,                  "DIA-NN proteins"),
    (_sig_diann_peptide,           diann_peptide_data,          "DIA-NN peptides"),
    (_sig_spectronaut_protein,     spectronaut_protein_data,    "Spectronaut proteins"),
    (_sig_spectronaut_peptide,     spectronaut_peptide_data,    "Spectronaut peptides"),
    (_sig_fragpipe,                fragpipe_data,               "FragPipe/MSFragger"),
    (_sig_proteome_discoverer,     proteome_discoverer_data,    "Proteome Discoverer"),
    (_sig_progenesis,              progenesis_data,             "Progenesis QI"),
    (_sig_peaks,                   peaks_data,                  "PEAKS Studio"),
    (_sig_featurecounts,           featurecounts_data,          "featureCounts"),
    (_sig_star,                    star_counts_data,            "STAR counts"),
    (_sig_htseq,                   htseq_counts_data,           "HTSeq counts"),
    (_sig_salmon,                  salmon_kallisto_data,        "Salmon quant"),
    (_sig_kallisto,                salmon_kallisto_data,        "kallisto abundance"),
    (_sig_xcms_mzmine,             xcms_mzmine_data,            "XCMS/MZmine"),
    (_sig_metaboanalyst,           metaboanalyst_data,          "MetaboAnalyst"),
]


def detect_omics_format(file) -> Optional[str]:
    """
    Peek at the file and return the most likely format label string,
    or None if no match is found.
    """
    try:
        file.seek(0)
        fname = getattr(file, "name", "").lower()
        raw   = file.read(8192); file.seek(0)
        sep   = _detect_delimiter(raw)
        df    = pd.read_csv(io.BytesIO(raw), sep=sep, engine="python",
                            nrows=5, on_bad_lines="skip")
        df    = _clean_col_names(df)
        for detector, _, label in _FORMAT_REGISTRY:
            if detector(df, fname):
                return label
    except Exception:
        pass
    return None


def load_omics_auto(file, rename_mapping=None):
    """
    Automatically detect and parse any supported omics file format.
    Returns (df, format_label).  Raises ValueError if no format matched.
    """
    file.seek(0)
    fname  = getattr(file, "name", "").lower()
    raw    = file.read(); file.seek(0)
    sep    = _detect_delimiter(raw[:8192])
    try:
        df_peek = pd.read_csv(io.BytesIO(raw), sep=sep, engine="python",
                              nrows=5, on_bad_lines="skip")
        df_peek = _clean_col_names(df_peek)
    except Exception as e:
        raise ValueError(f"Cannot read file for format detection: {e}")

    for detector, parser, label in _FORMAT_REGISTRY:
        if detector(df_peek, fname):
            file.seek(0)
            df = parser(file, rename_mapping=rename_mapping)
            return df, label

    raise ValueError(
        "Could not automatically detect the file format. "
        "Supported formats: MaxQuant, DIA-NN, Spectronaut, FragPipe, "
        "Proteome Discoverer, Progenesis, PEAKS Studio (legacy & 13.1+), "
        "featureCounts, STAR, HTSeq, Salmon, kallisto, XCMS/MZmine, MetaboAnalyst."
    )
