"""
profiler_longitudinal.py
═══════════════════════════════════════════════════════════════════════════════
Longitudinal / repeated-measures omics analysis module for Profiler.

Provides:
  - validate_longitudinal_df()    → check & extract Subject/Time/Class cols
  - plot_trajectory()             → spaghetti + mean trajectory per feature
  - plot_longitudinal_heatmap()   → samples × features heatmap faceted by timepoint
  - delta_features()              → compute Δ (T1−T0) or fold-change between timepoints
  - run_lmm()                     → linear mixed-effects model (statsmodels)
  - run_rm_anova()                → repeated-measures ANOVA (pingouin)
  - volcano_longitudinal()        → volcano of LMM coefficients
  - summarise_longitudinal()      → markdown summary table of significant features
  - render_longitudinal_tab()     → full Streamlit UI sub-tab (called from Profiler.py)

Dependencies (all available in the Profiler conda env):
  pandas, numpy, plotly, scipy, statsmodels, pingouin (optional), streamlit
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Optional
from statsmodels.stats.multitest import multipletests
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

SUBJECT_ALIASES = {
    "subject_id", "subjectid", "patient_id", "patientid",
    "subject", "patient", "individual", "animal_id",
    "participant", "participant_id", "sample_id",
}
TIME_ALIASES = {
    "time", "timepoint", "visit", "week", "day",
    "month", "year", "t", "tp", "time_point",
}
NON_FEATURE_COLS = {"Class", "ID", "Subject_ID", "Time", "File", "RT", "Sum"}


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, aliases: set) -> Optional[str]:
    """Return the first column whose lower-stripped name is in aliases."""
    for c in df.columns:
        if c.lower().replace(" ", "_") in aliases:
            return c
    return None


def validate_longitudinal_df(
    df: pd.DataFrame,
    subject_col: Optional[str] = None,
    time_col: Optional[str] = None,
) -> dict:
    """
    Validate and extract structural columns from a longitudinal DataFrame.

    Returns a dict with keys:
      subject_col, time_col, class_col, feature_cols,
      n_subjects, n_timepoints, timepoints, is_valid, warnings
    """
    result: dict = {
        "subject_col": None, "time_col": None, "class_col": None,
        "feature_cols": [], "n_subjects": 0, "n_timepoints": 0,
        "timepoints": [], "is_valid": False, "warnings": [],
    }

    # Auto-detect if not provided
    sc = subject_col or _find_col(df, SUBJECT_ALIASES)
    tc = time_col   or _find_col(df, TIME_ALIASES)

    if sc is None:
        result["warnings"].append(
            "No Subject_ID column found. Add a column named 'Subject_ID' or 'Patient_ID'."
        )
    if tc is None:
        result["warnings"].append(
            "No Time column found. Add a column named 'Time' or 'Timepoint'."
        )
    if sc is None or tc is None:
        return result

    result["subject_col"] = sc
    result["time_col"]    = tc
    result["class_col"]   = "Class" if "Class" in df.columns else None

    meta_cols  = [c for c in df.columns if str(c).endswith("_meta")]
    excl       = NON_FEATURE_COLS | set(meta_cols) | {sc, tc}
    feat_cols  = [c for c in df.columns
                  if c not in excl and pd.api.types.is_numeric_dtype(df[c])]

    if len(feat_cols) == 0:
        result["warnings"].append("No numeric feature columns found.")
        return result

    result["feature_cols"]  = feat_cols
    result["n_subjects"]    = df[sc].nunique()
    result["timepoints"]    = sorted(df[tc].dropna().unique().tolist(),
                                     key=lambda x: (str(x).lstrip("TtWwDdMm"), x))
    result["n_timepoints"]  = len(result["timepoints"])
    result["is_valid"]      = True

    if result["n_timepoints"] < 2:
        result["warnings"].append("Only one timepoint detected — need ≥ 2 for longitudinal analysis.")
        result["is_valid"] = False

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Plot: trajectory (spaghetti + mean)
# ─────────────────────────────────────────────────────────────────────────────

def plot_trajectory(
    df: pd.DataFrame,
    feature: str,
    subject_col: str,
    time_col: str,
    class_col: Optional[str] = None,
    show_individual: bool = True,
    show_mean: bool = True,
    normalize: bool = False,
) -> go.Figure:
    """
    Spaghetti plot of individual trajectories + group mean ± SD.
    """
    df = df.copy()
    df[feature] = pd.to_numeric(df[feature], errors="coerce")
    if normalize:
        # Normalize each subject to its own T0 value
        t0 = df[time_col].min()
        ref = df[df[time_col] == t0].set_index(subject_col)[feature].rename("ref")
        df  = df.join(ref, on=subject_col)
        df[feature] = df[feature] / df["ref"].replace(0, np.nan)

    colors = px.colors.qualitative.Set2
    class_list = sorted(df[class_col].dropna().unique()) if class_col else ["All"]
    color_map  = {g: colors[i % len(colors)] for i, g in enumerate(class_list)}

    fig = go.Figure()

    for group in class_list:
        sub = df[df[class_col] == group] if class_col else df
        grp_color = color_map[group]

        if show_individual:
            for subj in sub[subject_col].unique():
                sdf = sub[sub[subject_col] == subj].sort_values(time_col)
                fig.add_trace(go.Scatter(
                    x=sdf[time_col].astype(str), y=sdf[feature],
                    mode="lines+markers",
                    line=dict(color=grp_color, width=0.8),
                    marker=dict(size=4, color=grp_color),
                    opacity=0.35,
                    name=group,
                    legendgroup=group,
                    showlegend=False,
                    hovertemplate=f"<b>{subj}</b><br>{time_col}: %{{x}}<br>{feature}: %{{y:.3f}}<extra></extra>",
                ))

        if show_mean:
            agg = (sub.groupby(time_col)[feature]
                      .agg(["mean", "std", "count"])
                      .reset_index()
                      .sort_values(time_col))
            agg["se"] = agg["std"] / np.sqrt(agg["count"])
            fig.add_trace(go.Scatter(
                x=agg[time_col].astype(str), y=agg["mean"],
                mode="lines+markers",
                line=dict(color=grp_color, width=2.5),
                marker=dict(size=9, color=grp_color, line=dict(width=1.5, color="white")),
                error_y=dict(type="data", array=agg["sd"].tolist()
                             if "sd" in agg.columns else agg["std"].tolist(),
                             visible=True, color=grp_color, thickness=1.5, width=4),
                name=group,
                legendgroup=group,
                showlegend=True,
                hovertemplate=f"<b>{group}</b><br>{time_col}: %{{x}}<br>Mean: %{{y:.3f}}<br>SD: %{{error_y.array:.3f}}<extra></extra>",
            ))

    ylabel = f"{feature} (fold-change vs T0)" if normalize else feature
    fig.update_layout(
        title=dict(text=f"<b>{feature}</b> — Longitudinal trajectory", font_size=14),
        xaxis_title=time_col,
        yaxis_title=ylabel,
        legend_title=class_col or "",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=420,
        margin=dict(l=50, r=20, t=50, b=50),
        font=dict(size=11),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", zeroline=False)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Plot: multi-feature trajectory overview (small multiples)
# ─────────────────────────────────────────────────────────────────────────────

def plot_multi_trajectory(
    df: pd.DataFrame,
    features: list,
    subject_col: str,
    time_col: str,
    class_col: Optional[str] = None,
    ncols: int = 3,
    normalize: bool = False,
) -> go.Figure:
    """Small-multiples grid of mean trajectories for the top N features."""
    features = features[:12]   # cap at 12 panels
    nrows = int(np.ceil(len(features) / ncols))
    fig   = make_subplots(rows=nrows, cols=ncols,
                          subplot_titles=[f"<b>{f}</b>" for f in features],
                          shared_xaxes=False, vertical_spacing=0.12,
                          horizontal_spacing=0.08)

    colors    = px.colors.qualitative.Set2
    class_list = sorted(df[class_col].dropna().unique()) if class_col else ["All"]
    color_map  = {g: colors[i % len(colors)] for i, g in enumerate(class_list)}
    shown      = set()

    for idx, feat in enumerate(features):
        r = idx // ncols + 1
        c = idx %  ncols + 1
        df2 = df.copy()
        df2[feat] = pd.to_numeric(df2[feat], errors="coerce")

        if normalize:
            t0  = df2[time_col].min()
            ref = df2[df2[time_col] == t0].set_index(subject_col)[feat]
            df2 = df2.join(ref.rename("_ref"), on=subject_col)
            df2[feat] = df2[feat] / df2["_ref"].replace(0, np.nan)

        for group in class_list:
            sub = df2[df2[class_col] == group] if class_col else df2
            agg = (sub.groupby(time_col)[feat]
                      .agg(["mean", "std"]).reset_index()
                      .sort_values(time_col))
            show_leg = group not in shown
            shown.add(group)
            fig.add_trace(go.Scatter(
                x=agg[time_col].astype(str), y=agg["mean"],
                mode="lines+markers",
                line=dict(color=color_map[group], width=2),
                marker=dict(size=6, color=color_map[group]),
                name=group, legendgroup=group,
                showlegend=show_leg,
                error_y=dict(type="data", array=agg["std"].tolist(),
                             visible=True, color=color_map[group], thickness=1, width=3),
            ), row=r, col=c)

    fig.update_layout(
        height=280 * nrows,
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(size=10),
        legend_title=class_col or "",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    for ax in fig.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].update(showgrid=True, gridcolor="#f1f5f9", zeroline=False)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Delta (fold-change / difference between timepoints)
# ─────────────────────────────────────────────────────────────────────────────

def delta_features(
    df: pd.DataFrame,
    feature_cols: list,
    subject_col: str,
    time_col: str,
    t0,
    t1,
    method: str = "difference",   # "difference" | "log2fc" | "pct_change"
) -> pd.DataFrame:
    """
    Compute per-subject delta between two timepoints.
    Returns a wide DataFrame: one row per subject, cols = δ(features).
    """
    sub0 = df[df[time_col] == t0].set_index(subject_col)[feature_cols]
    sub1 = df[df[time_col] == t1].set_index(subject_col)[feature_cols]
    common = sub0.index.intersection(sub1.index)
    sub0, sub1 = sub0.loc[common], sub1.loc[common]

    if method == "log2fc":
        delta = np.log2((sub1 + 1e-9) / (sub0 + 1e-9))
    elif method == "pct_change":
        delta = (sub1 - sub0) / (sub0.abs() + 1e-9) * 100
    else:
        delta = sub1 - sub0

    delta.columns = [f"Δ_{c}" for c in delta.columns]
    # Attach Class if present
    if "Class" in df.columns:
        cls = df[df[time_col] == t0].set_index(subject_col)["Class"]
        delta = delta.join(cls.loc[common].rename("Class"))
    delta.index.name = subject_col
    return delta.reset_index()


# ─────────────────────────────────────────────────────────────────────────────
#  Statistical models
# ─────────────────────────────────────────────────────────────────────────────

def run_lmm(
    df: pd.DataFrame,
    feature_cols: list,
    subject_col: str,
    time_col: str,
    class_col: Optional[str] = None,
    max_features: int = 500,
) -> pd.DataFrame:
    """
    Fit a Linear Mixed-Effects Model for each feature:
        feature ~ Time [+ Class] + (1 | Subject_ID)

    Returns a results DataFrame with columns:
        feature, coef_Time, se_Time, pval_Time,
        coef_Class (if class_col), pval_Class (if class_col),
        pval_interaction (if class_col), padj_Time (BH)
    """
    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        st.error("statsmodels is required for LMM. Install with: pip install statsmodels")
        return pd.DataFrame()

    df = df.copy()
    # Ensure time is numeric if possible
    try:
        df[time_col] = pd.to_numeric(df[time_col])
    except Exception:
        df[time_col] = pd.Categorical(df[time_col]).codes

    features = feature_cols[:max_features]
    rows = []
    progress = st.progress(0, text="Fitting linear mixed models…")

    for k, feat in enumerate(features):
        progress.progress((k + 1) / len(features),
                          text=f"LMM: {feat} ({k+1}/{len(features)})")
        try:
            sub = df[[subject_col, time_col, feat]
                      + ([class_col] if class_col else [])].dropna()
            sub = sub.rename(columns={feat: "_y", subject_col: "_subj",
                                      time_col: "_t"})
            if class_col:
                sub = sub.rename(columns={class_col: "_cls"})
                formula = "_y ~ _t + C(_cls) + _t:C(_cls)"
            else:
                formula = "_y ~ _t"

            md  = smf.mixedlm(formula, sub, groups=sub["_subj"])
            mdf = md.fit(method="lbfgs", warn_convergence=False)

            row = {"feature": feat}
            # Time coefficient
            t_key = [k for k in mdf.params.index if "_t" in k and ":" not in k]
            if t_key:
                row["coef_Time"] = round(mdf.params[t_key[0]], 6)
                row["se_Time"]   = round(mdf.bse[t_key[0]], 6)
                row["pval_Time"] = round(mdf.pvalues[t_key[0]], 6)
            # Class coefficient
            if class_col:
                cls_keys = [k for k in mdf.params.index if "_cls" in k and ":" not in k]
                inter_keys = [k for k in mdf.params.index if ":" in k]
                if cls_keys:
                    row["coef_Class"] = round(mdf.params[cls_keys[0]], 6)
                    row["pval_Class"] = round(mdf.pvalues[cls_keys[0]], 6)
                if inter_keys:
                    row["pval_interaction"] = round(min(mdf.pvalues[inter_keys]), 6)
            rows.append(row)
        except Exception:
            rows.append({"feature": feat, "coef_Time": np.nan,
                         "se_Time": np.nan, "pval_Time": np.nan})

    progress.empty()
    if not rows:
        return pd.DataFrame()

    res = pd.DataFrame(rows)
    if "pval_Time" in res.columns:
        valid = res["pval_Time"].notna()
        res.loc[valid, "padj_Time"] = multipletests(
            res.loc[valid, "pval_Time"], method="fdr_bh"
        )[1].round(6)
    return res.sort_values("pval_Time", na_position="last").reset_index(drop=True)


def run_rm_anova(
    df: pd.DataFrame,
    feature_cols: list,
    subject_col: str,
    time_col: str,
    max_features: int = 200,
) -> pd.DataFrame:
    """
    Repeated-measures ANOVA for each feature using pingouin.
    Returns feature, F, ddof1, ddof2, p-unc, p-GG-corr, padj (BH).
    """
    try:
        import pingouin as pg
    except ImportError:
        # st.error("pingouin & statsmodels required. pip install pingouin statsmodels")
        return pd.DataFrame()

    features = feature_cols[:max_features]
    rows = []
    progress = st.progress(0, text="Repeated-measures ANOVA…")

    for k, feat in enumerate(features):
        progress.progress((k + 1) / len(features),
                          text=f"RM-ANOVA: {feat} ({k+1}/{len(features)})")
        try:
            sub = df[[subject_col, time_col, feat]].dropna()
            aov = pg.rm_anova(data=sub, dv=feat,
                              within=time_col, subject=subject_col,
                              correction=True, detailed=False)
            r = aov.iloc[0]
            rows.append({
                "feature":  feat,
                "F":        round(r.get("F", np.nan), 4),
                "ddof1":    r.get("ddof1", np.nan),
                "ddof2":    r.get("ddof2", np.nan),
                "p_unc":    round(r.get("p-unc", np.nan), 6),
                "p_GG":     round(r.get("p-GG-corr", r.get("p-unc", np.nan)), 6),
                "eta_sq":   round(r.get("ng2", np.nan), 4),
            })
        except Exception:
            rows.append({"feature": feat, "F": np.nan, "p_unc": np.nan, "p_GG": np.nan})

    progress.empty()
    if not rows:
        return pd.DataFrame()

    res = pd.DataFrame(rows)
    valid = res["p_GG"].notna()
    if valid.any():
        res.loc[valid, "padj"] = multipletests(
            res.loc[valid, "p_GG"], method="fdr_bh"
        )[1].round(6)
    return res.sort_values("p_GG", na_position="last").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Volcano of LMM results
# ─────────────────────────────────────────────────────────────────────────────

def volcano_longitudinal(
    results: pd.DataFrame,
    coef_col: str = "coef_Time",
    pval_col: str = "padj_Time",
    pval_thr: float = 0.05,
    coef_thr: float = 0.0,
    top_n: int = 15,
) -> go.Figure:
    """Volcano plot of LMM / RM-ANOVA results."""
    df = results.copy()
    df["-log10p"] = -np.log10(df[pval_col].clip(lower=1e-300))
    df["sig"] = (df[pval_col] < pval_thr) & (df[coef_col].abs() > coef_thr)
    df["color"] = df.apply(
        lambda r: ("#ef4444" if r[coef_col] > coef_thr else "#3b82f6")
                  if r["sig"] else "#94a3b8", axis=1
    )

    fig = go.Figure()
    for color, label in [("#94a3b8", "NS"), ("#ef4444", "↑ over time"), ("#3b82f6", "↓ over time")]:
        sub = df[df["color"] == color]
        fig.add_trace(go.Scatter(
            x=sub[coef_col], y=sub["-log10p"],
            mode="markers",
            marker=dict(color=color, size=6, opacity=0.75,
                        line=dict(width=0.4, color="white")),
            name=label,
            hovertemplate="<b>%{customdata}</b><br>Coef: %{x:.4f}<br>-log10(padj): %{y:.2f}<extra></extra>",
            customdata=sub["feature"],
        ))

    # Label top features
    top = df[df["sig"]].nsmallest(top_n, pval_col)
    for _, row in top.iterrows():
        fig.add_annotation(
            x=row[coef_col], y=row["-log10p"],
            text=f"<b>{row['feature']}</b>",
            showarrow=True, arrowhead=2, arrowsize=0.8,
            arrowcolor="#475569", ax=25, ay=-18,
            font=dict(size=9, color="#1e293b"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#cbd5e1", borderwidth=0.5,
        )

    fig.add_hline(y=-np.log10(pval_thr), line=dict(color="#f59e0b", width=1.2, dash="dash"))
    fig.update_layout(
        title="<b>Longitudinal Volcano Plot</b> — Feature significance over time",
        xaxis_title=f"Coefficient ({coef_col})",
        yaxis_title=f"-log₁₀(adjusted p-value)",
        plot_bgcolor="white", paper_bgcolor="white",
        height=480, font=dict(size=11),
        margin=dict(l=60, r=30, t=60, b=60),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9", zeroline=True, zerolinecolor="#e2e8f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Longitudinal heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_longitudinal_heatmap(
    df: pd.DataFrame,
    feature_cols: list,
    subject_col: str,
    time_col: str,
    top_n: int = 40,
    sort_by_variance: bool = True,
) -> go.Figure:
    """
    Heatmap: features (rows) × subjects@timepoints (cols), faceted by timepoint.
    """
    feats = feature_cols[:top_n]
    timepoints = sorted(df[time_col].dropna().unique(),
                        key=lambda x: (str(x).lstrip("TtWwDdMm"), x))
    ncols = len(timepoints)
    fig   = make_subplots(
        rows=1, cols=ncols,
        subplot_titles=[f"<b>{t}</b>" for t in timepoints],
        shared_yaxes=True, horizontal_spacing=0.02,
    )

    # Z-score across all timepoints for consistent colour scale
    wide = df.set_index(subject_col)[feats]
    mu, sd = wide.mean(), wide.std().replace(0, 1)
    colorscale = "RdBu_r"

    for ci, tp in enumerate(timepoints, start=1):
        sub  = df[df[time_col] == tp].set_index(subject_col)[feats]
        zval = ((sub - mu) / sd).T       # features × subjects
        subjects = zval.columns.astype(str).tolist()

        if sort_by_variance and ci == 1:
            var_order = zval.var(axis=1).sort_values(ascending=False).index
            feats = [f for f in var_order if f in feats] + [f for f in feats if f not in var_order]
            zval = zval.loc[[f for f in feats if f in zval.index]]

        fig.add_trace(go.Heatmap(
            z=zval.values, x=subjects, y=zval.index.tolist(),
            colorscale=colorscale,
            zmin=-3, zmax=3,
            showscale=(ci == ncols),
            colorbar=dict(title="Z-score", thickness=10, len=0.6),
            hovertemplate="Feature: %{y}<br>Subject: %{x}<br>Z-score: %{z:.2f}<extra></extra>",
        ), row=1, col=ci)

    fig.update_layout(
        title="<b>Longitudinal Heatmap</b> — Z-scored features by timepoint",
        height=max(350, 14 * len(feats) + 100),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(size=10),
        margin=dict(l=120, r=60, t=80, b=40),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Summary table
# ─────────────────────────────────────────────────────────────────────────────

def summarise_longitudinal(
    results: pd.DataFrame,
    method: str = "LMM",
    pval_col: str = "padj_Time",
    coef_col: str = "coef_Time",
    pval_thr: float = 0.05,
) -> pd.DataFrame:
    """Return a filtered + formatted summary of significant features."""
    df = results.copy()
    if pval_col not in df.columns:
        return df
    sig = df[df[pval_col] < pval_thr].copy()
    if coef_col in sig.columns:
        sig["direction"] = sig[coef_col].apply(
            lambda v: "↑ increasing" if v > 0 else "↓ decreasing"
        )
    sig.insert(0, "method", method)
    return sig.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Main Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

def render_longitudinal_tab(df: pd.DataFrame) -> None:
    """
    Full Streamlit UI for longitudinal omics analysis.
    Called inside a `with tab:` block in Profiler.py.
    """
    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown("""
<div style='background:linear-gradient(135deg,#fffbeb,#fef3c7);border-left:4px solid #d97706;
border-radius:8px;padding:10px 16px;margin-bottom:14px;'>
<span style='color:#92400e;font-size:0.82rem;font-weight:700;'></span>
Linear mixed-effects models (LMM) and repeated-measures ANOVA are available. 
Standard cross-sectional ML models (Supervised ML tab) treat each row independently.</span>
</div>
""", unsafe_allow_html=True)

    if df is None:
        st.info("👈 No data loaded. Load a longitudinal dataset from **Step 5** in the sidebar.")
        return

    # ── Detect structure ────────────────────────────────────────────────────
    sc = st.session_state.get("long_subj_col_confirmed")
    tc = st.session_state.get("long_time_col_confirmed")
    info = validate_longitudinal_df(df, sc, tc)

    if info["warnings"]:
        for w in info["warnings"]:
            st.warning(w)
    if not info["is_valid"]:
        st.error("Cannot run longitudinal analysis — check the warnings above.")
        return

    subject_col  = info["subject_col"]
    time_col     = info["time_col"]
    class_col    = info["class_col"]
    feature_cols = info["feature_cols"]
    timepoints   = info["timepoints"]

    # ── Dataset summary card ─────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Subjects",    info["n_subjects"])
    c2.metric("Timepoints",  info["n_timepoints"])
    c3.metric("Features",    len(feature_cols))
    c4.metric("Groups",      df[class_col].nunique() if class_col else "—")
    st.markdown("---")

    # ── Sub-tabs ─────────────────────────────────────────────────────────────
    tab_traj, tab_delta, tab_lmm, tab_rmanova, tab_heatmap = st.tabs([
        "📈 Trajectories",
        "Δ Delta Analysis",
        "🧮 Linear Mixed Model",
        "🧮 RM-ANOVA",
        "🔥 Heatmap",
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — TRAJECTORIES
    # ════════════════════════════════════════════════════════════════════════
    with tab_traj:
        st.markdown("#### Individual trajectories + group mean ± SD")
        col_a, col_b, col_c = st.columns([3, 1, 1])
        with col_a:
            selected_feat = st.selectbox(
                "Feature to plot",
                feature_cols,
                key="long_traj_feat",
                help="Choose one feature to visualise its trajectory over time."
            )
        with col_b:
            show_ind = st.checkbox("Show individuals", value=True, key="long_show_ind")
        with col_c:
            normalize = st.checkbox("Normalize to T0", value=False, key="long_norm_t0",
                                    help="Divide each subject's value by its T0 value (fold-change).")

        fig = plot_trajectory(
            df, selected_feat, subject_col, time_col,
            class_col=class_col,
            show_individual=show_ind,
            show_mean=True,
            normalize=normalize,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Multi-feature overview (top 12 by variance)")
        col_m1, col_m2 = st.columns([2, 1])
        with col_m1:
            n_top = st.slider("Number of features to show", 3, 12, 6, key="long_multi_n")
        with col_m2:
            norm_multi = st.checkbox("Normalize to T0", key="long_norm_multi")

        variances   = df[feature_cols].var().sort_values(ascending=False)
        top_feats   = variances.head(n_top).index.tolist()
        fig_multi   = plot_multi_trajectory(
            df, top_feats, subject_col, time_col,
            class_col=class_col, normalize=norm_multi,
        )
        st.plotly_chart(fig_multi, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — DELTA ANALYSIS
    # ════════════════════════════════════════════════════════════════════════
    with tab_delta:
        st.markdown("#### Δ between two timepoints")
        st.caption(
            "Compute per-subject change between a reference and a target timepoint. "
            "The resulting Δ-matrix can then be analysed with standard ML."
        )
        col_t0, col_t1, col_meth = st.columns(3)
        with col_t0:
            tp_ref = st.selectbox("Reference timepoint (T0)",
                                  timepoints, index=0, key="long_tp_ref")
        with col_t1:
            tp_tgt = st.selectbox("Target timepoint (T1)",
                                  timepoints, index=min(1, len(timepoints)-1), key="long_tp_tgt")
        with col_meth:
            delta_method = st.selectbox(
                "Method",
                ["difference", "log2fc", "pct_change"],
                key="long_delta_meth",
                help="difference = T1−T0 · log2fc = log₂(T1/T0) · pct_change = (T1−T0)/|T0| × 100"
            )

        if tp_ref == tp_tgt:
            st.warning("Reference and target timepoints must be different.")
        else:
            df_delta = delta_features(
                df, feature_cols, subject_col, time_col,
                t0=tp_ref, t1=tp_tgt, method=delta_method
            )
            st.success(f"Δ matrix: {df_delta.shape[0]} subjects × {df_delta.shape[1]-2} features")
            st.dataframe(df_delta.head(10), use_container_width=True)

            # Quick bar chart: top changing features
            delta_feats = [c for c in df_delta.columns if c.startswith("Δ_")]
            means   = df_delta[delta_feats].mean().sort_values(key=abs, ascending=False)
            top10   = means.head(10)
            fig_bar = go.Figure(go.Bar(
                x=top10.values, y=top10.index,
                orientation="h",
                marker_color=["#ef4444" if v > 0 else "#3b82f6" for v in top10.values],
            ))
            fig_bar.update_layout(
                title=f"<b>Top 10 features by mean Δ</b> ({tp_ref} → {tp_tgt})",
                xaxis_title=f"Mean Δ ({delta_method})",
                yaxis=dict(autorange="reversed"),
                height=360, plot_bgcolor="white", paper_bgcolor="white",
                font=dict(size=11), margin=dict(l=150, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Download Δ matrix
            csv_buf = df_delta.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download Δ matrix (CSV)",
                csv_buf,
                file_name=f"delta_{tp_ref}_vs_{tp_tgt}.csv",
                mime="text/csv",
                key="dl_delta_csv",
            )
            st.info(
                "💡 **Tip:** Load the Δ matrix into Profiler's **Supervised ML** tab "
                "to classify responders vs non-responders based on their change over time."
            )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — LINEAR MIXED MODEL
    # ════════════════════════════════════════════════════════════════════════
    with tab_lmm:
        st.markdown("#### Linear Mixed-Effects Model")
        st.markdown(
            "Model: `feature ~ Time [+ Class + Time×Class] + (1 | Subject_ID)`  \n"
            "Tests whether each feature changes significantly over time, "
            "controlling for repeated measures within subjects."
        )

        col_lm1, col_lm2, col_lm3 = st.columns(3)
        with col_lm1:
            _lmm_min = min(10, len(feature_cols))
            _lmm_step = max(1, min(10, len(feature_cols) // 2))
            lmm_max = st.number_input(
                "Max features to test",
                min_value=_lmm_min,
                max_value=len(feature_cols),
                value=min(200, len(feature_cols)),
                step=_lmm_step,
                key="lmm_max_feat",
                help="LMM is slow for large feature sets — limit to the most variable."
            )
        with col_lm2:
            lmm_padj = st.number_input(
                "Adj. p-value threshold", min_value=0.001, max_value=0.5,
                value=0.05, step=0.01, key="lmm_padj",
                format="%.3f"
            )
        with col_lm3:
            lmm_use_class = st.checkbox(
                "Include group effect", value=bool(class_col),
                key="lmm_use_class",
                help="Add Class + Time×Class interaction term."
            )

        # Pre-filter to most variable features
        var_order = df[feature_cols].var().sort_values(ascending=False)
        top_feats = var_order.head(lmm_max).index.tolist()
        st.caption(f"Testing top {len(top_feats)} features by variance.")

        if st.button("🚀 Run LMM", key="run_lmm_btn", type="primary", use_container_width=True):
            with st.spinner("Fitting LMM (this may take a minute for large datasets)…"):
                lmm_res = run_lmm(
                    df, top_feats, subject_col, time_col,
                    class_col=class_col if lmm_use_class else None,
                    max_features=lmm_max,
                )
                st.session_state["lmm_results"] = lmm_res

        if "lmm_results" in st.session_state and st.session_state["lmm_results"] is not None:
            lmm_res = st.session_state["lmm_results"]
            sig = lmm_res[lmm_res.get("padj_Time", lmm_res.get("pval_Time", pd.Series(1))) < lmm_padj]
            st.success(
                f"**{len(sig)}** significant features (adj. p < {lmm_padj}) "
                f"out of {len(lmm_res)} tested."
            )

            col_res1, col_res2 = st.columns([3, 2])
            with col_res1:
                st.dataframe(lmm_res.head(30), use_container_width=True)
            with col_res2:
                if "coef_Time" in lmm_res.columns and "padj_Time" in lmm_res.columns:
                    fig_v = volcano_longitudinal(lmm_res, pval_thr=lmm_padj)
                    st.plotly_chart(fig_v, use_container_width=True)

            # Trajectory of top significant feature
            if len(sig) > 0:
                top_sig = sig.iloc[0]["feature"]
                st.markdown(f"**Trajectory of top hit: `{top_sig}`**")
                st.plotly_chart(
                    plot_trajectory(df, top_sig, subject_col, time_col, class_col),
                    use_container_width=True
                )

            # Download
            st.download_button(
                "⬇️ Download LMM results (CSV)",
                lmm_res.to_csv(index=False).encode(),
                file_name="lmm_longitudinal_results.csv",
                mime="text/csv",
                key="dl_lmm_csv",
            )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — RM-ANOVA
    # ════════════════════════════════════════════════════════════════════════
    with tab_rmanova:
        st.markdown("#### Repeated-Measures ANOVA")
        st.markdown(
            "One-way within-subjects ANOVA with Greenhouse-Geisser correction. "
            "Tests whether each feature differs across timepoints. "
            "Requires **pingouin** (`pip install pingouin`)."
        )

        col_ra1, col_ra2 = st.columns(2)
        with col_ra1:
            _rm_min = min(10, len(feature_cols))
            _rm_step = max(1, min(10, len(feature_cols) // 2))
            rm_max = st.number_input(
                "Max features to test",
                min_value=_rm_min,
                max_value=len(feature_cols),
                value=min(200, len(feature_cols)),
                step=_rm_step,
                key="rma_max_feat",
            )
        with col_ra2:
            rm_padj = st.number_input(
                "Adj. p-value threshold", min_value=0.001, max_value=0.5,
                value=0.05, step=0.01, key="rma_padj", format="%.3f"
            )

        var_order2 = df[feature_cols].var().sort_values(ascending=False)
        top_feats2 = var_order2.head(rm_max).index.tolist()

        if st.button("🚀 Run RM-ANOVA", key="run_rmanova_btn", type="primary", use_container_width=True):
            with st.spinner("Running repeated-measures ANOVA…"):
                rm_res = run_rm_anova(df, top_feats2, subject_col, time_col, max_features=rm_max)
                st.session_state["rm_anova_results"] = rm_res

        if "rm_anova_results" in st.session_state and st.session_state["rm_anova_results"] is not None:
            rm_res = st.session_state["rm_anova_results"]
            padj_col = "padj" if "padj" in rm_res.columns else "p_GG"
            sig_rm   = rm_res[rm_res[padj_col] < rm_padj]
            st.success(
                f"**{len(sig_rm)}** significant features (adj. p < {rm_padj}) "
                f"out of {len(rm_res)} tested."
            )
            st.dataframe(rm_res.head(30), use_container_width=True)

            # F-value barplot of top 15
            if "F" in rm_res.columns:
                top15  = rm_res.head(15)
                fig_f  = go.Figure(go.Bar(
                    x=top15["F"], y=top15["feature"],
                    orientation="h",
                    marker_color=["#6366f1" if p < rm_padj else "#cbd5e1"
                                  for p in top15[padj_col]],
                    text=top15[padj_col].round(4).astype(str),
                    textposition="outside",
                ))
                fig_f.update_layout(
                    title="<b>Top 15 features — F-statistic (RM-ANOVA)</b>",
                    xaxis_title="F-statistic",
                    yaxis=dict(autorange="reversed"),
                    height=380, plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(size=11), margin=dict(l=150, r=60, t=50, b=40),
                )
                st.plotly_chart(fig_f, use_container_width=True)

            st.download_button(
                "⬇️ Download RM-ANOVA results (CSV)",
                rm_res.to_csv(index=False).encode(),
                file_name="rmanova_longitudinal_results.csv",
                mime="text/csv",
                key="dl_rmanova_csv",
            )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 5 — HEATMAP
    # ════════════════════════════════════════════════════════════════════════
    with tab_heatmap:
        st.markdown("#### Z-scored heatmap faceted by timepoint")
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            heat_n = st.slider("Number of features", 5, 60, 30, key="long_heat_n")
        with col_h2:
            heat_sort = st.checkbox("Sort by variance", value=True, key="long_heat_sort")

        fig_heat = plot_longitudinal_heatmap(
            df, feature_cols, subject_col, time_col,
            top_n=heat_n, sort_by_variance=heat_sort,
        )
        st.plotly_chart(fig_heat, use_container_width=True)
