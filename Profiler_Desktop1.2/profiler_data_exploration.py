
"""
Software Name: Profiler
Module name : data_exploration
Author: Yanis Zirem
Email : yanis.zirem@yahoo.com / yanis.zirem@univ-lille.fr
Creation Date: 15/01/2025
Last Updated: 05/03/2026
Version: 1.2.0

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

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from venn import venn
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import gc
from upsetplot import UpSet, from_memberships
import pandas as pd

def _capture_matplotlib(fig, key: str):
    """Convert a Matplotlib figure to base64 PNG and store for the HTML report."""
    if fig is not None:
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=130)
            buf.seek(0)
            st.session_state[f"_report_{key}"] = ("b64", base64.b64encode(buf.read()).decode())
        except Exception:
            pass




def _capture_plotly(fig, key: str):
    """Store a Plotly figure in session_state for the HTML report."""
    if fig is not None:
        st.session_state[f"_report_{key}"] = ("plotly", fig)


def plot_feature_distribution(data, feature, class_colors, histfunc='sum', capture_name=None):
    if feature == 'Class':
        fig = px.histogram(
            data,
            x='Class',
            title=f"Distribution of {feature}",
            color='Class',
            color_discrete_map=class_colors
        )
    else:
        fig = px.histogram(
            data,
            x='Class',
            y=feature,
            histfunc=histfunc,
            title=f"{histfunc.capitalize()} of {feature} intensities by class",
            color='Class',
            color_discrete_map=class_colors
        )

    # fig.update_layout(
    #     title_font_size=22,
    #     font=dict(color="black", size=18),
    #     legend=dict(
    #         title_font=dict(size=18, color='black'),
    #         font=dict(size=16, color='black')
    #     ),
    #     xaxis=dict(
    #         title='Class',
    #         titlefont=dict(size=20, color='black'),
    #         tickfont=dict(size=16, color='black')
    #     ),
    #     yaxis=dict(
    #         title='Intensity' if feature != 'Class' else 'Count',
    #         titlefont=dict(size=20, color='black'),
    #         tickfont=dict(size=16, color='black')
    #     ),
    #     plot_bgcolor='white',
    #     paper_bgcolor='white'
    # )
    fig.update_layout(
        title=dict(
            text=fig.layout.title.text,
            font=dict(size=22)
        ),
        font=dict(color="black", size=18),
        legend=dict(
            title=dict(font=dict(size=18, color='black')),
            font=dict(size=16, color='black')
        ),
        xaxis=dict(
            title=dict(text='Class', font=dict(size=20, color='black')),
            tickfont=dict(size=16, color='black')
        ),
        yaxis=dict(
            title=dict(text='Intensity' if feature != 'Class' else 'Count', font=dict(size=20, color='black')),
            tickfont=dict(size=16, color='black')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    if capture_name:
        _capture_plotly(fig, capture_name)
    return fig







def plot_multiple_features_line(data, features, feature_colors, histfunc='sum', error_type=None, capture_name=None):

    melted = data.melt(id_vars='Class', value_vars=features, var_name='Feature', value_name='Value')

    if histfunc == 'percentage':
        melted = melted.groupby(['Class', 'Feature'])['Value'].sum().reset_index()
        total_by_class = melted.groupby('Class')['Value'].transform('sum')
        melted['Value'] = (melted['Value'] / total_by_class) * 100
        error_df = None  # No error bars for percentage
    else:
        agg_df = melted.groupby(['Class', 'Feature'])['Value'].agg(histfunc).reset_index()
        if error_type == 'sem':
            error_df = melted.groupby(['Class', 'Feature'])['Value'].sem().reset_index().rename(columns={'Value': 'Error'})
        elif error_type == 'std':
            error_df = melted.groupby(['Class', 'Feature'])['Value'].std().reset_index().rename(columns={'Value': 'Error'})
        else:
            error_df = None
        melted = agg_df

    fig = go.Figure()

    for cls in melted['Class'].unique():
        df_c = melted[melted['Class'] == cls].sort_values(by='Feature')
        error_c = None
        if error_df is not None:
            error_c = error_df[error_df['Class'] == cls].sort_values(by='Feature')['Error']

        fig.add_trace(go.Scatter(
            x=df_c['Feature'],
            y=df_c['Value'],
            mode='lines+markers',
            name=f"Class {cls}",
            line=dict(color=feature_colors.get(cls, '#333333')),
            error_y=dict(
                type='data',
                array=error_c if error_c is not None else None,
                visible=error_c is not None
            )
        ))

    fig.update_layout(
        title=f"{'Percentage' if histfunc == 'percentage' else histfunc.capitalize()} Line Chart",
        font=dict(color='black', size=16),
        title_font=dict(size=20, color='black'),
        xaxis=dict(
            title='Feature',
            title_font=dict(size=16, color='black'),
            tickfont=dict(size=14, color='black'),
            linecolor='black',
            showline=True
        ),
        yaxis=dict(
            title=f"{'Percentage (%)' if histfunc == 'percentage' else histfunc.capitalize()}",
            title_font=dict(size=16, color='black'),
            tickfont=dict(size=14, color='black'),
            linecolor='black',
            showline=True
        ),
        legend=dict(
            title_font=dict(size=16, color='black'),
            font=dict(size=14, color='black')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    if capture_name:
        _capture_plotly(fig, capture_name)
    # la figure (pas plt)
    return fig

def plot_multiple_features_radar(data, features, feature_colors, histfunc='sum', capture_name=None):
    melted = data.melt(id_vars='Class', value_vars=features, var_name='Feature', value_name='Value')

    if histfunc == 'percentage':
        melted = melted.groupby(['Class', 'Feature'])['Value'].sum().reset_index()
        total_by_class = melted.groupby('Class')['Value'].transform('sum')
        melted['Value'] = (melted['Value'] / total_by_class) * 100
    else:
        melted = melted.groupby(['Class', 'Feature'])['Value'].agg(histfunc).reset_index()

    fig = go.Figure()

    for cls in melted['Class'].unique():
        df_c = melted[melted['Class'] == cls].sort_values(by='Feature')
        fig.add_trace(go.Scatterpolar(
            r=df_c['Value'],
            theta=df_c['Feature'],
            fill='toself',
            name=f"Class {cls}",
            line=dict(color=feature_colors.get(cls, '#333333'))
        ))

    fig.update_layout(
        title=f"{'Percentage' if histfunc == 'percentage' else histfunc.capitalize()} Radar Chart",
        font=dict(color='black', size=16),
        title_font=dict(size=20, color='black'),
        polar=dict(
            radialaxis=dict(visible=True, color='black', tickfont=dict(color='black')),
            angularaxis=dict(tickfont=dict(color='black'))
        ),
        legend=dict(
            title_font=dict(size=16, color='black'),
            font=dict(size=14, color='black')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    if capture_name:
        _capture_plotly(fig, capture_name)
    
    return fig

def plot_multiple_features_distribution(data, features, feature_colors, histfunc='sum', capture_name=None):
    melted = data.melt(id_vars='Class', value_vars=features, var_name='Feature', value_name='Value')

    if histfunc == 'percentage':
        melted = melted.groupby(['Class', 'Feature'])['Value'].sum().reset_index()
        total_by_class = melted.groupby('Class')['Value'].transform('sum')
        melted['Value'] = (melted['Value'] / total_by_class) * 100
    else:
        melted = melted.groupby(['Class', 'Feature'])['Value'].agg(histfunc).reset_index()

    fig = px.bar(
        melted,
        x='Class',
        y='Value',
        color='Feature',
        barmode='group',
        title=f"{'Percentage' if histfunc == 'percentage' else histfunc.capitalize()} Bar Chart",
        color_discrete_map=feature_colors,
        labels={
            'Class': 'Class',
            'Value': f"{'Percentage (%)' if histfunc == 'percentage' else histfunc.capitalize()}",
            'Feature': 'Feature'
        }
    )

    unique_classes = melted['Class'].nunique()

    if unique_classes <= 2:
        fig.update_layout(
            width=600,
            height=600,
            bargap=0.6
        )
        fig.update_traces(width=0.1)
    else:
        fig.update_layout(
            width=900,
            height=600,
            bargap=0.2
        )
        fig.update_traces(width=0.2)

    fig.update_layout(
        font=dict(color='black', size=22),
        title_font=dict(size=26, color='black'),
        xaxis=dict(
            title_font=dict(size=26, color='black'),
            tickfont=dict(size=22, color='black'),
            linecolor='black',
            showline=True,
            showgrid=False
        ),
        yaxis=dict(
            title_font=dict(size=26, color='black'),
            tickfont=dict(size=22, color='black'),
            linecolor='black',
            showline=True,
            showgrid=False
        ),
        legend=dict(
            title_font=dict(size=26, color='black'),
            font=dict(size=22, color='black')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    if capture_name:
        _capture_plotly(fig, capture_name)
    return fig





import io
import base64
import gc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.colors import to_rgba
import matplotlib
matplotlib.use("Agg")


# ─────────────────────────────────────────────────────────────────────────────
# Capture helpers
# ─────────────────────────────────────────────────────────────────────────────

def _capture_matplotlib(fig, capture_name):
    """Capture a matplotlib figure as base64 PNG and store for HTML report."""
    if fig is not None:
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
            buf.seek(0)
            st.session_state[f"_report_{capture_name}"] = ("b64", base64.b64encode(buf.read()).decode())
        except Exception:
            pass


def _capture_plotly(fig, capture_name):
    """Store a Plotly figure in session_state for the HTML report."""
    if fig is not None:
        st.session_state[f"_report_{capture_name}"] = ("plotly", fig)


# ─────────────────────────────────────────────────────────────────────────────
# VENN DIAGRAM — interactive Plotly
# ─────────────────────────────────────────────────────────────────────────────

def _hex_to_rgba(hex_color, alpha=0.28):
    r, g, b, _ = to_rgba(hex_color)
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha})"


def _circle_points(cx, cy, r, n=300):
    t = np.linspace(0, 2 * np.pi, n)
    return cx + r * np.cos(t), cy + r * np.sin(t)


_VENN_LAYOUTS = {
    2: [(-0.35, 0),  (0.35, 0)],
    3: [(0, 0.30),   (-0.30, -0.20), (0.30, -0.20)],
    4: [(-0.30, 0.25),(0.30, 0.25),  (-0.30,-0.25),(0.30,-0.25)],
    5: [(0, 0.40),(0.38, 0.12),(0.24,-0.35),(-0.24,-0.35),(-0.38, 0.12)],
    6: [(0, 0.42),(0.37, 0.21),(0.37,-0.21),(0,-0.42),(-0.37,-0.21),(-0.37, 0.21)],
}
_VENN_RADIUS = {2: 0.55, 3: 0.48, 4: 0.40, 5: 0.36, 6: 0.32}


def _compute_set_intersections(detected_features):
    """Return dict: frozenset(classes) → set of features exclusive to that exact combination."""
    classes = list(detected_features.keys())
    n = len(classes)
    result = {}
    for mask in range(1, 2**n):
        members = frozenset(classes[i] for i in range(n) if mask >> i & 1)
        in_all = set.intersection(*(detected_features[c] for c in members))
        not_in_others = (
            set.union(*(detected_features[c] for c in classes if c not in members))
            if len(members) < n else set()
        )
        result[members] = in_all - not_in_others
    return result


def plot_venn_diagram(data, class_column, color_palette, source=None, capture_name=None):
    if source != "Raw Data":
        data = data.replace(0, np.nan)

    # ⚡ Vectorized: compute notna mask once, then slice per class
    _feat_cols = [c for c in data.columns if c != class_column]
    _notna_mask = data[_feat_cols].notna()  # bool DataFrame, computed once
    detected_features = {
        cn: set(_feat_cols[i] for i in range(len(_feat_cols))
                if _notna_mask.loc[data[class_column] == cn].any().iloc[i])
        for cn in data[class_column].unique()
    }

    num_classes = len(detected_features)
    if num_classes < 2 or num_classes > 6:
        st.error("⚠️ Venn diagram supports only 2 to 6 classes.")
        return None

    classes = list(detected_features.keys())

    # ── Compute intersections & store in session_state ────────────────────────
    exclusive_features = {}
    for cls in classes:
        other = set().union(*[detected_features[c] for c in classes if c != cls])
        exclusive_features[cls] = detected_features[cls] - other
    common_features = set.intersection(*detected_features.values())
    st.session_state["common_features"] = common_features
    st.session_state["exclusive_features"] = exclusive_features

    intersections = _compute_set_intersections(detected_features)

    # ── Build Plotly figure ───────────────────────────────────────────────────
    positions = _VENN_LAYOUTS[num_classes]
    radius    = _VENN_RADIUS[num_classes]

    fig = go.Figure()
    fig.update_layout(
        width=None, height=560,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(visible=False, range=[-1.1, 1.1]),
        yaxis=dict(visible=False, range=[-1.0, 1.0], scaleanchor="x"),
        margin=dict(l=20, r=20, t=55, b=20),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.08,
            xanchor="center", x=0.5, font=dict(size=12),
        ),
        title=dict(
            text=f"<b>Venn Diagram</b>{' — ' + source if source else ''}",
            x=0.5, font=dict(size=16, color="#222"),
        ),
        font=dict(family="Arial"),
    )

    # Draw filled circles
    for i, (cls, (cx, cy)) in enumerate(zip(classes, positions)):
        color      = color_palette.get(cls, "#888888")
        fill_color = _hex_to_rgba(color, alpha=0.28)
        xs, ys = _circle_points(cx, cy, radius)
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            fill="toself", fillcolor=fill_color,
            line=dict(color=color, width=2.5),
            mode="lines", name=cls,
            hoverinfo="skip", showlegend=True,
        ))
        # Class label outside circle
        lx = max(-0.95, min(0.95, cx * 1.55))
        ly = max(-0.88, min(0.88, cy * 1.55))
        n_feat = len(detected_features[cls])
        fig.add_annotation(
            x=lx, y=ly,
            text=f"<b>{cls}</b><br><span style='font-size:11px;color:#555'>{n_feat} features</span>",
            showarrow=False, font=dict(size=13, color=color), align="center",
        )

    # Intersection hover points
    for members, feats in intersections.items():
        if not feats:
            continue
        idxs = [classes.index(c) for c in members]
        cx = np.mean([positions[i][0] for i in idxs])
        cy = np.mean([positions[i][1] for i in idxs])
        label_lines = [
            f"<b>{'∩ '.join(sorted(members))}</b>",
            f"<b>{len(feats)} features</b>", "─────────",
        ] + sorted(feats)[:20]
        if len(feats) > 20:
            label_lines.append(f"… +{len(feats)-20} more")
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy],
            mode="markers+text",
            marker=dict(size=1, color="rgba(0,0,0,0)"),
            text=[str(len(feats))],
            textfont=dict(size=14, color="#111", family="Arial Black"),
            textposition="middle center",
            hovertemplate="<br>".join(label_lines) + "<extra></extra>",
            showlegend=False, name="",
        ))

    # ── Render ────────────────────────────────────────────────────────────────
    st.plotly_chart(fig, use_container_width=True, config={
        "displayModeBar": False,
        "toImageButtonOptions": {"format": "png", "scale": 3},
    })

    # ── Intersection tables ───────────────────────────────────────────────────
    _show_intersection_tables(detected_features, intersections, exclusive_features, common_features)

    # ── Capture for HTML report ───────────────────────────────────────────────
    if capture_name:
        _capture_plotly(fig, capture_name)

    return fig


def _show_intersection_tables(detected_features, intersections, exclusive_features, common_features):
    """Display downloadable tables for intersections, exclusives, and commons."""
    classes = list(detected_features.keys())

    # --- Exclusive features table ---
    excl_rows = []
    for cls, feats in exclusive_features.items():
        for f in sorted(feats):
            excl_rows.append({"Class": cls, "Exclusive Feature": f})
    if excl_rows:
        df_excl = pd.DataFrame(excl_rows)
        st.markdown(f"**📋 Exclusive features ({len(excl_rows)} total)**")
        st.dataframe(df_excl, use_container_width=True, height=220)
        csv = df_excl.to_csv(index=False, sep=";", encoding="utf-8-sig")
        st.download_button("📥 Download exclusive features", csv.encode("utf-8-sig"),
                           "exclusive_features.csv", "text/csv", use_container_width=True)

    # --- Common to ALL classes ---
    if common_features:
        df_common = pd.DataFrame(sorted(common_features), columns=["Common Feature (all classes)"])
        st.markdown(f"**📋 Common to ALL classes ({len(common_features)} features)**")
        st.dataframe(df_common, use_container_width=True, height=220)
        csv = df_common.to_csv(index=False, sep=";", encoding="utf-8-sig")
        st.download_button("📥 Download common features", csv.encode("utf-8-sig"),
                           "common_features_all.csv", "text/csv", use_container_width=True)

    # --- All pairwise / multi intersections ---
    inter_rows = []
    for members, feats in intersections.items():
        if len(members) < 2 or not feats:
            continue
        label = " ∩ ".join(sorted(members))
        for f in sorted(feats):
            inter_rows.append({"Intersection": label, "N classes": len(members), "Feature": f})
    if inter_rows:
        df_inter = pd.DataFrame(inter_rows)
        st.markdown(f"**📋 All pairwise intersections ({len(inter_rows)} entries)**")
        st.dataframe(df_inter, use_container_width=True, height=220)
        csv = df_inter.to_csv(index=False, sep=";", encoding="utf-8-sig")
        st.download_button("📥 Download intersections", csv.encode("utf-8-sig"),
                           "intersections.csv", "text/csv", use_container_width=True)


def _venn_capture_static(data, class_column, color_palette, source, capture_name):
    """Fallback: generate static matplotlib Venn for HTML report capture."""
    try:
        # ⚡ vectorized version
        _feat_cols_c = [c for c in data.columns if c != class_column]
        _nm = data[_feat_cols_c].notna()
        detected_features = {
            cn: set(c for c in _feat_cols_c if _nm.loc[data[class_column]==cn, c].any())
            for cn in data[class_column].unique()
        }
        colors = [color_palette.get(c, "#000") for c in detected_features]
        fig_mpl, ax = plt.subplots(figsize=(10, 8))
        venn(detected_features, cmap=colors, ax=ax)
        ax.set_title(f"Venn Diagram ({source})" if source else "Venn Diagram")
        _capture_matplotlib(fig_mpl, capture_name)
        plt.close(fig_mpl)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# UPSET PLOT — interactive Plotly
# ─────────────────────────────────────────────────────────────────────────────

def plot_upset(data, class_column, source=None, capture_name=None, class_colors=None):
    try:
        if source != "Raw Data":
            data = data.replace(0, np.nan)

        clean_data = data.drop(columns=["File", "RT", "Sum"], errors="ignore")
        if class_column not in clean_data.columns:
            st.error(f"Column '{class_column}' not found in the data.")
            return None

        clean_data[class_column] = clean_data[class_column].astype(str)

        # ⚡ Vectorized: pivot instead of groupby.apply
        _feat_c = [c for c in clean_data.columns if c != class_column]
        _notna = clean_data[_feat_c].notna()  # shape (n_samples, n_features)
        _classes_u = clean_data[class_column].unique()
        upset_input = pd.DataFrame(
            {cn: _notna.loc[clean_data[class_column] == cn].any(axis=0).astype(int)
             for cn in _classes_u},
            index=_feat_c
        )
        del _notna, _feat_c
        gc.collect()

        if upset_input.empty:
            st.warning("⚠️ No features available for UpSet plot.")
            return None

        classes = list(upset_input.columns)
        n_classes = len(classes)

        # ── Class color setup ─────────────────────────────────────────────────
        _class_colors = class_colors or {}
        default_palette = ["#1a1a2e","#e63946","#2a9d8f","#e9c46a","#f4a261","#264653","#8338ec"]
        class_color_map = {
            cls: _class_colors.get(cls, default_palette[i % len(default_palette)])
            for i, cls in enumerate(classes)
        }
        dot_inactive = "#e8e8e8"

        # ── Build subset counts ───────────────────────────────────────────────
        subset_dict = {}
        feature_map = {}   # key → list of features in that subset
        for feat, row in zip(upset_input.index, upset_input.values):
            key = tuple(int(v) for v in row)
            subset_dict[key] = subset_dict.get(key, 0) + 1
            feature_map.setdefault(key, []).append(feat)

        min_size = max(1, len(upset_input) // 500)
        subset_dict = {k: v for k, v in subset_dict.items() if v >= min_size}

        subsets = sorted(subset_dict.items(), key=lambda x: -x[1])
        n_subsets = len(subsets)

        if n_subsets == 0:
            st.warning("No subsets found after filtering.")
            return None

        counts = [v for _, v in subsets]
        keys   = [k for k, _ in subsets]

        # ── Plotly figure ─────────────────────────────────────────────────────
        bar_h    = 0.55
        matrix_h = 0.45

        fig = go.Figure()
        x_vals = list(range(n_subsets))

        # Hover texts for bars
        hover_texts = []
        for key, count in subsets:
            members  = [classes[i] for i, v in enumerate(key) if v]
            feats    = feature_map.get(key, [])
            feat_preview = ", ".join(sorted(feats)[:10])
            if len(feats) > 10:
                feat_preview += f" … +{len(feats)-10} more"
            hover_texts.append(
                f"<b>{'  ∩  '.join(members) if members else 'None'}</b><br>"
                f"<b>Count: {count}</b><br>"
                f"<span style='font-size:11px'>{feat_preview}</span>"
            )

        # Bar chart
        fig.add_trace(go.Bar(
            x=x_vals, y=counts,
            marker=dict(
                color=counts,
                colorscale=[[0, "#b8c6db"], [1, "#1a1a2e"]],
                line=dict(width=0),
            ),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_texts,
            showlegend=False,
            yaxis="y1", xaxis="x1",
        ))

        # Count labels on bars
        for xi, cnt in zip(x_vals, counts):
            fig.add_annotation(
                x=xi, y=cnt, text=str(cnt),
                showarrow=False, yref="y1", xref="x1",
                font=dict(size=9, color="#333"), yanchor="bottom",
            )

        # Dot matrix — colored by class
        for ci, cls in enumerate(classes):
            cls_col = class_color_map[cls]
            for xi, key in enumerate(keys):
                active = bool(key[ci])
                fig.add_trace(go.Scatter(
                    x=[xi], y=[ci],
                    mode="markers",
                    marker=dict(
                        size=12,
                        color=cls_col if active else dot_inactive,
                        line=dict(color="#aaa" if not active else cls_col, width=1),
                    ),
                    hoverinfo="skip", showlegend=False,
                    xaxis="x1", yaxis="y2",
                ))

        # Connecting lines — use color of the top active class
        for xi, key in enumerate(keys):
            active_rows = [ci for ci, v in enumerate(key) if v]
            if len(active_rows) >= 2:
                top_cls = classes[active_rows[0]]
                fig.add_shape(
                    type="line",
                    x0=xi, x1=xi, y0=min(active_rows), y1=max(active_rows),
                    line=dict(color=class_color_map[top_cls], width=3),
                    xref="x1", yref="y2",
                )

        # Layout
        plot_width  = min(1200, max(700, 80 + n_subsets * 44))
        plot_height = min(700, max(380, 200 + n_classes * 40))
        set_sizes   = [int(upset_input[cls].sum()) for cls in classes]

        fig.update_layout(
            width=plot_width, height=plot_height,
            paper_bgcolor="white", plot_bgcolor="white",
            bargap=0.25,
            title=dict(
                text=f"<b>UpSet Plot</b>{' — ' + source if source else ''}",
                x=0.5, font=dict(size=16, color="#222"),
            ),
            font=dict(family="Arial"),
            showlegend=False,
            margin=dict(l=110, r=30, t=60, b=20),
            yaxis=dict(
                domain=[matrix_h + 0.06, 1.0],
                title=dict(text="Intersection size", font=dict(size=11)),
                showgrid=True, gridcolor="#eee", zeroline=False,
                tickfont=dict(size=9),
            ),
            xaxis=dict(
                domain=[0, 1], showticklabels=False,
                showgrid=False, zeroline=False,
                range=[-0.5, n_subsets - 0.5],
            ),
            yaxis2=dict(
                domain=[0, matrix_h],
                tickmode="array",
                tickvals=list(range(n_classes)),
                ticktext=[f"{cls}  ({sz})" for cls, sz in zip(classes, set_sizes)],
                tickfont=dict(size=11, color="#222"),
                showgrid=False, zeroline=False,
                range=[-0.6, n_classes - 0.4],
            ),
        )

        del upset_input, clean_data
        gc.collect()

        st.plotly_chart(fig, use_container_width=True, config={
            "scrollZoom": False,
            "displayModeBar": True,
            "toImageButtonOptions": {"format": "png", "scale": 3},
        })

        # ── Intersection table ────────────────────────────────────────────────
        _show_upset_table(keys, counts, classes, feature_map)

        # ── Capture for HTML report ───────────────────────────────────────────
        if capture_name:
            _capture_plotly(fig, capture_name)

        return fig   # Plotly figure — caller must NOT call plt.close() on it

    except Exception as e:
        import traceback
        st.error(f"❌ Error while generating UpSet plot: {e}")
        st.text(traceback.format_exc())
        gc.collect()
        return None


def _show_upset_table(keys, counts, classes, feature_map):
    """Display downloadable table of all subsets with their features."""
    rows = []
    for key, count in zip(keys, counts):
        members = " ∩ ".join([classes[i] for i, v in enumerate(key) if v]) or "None"
        feats   = feature_map.get(key, [])
        for f in sorted(feats):
            rows.append({"Subset": members, "Count": count, "Feature": f})
    if not rows:
        return
    df = pd.DataFrame(rows)
    st.markdown(f"**📋 UpSet subset features ({len(df)} entries)**")
    st.dataframe(df, use_container_width=True, height=250)
    csv = df.to_csv(index=False, sep=";", encoding="utf-8-sig")
    st.download_button("📥 Download subset table", csv.encode("utf-8-sig"),
                       "upset_subsets.csv", "text/csv", use_container_width=True)

