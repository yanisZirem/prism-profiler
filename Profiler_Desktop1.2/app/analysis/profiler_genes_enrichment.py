"""
profiler_genes_enrichement.py
Enrichment analysis module for Profiler — 100% Plotly interactive.

Two modes in render_enrichment_tab():
  ORA  — Over-Representation Analysis via Enrichr / local GMT
  GSEA — Gene Set Enrichment Analysis via gseapy.prerank()

Key design decisions:
  - ORA gene inputs inside st.form (prevents live-rerender freeze)
  - GSEA fully outside any form (ranked list + database = live)
  - Shared _render_db_selector() for DB choice in both tabs
  - All plots: axes/labels/legend in large black text, high quality
  - ASCII sanitisation before every gseapy call (fixes codec errors)
"""

import gc
import re
import unicodedata
import tempfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import gseapy as gp
import networkx as nx
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════

_CSS = """
<style>
.enrich-section {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px 20px 14px 20px;
    margin-bottom: 16px;
}
.enrich-section-title {
    margin: 0 0 12px 0;
    font-size: 14px;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: 0.02em;
}
.badge-ok   { background:#dcfce7; color:#14532d; border-radius:6px;
              padding:4px 10px; font-size:12px; font-weight:700; display:inline-block; }
.badge-warn { background:#fef9c3; color:#713f12; border-radius:6px;
              padding:4px 10px; font-size:12px; font-weight:700; display:inline-block; }
.badge-err  { background:#fee2e2; color:#7f1d1d; border-radius:6px;
              padding:4px 10px; font-size:12px; font-weight:700; display:inline-block; }
</style>
"""

def _inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)

def _section(icon: str, title: str):
    st.markdown(
        f'<div class="enrich-section">'
        f'<p class="enrich-section-title">{icon}&nbsp;&nbsp;{title}</p>',
        unsafe_allow_html=True,
    )

def _section_end():
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ASCII SANITISATION  (fixes gseapy codec errors with Greek / special chars)
# ══════════════════════════════════════════════════════════════════════════════

def _ascii_safe(s: str) -> str:
    """
    Convert a string to pure ASCII:
      1. NFKD decomposition (e.g. é → e + combining accent)
      2. Drop non-ASCII combining characters
      3. Replace any remaining non-ASCII chars with '_'
    Preserves alphanumerics, hyphens, underscores, dots.
    """
    nfkd = unicodedata.normalize("NFKD", str(s))
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    return ascii_str.encode("ascii", errors="replace").decode("ascii").replace("?", "_")


def _sanitise_gene_list(genes: list) -> list:
    return [_ascii_safe(g) for g in genes if g and g.strip()]


def _sanitise_ranked_series(series: pd.Series) -> pd.Series:
    """Sanitise index (feature names) to ASCII and drop duplicates."""
    s = series.copy()
    s.index = [_ascii_safe(i) for i in s.index]
    s = s[~s.index.duplicated(keep="first")]
    return s


# ══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _capture_plotly(fig, key: str):
    if fig is not None:
        st.session_state[f"_report_{key}"] = ("plotly", fig)


# ── High-quality shared layout ────────────────────────────────────────────────
_FONT_FAMILY = "Arial Black, Arial, sans-serif"
_TICK_FONT   = dict(family=_FONT_FAMILY, size=13, color="#0f172a")
_TITLE_FONT  = dict(family=_FONT_FAMILY, size=16, color="#0f172a")
_AXIS_TITLE  = dict(family=_FONT_FAMILY, size=14, color="#0f172a")
_LEGEND_FONT = dict(family=_FONT_FAMILY, size=13, color="#0f172a")


def _layout(fig, title: str = "", height: int = 500) -> go.Figure:
    """Apply high-quality layout: black labels, large fonts, clean white bg."""
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=_TITLE_FONT, x=0.01),
        font=dict(family=_FONT_FAMILY, color="#0f172a"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=55, b=20),
        height=height,
        legend=dict(
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#cbd5e1",
            borderwidth=1.5,
            font=_LEGEND_FONT,
            title_font=_LEGEND_FONT,
        ),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor="#e2e8f0", gridwidth=1,
        linecolor="#0f172a", linewidth=1.5,
        tickfont=_TICK_FONT,
        title_font=_AXIS_TITLE,
        zeroline=False,
        showline=True,
        ticks="outside", ticklen=5, tickwidth=1.5, tickcolor="#0f172a",
    )
    fig.update_yaxes(
        showgrid=False,
        linecolor="#0f172a", linewidth=1.5,
        tickfont=_TICK_FONT,
        title_font=_AXIS_TITLE,
        showline=True,
        ticks="outside", ticklen=5, tickwidth=1.5, tickcolor="#0f172a",
    )
    return fig


def load_gene_sets() -> dict:
    gene_sets = gp.get_library_name()
    cats = {
        "KEGG": [], "Reactome": [],
        "GO Biological Process": [], "GO Molecular Function": [],
        "GO Cellular Component": [], "Drug": [], "Atlas": [], "Other": [],
    }
    for name in gene_sets:
        if "KEGG" in name:                               cats["KEGG"].append(name)
        elif "Reactome" in name:                         cats["Reactome"].append(name)
        elif "GO_Biological_Process" in name:            cats["GO Biological Process"].append(name)
        elif "GO_Molecular_Function" in name:            cats["GO Molecular Function"].append(name)
        elif "GO_Cellular_Component" in name:            cats["GO Cellular Component"].append(name)
        elif any(k in name for k in ("Drug","Pharm","Chem")): cats["Drug"].append(name)
        elif "Atlas" in name:                            cats["Atlas"].append(name)
        else:                                            cats["Other"].append(name)
    return cats


def load_gene_sets_offline() -> dict:
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# SHARED DB SELECTOR (identical widget set in ORA and GSEA)
# ══════════════════════════════════════════════════════════════════════════════

def _render_db_selector(tab_prefix: str):
    """
    Returns (db_source_str, gene_set_value, gmt_file_path, organism).
    Fully outside st.form — always live.
    """
    db_source = st.radio(
        "Database source",
        ["🌐 Online — Enrichr API", "📁 Offline — local .gmt"],
        horizontal=True,
        key=f"{tab_prefix}_db_source",
    )
    gene_set_value = None
    gmt_file_path  = None
    organism       = "Human"

    if "Offline" in db_source:
        gmt_folder = Path("gmt_databases")
        local_gmts = sorted(gmt_folder.glob("*.gmt")) if gmt_folder.exists() else []
        co1, co2   = st.columns(2)
        with co1:
            if local_gmts:
                choices = ["— select installed GMT —"] + [f.name for f in local_gmts]
                sel     = st.selectbox("Installed databases", choices,
                                       key=f"{tab_prefix}_local_gmt")
                if sel != choices[0]:
                    gmt_file_path  = gmt_folder / sel
                    gene_set_value = str(gmt_file_path)
            else:
                st.caption("No .gmt files found in `gmt_databases/`")
        with co2:
            up = st.file_uploader("Or upload a .gmt file", type=["gmt"],
                                  key=f"{tab_prefix}_upload_gmt")
            if up:
                gmt_file_path  = up
                gene_set_value = up
        if gene_set_value:
            name = getattr(gene_set_value, "name", None) or Path(str(gene_set_value)).name
            st.markdown(f'<span class="badge-ok">✓ GMT ready: {name}</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-warn">⚠ No GMT file selected</span>',
                        unsafe_allow_html=True)
    else:
        # Online
        c_btn, c_status = st.columns([2, 3])
        with c_btn:
            if st.button("📥 Load category list", key=f"{tab_prefix}_load_cats"):
                with st.spinner("Fetching from Enrichr API…"):
                    try:
                        cats = load_gene_sets()
                        if isinstance(cats, dict):
                            st.session_state["_enrich_categories"]  = cats
                            st.session_state["_enrich_cats_loaded"] = True
                            st.rerun()
                        else:
                            st.error("Unexpected API response.")
                    except Exception as e:
                        st.error(f"API error: {e}")
        with c_status:
            if st.session_state.get("_enrich_cats_loaded"):
                n = len(st.session_state["_enrich_categories"])
                st.markdown(f'<span class="badge-ok">✓ {n} categories loaded</span>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-warn">Not loaded yet — click button</span>',
                            unsafe_allow_html=True)

        cats_loaded = st.session_state.get("_enrich_cats_loaded", False)
        cats_dict   = st.session_state.get("_enrich_categories", {})

        c_cat, c_db, c_org = st.columns([2, 3, 1])
        with c_cat:
            cat_opts = (["— select category —"] + list(cats_dict.keys())
                        if cats_loaded else ["— load categories first —"])
            sel_cat  = st.selectbox("Category", cat_opts, key=f"{tab_prefix}_category")
        with c_db:
            if cats_loaded and sel_cat not in ("— select category —", "— load categories first —"):
                db_list = cats_dict.get(sel_cat, [])
                if db_list:
                    gene_set_value = st.selectbox("Library", db_list,
                                                  key=f"{tab_prefix}_library")
                else:
                    st.caption("No libraries in this category.")
            else:
                gene_set_value = st.text_input(
                    "Library name (type directly)",
                    value="KEGG_2021_Human",
                    key=f"{tab_prefix}_lib_text",
                    placeholder="e.g. KEGG_2021_Human",
                )
        with c_org:
            organism = st.selectbox(
                "Organism",
                ["Human","Mouse","Rat","Yeast","Fly","Worm","Fish"],
                key=f"{tab_prefix}_organism",
            )

    return db_source, gene_set_value, gmt_file_path, organism


# ══════════════════════════════════════════════════════════════════════════════
# ORA PLOT HELPERS — high-quality axes/labels
# ══════════════════════════════════════════════════════════════════════════════

def _plot_bar(df, gene_set, color_map):
    n   = df["Term"].nunique()
    fig = px.bar(
        df, x="Combined Score", y="Term", color="Class",
        orientation="h", barmode="overlay", opacity=0.80,
        color_discrete_map=color_map,
        hover_data={"Combined Score": ":.2f"},
    )
    fig = _layout(fig, f"Top Enriched Pathways — {gene_set}", height=max(450, n*30+120))
    fig.update_xaxes(title_text="Combined Score")
    fig.update_yaxes(title_text="Pathway", tickfont=dict(size=12))
    return fig


def _plot_enrichment_heatmap(df):
    pivot = df.pivot_table(index="Class", columns="Term",
                           values="Combined Score", fill_value=0)
    fig   = px.imshow(
        pivot,
        labels=dict(x="Pathway", y="Class", color="Combined Score"),
        color_continuous_scale="Reds", aspect="auto", text_auto=".0f",
    )
    fig = _layout(fig, "Pathway Enrichment Heatmap",
                  height=max(350, pivot.shape[0]*70+180))
    fig.update_xaxes(tickangle=45, tickfont=dict(size=11), title_text="Pathway")
    fig.update_yaxes(title_text="Class")
    fig.update_coloraxes(colorbar=dict(
        title=dict(text="Combined Score", font=_AXIS_TITLE),
        tickfont=_TICK_FONT,
    ))
    return fig


def _plot_gene_count(gene_mapping, color_map):
    gm = pd.DataFrame(gene_mapping).copy()
    gm["Gene Count"] = gm["Genes"].apply(len)

    # Compute total unique genes per pathway (across all classes)
    pathway_total = (
        gm.groupby("Pathway")["Genes"]
        .apply(lambda s: len(set(g for lst in s for g in lst)))
        .rename("Total Genes in Pathway")
        .reset_index()
    )
    gm = gm.merge(pathway_total, on="Pathway", how="left")
    gm["Gene %"] = (gm["Gene Count"] / gm["Total Genes in Pathway"].replace(0, 1) * 100).round(1)

    n = gm["Pathway"].nunique()

    # Bar: absolute count
    fig = px.bar(gm, x="Gene Count", y="Pathway", color="Class",
                 orientation="h", color_discrete_map=color_map,
                 hover_data={"Gene %": True, "Gene Count": True, "Total Genes in Pathway": True})
    fig = _layout(fig, "Gene Count per Enriched Pathway", height=max(450, n*30+120))
    fig.update_xaxes(title_text="Number of Genes")
    fig.update_yaxes(title_text="Pathway", tickfont=dict(size=12))
    return fig


def _plot_gene_pct(gene_mapping, color_map, combined_df=None):
    """
    % of pathway genes detected per class.
    Uses the Overlap column (e.g. '2/104') from ORA results to compute the
    true coverage: detected_genes / total_genes_in_pathway * 100.
    Falls back to gene_mapping counts if combined_df is not available.
    """
    gm = pd.DataFrame(gene_mapping).copy()
    gm["Gene Count"] = gm["Genes"].apply(len)

    if combined_df is not None and "Overlap" in combined_df.columns:
        # Parse Overlap "detected/total" → total pathway size per (Term, Class)
        def _parse_overlap(ov):
            try:
                parts = str(ov).split("/")
                if len(parts) == 2:
                    return int(parts[0]), int(parts[1])
            except Exception:
                pass
            return None, None

        _ov = combined_df[["Term", "Class", "Overlap"]].copy()
        _ov[["Detected", "Total"]] = _ov["Overlap"].apply(
            lambda x: pd.Series(_parse_overlap(x))
        )
        _ov = _ov.dropna(subset=["Detected", "Total"])
        _ov["Detected"] = _ov["Detected"].astype(int)
        _ov["Total"]    = _ov["Total"].astype(int)
        _ov["Gene %"]   = (_ov["Detected"] / _ov["Total"].replace(0, 1) * 100).round(1)

        # Optionally add adjusted p-value for hover
        hover_extra = {}
        if "Adjusted P-value" in combined_df.columns:
            _ov = _ov.merge(
                combined_df[["Term", "Class", "Adjusted P-value"]].drop_duplicates(),
                on=["Term", "Class"], how="left"
            )
            hover_extra["Adjusted P-value"] = ":.4f"

        n = _ov["Term"].nunique()
        fig = px.bar(
            _ov, x="Gene %", y="Term", color="Class",
            orientation="h", color_discrete_map=color_map,
            text="Gene %",
            hover_data={"Detected": True, "Total": True, **hover_extra},
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig = _layout(fig, "% of Pathway Genes Detected per Class", height=max(450, n*30+120))
        fig.update_xaxes(title_text="Gene Coverage (%)", range=[0, 110])
        fig.update_yaxes(title_text="Pathway", tickfont=dict(size=12))
        return fig

    # Fallback: use gene_mapping only (less accurate — total = union across classes)
    pathway_total = (
        gm.groupby("Pathway")["Genes"]
        .apply(lambda s: len(set(g for lst in s for g in lst)))
        .rename("Total Genes in Pathway")
        .reset_index()
    )
    gm = gm.merge(pathway_total, on="Pathway", how="left")
    gm["Gene %"] = (gm["Gene Count"] / gm["Total Genes in Pathway"].replace(0, 1) * 100).round(1)

    n = gm["Pathway"].nunique()
    fig = px.bar(
        gm, x="Gene %", y="Pathway", color="Class",
        orientation="h", color_discrete_map=color_map,
        text="Gene %",
        hover_data={"Gene Count": True, "Total Genes in Pathway": True},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig = _layout(fig, "% of Pathway Genes Detected per Class", height=max(450, n*30+120))
    fig.update_xaxes(title_text="Gene Coverage (%)", range=[0, 110])
    fig.update_yaxes(title_text="Pathway", tickfont=dict(size=12))
    return fig


def _plot_dot(combined, gene_mapping, color_map):
    gm  = pd.DataFrame(gene_mapping).copy()
    gm["Gene Count"] = gm["Genes"].apply(len)
    total_per_class  = (
        gm.groupby("Class")["Genes"]
        .apply(lambda s: len(set(g for lst in s for g in lst))).to_dict()
    )
    dot = combined.merge(
        gm[["Pathway","Class","Gene Count"]].rename(columns={"Pathway":"Term"}),
        on=["Term","Class"], how="left",
    )
    dot["Gene Count"] = dot["Gene Count"].fillna(0).astype(int)
    dot["GeneRatio"]  = dot.apply(
        lambda r: r["Gene Count"] / max(total_per_class.get(r["Class"],1),1), axis=1)
    dot = dot.sort_values("Combined Score", ascending=True)
    max_count = max(dot["Gene Count"].max(), 1)
    sizeref   = 2.0 * max_count / (40.0**2)
    fig = go.Figure()
    for cls in dot["Class"].unique():
        sub = dot[dot["Class"] == cls]
        fig.add_trace(go.Scatter(
            x=sub["GeneRatio"], y=sub["Term"],
            mode="markers", name=cls,
            marker=dict(
                size=sub["Gene Count"].clip(lower=3)*4,
                sizemode="area", sizeref=sizeref,
                color=sub["Combined Score"], colorscale="RdYlBu_r",
                showscale=True,
                colorbar=dict(
                    title=dict(text="Combined Score", font=_AXIS_TITLE),
                    tickfont=_TICK_FONT,
                    thickness=16, len=0.65, x=1.01,
                ),
                line=dict(width=0.8, color="white"), opacity=0.85,
            ),
            customdata=np.stack([sub["Combined Score"], sub["Gene Count"],
                                 sub["GeneRatio"], sub["Class"]], axis=1),
            hovertemplate=(
                "<b>%{y}</b><br>Class: %{customdata[3]}<br>"
                "Combined Score: %{customdata[0]:.2f}<br>"
                "Gene Count: %{customdata[1]}<br>"
                "Gene Ratio: %{customdata[2]:.3f}<extra></extra>"
            ),
        ))
    n   = dot["Term"].nunique()
    fig = _layout(fig, "Dot Plot — Enrichment (clusterProfiler style)",
                  height=max(500, n*28+150))
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=12), title_text="Pathway")
    fig.update_xaxes(title_text="Gene Ratio", tickformat=".2f")
    fig.add_annotation(
        xref="paper", yref="paper", x=1.13, y=0.02,
        text="● size = gene count", showarrow=False,
        font=dict(family=_FONT_FAMILY, size=11, color="#374151"),
    )
    return fig


def _plot_gene_pathway_heatmap(gene_mapping):
    all_genes    = sorted(set(g for row in gene_mapping for g in row["Genes"]))
    all_pathways = sorted(set(row["Pathway"] for row in gene_mapping))
    if not all_genes or not all_pathways:
        return None
    mat = pd.DataFrame(0, index=all_genes, columns=all_pathways)
    for row in gene_mapping:
        for g in row["Genes"]:
            if g in mat.index:
                mat.loc[g, row["Pathway"]] = 1
    fig = px.imshow(
        mat,
        labels=dict(x="Pathway", y="Gene", color="Present"),
        color_continuous_scale=[[0,"#f8fafc"],[1,"#1d4ed8"]],
        aspect="auto",
    )
    fig = _layout(fig, "Gene Involvement Across Pathways",
                  height=max(450, len(all_genes)*20+180))
    fig.update_xaxes(tickangle=45, tickfont=dict(size=10), title_text="Pathway")
    fig.update_yaxes(tickfont=dict(size=10), title_text="Gene")
    return fig


def _plot_network(gene_mapping, color_map):
    G = nx.Graph()
    pathway_genes = {}
    for row in gene_mapping:
        pw = row["Pathway"]
        pathway_genes.setdefault(pw, set()).update(row["Genes"])
        if not G.has_node(pw):
            G.add_node(pw, node_type="pathway", cls=row.get("Class",""))
    if len(set(g for gs in pathway_genes.values() for g in gs)) < 2:
        return None
    for pw, genes in pathway_genes.items():
        for g in genes:
            G.add_node(g, node_type="gene", cls="")
            G.add_edge(pw, g)
    pos = nx.spring_layout(G, seed=42, k=0.8)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0,y0=pos[u]; x1,y1=pos[v]
        edge_x += [x0,x1,None]; edge_y += [y0,y1,None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(width=0.7, color="#94a3b8"), hoverinfo="none"))
    for ntype, symbol, sz in [("pathway","diamond",16),("gene","circle",8)]:
        nodes  = [n for n,d in G.nodes(data=True) if d.get("node_type")==ntype]
        if not nodes: continue
        x_     = [pos[n][0] for n in nodes]
        y_     = [pos[n][1] for n in nodes]
        colors = [color_map.get(G.nodes[n].get("cls",""),"#94a3b8") for n in nodes]
        fig.add_trace(go.Scatter(
            x=x_, y=y_,
            mode="markers+text",
            text=nodes,
            textposition="top center" if ntype == "pathway" else "bottom center",
            textfont=dict(
                family=_FONT_FAMILY,
                size=9 if ntype == "pathway" else 7,
                color="#0f172a" if ntype == "pathway" else "#475569"
            ),
            marker=dict(size=sz, symbol=symbol, color=colors,
                        line=dict(width=1, color="white")),
            name=ntype.capitalize(), hovertext=nodes, hoverinfo="text",
        ))
    fig = _layout(fig, "Gene Co-Pathway Network", height=650)
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, showline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, showline=False)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# ORA RENDER PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def _render_ora_results(combined, gene_mapping, gene_set_label, color_map, class_names):
    with st.expander("📋 Full Results Table", expanded=False):
        st.dataframe(combined[[c for c in combined.columns if c != "Genes"]],
                     use_container_width=True)
        st.download_button("📥 Download CSV", combined.to_csv(index=False).encode(),
                           "ora_results.csv", "text/csv")

    if gene_mapping:
        with st.expander("🧬 Gene Involvement per Pathway & Class", expanded=False):
            rows = [{"Pathway": r["Pathway"], "Class": r["Class"], "Gene": g}
                    for r in gene_mapping for g in r["Genes"]]
            gene_table = (pd.DataFrame(rows) if rows
                          else pd.DataFrame(columns=["Pathway","Class","Gene"]))
            gene_wide  = (
                gene_table.groupby(["Pathway","Class"])["Gene"]
                .apply(lambda x: "; ".join(sorted(x))).reset_index()
                .rename(columns={"Gene":"Genes"})
            )
            gene_wide["Gene Count"] = gene_wide["Genes"].apply(
                lambda x: len(x.split("; ")) if x else 0)
            st.dataframe(gene_wide, use_container_width=True)
            st.download_button("📥 Download Gene Table",
                               gene_wide.to_csv(index=False).encode(),
                               "ora_gene_table.csv", "text/csv",
                               key="dl_ora_gene_table")
            st.session_state["enrichment_gene_table"] = gene_wide

    st.markdown("**Top Enriched Pathways**")
    fig = _plot_bar(combined[["Term","Combined Score","Class"]], gene_set_label, color_map)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
    _capture_plotly(fig, "enrichment_bar"); del fig; gc.collect()

    try:
        fig = _plot_dot(combined[["Term","Combined Score","Class"]], gene_mapping, color_map)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
        _capture_plotly(fig, "enrichment_dot"); del fig; gc.collect()
    except Exception as e:
        st.warning(f"Dot plot skipped: {e}")

    if len(class_names) > 1:
        fig = _plot_enrichment_heatmap(combined[["Term","Combined Score","Class"]])
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
        _capture_plotly(fig, "enrichment_heatmap"); del fig; gc.collect()

    if gene_mapping:
        fig = _plot_gene_count(gene_mapping, color_map)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
        _capture_plotly(fig, "enrichment_gene_count"); del fig; gc.collect()

        try:
            fig_pct = _plot_gene_pct(gene_mapping, color_map, combined_df=combined)
            st.plotly_chart(fig_pct, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
            _capture_plotly(fig_pct, "enrichment_gene_pct"); del fig_pct; gc.collect()
        except Exception as _e:
            st.warning(f"Gene % plot skipped: {_e}")

    with st.expander("Gene × Pathway Heatmap", expanded=False):
        if gene_mapping:
            fig = _plot_gene_pathway_heatmap(gene_mapping)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
                _capture_plotly(fig, "enrichment_gene_pathway"); del fig; gc.collect()
        else:
            st.info("No gene mapping available.")

    with st.expander("🕸️ Gene Co-Pathway Network", expanded=False):
        if gene_mapping:
            fig = _plot_network(gene_mapping, color_map)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
                _capture_plotly(fig, "enrichment_network"); del fig; gc.collect()
            else:
                st.info("Not enough shared genes to build a network.")
        else:
            st.info("No gene mapping available.")


# ══════════════════════════════════════════════════════════════════════════════
# ORA RUNNERS  (backward-compatible signatures)
# ══════════════════════════════════════════════════════════════════════════════

def perform_gsea(gene_lists, class_names, gene_set, organism, num_pathways):
    """ORA via Enrichr online."""
    results, gene_mapping = [], []
    for i, gene_list in enumerate(gene_lists):
        if not gene_list: continue
        clean_list = _sanitise_gene_list(gene_list)
        try:
            enr = gp.enrichr(gene_list=clean_list, gene_sets=[gene_set], organism=organism)
            if enr.results.empty:
                st.warning(f"No results for '{class_names[i]}'."); continue
            top = enr.results.head(num_pathways).copy()
            top["Class"] = class_names[i]
            keep = [c for c in ["Term","Combined Score","Class","Adjusted P-value","Overlap","Genes"]
                    if c in top.columns]
            results.append(top[keep])
            for pathway, genes_str in zip(top["Term"], top.get("Genes", [""]*len(top))):
                gene_mapping.append({
                    "Pathway": pathway,
                    "Genes": [g.strip() for g in str(genes_str).split(";") if g.strip()],
                    "Class": class_names[i],
                })
        except Exception as e:
            st.error(f"Enrichr error for '{class_names[i]}': {e}")
    if not results:
        st.warning("No enrichment results found."); return
    combined  = pd.concat(results, ignore_index=True)
    color_map = st.session_state.get("class_colors", {})
    _render_ora_results(combined, gene_mapping, gene_set, color_map, class_names)
    del results, combined, gene_mapping; gc.collect()


def perform_gsea_offline(gene_lists, class_names, gene_set_path, organism, num_pathways):
    """ORA via local GMT."""
    import io

    # ── Resolve gene_set_path to an actual filesystem path ───────────────────
    # gene_set_path may be:
    #   (a) a pathlib.Path / str  → already a valid path, use as-is
    #   (b) a Streamlit UploadedFile (BytesIO-like) → must be written to a
    #       temporary file first, otherwise str() gives "<BytesIO object …>"
    #       and gseapy tries to hit the Enrichr API instead.
    _tmp_gmt_file = None  # keep reference so it stays alive during the loop
    if hasattr(gene_set_path, "read"):
        # It's a file-like object (UploadedFile or similar)
        gene_set_path.seek(0)
        _tmp_gmt_file = tempfile.NamedTemporaryFile(
            suffix=".gmt", delete=False, mode="wb"
        )
        _tmp_gmt_file.write(gene_set_path.read())
        _tmp_gmt_file.flush()
        _tmp_gmt_file.close()
        _resolved_gmt = _tmp_gmt_file.name
    else:
        _resolved_gmt = str(gene_set_path)

    results, gene_mapping = [], []
    for i, gene_list in enumerate(gene_lists):
        if not gene_list: continue
        clean_list = _sanitise_gene_list(gene_list)
        try:
            enr = gp.enrichr(gene_list=clean_list,
                             gene_sets=_resolved_gmt,
                             organism=organism or "Human")
            if enr.results.empty:
                st.warning(f"No results for '{class_names[i]}'."); continue
            top = enr.results.head(num_pathways).copy()
            top["Class"] = class_names[i]
            keep = [c for c in ["Term","Combined Score","Class","Adjusted P-value","Overlap","Genes"]
                    if c in top.columns]
            results.append(top[keep])
            for pathway, genes_str in zip(top["Term"], top.get("Genes", [""]*len(top))):
                gene_mapping.append({
                    "Pathway": pathway,
                    "Genes": [g.strip() for g in str(genes_str).split(";") if g.strip()],
                    "Class": class_names[i],
                })
        except Exception as e:
            st.error(f"Enrichment error for '{class_names[i]}': {e}")
    if not results:
        st.warning("No enrichment results found.")
        if _tmp_gmt_file is not None:
            import os
            try: os.unlink(_tmp_gmt_file.name)
            except Exception: pass
        return
    combined  = pd.concat(results, ignore_index=True)
    color_map = st.session_state.get("class_colors", {})
    label = (getattr(gene_set_path, "name", None)
             or Path(_resolved_gmt).name)
    _render_ora_results(combined, gene_mapping, label, color_map, class_names)
    del results, combined, gene_mapping
    if _tmp_gmt_file is not None:
        import os
        try: os.unlink(_tmp_gmt_file.name)
        except Exception: pass
    gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# GSEA HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _normalise_gsea_cols(df):
    """
    Rename gseapy res2d columns to canonical names.
    Handles duplicates: if a target name already exists verbatim in the df
    (gseapy may already have 'Term', 'NES', etc.), that target is marked as
    claimed so no other column gets renamed to the same name.
    After renaming, any remaining duplicate column names are dropped.
    """
    _TARGETS = {
        "NES":        {"nes", "normalized_enrichment_score"},
        "FDR":        {"fdr", "fdr_q_val", "fdr_q-val"},
        "pval":       {"pval", "p_val", "nom_p_val", "nom_p-val"},
        "Term":       {"term", "name", "pathway"},
        "ES":         {"es", "enrichment_score"},
        "Lead_genes": {"lead_genes", "lead_overlap", "leading_edge"},
    }
    # Mark targets already present verbatim — they must not be renamed-to again
    already_used = {t for t in _TARGETS if t in df.columns}
    col_map = {}
    for c in df.columns:
        cl = c.lower().replace(" ", "_").replace("-", "_")
        for target, variants in _TARGETS.items():
            if cl in variants and target not in already_used:
                if c != target:
                    col_map[c] = target
                already_used.add(target)
                break
    df = df.rename(columns=col_map)
    # Drop any remaining duplicate column names (keep first occurrence)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df


def _render_gsea_results(res_df, run_label):
    res_df = _normalise_gsea_cols(res_df.copy())
    if res_df.empty:
        st.warning("No GSEA results found."); return

    # ── Force numeric types on all numeric columns ────────────────────────────
    # gseapy sometimes returns "1.0", "NA", "" as strings
    for _col in ("NES", "ES", "FDR", "pval"):
        if _col in res_df.columns:
            res_df[_col] = pd.to_numeric(res_df[_col], errors="coerce")

    if "FDR" in res_df.columns:
        sig = res_df[res_df["FDR"] < 0.25].copy()
        if sig.empty:
            st.info("No pathways passed FDR < 0.25 — showing top 20 by |NES|.")
            sig = (res_df.reindex(res_df["NES"].abs().nlargest(20).index)
                   if "NES" in res_df.columns else res_df.head(20))
    else:
        sig = res_df.head(20)

    key_safe = run_label.replace(" ","_")

    with st.expander("📋 GSEA Results Table", expanded=True):
        disp = [c for c in ["Term","NES","ES","pval","FDR","Lead_genes"] if c in sig.columns]
        st.dataframe(sig[disp], use_container_width=True)
        st.download_button("📥 Download GSEA Results",
                           sig.to_csv(index=False).encode(),
                           f"gsea_{key_safe}.csv", "text/csv",
                           key=f"dl_gsea_{key_safe}")
        st.session_state[f"gsea_table_{key_safe}"] = sig

    # ── NES bar chart ─────────────────────────────────────────────────────────
    if "NES" in sig.columns and "Term" in sig.columns:
        sig_s  = sig.dropna(subset=["NES"]).sort_values("NES")
        colors = ["#dc2626" if v > 0 else "#2563eb" for v in sig_s["NES"]]
        fig    = go.Figure(go.Bar(
            x=sig_s["NES"], y=sig_s["Term"],
            orientation="h", marker_color=colors,
            hovertemplate="<b>%{y}</b><br>NES: %{x:.3f}<extra></extra>",
        ))
        fig = _layout(fig, f"NES Bar Chart — {run_label}",
                      height=max(450, len(sig_s)*26+130))
        fig.update_xaxes(title_text="Normalized Enrichment Score (NES)")
        fig.update_yaxes(title_text="Pathway", tickfont=dict(size=12))
        fig.add_vline(x=0, line_dash="dash", line_color="#374151", line_width=1.5)
        # colour legend annotation
        fig.add_annotation(
            xref="paper", yref="paper", x=1.01, y=1.05,
            text="<b style='color:#dc2626'>■</b> Enriched  "
                 "<b style='color:#2563eb'>■</b> Depleted",
            showarrow=False,
            font=dict(family=_FONT_FAMILY, size=12, color="#0f172a"),
            align="left",
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToAdd': ['downloadImage'], 'toImageButtonOptions': {'format': 'png', 'scale': 2}})
        _capture_plotly(fig, f"gsea_nes_{key_safe}"); del fig; gc.collect()

    # ── Bubble plot ───────────────────────────────────────────────────────────
    if all(c in sig.columns for c in ("NES","FDR","Term")):
        sig2 = sig.dropna(subset=["NES","FDR"]).copy()
        sig2["FDR"]       = pd.to_numeric(sig2["FDR"], errors="coerce")
        sig2["NES"]       = pd.to_numeric(sig2["NES"], errors="coerce")
        sig2 = sig2.dropna(subset=["NES","FDR"])
        sig2["FDR_safe"]  = sig2["FDR"].astype(float).clip(lower=1e-10)
        sig2["-log10FDR"] = -np.log10(sig2["FDR_safe"].values.astype(float))

        fig2 = px.scatter(
            sig2, x="NES", y="-log10FDR",
            text="Term" if len(sig2) <= 30 else None,
            color="NES", color_continuous_scale="RdBu_r",
            hover_data=["Term","NES","FDR"],
            size=[12]*len(sig2),
        )
        fig2.update_traces(textposition="top center",
                           textfont=dict(family=_FONT_FAMILY, size=10, color="#0f172a"),
                           marker=dict(size=12, line=dict(width=1, color="#0f172a")))
        fig2 = _layout(fig2, f"GSEA Bubble Plot — {run_label}", height=520)
        fig2.update_xaxes(title_text="Normalized Enrichment Score (NES)")
        fig2.update_yaxes(title_text="-log\u2081\u2080(FDR)")
        fig2.add_vline(x=0, line_dash="dash", line_color="#374151", line_width=1.5)
        fig2.add_hline(
            y=-np.log10(0.25), line_dash="dot",
            line_color="#d97706", line_width=1.5,
            annotation_text="FDR = 0.25",
            annotation_font=dict(family=_FONT_FAMILY, size=12, color="#d97706"),
        )
        fig2.update_coloraxes(colorbar=dict(
            title=dict(text="NES", font=_AXIS_TITLE),
            tickfont=_TICK_FONT,
        ))
        st.plotly_chart(fig2, use_container_width=True)
        _capture_plotly(fig2, f"gsea_bubble_{key_safe}"); del fig2; gc.collect()


def _build_ranked_volcano():
    vd = st.session_state.get("volcano_data")
    if vd is None:
        return None, "No Volcano Plot data found — run Volcano Plot first."
    if "Log2 Fold Change" not in vd.columns or "Feature" not in vd.columns:
        return None, "Volcano data missing columns (Feature, Log2 Fold Change)."
    vd = vd.copy()
    if "-log10(p-value)" in vd.columns:
        vd["_s"] = np.sign(vd["Log2 Fold Change"]) * vd["-log10(p-value)"]
    elif "p-value" in vd.columns:
        vd["_s"] = (np.sign(vd["Log2 Fold Change"])
                    * (-np.log10(vd["p-value"].clip(lower=1e-300))))
    else:
        vd["_s"] = vd["Log2 Fold Change"]
    ranked = vd.groupby("Feature")["_s"].mean().sort_values(ascending=False)
    return ranked, f"{len(ranked)} features — score = sign(Log2FC) x -log10p (avg. across comparisons)"


def _build_ranked_heatmap(sel_cls: str):
    hm_sig = st.session_state.get("heatmap_significant_features")
    hm_df  = st.session_state.get("heatmap_data_source_df")
    if hm_sig is None or hm_df is None:
        return None, "No Heatmap data — run Heatmap with stat test first."
    valid = [f for f in hm_sig if f in hm_df.columns]
    if not valid:
        return None, "Heatmap significant features not found in the data source."
    cls_v   = hm_df[hm_df["Class"] == sel_cls][valid].mean()
    other_v = hm_df[hm_df["Class"] != sel_cls][valid].mean()
    ranked  = (cls_v - other_v).sort_values(ascending=False)
    return ranked, f"{len(ranked)} features — score = mean({sel_cls}) - mean(others)"


def _run_gsea_prerank(ranked_series, gene_set, run_label,
                      num_pathways, min_size, max_size, permutation_num, seed):
    # Sanitise to ASCII before passing to gseapy
    ranked_clean = _sanitise_ranked_series(ranked_series)

    rnk = ranked_clean.reset_index()
    rnk.columns = ["gene", "score"]
    rnk = rnk.sort_values("score", ascending=False).drop_duplicates("gene")

    with st.spinner(f"Running GSEA prerank — {len(rnk)} features, "
                    f"{permutation_num} permutations…"):
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                pre    = gp.prerank(
                    rnk=rnk, gene_sets=gene_set,
                    min_size=min_size, max_size=max_size,
                    permutation_num=permutation_num, seed=seed,
                    outdir=tmpdir, verbose=False,
                )
                res_df = pre.res2d.copy() if hasattr(pre, "res2d") else pd.DataFrame()
        except Exception as e:
            st.error(f"GSEA error: {e}"); return

    if res_df.empty:
        st.warning("No results returned. Try relaxing min/max size or verify "
                   "that feature names match the gene set library."); return

    st.session_state[f"gsea_res_{run_label.replace(' ','_')}"] = res_df
    _render_gsea_results(res_df.head(num_pathways * 3), run_label)
    gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def render_enrichment_tab():
    """
    Full Enrichment UI — two tabs: ORA and GSEA.
    Call inside `with _t6_enrich:` in Profiler.py.
    """
    _inject_css()

    # shared session init
    for k, v in [("_enrich_categories",{}), ("_enrich_cats_loaded",False),
                 ("_ora_num_classes", 1)]:
        if k not in st.session_state:
            st.session_state[k] = v

    _tab_ora, _tab_gsea = st.tabs([
        "ORA (Over-Representation Analysis)",
        "GSEA (Ranked List Enrichment)",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # ① ORA TAB
    # ══════════════════════════════════════════════════════════════════════════
    with _tab_ora:
        st.markdown(
            '<p style="color:#64748b;font-size:13px;margin-bottom:16px;">'
            'Tests whether a predefined gene set (pathway, GO term…) is '
            'over-represented among your <b>significant features</b>. '
            'Gene lists are auto-detected from your Volcano Plot or Heatmap, '
            'or entered manually.'
            '</p>', unsafe_allow_html=True)

        # ── Section 1 · Database ──────────────────────────────────────────────
        _section("1️⃣", "Database")
        ora_db_source, ora_gene_set, ora_gmt_path, ora_organism = _render_db_selector("ora")
        _section_end()

        # ── Section 2 · Gene List Source (mirrors GSEA design) ───────────────
        _section("2️⃣", "Gene / Protein Lists Source")

        has_volcano      = (st.session_state.get("volcano_up_by_comparison") is not None
                            or st.session_state.get("volcano_down_by_comparison") is not None)
        has_heatmap_over = bool(st.session_state.get("heatmap_overexpressed_features"))
        has_heatmap_exc  = bool(st.session_state.get("heatmap_exclusive_features"))
        has_heatmap      = has_heatmap_over or has_heatmap_exc
        has_venn         = bool(st.session_state.get("_venn_exclusive_features"))

        ora_src_opts = []
        if has_volcano:
            ora_src_opts.append("🌋 Volcano — Upregulated per comparison")
            ora_src_opts.append("🌋 Volcano — Downregulated per comparison")
        if has_heatmap_over:
            ora_src_opts.append("🗺️ Heatmap — Overexpressed per class")
        if has_heatmap_exc:
            ora_src_opts.append("🗺️ Heatmap — Exclusive per class")
        if has_venn:
            ora_src_opts.append("🔷 Venn/UpSet — Exclusive per class")
        ora_src_opts.append("✏️ Manual")

        if not has_volcano and not has_heatmap and not has_venn:
            st.info(
                "💡 Run a **Volcano Plot** or **Heatmap** in the Biomarkers tab, "
                "or a **Venn/UpSet** diagram in the Comparisons tab "
                "to auto-populate gene lists — or use manual entry.",
            )

        ora_list_source = st.radio(
            "Source:", ora_src_opts,
            key="ora_list_source", horizontal=True
        )

        # ── Build the prefill dict {class/comparison: [features]} ────────────
        _prefill: dict = {}

        def _badge_ok(msg):
            st.markdown(f'<span class="badge-ok">✓ {msg}</span>', unsafe_allow_html=True)
        def _badge_warn(msg):
            st.markdown(f'<span class="badge-warn">⚠ {msg}</span>', unsafe_allow_html=True)
        def _preview(d, label="detected lists"):
            with st.expander(f"👁 Preview {label}", expanded=False):
                for cls, feats in d.items():
                    st.caption(f"**{cls}** ({len(feats)}): {', '.join(str(f) for f in feats[:25])}"
                               + ("…" if len(feats) > 25 else ""))

        if "🌋 Volcano — Upregulated" in ora_list_source:
            _up = st.session_state.get("volcano_up_by_comparison", {})
            _prefill = {comp: feats for comp, feats in _up.items() if feats}
            if _prefill:
                _badge_ok(f"{sum(len(v) for v in _prefill.values())} upregulated features "
                          f"across {len(_prefill)} comparison(s)")
                _preview(_prefill, "upregulated lists")
            else:
                _badge_warn("No upregulated features found in Volcano data")

        elif "🌋 Volcano — Downregulated" in ora_list_source:
            _down = st.session_state.get("volcano_down_by_comparison", {})
            _prefill = {comp: feats for comp, feats in _down.items() if feats}
            if _prefill:
                _badge_ok(f"{sum(len(v) for v in _prefill.values())} downregulated features "
                          f"across {len(_prefill)} comparison(s)")
                _preview(_prefill, "downregulated lists")
            else:
                _badge_warn("No downregulated features found in Volcano data")

        elif "🗺️ Heatmap — Overexpressed" in ora_list_source:
            _over = st.session_state.get("heatmap_overexpressed_features", {})
            _prefill = {cls: feats for cls, feats in _over.items() if feats}
            if _prefill:
                _badge_ok(f"{sum(len(v) for v in _prefill.values())} overexpressed features "
                          f"across {len(_prefill)} class(es)")
                _preview(_prefill, "overexpressed lists")
            else:
                _badge_warn("No overexpressed features found — run Heatmap first")

        elif "🗺️ Heatmap — Exclusive" in ora_list_source:
            _exc_hm = st.session_state.get("heatmap_exclusive_features", {})
            _prefill = {cls: feats for cls, feats in _exc_hm.items() if feats}
            if _prefill:
                _badge_ok(f"{sum(len(v) for v in _prefill.values())} exclusive features "
                          f"across {len(_prefill)} class(es) — only non-zero in that class")
                _preview(_prefill, "exclusive heatmap lists")
            else:
                _badge_warn("No exclusive heatmap features found — run Heatmap first")

        elif "🔷 Venn/UpSet" in ora_list_source:
            _venn_exc = st.session_state.get("_venn_exclusive_features", {})
            _prefill  = {cls: sorted(feats) for cls, feats in _venn_exc.items() if feats}
            if _prefill:
                _badge_ok(f"{sum(len(v) for v in _prefill.values())} exclusive features "
                          f"across {len(_prefill)} class(es) — from Venn/UpSet diagram")
                _preview(_prefill, "exclusive feature lists")
                # Also show shared features info
                _shared = st.session_state.get("_venn_shared_features", set())
                if _shared:
                    st.caption(f"ℹ️ {len(_shared)} features shared by all classes (not included — exclusive only)")
            else:
                _badge_warn("No exclusive features found — run Venn/UpSet diagram in Comparisons first")

        # Manual: no prefill, user types everything
        _section_end()

        # ── Section 3 · Gene lists form ───────────────────────────────────────
        _section("3️⃣", "Gene / Protein Lists & Parameters")

        _prefill_classes = list(_prefill.keys())
        _n_auto = len(_prefill_classes)

        # Number of classes: auto-set from detected source, or manual
        if _n_auto > 0:
            _num_cls_default = _n_auto
        else:
            _num_cls_default = st.session_state.get("_ora_num_classes", 1)

        num_cls = st.number_input(
            "Number of classes / comparisons", min_value=1, max_value=50,
            value=_num_cls_default, step=1,
            key="ora_num_classes",
            help="Automatically set from detected source — adjust if needed",
        )
        st.session_state["_ora_num_classes"] = num_cls
        _section_end()

        with st.form("ora_gene_form", clear_on_submit=False):
            st.markdown("**Gene / protein lists** — auto-filled from selected source, edit freely")
            gene_lists, class_names = [], []
            for i in range(num_cls):
                if num_cls > 1:
                    st.markdown(f"**— Entry {i+1} —**")
                c1, c2 = st.columns([1, 3])

                _default_cls      = _prefill_classes[i] if i < len(_prefill_classes) else f"Class_{i+1}"
                _prefill_genes_i  = ", ".join(str(g) for g in _prefill.get(_default_cls, []))

                with c1:
                    class_name = st.text_input(
                        "Class / comparison name", key=f"ora_cls_name_{i}",
                        value=_default_cls,
                        placeholder="e.g. Upregulated")
                with c2:
                    raw = st.text_area(
                        "Genes / Proteins", key=f"ora_genes_{i}",
                        value=_prefill_genes_i,
                        placeholder="Comma, space or newline separated\n"
                                    "e.g.  TP53, EGFR, BRCA1",
                        height=90)
                    genes = [g.strip() for g in re.split(r"[,\s]+", raw) if g.strip()]
                class_names.append(class_name)
                gene_lists.append(genes)
                if genes:
                    st.caption(f"✓ {len(genes)} genes/features")
                else:
                    st.caption("⚠️ No genes entered yet")
                if i < num_cls - 1:
                    st.divider()

            st.divider()
            ora_num_paths = st.slider("Max pathways to display", 1, 100, 10, key="ora_npaths")
            st.divider()

            submitted_ora = st.form_submit_button("✅ Confirm gene lists", use_container_width=True)

        # ── Run button OUTSIDE form ───────────────────────────────────────────
        if submitted_ora:
            st.session_state["_ora_submitted"] = True
            gene_lists, class_names = [], []
            for i in range(num_cls):
                raw   = st.session_state.get(f"ora_genes_{i}", "")
                genes = [g.strip() for g in re.split(r"[,\s]+", raw) if g.strip()]
                gene_lists.append(genes)
                class_names.append(st.session_state.get(f"ora_cls_name_{i}", f"Class_{i+1}"))
            ora_num_paths = st.session_state.get("ora_npaths", 10)

        if st.session_state.get("_ora_submitted"):
            missing = []
            if not ora_gene_set:                              missing.append("database (select above)")
            if not all(len(gl) > 0 for gl in gene_lists):    missing.append("gene lists")
            if missing:
                st.caption(f"⬆️ Still needed: {', '.join(missing)}")

            if st.button("▶ Run ORA", key="run_ora_btn", type="primary",
                         disabled=bool(missing), use_container_width=True):
                with st.spinner("Running ORA…"):
                    if "Offline" in ora_db_source:
                        perform_gsea_offline(gene_lists, class_names,
                                             ora_gmt_path, None, ora_num_paths)
                    else:
                        perform_gsea(gene_lists, class_names,
                                     ora_gene_set, ora_organism, ora_num_paths)
                st.success("✅ ORA completed!")
        else:
            st.info("👆 Select a source and confirm gene lists above to enable the Run button.")

    # ══════════════════════════════════════════════════════════════════════════
    # ② GSEA TAB  (fully outside any form)
    # ══════════════════════════════════════════════════════════════════════════
    with _tab_gsea:
        st.markdown(
            '<p style="color:#64748b;font-size:13px;margin-bottom:16px;">'
            'Uses a <b>ranked list of ALL features</b> to detect pathways enriched '
            'at the top or bottom of the ranking. The ranked list is auto-detected '
            'from your Volcano Plot or Heatmap results, or entered manually.'
            '</p>', unsafe_allow_html=True)

        # ── Section 1 · Ranked List ──────────────────────────────────────────
        _section("1️⃣", "Ranked List")

        has_volcano = st.session_state.get("volcano_data") is not None
        has_heatmap = (st.session_state.get("heatmap_significant_features") is not None
                       and st.session_state.get("heatmap_data_source_df") is not None)
        has_venn_g       = bool(st.session_state.get("_venn_exclusive_features"))
        has_heatmap_exc_g = bool(st.session_state.get("heatmap_exclusive_features"))

        source_opts = []
        if has_volcano:          source_opts.append("🌋 Volcano Plot")
        if has_heatmap:          source_opts.append("🗺️ Heatmap — Overexpressed")
        if has_heatmap_exc_g:    source_opts.append("🗺️ Heatmap — Exclusive (pick class)")
        if has_venn_g:           source_opts.append("🔷 Venn/UpSet — Exclusive (pick class)")
        source_opts.append("✏️ Manual")

        if not has_volcano and not has_heatmap and not has_venn_g and not has_heatmap_exc_g:
            st.info(
                "💡 Run a **Volcano Plot**, a **Heatmap**, "
                "or a **Venn/UpSet** in Comparisons to auto-populate the ranked list."
            )

        rank_source   = st.radio("Source:", source_opts,
                                 key="gsea_rank_source", horizontal=True)
        ranked_series = None
        rank_desc     = ""
        rank_ok       = False

        if "🌋 Volcano" in rank_source:
            ranked_series, rank_desc = _build_ranked_volcano()
            rank_ok = ranked_series is not None and not ranked_series.empty

        elif "🗺️ Heatmap — Overexpressed" in rank_source:
            hm_df = st.session_state.get("heatmap_data_source_df")
            if hm_df is not None:
                classes = sorted(hm_df["Class"].unique().tolist())
                sel_cls = st.selectbox(
                    "Target class (ranked higher when overexpressed vs others):",
                    classes, key="gsea_heatmap_cls")
                ranked_series, rank_desc = _build_ranked_heatmap(sel_cls)
                rank_ok = ranked_series is not None and not ranked_series.empty
            else:
                rank_desc = "No Heatmap data — run Heatmap with stat test first."

        elif "🗺️ Heatmap — Exclusive" in rank_source:
            _hm_exc_g = st.session_state.get("heatmap_exclusive_features", {})
            _hm_exc_classes = [cls for cls, feats in _hm_exc_g.items() if feats]
            if _hm_exc_classes:
                sel_cls_hm = st.selectbox(
                    "Class (exclusive features ranked 1.0, others 0.0):",
                    _hm_exc_classes, key="gsea_heatmap_exc_cls")
                _exc_feats_hm = _hm_exc_g.get(sel_cls_hm, [])
                _all_hm = sorted(set(f for feats in _hm_exc_g.values() for f in feats))
                _scores_hm = {f: (1.0 if f in _exc_feats_hm else 0.0) for f in _all_hm}
                ranked_series = pd.Series(_scores_hm).sort_values(ascending=False)
                rank_desc = (f"{len(_exc_feats_hm)} exclusive features for {sel_cls_hm} "
                             f"(ranked 1.0) vs {len(_all_hm)-len(_exc_feats_hm)} others (0.0)")
                rank_ok = bool(_exc_feats_hm)
            else:
                rank_desc = "No exclusive heatmap features — run Heatmap first."

        elif "🔷 Venn/UpSet" in rank_source:
            _venn_exc = st.session_state.get("_venn_exclusive_features", {})
            _venn_classes = [cls for cls, feats in _venn_exc.items() if feats]
            if _venn_classes:
                sel_cls_v = st.selectbox(
                    "Class to rank (exclusive features ranked highest, others 0):",
                    _venn_classes, key="gsea_venn_cls"
                )
                _exc_feats = sorted(_venn_exc.get(sel_cls_v, []))
                if _exc_feats:
                    # Build ranked series: exclusive features get score 1.0,
                    # all other features from the union get 0.0
                    _all_feats = sorted(set.union(*[set(f) for f in _venn_exc.values()]))
                    _scores    = {f: (1.0 if f in _exc_feats else 0.0) for f in _all_feats}
                    ranked_series = pd.Series(_scores).sort_values(ascending=False)
                    rank_desc = (f"{len(_exc_feats)} exclusive features for {sel_cls_v} "
                                 f"(ranked 1.0) vs {len(_all_feats)-len(_exc_feats)} shared/other (ranked 0.0)")
                    rank_ok = True
                else:
                    rank_desc = f"No exclusive features for {sel_cls_v}"
            else:
                rank_desc = "No exclusive features found — run Venn/UpSet in Comparisons first."

        else:  # Manual
            manual_text = st.text_area(
                "Paste ranked list — one entry per line: `feature_name TAB score`",
                height=160, key="gsea_manual_ranked",
                placeholder="TP53\t2.45\nEGFR\t1.87\nBRCA1\t-1.23")
            if manual_text.strip():
                try:
                    rows = []
                    for line in manual_text.strip().splitlines():
                        parts = line.replace(",","\t").split()
                        if len(parts) >= 2:  rows.append((parts[0], float(parts[1])))
                        elif len(parts) == 1: rows.append((parts[0], 0.0))
                    if rows:
                        ranked_series = pd.Series(dict(rows)).sort_values(ascending=False)
                        rank_desc = f"{len(ranked_series)} features entered manually"
                        rank_ok   = True
                except Exception as e:
                    st.error(f"Parse error: {e}")

        if rank_ok and ranked_series is not None:
            st.markdown(f'<span class="badge-ok">✓ {rank_desc}</span>',
                        unsafe_allow_html=True)
            with st.expander("👁️ Preview ranked list (top / bottom 10)", expanded=False):
                preview = pd.DataFrame({"Feature": ranked_series.index,
                                        "Score":   ranked_series.values})
                ct, cb = st.columns(2)
                with ct:
                    st.caption("**Top 10**")
                    st.dataframe(preview.head(10), use_container_width=True)
                with cb:
                    st.caption("**Bottom 10**")
                    st.dataframe(preview.tail(10), use_container_width=True)
        elif rank_desc:
            st.markdown(f'<span class="badge-warn">⚠ {rank_desc}</span>',
                        unsafe_allow_html=True)
        _section_end()

        # ── Section 2 · Database ─────────────────────────────────────────────
        _section("2️⃣", "Database")
        gsea_db_source, gsea_gene_set, gsea_gmt_path, _ = _render_db_selector("gsea")
        _section_end()

        # ── Section 3 · Parameters ───────────────────────────────────────────
        _section("3️⃣", "Parameters")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            gsea_min  = st.number_input("Min set size",  5, 500, 15, key="gsea_min",
                                        help="Min genes per set to test")
        with c2:
            gsea_max  = st.number_input("Max set size", 10, 5000, 500, step=50,
                                        key="gsea_max", help="Max genes per set to test")
        with c3:
            gsea_perm = st.number_input("Permutations", 50, 2000, 100, step=50,
                                        key="gsea_perm",
                                        help="More = more accurate p-values, slower")
        with c4:
            gsea_topn = st.number_input("Top pathways", 5, 100, 20, key="gsea_topn",
                                        help="Max pathways shown in plots")
        c5, c6 = st.columns([1, 3])
        with c5:
            gsea_seed = st.number_input("Random seed", 0, 9999, 42, key="gsea_seed")
        with c6:
            gsea_lbl  = st.text_input("Run label (used in filenames)",
                                      value="GSEA_run", key="gsea_label")
        _section_end()

        # ── Section 4 · Run ──────────────────────────────────────────────────
        _section("4️⃣", "Run Analysis")
        gsea_ready = rank_ok and bool(gsea_gene_set)
        if not gsea_ready:
            missing = []
            if not rank_ok:       missing.append("ranked list")
            if not gsea_gene_set: missing.append("database")
            st.caption(f"⬆️ Still needed: {', '.join(missing)}")

        if st.button("▶ Run GSEA", key="run_gsea_btn",
                     type="primary", disabled=not gsea_ready,
                     use_container_width=True):
            _run_gsea_prerank(
                ranked_series   = ranked_series,
                gene_set        = gsea_gene_set,
                run_label       = str(gsea_lbl),
                num_pathways    = int(gsea_topn),
                min_size        = int(gsea_min),
                max_size        = int(gsea_max),
                permutation_num = int(gsea_perm),
                seed            = int(gsea_seed),
            )
            st.success("✅ GSEA completed!")
        _section_end()
