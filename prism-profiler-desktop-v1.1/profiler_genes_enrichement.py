"""
Software Name: Profiler
Module Name: Enrichement
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

import gseapy as gp
import pandas as pd
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import networkx as nx
import gc
from sklearn.metrics import jaccard_score
import numpy as np



def load_gene_sets():
    all_sets = gp.get_library_name()
    categories = {
        "KEGG": [gs for gs in all_sets if "KEGG" in gs],
        "GO": [gs for gs in all_sets if "GO" in gs],
        "Reactome": [gs for gs in all_sets if "Reactome" in gs],
        "ARCHS4": [gs for gs in all_sets if "ARCHS4" in gs],
        "Drug": [gs for gs in all_sets if "Drug" in gs],
        "MSigDB": [gs for gs in all_sets if "MSig" in gs],
        "Other": [gs for gs in all_sets if all(x not in gs for x in ["KEGG", "GO", "REACTOME", "ARCHS4", "Drug", "MSig"])]
    }
    return categories


def perform_gsea(gene_lists, class_names, gene_set, organism, num_pathways):
    results = []
    gene_mapping = []

    for i, gene_list in enumerate(gene_lists):
        enrichment_results = gp.enrichr(
            gene_list=gene_list,
            gene_sets=[gene_set],
            organism=organism
        )

        if not enrichment_results.results.empty:
            top_results = enrichment_results.results.head(num_pathways).copy()
            top_results["Class"] = class_names[i]
            results.append(top_results[['Term', 'Combined Score', 'Class']])

            for _, row in top_results.iterrows():
                genes = row["Genes"].split(";") if "Genes" in row else []
                gene_mapping.append({"Pathway": row["Term"], "Genes": genes, "Class": class_names[i]})

    if results:
        combined_results = pd.concat(results, ignore_index=True)
        gene_df = pd.DataFrame(gene_mapping)
    else:
        st.write("No enrichment results found for the provided gene lists.")
        return

    color_map = st.session_state.get('class_colors', {})



    fig = px.bar(
        combined_results,
        x='Combined Score',
        y='Term',
        color='Class',
        title=f"Top Enriched Pathways using {gene_set}",
        labels={'Combined Score': 'Combined Score', 'Term': 'Pathway'},
        orientation='h',
        barmode='overlay',
        opacity=0.7,
        color_discrete_map=color_map
    )

    # fig.update_layout(
    #     legend=dict(
    #         font=dict(
    #             size=16,
    #             color='black'
    #         ),
    #         title_font=dict(
    #             size=16,
    #             color='black'
    #         )
    #     ),
    #     xaxis=dict(
    #         title_font=dict(
    #             size=16,
    #             color='black',
    #             family="Arial, bold"
    #         ),
    #         tickfont=dict(
    #             size=14,
    #             color='black',
    #             family="Arial, bold"
    #         )
    #     ),
    #     yaxis=dict(
    #         title_font=dict(
    #             size=16,
    #             color='black',
    #             family="Arial, bold"
    #         ),
    #         tickfont=dict(
    #             size=14,
    #             color='black',
    #             family="Arial, bold"
    #         )
    #     )
    # )
    fig.update_layout(
        legend=dict(
            font=dict(
                size=16,
                color='black'
            ),
            title=dict(
                text=fig.layout.legend.title.text if fig.layout.legend.title.text else "Legend",
                font=dict(size=16, color='black')
            )
        ),
        xaxis=dict(
            title=dict(
                text='Combiend Score',  # remplace par ton texte
                font=dict(size=16, color='black', family="Arial, bold")
            ),
            tickfont=dict(
                size=14,
                color='black',
                family="Arial, bold"
            )
        ),
        yaxis=dict(
            title=dict(
                text='Pathways',  # remplace par ton texte
                font=dict(size=16, color='black', family="Arial, bold")
            ),
            tickfont=dict(
                size=14,
                color='black',
                family="Arial, bold"
            )
        )
    )

    st.plotly_chart(fig)

    # st.markdown("**Heatmap of Pathway Enrichment per Class**")

    # Pivot table: rows = Class, columns = Pathway, values = Combined Score
    heatmap_df = pd.concat(results, ignore_index=True).pivot_table(
        index='Class',
        columns='Term',
        values='Combined Score',
        fill_value=0
    )

    fig = px.imshow(
        heatmap_df,
        labels=dict(x="Pathway", y="Class", color="Combined Score"),
        color_continuous_scale="Reds",
        aspect="auto"
    )


    # # st.plotly_chart(fig)
    # fig.update_layout(
    #     title=dict(
    #         text="Pathway Enrichment Heatmap",
    #         font=dict(size=18, color='black', family="Arial, bold")
    #     ),
    #     xaxis=dict(
    #         title="Pathway",
    #         tickangle=45,
    #         tickfont=dict(size=12, color='black', family="Arial, bold"),
    #         title_font=dict(size=16, color='black', family="Arial, bold")
    #     ),
    #     yaxis=dict(
    #         title="Class",
    #         tickfont=dict(size=14, color='black', family="Arial, bold"),
    #         title_font=dict(size=16, color='black', family="Arial, bold")
    #     ),
    #     coloraxis_colorbar=dict(
    #         title='Combined Score',
    #         titlefont=dict(size=14, color='black', family="Arial, bold"),
    #         tickfont=dict(size=12, color='black', family="Arial")
    #     ),
    #     margin=dict(l=50, r=50, t=60, b=150),
    #     width=900,
    #     height=400 + 20 * len(heatmap_df)
    # )
    fig.update_layout(
        title=dict(
            text="Pathway Enrichment Heatmap",
            font=dict(size=18, color='black', family="Arial, bold")
        ),
        xaxis=dict(
            title=dict(
                text="Pathway",
                font=dict(size=16, color='black', family="Arial, bold")
            ),
            tickangle=45,
            tickfont=dict(size=12, color='black', family="Arial, bold")
        ),
        yaxis=dict(
            title=dict(
                text="Class",
                font=dict(size=16, color='black', family="Arial, bold")
            ),
            tickfont=dict(size=14, color='black', family="Arial, bold")
        ),
        coloraxis_colorbar=dict(
            title=dict(
                text='Combined Score',
                font=dict(size=14, color='black', family="Arial, bold")
            ),
            tickfont=dict(size=12, color='black', family="Arial")
        ),
        margin=dict(l=50, r=50, t=60, b=150),
        width=900,
        height=400 + 20 * len(heatmap_df)
    )

    st.plotly_chart(fig)


    gene_counts = pd.DataFrame(gene_mapping)
    gene_counts['Gene Count'] = gene_counts['Genes'].apply(len)

    fig = px.bar(
        gene_counts,
        x='Gene Count',
        y='Pathway',
        color='Class',
        orientation='h',
        title="Gene Count per Enriched Pathway",
        color_discrete_map=color_map
    )

    fig.update_layout(
        height=600,
        title=dict(
            font=dict(size=18, color='black', family="Arial, bold")
        ),
        legend=dict(
            font=dict(
                size=16,
                color='black'
            ),
            title_font=dict(
                size=16,
                color='black'
            )
        ),
        xaxis=dict(
            title='Gene Count',
            title_font=dict(
                size=16,
                color='black',
                family="Arial, bold"
            ),
            tickfont=dict(
                size=14,
                color='black',
                family="Arial, bold"
            )
        ),
        yaxis=dict(
            title='Pathway',
            title_font=dict(
                size=16,
                color='black',
                family="Arial, bold"
            ),
            tickfont=dict(
                size=14,
                color='black',
                family="Arial, bold"
            )
        )
    )

    st.plotly_chart(fig)



    # Création de la matrice binaire Gène x Pathway
    all_genes = sorted(set(g for row in gene_mapping for g in row["Genes"]))
    all_pathways = sorted(set(row["Pathway"] for row in gene_mapping))
    matrix = pd.DataFrame(0, index=all_genes, columns=all_pathways)

    for row in gene_mapping:
        for gene in row["Genes"]:
            matrix.loc[gene, row["Pathway"]] = 1

    # heatmap
    fig = px.imshow(
        matrix,
        labels=dict(x="Pathways", y="Genes", color="Presence"),
        x=matrix.columns,
        y=matrix.index,
        color_continuous_scale="Blues",
        aspect="auto"
    )

    
    # fig.update_layout(
    #     title=dict(
    #         text="Gene Involvement Across Pathways",
    #         font=dict(size=24, color='black', family="Arial Bold")
    #     ),
    #     xaxis=dict(
    #         title="Pathways",
    #         tickangle=45,
    #         tickfont=dict(size=14, color='black', family="Arial Bold"),
    #         title_font=dict(size=18, color='black', family="Arial Bold")
    #     ),
    #     yaxis=dict(
    #         title="Genes",
    #         tickfont=dict(size=14, color='black', family="Arial Bold"),
    #         title_font=dict(size=18, color='black', family="Arial Bold")
    #     ),
    #     coloraxis_colorbar=dict(
    #         title='Presence',
    #         titlefont=dict(size=16, color='black', family="Arial Bold"),
    #         tickfont=dict(size=14, color='black', family="Arial")
    #     ),
    #     margin=dict(l=120, r=30, t=80, b=120),
    #     width=1000,
    #     height=800
    # )

    fig.update_layout(
        title=dict(
            text="Gene Involvement Across Pathways",
            font=dict(size=24, color='black', family="Arial Bold")
        ),
        xaxis=dict(
            title=dict(
                text="Pathways",
                font=dict(size=18, color='black', family="Arial Bold")
            ),
            tickangle=45,
            tickfont=dict(size=14, color='black', family="Arial Bold")
        ),
        yaxis=dict(
            title=dict(
                text="Genes",
                font=dict(size=18, color='black', family="Arial Bold")
            ),
            tickfont=dict(size=14, color='black', family="Arial Bold")
        ),
        coloraxis_colorbar=dict(
            title=dict(
                text='Presence',
                font=dict(size=16, color='black', family="Arial Bold")
            ),
            tickfont=dict(size=14, color='black', family="Arial")
        ),
        margin=dict(l=120, r=30, t=80, b=120),
        width=1000,
        height=800
    )



    # Affichage
    st.plotly_chart(fig, use_container_width=True)



    st.markdown("**Genes involved in each pathway**")
    st.dataframe(gene_df)
    csv = gene_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download table as CSV",
        data=csv,
        file_name="genes_in_pathways.csv",
        mime="text/csv"
    )

    st.markdown("**Interactive Gene Interaction Network**")
    G = nx.Graph()

    gene_to_color = {}
    for row in gene_mapping:
        protein_type = row["Class"]
        color = color_map.get(protein_type, '#00BFFF')
        for gene in row["Genes"]:
            gene_to_color[gene] = color

    for row in gene_mapping:
        genes = row["Genes"]
        for gene in genes:
            G.add_node(gene)
        for i in range(len(genes)):
            for j in range(i+1, len(genes)):
                G.add_edge(genes[i], genes[j])

    pos = nx.spring_layout(G)

    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)

    # node_trace = go.Scatter(
    #     x=[],
    #     y=[],
    #     text=[],
    #     mode='markers+text',
    #     textposition='top center',
    #     hoverinfo='text',
    #     marker=dict(
    #         showscale=False,
    #         colorscale='jet',
    #         size=10,
    #         colorbar=dict(
    #             thickness=15,
    #             title='Node Connections',
    #             xanchor='left',
    #             titleside='right'
    #         ),
    #         color=[],
    #     )
    # )
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        textposition='top center',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            colorscale='jet',
            size=10,
            colorbar=dict(
                thickness=15,
                title=dict(
                    text='Node Connections',
                    side='right'
                ),
                xanchor='left'
            ),
            color=[],
        )
    )
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_trace['text'] += (node,)
        node_trace['marker']['color'] += (gene_to_color[node],)

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False)
                    ))

    st.plotly_chart(fig)

    del results, gene_mapping, combined_results, gene_df, G, pos, edge_trace, node_trace
    gc.collect()


