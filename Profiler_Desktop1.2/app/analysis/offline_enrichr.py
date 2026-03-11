"""
Software Name: Profiler
Module Name: Enrichment (Offline Version)
Author: Yanis Zirem
Email : yanis.zirem@yahoo.com / yanis.zirem@univ-lille.fr
Creation Date: 15/01/2025
Last Updated: 05/12/2025
Version: 1.1.0 (Offline)

License: l'Agence pour la Protection des Programmes IDDN
"""

import gseapy as gp
import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objs as go
import networkx as nx
import gc
import os
from pathlib import Path


def get_local_gmt_files():
    """
    Charge les fichiers GMT locaux depuis le dossier 'gmt_databases'
    Retourne un dictionnaire organisé par catégories
    """
    gmt_folder = Path("gmt_databases")
    
    if not gmt_folder.exists():
        st.warning("⚠️ Le dossier 'gmt_databases' n'existe pas.")
        st.info("""
        **Pour utiliser cette fonctionnalité :**
        1. Créez un dossier 'gmt_databases' à la racine
        2. Utilisez le script de téléchargement fourni
        3. Ou téléchargez manuellement les fichiers GMT depuis MSigDB ou Enrichr
        """)
        return {}
    
    gmt_files = list(gmt_folder.glob("*.gmt"))
    
    if not gmt_files:
        st.warning("⚠️ Aucun fichier GMT trouvé dans 'gmt_databases'")
        return {}
    
    # Organiser par catégories basées sur les noms de fichiers
    categories = {
        "KEGG": [],
        "GO Biological Process": [],
        "GO Molecular Function": [],
        "GO Cellular Component": [],
        "Reactome": [],
        "WikiPathways": [],
        "MSigDB": [],
        "Disease": [],
        "Drug": [],
        "Tissue": [],
        "Cell Lines": [],
        "Transcription Factors": [],
        "Other": []
    }
    
    for gmt_file in gmt_files:
        file_name = gmt_file.stem
        file_path = str(gmt_file)
        
        # Catégorisation intelligente
        if "KEGG" in file_name:
            categories["KEGG"].append((file_name, file_path))
        elif "GO_Biological_Process" in file_name:
            categories["GO Biological Process"].append((file_name, file_path))
        elif "GO_Molecular_Function" in file_name:
            categories["GO Molecular Function"].append((file_name, file_path))
        elif "GO_Cellular_Component" in file_name:
            categories["GO Cellular Component"].append((file_name, file_path))
        elif "Reactome" in file_name:
            categories["Reactome"].append((file_name, file_path))
        elif "WikiPathway" in file_name:
            categories["WikiPathways"].append((file_name, file_path))
        elif "MSigDB" in file_name or "Hallmark" in file_name:
            categories["MSigDB"].append((file_name, file_path))
        elif any(x in file_name for x in ["Disease", "DisGeNET", "OMIM"]):
            categories["Disease"].append((file_name, file_path))
        elif "Drug" in file_name:
            categories["Drug"].append((file_name, file_path))
        elif any(x in file_name for x in ["Tissue", "GTEx", "Human_Gene_Atlas"]):
            categories["Tissue"].append((file_name, file_path))
        elif any(x in file_name for x in ["Cell", "CCLE", "ARCHS4_Cell"]):
            categories["Cell Lines"].append((file_name, file_path))
        elif any(x in file_name for x in ["ChEA", "ENCODE_TF", "TRRUST", "TF"]):
            categories["Transcription Factors"].append((file_name, file_path))
        else:
            categories["Other"].append((file_name, file_path))
    
    # Retirer les catégories vides et trier
    categories = {k: sorted(v) for k, v in categories.items() if v}
    
    return categories


def load_gene_sets_offline():
    """
    Version offline : charge les bases de données GMT locales
    """
    local_databases = get_local_gmt_files()
    
    if not local_databases:
        return {}
    
    total_files = sum(len(v) for v in local_databases.values())
    st.success(f"✅ {total_files} fichiers GMT chargés depuis le mode offline !")
    
    return local_databases


def perform_gsea_offline(gene_lists, class_names, gene_set_path, organism, num_pathways):
    """
    Version offline utilisant des fichiers GMT locaux
    
    Args:
        gene_lists: Liste de listes de gènes
        class_names: Noms des classes
        gene_set_path: Chemin vers le fichier GMT local
        organism: Organisme (non utilisé en mode local)
        num_pathways: Nombre de pathways à afficher
    """
    # Vérifier si le fichier existe
    if not os.path.exists(gene_set_path):
        st.error(f"❌ Le fichier {gene_set_path} n'existe pas.")
        return
    
    results = []
    gene_mapping = []
    
    st.info(f"🔬 Analyse d'enrichissement avec: {Path(gene_set_path).stem}")
    
    for i, gene_list in enumerate(gene_lists):
        if not gene_list:
            st.warning(f"⚠️ La classe '{class_names[i]}' est vide, ignorée.")
            continue
            
        try:
            with st.spinner(f"Analyse de la classe '{class_names[i]}'..."):
                # Enrichissement avec fichier GMT local
                enrichment_results = gp.enrichr(
                    gene_list=gene_list,
                    gene_sets=gene_set_path,
                    organism='Human',  # Paramètre ignoré pour fichiers locaux
                    outdir=None,
                    cutoff=0.05
                )
                
                if not enrichment_results.results.empty:
                    top_results = enrichment_results.results.head(num_pathways).copy()
                    top_results["Class"] = class_names[i]
                    results.append(top_results[['Term', 'Combined Score', 'Adjusted P-value', 'Class']])
                    
                    for _, row in top_results.iterrows():
                        genes = row["Genes"].split(";") if "Genes" in row else []
                        gene_mapping.append({
                            "Pathway": row["Term"],
                            "Genes": genes,
                            "Class": class_names[i],
                            "P-value": row.get("Adjusted P-value", "N/A")
                        })
                else:
                    st.warning(f"⚠️ Aucun résultat significatif pour '{class_names[i]}'")
        
        except Exception as e:
            st.error(f"❌ Erreur lors de l'enrichissement pour '{class_names[i]}': {str(e)}")
            continue
    
    if not results:
        st.warning("⚠️ Aucun résultat d'enrichissement trouvé pour les listes de gènes fournies.")
        st.info("💡 Suggestions: Vérifiez que vos gènes correspondent au format de la base de données (symboles, IDs, etc.)")
        return
    
    combined_results = pd.concat(results, ignore_index=True)
    gene_df = pd.DataFrame(gene_mapping)
    
    color_map = st.session_state.get('class_colors', {})
    
    st.success(f"✅ Analyse terminée ! {len(combined_results)} pathways enrichis trouvés.")
    
    # === Graphique 1 : Bar plot des pathways enrichis ===
    st.markdown("### 📊 Top Enriched Pathways")
    
    fig = px.bar(
        combined_results,
        x='Combined Score',
        y='Term',
        color='Class',
        title=f"Top {num_pathways} Enriched Pathways per Class",
        labels={'Combined Score': 'Combined Score', 'Term': 'Pathway'},
        orientation='h',
        barmode='group',
        color_discrete_map=color_map,
        hover_data=['Adjusted P-value']
    )
    
    fig.update_layout(
        height=max(400, len(combined_results) * 20),
        legend=dict(
            font=dict(size=14, color='black'),
            title=dict(text="Class", font=dict(size=14, color='black'))
        ),
        xaxis=dict(
            title=dict(text='Combined Score', font=dict(size=14, color='black', family="Arial, bold")),
            tickfont=dict(size=12, color='black')
        ),
        yaxis=dict(
            title=dict(text='Pathways', font=dict(size=14, color='black', family="Arial, bold")),
            tickfont=dict(size=11, color='black')
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # === Graphique 2 : Heatmap ===
    st.markdown("### 🔥 Pathway Enrichment Heatmap")
    
    heatmap_df = combined_results.pivot_table(
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
    
    fig.update_layout(
        title=dict(
            text="Pathway Enrichment Heatmap",
            font=dict(size=16, color='black', family="Arial, bold")
        ),
        xaxis=dict(
            title=dict(text="Pathway", font=dict(size=14, color='black', family="Arial, bold")),
            tickangle=45,
            tickfont=dict(size=10, color='black')
        ),
        yaxis=dict(
            title=dict(text="Class", font=dict(size=14, color='black', family="Arial, bold")),
            tickfont=dict(size=12, color='black')
        ),
        coloraxis_colorbar=dict(
            title=dict(text='Score', font=dict(size=12, color='black')),
            tickfont=dict(size=10, color='black')
        ),
        height=max(300, 50 * len(heatmap_df))
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # === Graphique 3 : Gene Count ===
    st.markdown("### 🧬 Gene Count per Pathway")
    
    gene_counts = pd.DataFrame(gene_mapping)
    gene_counts['Gene Count'] = gene_counts['Genes'].apply(len)
    
    fig = px.bar(
        gene_counts,
        x='Gene Count',
        y='Pathway',
        color='Class',
        orientation='h',
        title="Number of Genes per Enriched Pathway",
        color_discrete_map=color_map
    )
    
    fig.update_layout(
        height=max(400, len(gene_counts) * 20),
        xaxis=dict(
            title='Gene Count',
            title_font=dict(size=14, color='black', family="Arial, bold"),
            tickfont=dict(size=12, color='black')
        ),
        yaxis=dict(
            title='Pathway',
            title_font=dict(size=14, color='black', family="Arial, bold"),
            tickfont=dict(size=11, color='black')
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # === Graphique 4 : Matrice Gène x Pathway ===
    st.markdown("### 🎯 Gene Involvement Matrix")
    
    all_genes = sorted(set(g for row in gene_mapping for g in row["Genes"]))
    all_pathways = sorted(set(row["Pathway"] for row in gene_mapping))
    
    # Limiter la taille pour les grandes matrices
    max_display = 100
    if len(all_genes) > max_display:
        st.warning(f"⚠️ Trop de gènes ({len(all_genes)}). Affichage limité aux {max_display} premiers.")
        all_genes = all_genes[:max_display]
    
    matrix = pd.DataFrame(0, index=all_genes, columns=all_pathways)
    
    for row in gene_mapping:
        for gene in row["Genes"]:
            if gene in matrix.index:
                matrix.loc[gene, row["Pathway"]] = 1
    
    fig = px.imshow(
        matrix,
        labels=dict(x="Pathways", y="Genes", color="Present"),
        x=matrix.columns,
        y=matrix.index,
        color_continuous_scale="Blues",
        aspect="auto"
    )
    
    fig.update_layout(
        title=dict(
            text="Gene Involvement Across Pathways",
            font=dict(size=16, color='black', family="Arial Bold")
        ),
        xaxis=dict(
            title=dict(text="Pathways", font=dict(size=14, color='black', family="Arial Bold")),
            tickangle=45,
            tickfont=dict(size=9, color='black')
        ),
        yaxis=dict(
            title=dict(text="Genes", font=dict(size=14, color='black', family="Arial Bold")),
            tickfont=dict(size=8, color='black')
        ),
        height=max(600, len(all_genes) * 10)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # === Graphique 5 : Network ===
    st.markdown("### 🕸️ Gene Interaction Network")
    
    # Limiter le réseau pour de meilleures performances
    max_genes_network = 50
    if len(all_genes) > max_genes_network:
        st.info(f"ℹ️ Réseau limité aux {max_genes_network} premiers gènes pour des raisons de performance")
        genes_for_network = all_genes[:max_genes_network]
    else:
        genes_for_network = all_genes
    
    G = nx.Graph()
    gene_to_color = {}
    
    for row in gene_mapping:
        protein_type = row["Class"]
        color = color_map.get(protein_type, '#00BFFF')
        for gene in row["Genes"]:
            if gene in genes_for_network:
                gene_to_color[gene] = color
                G.add_node(gene)
    
    # Ajouter les arêtes entre gènes du même pathway
    for row in gene_mapping:
        genes = [g for g in row["Genes"] if g in genes_for_network]
        for i in range(len(genes)):
            for j in range(i+1, len(genes)):
                G.add_edge(genes[i], genes[j])
    
    if len(G.nodes()) == 0:
        st.warning("⚠️ Pas assez de gènes pour créer un réseau")
    else:
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += (x0, x1, None)
            edge_trace['y'] += (y0, y1, None)
        
        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers+text',
            textposition='top center',
            hoverinfo='text',
            marker=dict(
                size=10,
                color=[],
                line=dict(width=1, color='white')
            ),
            textfont=dict(size=8)
        )
        
        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += (x,)
            node_trace['y'] += (y,)
            node_trace['text'] += (node,)
            node_trace['marker']['color'] += (gene_to_color[node],)
        
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=0),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=600
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # === Tableau des résultats ===
    st.markdown("### 📋 Detailed Results")
    
    with st.expander("View genes per pathway", expanded=False):
        st.dataframe(gene_df, use_container_width=True)
    
    with st.expander("Download results", expanded=False):
        csv = combined_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download enrichment results (CSV)",
            data=csv,
            file_name="enrichment_results.csv",
            mime="text/csv"
        )
    
    # Nettoyage mémoire
    del results, gene_mapping, combined_results, gene_df, G
    gc.collect()