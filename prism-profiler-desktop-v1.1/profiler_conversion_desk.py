"""
Software Name: Profiler
Module name : conversion
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
import subprocess
import streamlit as st
import numpy as np

def convert_to_float32(df):
    """Convert all numeric columns to float32."""
    for col in df.select_dtypes(include=[np.float64]).columns:
        df[col] = df[col].astype(np.float32)
    return df



def convert_raw_to_mzml(raw_files_path, output_dir, file_type, mass_range=None, peak_picking=False, lock_mass=None, output_format='mzML'):
    """
    Convertit des fichiers bruts de différents fabricants (Thermo, Waters, Bruker) en mzML ou mzXML.

    :param raw_files_path: Chemin du dossier contenant les fichiers bruts.
    :param output_dir: Dossier de sortie pour les fichiers convertis.
    :param file_type: Type de fichier brut ('thermo', 'waters', 'bruker').
    :param mass_range: Plage de masse à utiliser (ex: "400-1200").
    :param peak_picking: Activer/désactiver le peak picking.
    :param lock_mass: Valeur de lock mass (uniquement pour Waters).
    :param output_format: Format de sortie ('mzML', 'mzXML').
    """
    if not os.path.isdir(raw_files_path):
        st.error("Chemin du dossier invalide.")
        return

    file_extensions = {
        "thermo": ".raw",
        "waters": ".raw",
        "bruker": ".d"
    }
    
    if file_type not in file_extensions:
        st.error(f"Type de fichier non supporté : {file_type}")
        return

    file_ext = file_extensions[file_type]
    raw_files = [f for f in os.listdir(raw_files_path) if f.lower().endswith(file_ext)]
    
    if not raw_files:
        st.error(f"Aucun fichier {file_ext} trouvé dans le dossier.")
        return

    st.write(f"Conversion des fichiers {file_type} en {output_format}...")
    progress_bar = st.progress(0)

    for i, raw_file in enumerate(raw_files):
        raw_file_path = os.path.join(raw_files_path, raw_file)


        command = [
            "msconvert",
            raw_file_path,
            "-o", output_dir,
            "--" + output_format
        ]

        if file_type == "waters" and lock_mass:
            command.extend(["--filter", f"lockmassRefiner mz={lock_mass} tol=1.0"])
        
        if mass_range:
            command.extend(["--filter", f"mzWindow {mass_range}"])
        
        if peak_picking:
            command.extend(["--filter", "peakPicking true 1-"])

        try:
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            st.write(f"Sortie pour {raw_file} :\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            st.error(f"Erreur lors de la conversion de {raw_file} : {e.stderr}")

        progress_bar.progress((i + 1) / len(raw_files))

    st.success(f"Conversion completed for {len(raw_files)} files {file_type} into {output_format}.")
