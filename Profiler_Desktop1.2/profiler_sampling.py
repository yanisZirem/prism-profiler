"""
Software Name: Profiler
Module Name: Features importance
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

import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
import streamlit as st


def apply_sampling(df, technique='none', _progress_bar=None):
    X = df.drop(['Class', 'File', 'RT', 'Sum'], axis=1, errors='ignore')
    y = df['Class']
    sampler = None

    # Vérifiez la distribution des classes avant le suréchantillonnage
    class_counts_before = y.value_counts()
    st.write("Class Distribution Before Oversampling :")
    st.write(class_counts_before)

    if technique == 'smote':
        sampler = SMOTE(k_neighbors=1, sampling_strategy='not majority', n_jobs=-1)
    elif technique == 'adasyn':
        sampler = ADASYN(n_neighbors=2, sampling_strategy='minority', n_jobs=-1)

    if sampler:
        try:
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            resampled_df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=['Class'])], axis=1)

            # Vérifier l'équilibre des classes après le suréchantillonnage
            class_counts_after = resampled_df['Class'].value_counts()
            st.write("Class Distribution After Oversampling :")
            st.write(class_counts_after)

            _progress_bar.progress(1.0)
            return resampled_df
        except ValueError as e:
            st.error(f"Erreur lors du suréchantillonnage : {str(e)}")
            _progress_bar.progress(1.0)
            return df
    else:
        _progress_bar.progress(1.0)
        return df
