
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
import streamlit as st



# @st.cache_data
def apply_sampling(df, technique='none', _progress_bar=None):
    NON_FEATURE_COLS = ['Class', 'ID', 'File', 'RT', 'Sum', 'Original_index']
    meta_cols = [c for c in df.columns if str(c).endswith('_meta')]
    cols_to_drop = [c for c in NON_FEATURE_COLS + meta_cols if c in df.columns]
    X = df.drop(cols_to_drop, axis=1, errors='ignore').select_dtypes(include='number')
    y = df['Class']
    sampler = None

    # Vérifiez la distribution des classes avant le suréchantillonnage
    class_counts_before = y.value_counts()
    st.write("Class Distribution Before Oversampling :")
    st.write(class_counts_before)

    if technique == 'smote':
        # ⚡ n_jobs=-1 → parallélise le calcul des voisins
        sampler = SMOTE(k_neighbors=1, sampling_strategy='not majority', n_jobs=-1)
    elif technique == 'adasyn':
        # ⚡ n_jobs=-1
        sampler = ADASYN(n_neighbors=2, sampling_strategy='minority', n_jobs=-1)

    if sampler:
        # ── NaN guard ────────────────────────────────────────────────────────
        nan_cols = X.columns[X.isna().any()].tolist()
        if nan_cols:
            st.error(
                "⚠️ **Missing values detected in your dataset.**\n\n"
                "Oversampling algorithms (SMOTE / ADASYN) cannot handle NaN values. "
                "Please follow these steps before applying oversampling:\n\n"
                "1. Go to the **Preprocessing** step\n"
                "2. Apply **Imputation** (e.g. mean, median, or KNN imputer...)\n"
                "3. Come back here and select the **preprocessed dataset**\n"
                "4. Then apply oversampling\n\n"
                f"Columns with missing values: `{', '.join(nan_cols)}`"
            )
            _progress_bar.progress(1.0)
            return df
        # ─────────────────────────────────────────────────────────────────────
        try:
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            resampled_df = pd.concat([
                pd.DataFrame(X_resampled, columns=X.columns),
                pd.DataFrame(y_resampled, columns=['Class'])
            ], axis=1)

            # Vérifier l'équilibre des classes après le suréchantillonnage
            class_counts_after = resampled_df['Class'].value_counts()
            st.write("Class Distribution After Oversampling :")
            st.write(class_counts_after)

            _progress_bar.progress(1.0)
            return resampled_df
        except ValueError as e:
            st.error(f"❌ Oversampling error: {str(e)}")
            _progress_bar.progress(1.0)
            return df
    else:
        _progress_bar.progress(1.0)
        return df
