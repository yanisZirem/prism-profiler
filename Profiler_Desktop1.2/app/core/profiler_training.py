"""
Software Name: Profiler
Module Name: Features importance
Author: Yanis Zirem
Email : yanis.zirem@yahoo.com / yanis.zirem@univ-lille.fr
Creation Date: 15/01/2025
Last Updated: 30/04/2026
Version: 1.2.0

Context:
This module is part of the "Profiler" project, originally developed for a web version (https://prism-profiler.univ-lille.fr) and now adapted for a desktop version (profiler_desktop_GUI).
It is designed for archiving on Zenodo and integration into GitHub releases.

License: l’Agence pour la Protection des Programmes IDDN (InterDeposit Digital Number) : FR2 .0013 .0300044 .0005 .S6 .C7 .20258 .0009 .312301
Citation:
If Profiler or this module (a part of Profiler) is used in a publication, please cite:
Zirem, Y. (2025). Profiler: an open web platform for multi-omics analysis. Journal of Bioinformatics. doi:10.1093/bioinformatics/btaf644

Links:
- GitHub temporary Repository: https://github.com/yanisZirem/Profiler_v1_requests_datatests

"""



# === Standard Library ===
import gc
import os

# ── Desktop: détection automatique des CPUs ───────────────────────────────────
_N_CPUS   = os.cpu_count() or 2     # tous les cœurs physiques/logiques
_N_JOBS   = -1                       # sklearn/joblib → utilise tous les CPUs
_N_CV_JOBS = _N_CPUS                 # parallélisme des folds CV

# === Data manipulation ===
import numpy as np
import pandas as pd

# === Visualization ===
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs import Figure

# === Web app ===
import streamlit as st

# === Machine Learning - Scikit-learn ===
from sklearn.base import clone

# Models - Ensembles
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
    VotingClassifier
)

# Models - Linear
from sklearn.linear_model import (
    SGDClassifier,
    LogisticRegression,
    RidgeClassifier,
    PassiveAggressiveClassifier,
    Perceptron,
    Lasso
)

# Models - Naive Bayes
from sklearn.naive_bayes import GaussianNB, BernoulliNB

# Models - Trees
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

# Models - SVM
from sklearn.svm import SVC, NuSVC, LinearSVC

# Models - Neighbors
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

# Models - Other
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)

# === LightGBM ===
from lightgbm import LGBMClassifier

# === Preprocessing & Pipelines ===
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# === Model Selection & Evaluation ===
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    learning_curve
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score
)


from sklearn.calibration import CalibratedClassifierCV



def train_models(
    X, y,
    n_splits=3,
    progress_bar=None,
    n_jobs_cv=_N_CV_JOBS,   # desktop: tous les CPUs pour les folds CV
    calibrate=False
):
    # --- Encodage labels ---
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_

    # --- Préprocess ---
    preprocess_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ])
    X_processed = preprocess_pipeline.fit_transform(X)

    feature_names = X.columns.tolist() if hasattr(X, 'columns') else [
        f'feature_{i}' for i in range(X.shape[1])
    ]

    # --- Adapter KNN ---
    min_samples_per_class = np.min(np.bincount(y_encoded))
    adapted_n_neighbors = min(5, min_samples_per_class)
    adapted_n_neighbors = max(1, adapted_n_neighbors)

    # ── Desktop: n_jobs=-1 dans chaque modèle + parallélisme CV ─────────────
    # Les modèles internes utilisent tous les cœurs disponibles.
    # cross_val_predict parallélise en plus les folds (n_jobs_cv).
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=_N_JOBS),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, algorithm="SAMME"),
        'SGD': SGDClassifier(n_jobs=_N_JOBS),
        'NaiveBayes_Gaussian': GaussianNB(),
        'DecisionTree': DecisionTreeClassifier(max_depth=None),
        'LogisticRegression': LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=_N_JOBS),
        'Perceptron': Perceptron(n_jobs=_N_JOBS),
        'RidgeClassifier': RidgeClassifier(),
        'Lasso': LogisticRegression(
            penalty='l1',
            solver='saga',
            max_iter=2000,
            C=0.5,
            n_jobs=_N_JOBS,
        ),
        'ElasticNet': LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            l1_ratio=0.5,
            max_iter=3000,
            C=1.0,
            n_jobs=_N_JOBS,
        ),
        'PassiveAggressive': PassiveAggressiveClassifier(n_jobs=_N_JOBS),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=200, n_jobs=_N_JOBS, max_depth=None),
        'KNeighbors': KNeighborsClassifier(n_neighbors=adapted_n_neighbors, n_jobs=_N_JOBS),
        'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
        'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100),
        'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=200),
        'LGBMClassifier': LGBMClassifier(n_jobs=_N_JOBS, max_depth=-1, n_estimators=200),
    }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    results = {}
    total_models = len(models)

    # Modèles où calibration peut être utile
    models_requiring_calibration = (
        RidgeClassifier,
        SGDClassifier,
        Perceptron,
        PassiveAggressiveClassifier
    )

    for i, (model_name, model) in enumerate(models.items()):

        # --- Calibration optionnelle ---
        if calibrate and isinstance(model, models_requiring_calibration):
            # calibration interne = coûteux
            model_to_use = CalibratedClassifierCV(
                estimator=model,
                method="sigmoid",
                cv=3
            )
        else:
            model_to_use = model

        pipeline = Pipeline([
            ('model', model_to_use)
        ])

        # --- 1) y_pred via CV (1 seule passe) ---
        y_pred = cross_val_predict(
            pipeline,
            X_processed,
            y_encoded,
            cv=cv,
            method='predict',
            n_jobs=n_jobs_cv
        )

        # --- 2) confidence_scores + probas via CV (1 seule passe) ---
        confidence_scores = np.full(len(y_encoded), np.nan)
        probas = None  # will store (n_samples, n_classes) if available

        # on essaie predict_proba d'abord
        try:
            probas = cross_val_predict(
                pipeline,
                X_processed,
                y_encoded,
                cv=cv,
                method='predict_proba',
                n_jobs=n_jobs_cv
            )
            confidence_scores = np.max(probas, axis=1)

        except Exception:
            # sinon decision_function
            try:
                decision_scores = cross_val_predict(
                    pipeline,
                    X_processed,
                    y_encoded,
                    cv=cv,
                    method='decision_function',
                    n_jobs=n_jobs_cv
                )

                # normalisation simple fold-agnostic
                if decision_scores.ndim == 2:
                    # multi-class
                    ds_min = decision_scores.min(axis=1, keepdims=True)
                    ds_max = decision_scores.max(axis=1, keepdims=True)
                    norm = (decision_scores - ds_min) / (ds_max - ds_min + 1e-9)
                    confidence_scores = np.max(norm, axis=1)
                    # convert to pseudo-proba for ROC (softmax-like)
                    exp_n = np.exp(norm - norm.max(axis=1, keepdims=True))
                    probas = exp_n / exp_n.sum(axis=1, keepdims=True)
                else:
                    # binaire
                    ds_min = decision_scores.min()
                    ds_max = decision_scores.max()
                    norm_1d = (decision_scores - ds_min) / (ds_max - ds_min + 1e-9)
                    confidence_scores = norm_1d
                    probas = np.column_stack([1 - norm_1d, norm_1d])

            except Exception:
                confidence_scores = np.full(len(y_encoded), np.nan)

        # --- Metrics ---
        report = classification_report(
            y_encoded,
            y_pred,
            target_names=class_names,
            zero_division=0,
            output_dict=True
        )
        cm = confusion_matrix(y_encoded, y_pred)
        f1 = f1_score(y_encoded, y_pred, average='weighted')
        accuracy = accuracy_score(y_encoded, y_pred)

        # --- Fit final sur tout le dataset (pour usage après) ---
        pipeline.fit(X_processed, y_encoded)

        results[model_name] = {
            'classification_report': report,
            'confusion_matrix': cm,
            'label_encoder': le,
            'model': pipeline,  # entraîné final sur tout
            'f1_score': f1,
            'accuracy': accuracy,
            'features': feature_names,
            'confidence_scores': confidence_scores,
            'probas': probas,          # (n_samples, n_classes) for ROC
            'y_true': y_encoded,       # encoded labels
            'class_names': class_names
        }

        if progress_bar is not None:
            progress_bar.progress((i + 1) / total_models)

        gc.collect()

    return results




# ─────────────────────────────────────────────────────────────────────────────
# Shared publication-ready layout helper
# ─────────────────────────────────────────────────────────────────────────────
def _pub_layout(fig, title="", xlab="", ylab="", height=560, width=820):
    """Apply a clean, publication-quality Plotly layout in-place."""
    _ax = dict(
        titlefont=dict(size=17, color="black", family="Arial Black"),
        tickfont=dict(size=14, color="black", family="Arial"),
        showgrid=True, gridcolor="#ececec", zeroline=False,
        showline=True, linecolor="black", linewidth=1.5, mirror=True,
    )
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>",
                   font=dict(size=20, color="black", family="Arial Black"),
                   x=0.5, xanchor="center"),
        xaxis=dict(title=xlab, **_ax),
        yaxis=dict(title=ylab, **_ax),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            font=dict(size=13, color="black", family="Arial"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black", borderwidth=1,
        ),
        height=height, width=width,
        margin=dict(l=65, r=30, t=65, b=65),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Publication-ready Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(model_result: dict) -> "go.Figure":
    """
    Render a polished, annotated confusion matrix from a train_models() result dict.
    Returns a Plotly Figure with a download button in Streamlit.
    """
    cm          = model_result["confusion_matrix"]
    le          = model_result["label_encoder"]
    class_names = list(le.classes_)

    # Normalised version for colour scale
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    text = [[
        f"<b>{cm[i][j]}</b><br>({cm_norm[i][j]*100:.1f}%)"
        for j in range(len(class_names))]
        for i in range(len(class_names))]

    fig = go.Figure(go.Heatmap(
        z=cm_norm,
        x=class_names,
        y=class_names,
        text=text,
        texttemplate="%{text}",
        colorscale=[
            [0.0, "#f7fbff"],
            [0.3, "#6baed6"],
            [0.7, "#2171b5"],
            [1.0, "#08306b"],
        ],
        showscale=True,
        colorbar=dict(
            title=dict(text="Recall", font=dict(size=13, family="Arial")),
            tickfont=dict(size=12, family="Arial"),
        ),
        xgap=2, ygap=2,
    ))

    fig.update_layout(
        title=dict(
            text="<b>Confusion Matrix</b>",
            font=dict(size=20, color="black", family="Arial Black"),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(
            title="<b>Predicted</b>",
            titlefont=dict(size=16, color="black", family="Arial Black"),
            tickfont=dict(size=13, color="black", family="Arial"),
            side="bottom",
        ),
        yaxis=dict(
            title="<b>Actual</b>",
            titlefont=dict(size=16, color="black", family="Arial Black"),
            tickfont=dict(size=13, color="black", family="Arial"),
            autorange="reversed",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=max(400, 120 * len(class_names)),
        width=max(450, 130 * len(class_names)),
        margin=dict(l=60, r=30, t=70, b=60),
        font=dict(family="Arial"),
    )
    return fig


from sklearn.model_selection import StratifiedKFold, learning_curve

def plot_learning_curve(model, X, y, n_splits=5):

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)

    train_sizes, train_scores, test_scores = learning_curve(
        pipeline,
        X, y,
        cv=cv,
        scoring="f1_weighted",   # très conseillé
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 8)
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    test_mean  = test_scores.mean(axis=1)
    test_std   = test_scores.std(axis=1)

    fig = go.Figure()

    # ── CI bands ──────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
        fill='toself', fillcolor='rgba(31,119,180,0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip', showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([test_mean + test_std, (test_mean - test_std)[::-1]]),
        fill='toself', fillcolor='rgba(214,39,40,0.12)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip', showlegend=False,
    ))
    # ── Main lines ────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_mean,
        mode='lines+markers',
        name='Train Score',
        line=dict(color='#1f77b4', width=2.8),
        marker=dict(size=9, symbol='circle', line=dict(width=1.5, color='white')),
    ))
    fig.add_trace(go.Scatter(
        x=train_sizes, y=test_mean,
        mode='lines+markers',
        name='Validation Score',
        line=dict(color='#d62728', width=2.8, dash='dash'),
        marker=dict(size=9, symbol='diamond', line=dict(width=1.5, color='white')),
    ))

    _pub_layout(fig,
                title="Learning Curve (Weighted F1)",
                xlab="Training Set Size",
                ylab="F1 Score",
                height=560, width=820)
    return fig



def compare_models(model_results):
    """Compares models based on their performance metrics."""
    data = []
    
    for model_name, result in model_results.items():
        cm = result['confusion_matrix']
        
        if cm.shape == (2, 2):
            # Binaire - Déballage en tn, fp, fn, tp
            tn, fp, fn, tp = cm.ravel()

            # Calcul des métriques pour la classification binaire
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            # Multiclasse - Traitement de la matrice de confusion
            sensitivity = []
            specificity = []

            # Calcul des métriques pour chaque classe dans la classification multiclasse
            for i in range(cm.shape[0]):
                tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
                fp = np.sum(cm[:, i]) - cm[i, i]
                fn = np.sum(cm[i, :]) - cm[i, i]
                tp = cm[i, i]
                
                sensitivity.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
                specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

            # Moyenne des métriques pour la classification multiclasse
            sensitivity = np.mean(sensitivity)
            specificity = np.mean(specificity)

        # Ajout des métriques dans le résultat du modèle
        result['sensitivity'] = sensitivity
        result['specificity'] = specificity

        # Ajouter les données à la liste
        data.append({
            'Model': model_name,
            'Accuracy': result['accuracy'],
            'F1 Score': result['f1_score'],
            'Sensibilité': sensitivity,
            'Spécificité': specificity
        })

    # Création du DataFrame avec les métriques de chaque modèle
    metrics_df = pd.DataFrame(data)

    # Tri pour obtenir les top 3 modèles
    sorted_models = sorted(
        model_results.items(),
        key=lambda item: item[1].get('f1_score', 0),
        reverse=True
    )
    top_models = sorted_models[:3]

    # Création du message à afficher dans st.info
    top_models_info = "\n\n".join([ 
        f"**{i+1}. {name}**\n"
        f"- 🎯 F1 Score: `{info['f1_score']:.4f}`\n"
        f"- ✅ Accuracy: `{info['accuracy']:.4f}`\n"
        f"- 💓 Sensitivity (Recall): `{info['sensitivity']:.4f}`\n"
        f"- 🛡️ Specificity: `{info['specificity']:.4f}`"
        for i, (name, info) in enumerate(top_models)
    ])

    # Afficher les meilleurs modèles
    st.info(f"🏆 **Top 3 models (by F1 Score):**\n\n{top_models_info}")

    # ── Sort by F1 for display ─────────────────────────────────────────────
    metrics_df = metrics_df.sort_values("F1 Score", ascending=False).reset_index(drop=True)

    # ── Colour palette: top-3 gold/silver/bronze, rest steel-blue ─────────
    bar_colors = []
    medals = ["#FFD700", "#C0C0C0", "#CD7F32"]
    for i in range(len(metrics_df)):
        bar_colors.append(medals[i] if i < 3 else "#4a90d9")

    fig = go.Figure()
    for metric, base_color in [("Accuracy", "#4a90d9"), ("F1 Score", "#e05252"),
                                ("Sensibilité", "#52b96e"), ("Spécificité", "#c47ed4")]:
        if metric not in metrics_df.columns:
            continue
        fig.add_trace(go.Bar(
            x=metrics_df["Model"],
            y=metrics_df[metric],
            name=metric,
            text=[f"{v:.3f}" for v in metrics_df[metric]],
            textposition="outside",
            textfont=dict(size=11, color="black", family="Arial"),
            marker=dict(line=dict(width=1.2, color="black")),
        ))

    _pub_layout(fig,
                title="Model Comparison — All Metrics",
                xlab="Model",
                ylab="Score",
                height=580, width=980)
    fig.update_layout(
        barmode="group",
        xaxis_tickangle=-38,
        yaxis=dict(range=[0, 1.12]),
        legend=dict(
            orientation="h", x=0.5, xanchor="center",
            y=1.06, font=dict(size=13, family="Arial"),
        ),
    )
    return fig

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os
import streamlit as st


def train_regression_models(X, y, n_splits=5, progress_bar=None, n_jobs=None):

    # Desktop: utilise tous les CPUs disponibles
    if n_jobs is None:
        n_jobs = _N_JOBS  # -1 → joblib détecte automatiquement

    # ==============================
    # Preprocessing
    # ==============================
    preprocess = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # ==============================
    # Models — paramètres optimisés pour machine locale puissante
    # ==============================
    models = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.01),
        'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5),

        'RandomForest': RandomForestRegressor(
            n_estimators=300,
            max_depth=None,       # pas de plafond → meilleure précision
            n_jobs=n_jobs,
            random_state=1,
        ),

        'HistGradientBoosting': HistGradientBoostingRegressor(
            max_iter=200,
            random_state=1,
        ),

        'LightGBM': LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            n_jobs=n_jobs,
            num_leaves=127,       # plus de feuilles → meilleure capacité
            random_state=1,
        ),
    }

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=1)

    results = {}

    # ==============================
    # Training Loop
    # ==============================
    for i, (name, model) in enumerate(models.items()):

        pipeline = Pipeline([
            ('preprocess', preprocess),
            ('model', model)
        ])

        # Cross-validation predictions
        # ⚡ n_jobs=-1 → parallélise les folds
        y_pred = cross_val_predict(
            pipeline,
            X,
            y,
            cv=cv,
            n_jobs=-1
        )

        # Metrics
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        # Fit final model
        pipeline.fit(X, y)

        results[name] = {
            "model": pipeline,
            "r2_score": r2,
            "mae": mae,
            "rmse": rmse,
            "y_true": y,
            "y_pred": y_pred
        }

        if progress_bar:
            progress_bar.progress((i + 1) / len(models))

    return results

def compare_regression_models(model_results):

    data = []

    for name, res in model_results.items():
        data.append({
            "Model": name,
            "R2 Score": res["r2_score"],
            "MAE": res["mae"],
            "RMSE": res["rmse"]
        })

    df = pd.DataFrame(data).sort_values(by="R2 Score", ascending=False)

    # ==============================
    # Top 3
    # ==============================
    top_info = "\n".join([
        f"{i+1}. {row['Model']} "
        f"(R²: `{row['R2 Score']:.4f}` | "
        f"MAE: `{row['MAE']:.4f}` | "
        f"RMSE: `{row['RMSE']:.4f}`)"
        for i, row in df.head(3).iterrows()
    ])

    st.success(f"🏆 **Top 3 Regression Models:**\n\n{top_info}")

    # ==============================
    fig_r2 = px.bar(df, x="Model", y="R2 Score",
                    title="Model Comparison — R² Score", text_auto=".3f")
    fig_rmse = px.bar(df, x="Model", y="RMSE",
                      title="Model Comparison — RMSE", text_auto=".3f")

    _pub_layout(fig_r2,   title="Model Comparison — R² Score", xlab="Model", ylab="R²")
    _pub_layout(fig_rmse, title="Model Comparison — RMSE",     xlab="Model", ylab="RMSE")
    for f in (fig_r2, fig_rmse):
        f.update_traces(textfont=dict(size=13, color="black", family="Arial"),
                        marker_line_width=1.2)
    return fig_r2, fig_rmse

# ─────────────────────────────────────────────────────────────────────────────
# ROC Curves (one-vs-rest, publication-ready)
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import plotly.graph_objects as go


def plot_roc_curves(model_results: dict, top_n: int = 5) -> "go.Figure":
    """
    Plot publication-ready ROC curves for the top_n models (by AUC).
    Supports binary and multi-class (OvR macro-average).

    Parameters
    ----------
    model_results : dict  output of train_models()
    top_n         : int   max models to overlay on the same figure

    Returns
    -------
    plotly Figure
    """
    import plotly.graph_objects as go
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    COLORS = px.colors.qualitative.Plotly

    fig = go.Figure()
    # diagonal chance line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='gray', width=2),
        name='Chance (AUC = 0.50)',
        showlegend=True
    ))

    # Sort models by best available AUC or F1
    ranked = sorted(
        model_results.items(),
        key=lambda kv: kv[1].get('f1_score', 0),
        reverse=True
    )[:top_n]

    for idx, (model_name, res) in enumerate(ranked):
        probas   = res.get('probas')
        y_true   = res.get('y_true')
        classes  = res.get('class_names', [])
        if probas is None or y_true is None:
            continue

        n_classes = len(classes)
        color = COLORS[idx % len(COLORS)]

        if n_classes == 2:
            # Binary: positive class = column 1
            fpr, tpr, _ = roc_curve(y_true, probas[:, 1])
            roc_auc = auc(fpr, tpr)
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines',
                line=dict(color=color, width=2.5),
                name=f"{model_name} (AUC={roc_auc:.3f})"
            ))
        else:
            # Multi-class: OvR macro-average
            y_bin = label_binarize(y_true, classes=list(range(n_classes)))
            fpr_all, tpr_all = [], []
            for ci in range(n_classes):
                if y_bin[:, ci].sum() == 0:
                    continue
                fpr_i, tpr_i, _ = roc_curve(y_bin[:, ci], probas[:, ci])
                fpr_all.append(fpr_i)
                tpr_all.append(tpr_i)

            if not fpr_all:
                continue

            # Interpolate on common grid for macro average
            import numpy as _np
            mean_fpr = _np.linspace(0, 1, 200)
            mean_tpr = _np.zeros_like(mean_fpr)
            for fpr_i, tpr_i in zip(fpr_all, tpr_all):
                mean_tpr += _np.interp(mean_fpr, fpr_i, tpr_i)
            mean_tpr /= len(fpr_all)
            roc_auc = auc(mean_fpr, mean_tpr)

            fig.add_trace(go.Scatter(
                x=mean_fpr, y=mean_tpr, mode='lines',
                line=dict(color=color, width=2.5),
                name=f"{model_name} (AUC={roc_auc:.3f})"
            ))

    # ── Publication-ready styling ──────────────────────────────────────────
    font_color = "black"
    axis_style = dict(
        titlefont=dict(size=18, color=font_color, family="Arial Black"),
        tickfont=dict(size=15, color=font_color, family="Arial"),
        showgrid=True, gridcolor="#e5e5e5", zeroline=False,
        showline=True, linecolor="black", linewidth=2,
        mirror=True
    )
    fig.update_layout(
        title=dict(text="ROC Curves (Cross-validation)",
                   font=dict(size=22, color=font_color, family="Arial Black")),
        xaxis=dict(title="False Positive Rate", range=[0, 1], **axis_style),
        yaxis=dict(title="True Positive Rate", range=[0, 1], **axis_style),
        legend=dict(font=dict(size=13, color=font_color, family="Arial"),
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="black", borderwidth=1),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=700, height=600,
        margin=dict(l=60, r=30, t=60, b=60)
    )
    return fig
