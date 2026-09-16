"""
profiler_plot_theme.py — thème graphique + accélérateurs de rendu Profiler
==========================================================================
Un seul endroit pour :

  1. LE STYLE  — un template Plotly « profiler » (publication-ready) appliqué
     à TOUS les graphes de l'application : même police, mêmes axes, même
     palette daltonien-safe, mêmes marges. Fini les tailles de police à 24
     dans un module et à 10 dans le suivant.

  2. LA VITESSE — trois leviers qui n'exigent aucun changement d'algorithme :
       • `to_webgl()`     : go.Scatter → go.Scattergl au-delà de N points.
                            Le SVG plafonne vers ~3 000 marqueurs ; la WebGL
                            en encaisse 500 000 sans transpirer.
       • `thin_line()`    : décimation LTTB des spectres / courbes. 100 000
                            points → 2 000, visuellement identique.
       • `show()`         : un seul `st.plotly_chart` centralisé qui applique
                            thème + WebGL + config d'export (PNG scale 3).

  3. LE CACHE MULTI-UTILISATEUR — `session_cache`, mémoïsation par SESSION
     (st.session_state), jamais partagée entre utilisateurs.

     ⚠️ Nuance importante sur `st.cache_data` : il n'est pas interdit en soi.
     Il est dangereux quand la clé de cache ne capture PAS les données de
     l'utilisateur — typiquement `def f(_df, k)` : le `_` dit à Streamlit
     d'ignorer `_df`, donc l'utilisateur B reçoit le résultat calculé sur les
     données de A. C'est CE cas qu'il faut bannir.
     `st.cache_data` sur une fonction sans argument utilisateur (ex.
     `load_gene_sets()` qui liste les librairies Enrichr) est au contraire
     exactement le bon outil : garde-le.
     Règle : données utilisateur → `session_cache`. Ressource publique et
     immuable → `st.cache_data` / `st.cache_resource`.

Usage minimal dans un module de plot :

    from profiler_plot_theme import style, show, to_webgl, PALETTE

    fig = px.scatter(df, x="PC1", y="PC2", color="Class")
    show(style(fig, title="PCA"), key="pca_main")
"""

from __future__ import annotations

import hashlib
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

try:
    import streamlit as st
except Exception:                                       # usage hors Streamlit
    st = None


# ═══════════════════════════════════════════════════════════════════════════
# 1. PALETTE & TEMPLATE
# ═══════════════════════════════════════════════════════════════════════════

# Okabe–Ito : discriminable par les trois formes de daltonisme, imprimable
# en niveaux de gris. Standard de fait dans les journaux à comité de lecture.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
           "#E69F00", "#56B4E9", "#F0E442", "#000000"]

# Divergente pour tout ce qui est signé (z-scores, log2FC, corrélations)
DIVERGING = "RdBu_r"
# Séquentielle pour tout ce qui est positif (intensités, abondances)
SEQUENTIAL = "Viridis"

FONT_FAMILY = "Arial, Helvetica, sans-serif"
INK = "#111111"
GRID = "#E9E9E9"
AXIS = "#333333"

_AXIS_BASE = dict(
    showline=True, linecolor=AXIS, linewidth=1,
    ticks="outside", ticklen=4, tickwidth=1, tickcolor=AXIS,
    tickfont=dict(size=11, family=FONT_FAMILY, color=INK),
    title=dict(font=dict(size=13, family=FONT_FAMILY, color=INK)),
    showgrid=True, gridcolor=GRID, gridwidth=1,
    zeroline=False, automargin=True,
)

PROFILER_TEMPLATE = go.layout.Template(
    layout=dict(
        font=dict(family=FONT_FAMILY, size=12, color=INK),
        title=dict(font=dict(family=FONT_FAMILY, size=15, color=INK), x=0.5,
                   xanchor="center"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        colorway=PALETTE,
        colorscale=dict(sequential=SEQUENTIAL, diverging=DIVERGING),
        xaxis=_AXIS_BASE, yaxis=_AXIS_BASE,
        legend=dict(font=dict(size=11, family=FONT_FAMILY, color=INK),
                    bgcolor="rgba(255,255,255,0)", borderwidth=0,
                    itemsizing="constant", tracegroupgap=8),
        hoverlabel=dict(font=dict(family=FONT_FAMILY, size=11),
                        bgcolor="white", bordercolor=AXIS),
        margin=dict(l=60, r=30, t=50, b=55),
        separators=".,",
    )
)

pio.templates["profiler"] = PROFILER_TEMPLATE


def use_profiler_theme() -> None:
    """À appeler UNE fois au démarrage (Profiler.py / Profiler_Desktop_Gui.py)."""
    pio.templates.default = "profiler"


# ═══════════════════════════════════════════════════════════════════════════
# 2. ACCÉLÉRATEURS DE RENDU
# ═══════════════════════════════════════════════════════════════════════════

WEBGL_MIN_POINTS = 1_200        # au-delà : bascule WebGL
LINE_MAX_POINTS = 4_000         # au-delà : décimation LTTB
_NO_GL_SHAPES = {"spline", "hv", "vh", "hvh", "vhv"}


def _n_points(tr) -> int:
    for attr in ("x", "y"):
        v = getattr(tr, attr, None)
        if v is not None:
            try:
                return len(v)
            except TypeError:
                pass
    return 0


def to_webgl(fig, min_points: int = WEBGL_MIN_POINTS):
    """
    Convertit les go.Scatter volumineux en go.Scattergl (rendu GPU).
    Renvoie une nouvelle figure si une conversion a eu lieu — utilisez
    toujours la valeur de retour : `fig = to_webgl(fig)`.
    Sans effet sur les petits graphes, les heatmaps, les box/violin, la 3D.
    Les formes de ligne non supportées en WebGL sont laissées en SVG.
    """
    try:
        new_data = []
        changed = False
        for tr in fig.data:
            if tr.type == "scatter" and _n_points(tr) >= min_points:
                shape = getattr(getattr(tr, "line", None), "shape", None)
                if shape in _NO_GL_SHAPES:
                    new_data.append(tr)
                    continue
                d = tr.to_plotly_json()
                d.pop("type", None)
                try:
                    new_data.append(go.Scattergl(**d))
                    changed = True
                    continue
                except Exception:
                    pass
            new_data.append(tr)
        if changed:
            # fig.data n'accepte pas un changement de TYPE de trace :
            # il faut reconstruire la figure autour du nouveau data.
            fig = go.Figure(data=new_data, layout=fig.layout)
            for shp in getattr(fig.layout, "shapes", ()) or ():
                pass
    except Exception:
        pass
    return fig


def lttb(x, y, n_out: int):
    """
    Largest-Triangle-Three-Buckets : décimation qui préserve les pics.
    Indispensable pour les spectres de masse — un `x[::k]` naïf supprime
    justement les pics, qui sont toute l'information.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n_out >= n or n_out < 3:
        return x, y

    out_x = np.empty(n_out)
    out_y = np.empty(n_out)
    out_x[0], out_y[0] = x[0], y[0]
    out_x[-1], out_y[-1] = x[-1], y[-1]

    every = (n - 2) / (n_out - 2)
    a = 0
    for i in range(n_out - 2):
        lo = int(np.floor((i + 1) * every)) + 1
        hi = min(int(np.floor((i + 2) * every)) + 1, n)
        avg_x = x[lo:hi].mean() if hi > lo else x[-1]
        avg_y = y[lo:hi].mean() if hi > lo else y[-1]

        r0 = int(np.floor(i * every)) + 1
        r1 = min(int(np.floor((i + 1) * every)) + 1, n)
        seg_x, seg_y = x[r0:r1], y[r0:r1]
        area = np.abs((x[a] - avg_x) * (seg_y - y[a])
                      - (x[a] - seg_x) * (avg_y - y[a]))
        k = int(np.argmax(area)) if area.size else 0
        a = r0 + k
        out_x[i + 1], out_y[i + 1] = x[a], y[a]
    return out_x, out_y


def thin_line(fig, max_points: int = LINE_MAX_POINTS):
    """Applique LTTB à toutes les traces ligne trop denses de la figure."""
    for tr in fig.data:
        if tr.type in ("scatter", "scattergl") and tr.mode and "lines" in tr.mode:
            if _n_points(tr) > max_points and tr.x is not None and tr.y is not None:
                try:
                    tr.x, tr.y = lttb(tr.x, tr.y, max_points)
                    tr.text = None
                    tr.customdata = None
                except Exception:
                    pass
    return fig


def cap_hover(fig, max_cells: int = 60_000):
    """
    Supprime les tableaux de hover pré-calculés des heatmaps géantes :
    c'est ce qui fait exploser le JSON envoyé au navigateur.
    """
    for tr in fig.data:
        if tr.type == "heatmap" and tr.z is not None:
            try:
                z = np.asarray(tr.z)
                if z.size > max_cells and tr.text is not None:
                    tr.text = None
                    tr.hovertemplate = "%{z:.3g}<extra></extra>"
            except Exception:
                pass
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 3. STYLE + AFFICHAGE
# ═══════════════════════════════════════════════════════════════════════════

def style(fig, title: str = None, xtitle: str = None, ytitle: str = None,
          legend_title: str = None, square: bool = False, height: int = None):
    """
    Passe finale de mise en forme. Idempotente : peut être appelée sur une
    figure déjà stylée sans dégât.
    """
    fig.update_layout(
        template="profiler",
        title=dict(
            x=0.5,
            xanchor="center",
            font=dict(size=15, family=FONT_FAMILY, color=INK),
        ),
    )

    # Titres et libellés des axes en noir foncé et bien visibles.
    fig.update_xaxes(
        title_font=dict(size=13, family=FONT_FAMILY, color=INK),
        tickfont=dict(size=11, family=FONT_FAMILY, color=INK),
    )
    fig.update_yaxes(
        title_font=dict(size=13, family=FONT_FAMILY, color=INK),
        tickfont=dict(size=11, family=FONT_FAMILY, color=INK),
    )

    if title is not None:
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                x=0.5,
                xanchor="center",
                font=dict(size=15, family=FONT_FAMILY, color=INK),
            )
        )
    if xtitle is not None:
        fig.update_xaxes(
            title_text=xtitle,
            title_font=dict(size=13, family=FONT_FAMILY, color=INK),
        )
    if ytitle is not None:
        fig.update_yaxes(
            title_text=ytitle,
            title_font=dict(size=13, family=FONT_FAMILY, color=INK),
        )
    if legend_title is not None:
        fig.update_layout(legend=dict(title=dict(text=f"<b>{legend_title}</b>")))
    if height:
        fig.update_layout(height=height)
    if square:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    # Les contours noirs épais sur chaque marqueur coûtent cher en SVG et
    # noient les nuages denses : on les affine au-delà de 300 points.
    for tr in fig.data:
        if tr.type in ("scatter", "scattergl") and _n_points(tr) > 300:
            try:
                tr.marker.line.width = 0
            except Exception:
                pass
    return fig


def plot_config(filename: str = "profiler_plot", scale: int = 3,
                static: bool = False) -> dict:
    """Config Plotly homogène : export ~300 dpi, pas de logo, pas de lasso."""
    return {
        "displaylogo": False,
        "responsive": True,
        "scrollZoom": True,
        "staticPlot": static,
        "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
        "toImageButtonOptions": {"format": "png", "scale": scale,
                                 "filename": filename},
    }


def show(fig, key: str = None, use_container_width: bool = True,
         filename: str = None, webgl: bool = True, thin: bool = True,
         static: bool = False):
    """
    Point d'entrée unique pour afficher une figure Plotly dans Profiler.
    Applique thème + WebGL + décimation + config d'export, puis rend.
    """
    if fig is None:
        return None
    fig.update_layout(template="profiler")

    # Force les titres et graduations en noir après application du template.
    # Cela évite que les propriétés locales d'un graphique ou un template
    # Plotly ne réintroduisent une couleur grise.
    fig.update_layout(
        title_font=dict(
            family=FONT_FAMILY,
            size=15,
            color=INK,
        )
    )
    fig.update_xaxes(
        title_font=dict(
            family=FONT_FAMILY,
            size=13,
            color=INK,
        ),
        tickfont=dict(
            family=FONT_FAMILY,
            size=11,
            color=INK,
        ),
    )
    fig.update_yaxes(
        title_font=dict(
            family=FONT_FAMILY,
            size=13,
            color=INK,
        ),
        tickfont=dict(
            family=FONT_FAMILY,
            size=11,
            color=INK,
        ),
    )

    if thin:
        fig = thin_line(fig)
    if webgl:
        fig = to_webgl(fig)          # peut renvoyer une NOUVELLE figure
    fig = cap_hover(fig)
    if st is not None:
        st.plotly_chart(fig, use_container_width=use_container_width, key=key,
                        config=plot_config(filename or key or "profiler_plot",
                                           static=static))
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 4. CACHE SÛR EN MULTI-UTILISATEUR
# ═══════════════════════════════════════════════════════════════════════════

def fingerprint(obj) -> str:
    """
    Empreinte bon marché et stable d'un DataFrame / ndarray / objet simple.
    Sur un DataFrame on hache la forme, les dtypes, les noms de colonnes et
    un échantillon de valeurs — O(1) en pratique, pas O(n).
    """
    h = hashlib.blake2b(digest_size=16)
    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            h.update(str(obj.shape).encode())
            h.update(str(list(obj.columns)[:200]).encode())
            h.update(str(obj.dtypes.astype(str).tolist()[:200]).encode())
            n = len(obj)
            if n:
                idx = np.unique(np.linspace(0, n - 1, min(n, 64)).astype(int))
                h.update(
                    pd.util.hash_pandas_object(obj.iloc[idx], index=False)
                    .values.tobytes()
                )
            return h.hexdigest()
        if isinstance(obj, pd.Series):
            return fingerprint(obj.to_frame())
    except Exception:
        pass
    if isinstance(obj, np.ndarray):
        h.update(str(obj.shape).encode())
        h.update(str(obj.dtype).encode())
        flat = obj.ravel()
        idx = np.unique(np.linspace(0, flat.size - 1,
                                    min(flat.size, 256)).astype(int)) if flat.size else []
        if len(idx):
            h.update(np.ascontiguousarray(flat[idx]).tobytes())
        return h.hexdigest()
    h.update(repr(obj)[:4000].encode("utf-8", "ignore"))
    return h.hexdigest()


def session_cache(namespace: str, max_entries: int = 8):
    """
    Mémoïsation par session — remplace `st.cache_data` pour tout ce qui touche
    aux données de l'utilisateur.

        @session_cache("umap")
        def compute_umap(X, n_components, seed):
            ...

    • Clé = namespace + empreinte de chaque argument → aucun risque qu'une
      session reçoive le résultat d'une autre.
    • Stockage dans st.session_state → libéré avec la session, pas de fuite
      mémoire cumulée côté serveur comme avec un cache global.
    • `max_entries` borne la mémoire par namespace (LRU simple).
    """
    def decorator(fn):
        def wrapper(*args, **kwargs):
            if st is None:
                return fn(*args, **kwargs)
            key = namespace + "|" + "|".join(
                [fingerprint(a) for a in args]
                + [f"{k}={fingerprint(v)}" for k, v in sorted(kwargs.items())]
            )
            store = st.session_state.setdefault("_profiler_cache", {})
            bucket = store.setdefault(namespace, {})
            if key in bucket:
                bucket[key] = bucket.pop(key)          # LRU touch
                return bucket[key]
            res = fn(*args, **kwargs)
            bucket[key] = res
            while len(bucket) > max_entries:
                bucket.pop(next(iter(bucket)))
            return res
        wrapper.__name__ = getattr(fn, "__name__", "wrapped")
        wrapper.__doc__ = fn.__doc__
        wrapper.clear = lambda: st.session_state.get(
            "_profiler_cache", {}).pop(namespace, None) if st else None
        return wrapper
    return decorator


def clear_session_cache(namespace: str = None) -> None:
    """Vide le cache de session (tout, ou un namespace). À appeler au reset."""
    if st is None:
        return
    if namespace is None:
        st.session_state.pop("_profiler_cache", None)
    else:
        st.session_state.get("_profiler_cache", {}).pop(namespace, None)


# ═══════════════════════════════════════════════════════════════════════════
# 5. BOOTSTRAP GLOBAL (2 lignes dans Profiler.py / Profiler_Desktop_Gui.py)
# ═══════════════════════════════════════════════════════════════════════════

_HOOK_INSTALLED = False


def install_fast_plotly_chart(webgl: bool = True, thin: bool = True,
                              scale: int = 3) -> None:
    """
    Enveloppe `st.plotly_chart` une fois pour toutes : chaque figure de
    l'application, quel que soit le module qui l'affiche, passe alors par
    WebGL + décimation + config d'export homogène — sans modifier les ~40
    appels existants un par un.

    Idempotent, et sans effet sur les figures déjà petites.
    Appeler juste après `use_profiler_theme()`.
    """
    global _HOOK_INSTALLED
    if st is None or _HOOK_INSTALLED:
        return
    original = st.plotly_chart

    def patched(figure_or_data, *args, **kwargs):
        fig = figure_or_data
        try:
            if hasattr(fig, "data") and hasattr(fig, "layout"):
                if thin:
                    fig = thin_line(fig)
                if webgl:
                    fig = to_webgl(fig)
                fig = cap_hover(fig)
                if fig.layout.template is None or not fig.layout.template.layout:
                    fig.update_layout(template="profiler")
                if "config" not in kwargs:
                    kwargs["config"] = plot_config(
                        kwargs.get("key") or "profiler_plot", scale=scale)
                else:
                    cfg = dict(kwargs["config"])
                    cfg.setdefault("displaylogo", False)
                    cfg.setdefault("toImageButtonOptions",
                                   {"format": "png", "scale": scale})
                    kwargs["config"] = cfg
        except Exception:
            fig = figure_or_data
        return original(fig, *args, **kwargs)

    patched._profiler_wrapped = True
    st.plotly_chart = patched
    _HOOK_INSTALLED = True


def bootstrap(fast: bool = True) -> None:
    """Thème + accélérateur en un appel. À placer au démarrage de l'app."""
    use_profiler_theme()
    if fast:
        install_fast_plotly_chart()
