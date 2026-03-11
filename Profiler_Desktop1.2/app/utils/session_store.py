"""
session_store.py — Profiler Web Session Persistence
=====================================================
Survie des données au refresh de page, isolation par utilisateur,
nettoyage automatique TTL, optimisation RAM/CPU.

Architecture :
- Chaque utilisateur a un dossier temporaire isolé : temp_sessions/<username_hash>/
- DataFrames sérialisés en Parquet (compression snappy) → ~5-10x moins de RAM disque
- Objets non-sérialisables (modèles ML/DL) stockés en pickle compressé
- Thread de nettoyage en arrière-plan : supprime les sessions inactives > TTL
- Limite par utilisateur configurable
- Aucune dépendance externe (pas de Redis)

Usage dans Profiler.py :
    from session_store import SessionStore
    store = SessionStore(username)
    store.save_df("data", df)
    df = store.load_df("data")
    store.save_obj("models", models_dict)
    models = store.load_obj("models")
    store.touch()   # Renouvelle le TTL à chaque interaction
    store.cleanup() # Force la suppression immédiate (logout)
"""

import os
import time
import pickle
import shutil
import hashlib
import threading
import logging
from pathlib import Path
from typing import Optional, Any

import pandas as pd

# ─── Configuration ────────────────────────────────────────────────────────────
SESSION_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_sessions")
SESSION_TTL_SECONDS = 5 * 60          # 5 minutes d'inactivité → suppression
CLEANUP_INTERVAL_SECONDS = 60         # Vérifie toutes les 60 secondes
MAX_SESSION_SIZE_GB = 10.0            # Limite par utilisateur (10 GB)
PARQUET_COMPRESSION = "snappy"        # snappy = rapide, gzip = plus compact

# ─── Logging ──────────────────────────────────────────────────────────────────
logger = logging.getLogger("session_store")

# ─── Clés qui sont des DataFrames pandas ──────────────────────────────────────
DATAFRAME_KEYS = {
    "data", "final_data", "preprocessed_data",
    "oversampled_data", "undersampled_data",
    "survival_data", "reduced_data", "compressed_data",
}

# ─── Clés d'objets Python sérialisables (non-DataFrame) ──────────────────────
OBJECT_KEYS = {
    "models", "dl_models", "label_encoder", "svd_model",
    "shap_values", "enrichment_results",
}


# ══════════════════════════════════════════════════════════════════════════════
# SessionStore — interface principale
# ══════════════════════════════════════════════════════════════════════════════

class SessionStore:
    """Gestionnaire de persistance de session pour un utilisateur."""

    def __init__(self, username: str):
        # Hash du username pour le nom du dossier (sécurité + path-safe)
        self._uid = hashlib.sha256(username.encode()).hexdigest()[:16]
        self._username = username
        self._dir = Path(SESSION_BASE_DIR) / self._uid
        self._dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self._dir / "_meta.pkl"
        self.touch()

    # ── Horodatage d'activité ──────────────────────────────────────────────
    def touch(self):
        """Renouvelle le TTL. À appeler à chaque interaction utilisateur."""
        try:
            with open(self._meta_path, "wb") as f:
                pickle.dump({
                    "username": self._username,
                    "last_active": time.time(),
                }, f)
        except Exception as e:
            logger.warning(f"[SessionStore] touch() failed for {self._uid}: {e}")

    def last_active(self) -> float:
        """Retourne le timestamp de la dernière activité."""
        try:
            if self._meta_path.exists():
                with open(self._meta_path, "rb") as f:
                    return pickle.load(f).get("last_active", 0.0)
        except Exception:
            pass
        return 0.0

    # ── DataFrames ────────────────────────────────────────────────────────
    def save_df(self, key: str, df: Optional[pd.DataFrame]):
        """Sauvegarde un DataFrame en Parquet compressé."""
        path = self._df_path(key)
        try:
            if df is None:
                if path.exists():
                    path.unlink()
                return
            # Conversion des colonnes object en string pour compatibilité Parquet
            df_save = df.copy()
            for col in df_save.select_dtypes(include=["object"]).columns:
                df_save[col] = df_save[col].astype(str)
            df_save.to_parquet(path, compression=PARQUET_COMPRESSION, index=True)
            self._check_quota()
        except Exception as e:
            logger.error(f"[SessionStore] save_df({key}) error: {e}")
            # Fallback: pickle si Parquet échoue (ex: types non supportés)
            try:
                import gzip
                with gzip.open(str(path) + ".pkl.gz", "wb") as f:
                    pickle.dump(df, f)
            except Exception as e2:
                logger.error(f"[SessionStore] fallback pickle also failed: {e2}")

    def load_df(self, key: str) -> Optional[pd.DataFrame]:
        """Charge un DataFrame depuis le cache disque."""
        path = self._df_path(key)
        pkl_path = Path(str(path) + ".pkl.gz")
        try:
            if path.exists():
                return pd.read_parquet(path)
            elif pkl_path.exists():
                import gzip
                with gzip.open(pkl_path, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"[SessionStore] load_df({key}) error: {e}")
        return None

    def has_df(self, key: str) -> bool:
        """Vérifie si un DataFrame est disponible en cache."""
        return self._df_path(key).exists() or Path(str(self._df_path(key)) + ".pkl.gz").exists()

    # ── Objets Python ─────────────────────────────────────────────────────
    def save_obj(self, key: str, obj: Any):
        """Sauvegarde un objet Python sérialisable (modèles, encodeurs...)."""
        path = self._obj_path(key)
        try:
            if obj is None:
                if path.exists():
                    path.unlink()
                return
            import gzip
            with gzip.open(path, "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.error(f"[SessionStore] save_obj({key}) error: {e}")

    def load_obj(self, key: str) -> Any:
        """Charge un objet Python depuis le cache disque."""
        path = self._obj_path(key)
        try:
            if path.exists():
                import gzip
                with gzip.open(path, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"[SessionStore] load_obj({key}) error: {e}")
        return None

    def has_obj(self, key: str) -> bool:
        return self._obj_path(key).exists()

    # ── Valeurs scalaires (paramètres, flags...) ──────────────────────────
    def save_scalar(self, key: str, value: Any):
        """Sauvegarde une valeur scalaire (int, str, bool, dict simple...)."""
        path = self._dir / f"scalar_{key}.pkl"
        try:
            with open(path, "wb") as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.error(f"[SessionStore] save_scalar({key}) error: {e}")

    def load_scalar(self, key: str, default: Any = None) -> Any:
        path = self._dir / f"scalar_{key}.pkl"
        try:
            if path.exists():
                with open(path, "rb") as f:
                    return pickle.load(f)
        except Exception:
            pass
        return default

    # ── Nettoyage ─────────────────────────────────────────────────────────
    def cleanup(self):
        """Supprime immédiatement toute la session (logout)."""
        try:
            if self._dir.exists():
                shutil.rmtree(self._dir)
        except Exception as e:
            logger.error(f"[SessionStore] cleanup() error: {e}")

    def session_size_mb(self) -> float:
        """Retourne la taille totale de la session sur disque en MB."""
        try:
            total = sum(f.stat().st_size for f in self._dir.rglob("*") if f.is_file())
            return total / (1024 * 1024)
        except Exception:
            return 0.0

    # ── Helpers ───────────────────────────────────────────────────────────
    def _df_path(self, key: str) -> Path:
        return self._dir / f"df_{key}.parquet"

    def _obj_path(self, key: str) -> Path:
        return self._dir / f"obj_{key}.pkl.gz"

    def _check_quota(self):
        size_gb = self.session_size_mb() / 1024
        if size_gb > MAX_SESSION_SIZE_GB:
            logger.warning(
                f"[SessionStore] User {self._username} exceeded quota: "
                f"{size_gb:.1f} GB > {MAX_SESSION_SIZE_GB} GB"
            )


# ══════════════════════════════════════════════════════════════════════════════
# Background cleanup thread — supprime les sessions expirées
# ══════════════════════════════════════════════════════════════════════════════

_cleanup_thread_started = False
_cleanup_lock = threading.Lock()


def _cleanup_worker():
    """Thread background : parcourt temp_sessions/ et supprime les sessions TTL expirées."""
    base = Path(SESSION_BASE_DIR)
    while True:
        try:
            time.sleep(CLEANUP_INTERVAL_SECONDS)
            if not base.exists():
                continue
            now = time.time()
            for session_dir in base.iterdir():
                if not session_dir.is_dir():
                    continue
                meta_path = session_dir / "_meta.pkl"
                try:
                    if meta_path.exists():
                        with open(meta_path, "rb") as f:
                            meta = pickle.load(f)
                        last_active = meta.get("last_active", 0)
                        if now - last_active > SESSION_TTL_SECONDS:
                            shutil.rmtree(session_dir)
                            logger.info(f"[SessionStore] Expired session removed: {session_dir.name}")
                    else:
                        # Dossier orphelin sans méta → supprimer
                        shutil.rmtree(session_dir)
                except Exception as e:
                    logger.warning(f"[SessionStore] Cleanup error for {session_dir}: {e}")
        except Exception as e:
            logger.error(f"[SessionStore] Cleanup worker error: {e}")


def start_cleanup_thread():
    """Démarre le thread de nettoyage (une seule fois par processus)."""
    global _cleanup_thread_started
    with _cleanup_lock:
        if not _cleanup_thread_started:
            t = threading.Thread(target=_cleanup_worker, daemon=True, name="session-cleanup")
            t.start()
            _cleanup_thread_started = True
            logger.info("[SessionStore] Cleanup thread started.")


# ══════════════════════════════════════════════════════════════════════════════
# Integration helpers — à utiliser dans main() de Profiler.py
# ══════════════════════════════════════════════════════════════════════════════

# Les clés DataFrame à restaurer depuis le disque dans session_state
PERSIST_DF_KEYS    = list(DATAFRAME_KEYS)
PERSIST_OBJ_KEYS   = list(OBJECT_KEYS)


def restore_session(store: SessionStore):
    """
    Restaure les données persistées dans st.session_state.
    À appeler juste après l'authentification réussie (ou à chaque refresh).
    Ne restaure que les clés qui sont None dans session_state.
    """
    import streamlit as st
    restored = []
    for key in PERSIST_DF_KEYS:
        if st.session_state.get(key) is None and store.has_df(key):
            st.session_state[key] = store.load_df(key)
            if st.session_state[key] is not None:
                restored.append(key)
    for key in PERSIST_OBJ_KEYS:
        if not st.session_state.get(key) and store.has_obj(key):
            st.session_state[key] = store.load_obj(key)
            if st.session_state[key]:
                restored.append(key)
    if restored:
        logger.info(f"[SessionStore] Restored for {store._username}: {restored}")
    return restored


def persist_session(store: SessionStore):
    """
    Sauvegarde les données de session sur disque.
    À appeler après chaque opération qui modifie les données.
    """
    import streamlit as st
    for key in PERSIST_DF_KEYS:
        val = st.session_state.get(key)
        if val is not None and isinstance(val, pd.DataFrame):
            store.save_df(key, val)
    for key in PERSIST_OBJ_KEYS:
        val = st.session_state.get(key)
        if val:
            store.save_obj(key, val)
    store.touch()


def get_store(username: str) -> "SessionStore":
    """
    Retourne (ou crée) un SessionStore pour l'utilisateur.
    Stocké dans st.session_state['_store'] pour éviter de recréer à chaque rerun.
    """
    import streamlit as st
    store_key = "_session_store"
    if store_key not in st.session_state or st.session_state[store_key]._username != username:
        st.session_state[store_key] = SessionStore(username)
    return st.session_state[store_key]
