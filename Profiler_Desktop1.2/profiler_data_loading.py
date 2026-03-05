"""
Software Name: Profiler
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

import os
import gc
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from pyopenms import MSExperiment, MzMLFile, MzXMLFile
from scipy.signal import find_peaks, detrend


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS  (overridden at runtime by UI parameters)
# ─────────────────────────────────────────────────────────────────────────────
_MZ_TOLERANCE_PPM = 10          # default ppm tolerance for m/z binning
_MIN_PEAK_WIDTH    = 2          # minimum scans that form a valid peak


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: convert ppm / Da tolerance to an absolute Da value at a given m/z
# ─────────────────────────────────────────────────────────────────────────────
def _tol_da(mz: float, tol_ppm: float = 0.0, tol_da: float = 0.0) -> float:
    """Return the absolute tolerance in Da. Uses the larger of ppm or Da window."""
    return max(mz * tol_ppm * 1e-6, tol_da)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: find chromatographic peak boundaries (start, apex, end in scan idx)
# ─────────────────────────────────────────────────────────────────────────────
def _auto_min_distance(sig: np.ndarray) -> int:
    """
    Estimate a sensible minimum distance between peak apices from the signal
    itself, so the user never has to set it manually.

    Strategy
    --------
    1. Find all local minima (valleys) in the detrended signal.
    2. Compute the median gap between consecutive valleys.
    3. Use half that gap as the minimum inter-apex distance.
       Clamp to [2, len(sig)//4] for safety.
    """
    if len(sig) < 6:
        return 2
    from scipy.signal import find_peaks as _fp
    valleys, _ = _fp(-sig)                      # invert to find minima
    if len(valleys) >= 2:
        gaps = np.diff(valleys)
        dist = max(2, int(np.median(gaps) // 2))
    else:
        dist = max(2, len(sig) // 10)           # fallback: 10 % of total scans
    return min(dist, max(2, len(sig) // 4))


def _find_peak_intervals(times, intensities,
                         min_apex_intensity_pct=1.0,
                         min_peak_width=2,
                         min_distance=0,          # 0 = auto-detect
                         min_prominence_pct=5.0):
    """
    Detect chromatographic peaks and return scan-index boundaries.

    Parameters
    ----------
    min_apex_intensity_pct : float
        Minimum apex intensity as % of global TIC max.
        Replaces the old threshold_pct: it is now a *noise floor*, not a
        "keep only the top N%" filter.  Default 1 % keeps everything above
        noise while rejecting baseline artifacts.
    min_peak_width         : int
        Minimum number of scans that form a valid peak.
    min_distance           : int
        Minimum scans between two apex candidates.
        0 (default) = auto-detected from valley spacing of the TIC.
    min_prominence_pct     : float
        Peak prominence as % of global TIC max.
        Filters shoulders and noise bumps that are not true peaks.

    Returns list of (start_idx, apex_idx, end_idx)
    where start and end delimit the full peak from foot to foot.
    """
    if len(intensities) < 3:
        return []

    sig = detrend(intensities.astype(np.float64))
    sig = np.clip(sig, 0, None)

    if sig.max() == 0:
        return []

    global_max        = sig.max()
    height_thresh     = global_max * (min_apex_intensity_pct / 100.0)
    prominence_thresh = global_max * (min_prominence_pct      / 100.0)

    # Auto-detect minimum distance if not provided
    effective_dist = _auto_min_distance(sig) if min_distance <= 0 else max(1, min_distance)

    apex_indices, _ = find_peaks(
        sig,
        height     = height_thresh,
        distance   = effective_dist,
        prominence = prominence_thresh,
    )

    if len(apex_indices) == 0:
        return []

    # ── Walk from apex outward to find true peak boundaries ───────────────────
    floor_frac = 0.05    # stop walking when signal < 5 % of apex value
    intervals  = []

    for apex in apex_indices:
        apex_val = sig[apex]
        floor    = apex_val * floor_frac

        left = apex
        while left > 0 and sig[left - 1] >= floor:
            left -= 1

        right = apex
        while right < len(sig) - 1 and sig[right + 1] >= floor:
            right += 1

        # Ensure minimum width
        if (right - left + 1) >= min_peak_width:
            intervals.append((left, apex, right))

    if not intervals:
        return []

    # Merge overlapping intervals (keeps stronger apex)
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for start, apex, end in intervals[1:]:
        ps, pa, pe = merged[-1]
        if start <= pe:
            merged[-1] = (
                min(ps, start),
                pa if sig[pa] >= sig[apex] else apex,
                max(pe, end)
            )
        else:
            merged.append((start, apex, end))

    return merged


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: sum all spectra within a peak window => one integrated spectrum
# ─────────────────────────────────────────────────────────────────────────────
def _integrate_peak(spectra_slice, tol_ppm=_MZ_TOLERANCE_PPM, tol_da=0.0,
                    apex_rt: float = None):
    """
    Sum ALL MS1 spectra from peak start to peak end (inclusive) into one
    intensity vector.  The scan range is exactly [scan_start, scan_end+1]
    as detected by _find_peak_intervals — no scan is missed or double-counted.

    Parameters
    ----------
    spectra_slice : list of pyOpenMS Spectrum objects, already sliced
                    spectra[scan_start : scan_end + 1]
    tol_ppm / tol_da : m/z binning tolerance
    apex_rt       : RT (seconds) of the apex scan.  If None, falls back to
                    the RT of the middle spectrum in the slice.
    """
    all_mz    = []
    all_inten = []

    # Use the true apex RT, not the middle of an asymmetric slice
    if apex_rt is not None:
        rt_apex = apex_rt
    else:
        rt_apex = spectra_slice[len(spectra_slice) // 2].getRT()

    for spec in spectra_slice:
        if spec.getMSLevel() not in (0, 1):   # MS1 only
            continue
        mzs, ints = spec.get_peaks()
        if len(mzs) == 0:
            continue
        all_mz.append(mzs)
        all_inten.append(ints)

    if not all_mz:
        return {}, rt_apex

    all_mz    = np.concatenate(all_mz).astype(np.float64)
    all_inten = np.concatenate(all_inten).astype(np.float64)

    order     = np.argsort(all_mz)
    all_mz    = all_mz[order]
    all_inten = all_inten[order]

    bins = {}
    i, n = 0, len(all_mz)

    while i < n:
        ref = all_mz[i]
        tol = _tol_da(ref, tol_ppm, tol_da)
        j   = i + 1
        while j < n and (all_mz[j] - ref) <= tol:
            j += 1
        bin_mzs  = all_mz[i:j]
        bin_ints = all_inten[i:j]
        w_sum    = bin_ints.sum()
        if w_sum > 0:
            rep_mz = float(np.round(np.average(bin_mzs, weights=bin_ints), 5))
        else:
            rep_mz = float(np.round(bin_mzs.mean(), 5))
        bins[rep_mz] = bins.get(rep_mz, 0.0) + float(w_sum)
        i = j

    return bins, rt_apex


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: build global consensus m/z axis from all per-file m/z sets
# ─────────────────────────────────────────────────────────────────────────────
def _align_mz_columns(all_mz_sets, tol_ppm=_MZ_TOLERANCE_PPM, tol_da=0.0):
    """Merge m/z keys into a single sorted consensus list."""
    all_vals = sorted({mz for s in all_mz_sets for mz in s})
    if not all_vals:
        return []

    arr       = np.array(all_vals, dtype=np.float64)
    consensus = []
    i = 0
    while i < len(arr):
        ref  = arr[i]
        tol  = _tol_da(ref, tol_ppm, tol_da)
        j    = i + 1
        while j < len(arr) and (arr[j] - ref) <= tol:
            j += 1
        consensus.append(float(np.round(arr[i:j].mean(), 5)))
        i = j

    return sorted(consensus)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: remap a local m/z dict onto the global consensus grid
# ─────────────────────────────────────────────────────────────────────────────
def _map_to_consensus(local_dict, consensus_arr, tol_ppm=_MZ_TOLERANCE_PPM, tol_da=0.0):
    """Map every (mz, intensity) pair onto the nearest consensus bin."""
    out = {}
    for mz, inten in local_dict.items():
        tol = _tol_da(mz, tol_ppm, tol_da)
        lo  = np.searchsorted(consensus_arr, mz - tol)
        hi  = np.searchsorted(consensus_arr, mz + tol, side='right')
        if lo < hi:
            best = lo + int(np.argmin(np.abs(consensus_arr[lo:hi] - mz)))
            key  = float(consensus_arr[best])
            out[key] = out.get(key, 0.0) + inten
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  CORE: process a single mzML/mzXML file
# ─────────────────────────────────────────────────────────────────────────────
def _process_single_file(uploaded_file, class_name,
                         min_apex_intensity_pct=1.0,
                         tol_ppm=_MZ_TOLERANCE_PPM, tol_da=0.0,
                         min_peak_width=2, min_distance=0, min_prominence_pct=5.0):
    """
    Load one mzML/mzXML, filter to MS1, detect peaks, integrate each peak.

    min_apex_intensity_pct : noise floor (% of global TIC max).
                             Replaces the old peak_height_threshold.
    min_distance           : 0 = auto-detected from valley spacing.

    Returns (rows, mz_set, meta_rows, chrom_data)
    chrom_data = {file, class, rt, tic, peaks_rt, auto_distance}
    """
    rows, mz_set, meta_rows = [], set(), []
    chrom_data = {
        'file': uploaded_file.name, 'class': class_name,
        'rt': np.array([]), 'tic': np.array([]), 'peaks_rt': [],
        'auto_distance': 0,
    }

    suffix   = ".mzML" if uploaded_file.name.lower().endswith(".mzml") else ".mzXML"
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        exp = MSExperiment()
        if suffix == ".mzML":
            MzMLFile().load(tmp_path, exp)
        else:
            MzXMLFile().load(tmp_path, exp)

        all_spectra   = exp.getSpectra()
        chromatograms = exp.getChromatograms()

        if not all_spectra:
            st.warning(f"No spectra in {uploaded_file.name}")
            return [], set(), [], chrom_data

        # Filter to MS1 only
        spectra = [s for s in all_spectra if s.getMSLevel() in (0, 1)]
        if not spectra:
            st.warning(
                f"{uploaded_file.name}: no MS1 spectra found. "
                "Falling back to all scan levels."
            )
            spectra = all_spectra

        spectra  = sorted(spectra, key=lambda s: s.getRT())
        spec_rts = np.array([s.getRT() for s in spectra], dtype=np.float64)

        # Reconstructed TIC from MS1
        tic_reconstructed = np.array([
            float(np.sum(s.get_peaks()[1])) if len(s.get_peaks()[0]) > 0 else 0.0
            for s in spectra
        ], dtype=np.float64)

        chrom_data["rt"]  = spec_rts
        chrom_data["tic"] = tic_reconstructed

        # Compute auto distance for display in sidebar
        _sig_detr = np.clip(
            __import__('scipy.signal', fromlist=['detrend']).detrend(
                tic_reconstructed.astype(np.float64)
            ), 0, None
        )
        chrom_data["auto_distance"] = _auto_min_distance(_sig_detr)

        intervals = []

        # Priority A: instrument-stored chromatogram
        if chromatograms:
            chrom          = chromatograms[0]
            times, tic_int = chrom.get_peaks()
            times   = np.array(times,   dtype=np.float64)
            tic_int = np.array(tic_int, dtype=np.float64)

            raw = _find_peak_intervals(
                times, tic_int,
                min_apex_intensity_pct=min_apex_intensity_pct,
                min_peak_width=min_peak_width,
                min_distance=min_distance,
                min_prominence_pct=min_prominence_pct,
            )
            for (lt, at, rt_i) in raw:
                t_start, t_end, t_apex = times[lt], times[rt_i], times[at]
                mask = (spec_rts >= t_start) & (spec_rts <= t_end)
                idxs = np.where(mask)[0]
                if len(idxs) >= 1:
                    apex_local = int(np.argmin(np.abs(spec_rts[idxs] - t_apex)))
                    intervals.append((int(idxs[0]), int(idxs[apex_local]), int(idxs[-1])))

            chrom_data["rt"]  = times
            chrom_data["tic"] = tic_int

        # Priority B: reconstructed TIC from MS1 spectra
        if not intervals:
            raw = _find_peak_intervals(
                spec_rts, tic_reconstructed,
                min_apex_intensity_pct=min_apex_intensity_pct,
                min_peak_width=min_peak_width,
                min_distance=min_distance,
                min_prominence_pct=min_prominence_pct,
            )
            for (lt, at, rt_i) in raw:
                intervals.append((lt, at, rt_i))

        # Last resort
        if not intervals:
            best = int(np.argmax(tic_reconstructed))
            intervals = [(best, best, best)]

        # Store apex RTs for chromatogram display
        chrom_data["peaks_rt"] = [
            float(spec_rts[apex]) for (_, apex, _) in intervals
            if apex < len(spec_rts)
        ]

        # Integrate each peak — window is exactly [scan_start, scan_end] inclusive
        for (scan_start, scan_apex, scan_end) in intervals:
            # All scans in the peak window (start to end, no gaps)
            window = spectra[scan_start: scan_end + 1]
            # Filter out completely empty scans but keep the range intact
            window = [s for s in window if len(s.get_peaks()[0]) > 0]
            if not window:
                continue

            # Pass the true apex RT so the feature row RT is accurate
            _apex_rt = spectra[scan_apex].getRT() if scan_apex < len(spectra) else None
            mz_bins, rt_apex = _integrate_peak(
                window, tol_ppm=tol_ppm, tol_da=tol_da, apex_rt=_apex_rt
            )
            if not mz_bins:
                continue

            mz_set.update(mz_bins.keys())
            rows.append(mz_bins)
            meta_rows.append({
                "Class": class_name, "File": uploaded_file.name,
                "RT": float(rt_apex), "Sum": float(sum(mz_bins.values())),
            })

        if not rows:
            st.warning(
                f"No valid peaks extracted from {uploaded_file.name}. "
                "Try lowering the peak threshold or the minimum prominence."
            )

    except Exception as e:
        st.error(f"Error loading {uploaded_file.name}: {e}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        try:
            del exp
        except Exception:
            pass

    return rows, mz_set, meta_rows, chrom_data


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def load_uploaded_files(grouped_files, progress_bar,
                        min_apex_intensity_pct=1.0,
                        tol_ppm=_MZ_TOLERANCE_PPM, tol_da=0.0,
                        min_peak_width=2, min_distance=0, min_prominence_pct=5.0):
    """
    Load and integrate mzML/mzXML MS1 files grouped by biological class.

    Parameters
    ----------
    min_apex_intensity_pct : float
        Noise floor: minimum apex intensity as % of global TIC max.
        Default 1 % — keeps all real peaks, rejects baseline artifacts.
        Replaces the old peak_height_threshold.
    tol_ppm / tol_da       : m/z binning tolerance (larger window wins).
    min_peak_width         : minimum scans per peak.
    min_distance           : minimum scans between apices.
                             0 (default) = auto-detected per file from
                             valley spacing of the TIC — no manual tuning needed.
    min_prominence_pct     : prominence threshold (% of global TIC max).

    Integration guarantee
    ---------------------
    Every scan from peak_start to peak_end (inclusive) is summed — no scan
    is missed or counted twice.  The apex RT stored in metadata is always
    the RT of the detected apex scan, not the midpoint of the window.

    Stores chromatogram data in st.session_state["mzml_chrom_data"].
    """
    total_files = sum(len(g.get("files", [])) for g in grouped_files)
    if total_files == 0:
        st.error("No files to load.")
        return pd.DataFrame()

    progress_placeholder = st.empty()
    current_file         = 0

    all_rows       = []
    all_meta       = []
    all_mz_sets    = []
    all_chrom_data = []

    for group in grouped_files:
        class_name = group.get("class_name", "Unknown")
        for uploaded_file in group.get("files", []):
            current_file += 1
            progress_bar.progress(current_file / total_files)
            progress_placeholder.write(
                f"Processing **{uploaded_file.name}** ({current_file}/{total_files})..."
            )

            rows, mz_set, meta_rows, chrom_data = _process_single_file(
                uploaded_file, class_name,
                min_apex_intensity_pct=min_apex_intensity_pct,
                tol_ppm=tol_ppm, tol_da=tol_da,
                min_peak_width=min_peak_width,
                min_distance=min_distance,
                min_prominence_pct=min_prominence_pct,
            )

            for r, m in zip(rows, meta_rows):
                all_rows.append(r)
                all_meta.append(m)
                all_mz_sets.append(mz_set)

            if chrom_data["tic"].size > 0:
                all_chrom_data.append(chrom_data)

    progress_placeholder.empty()

    if not all_rows:
        st.error(
            "No valid data extracted. Check file formats, signal quality, "
            "lower the peak threshold, or reduce the minimum prominence."
        )
        return pd.DataFrame()

    # Store chromatogram data for sidebar display
    st.session_state["mzml_chrom_data"] = all_chrom_data

    # Pass 2 – consensus m/z axis
    consensus_mz  = _align_mz_columns(all_mz_sets, tol_ppm=tol_ppm, tol_da=tol_da)
    consensus_arr = np.array(consensus_mz, dtype=np.float64)
    n_rows, n_cols = len(all_rows), len(consensus_mz)

    tol_str = f"{tol_ppm:.0f} ppm" + (f" / {tol_da:.4f} Da" if tol_da > 0 else "")
    st.info(
        f"Consensus m/z features: **{n_cols:,}**  |  "
        f"Integrated peaks: **{n_rows}**  |  "
        f"Files: **{current_file}**  |  Tolerance: **{tol_str}**"
    )

    # Pass 3 – float32 matrix
    matrix = np.full((n_rows, n_cols), np.nan, dtype=np.float32)

    for i, row_dict in enumerate(all_rows):
        mapped = _map_to_consensus(row_dict, consensus_arr, tol_ppm=tol_ppm, tol_da=tol_da)
        for mz, inten in mapped.items():
            idx = np.searchsorted(consensus_arr, mz)
            if idx < n_cols and consensus_arr[idx] == mz:
                matrix[i, idx] = np.float32(inten)

    meta_df  = pd.DataFrame(all_meta)
    feat_df  = pd.DataFrame(matrix, columns=consensus_mz)
    final_df = pd.concat(
        [meta_df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1
    )

    final_df["Class"] = final_df["Class"].astype(str)
    final_df["File"]  = final_df["File"].astype(str)
    final_df["RT"]    = final_df["RT"].astype(np.float32)
    final_df["Sum"]   = final_df["Sum"].astype(np.float32)

    st.session_state["class_colors"] = {
        cls: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
        for i, cls in enumerate(final_df["Class"].dropna().unique())
    }

    st.success(
        f"Loaded: **{n_rows}** integrated peak(s) x "
        f"**{n_cols:,}** m/z features across **{current_file}** file(s)."
    )

    del all_rows, matrix, feat_df, meta_df, all_mz_sets
    gc.collect()

    return final_df


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR CHROMATOGRAM DISPLAY
# ─────────────────────────────────────────────────────────────────────────────
def render_mzml_chromatogram_sidebar():
    """
    Render a compact TIC chromatogram with detected peak markers (triangles)
    in the Streamlit sidebar, immediately after mzML import.

    Reads  : st.session_state["mzml_chrom_data"]
    Colors : st.session_state["class_colors"]
    """
    import plotly.graph_objects as go

    chrom_list = st.session_state.get("mzml_chrom_data")
    if not chrom_list:
        return

    class_colors = st.session_state.get("class_colors", {})

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='font-size:0.68rem;font-weight:800;text-transform:uppercase;"
        "letter-spacing:0.10em;color:#1e3a5f;margin-bottom:4px;'>"
        "MS1 Chromatogram — selected peaks</div>",
        unsafe_allow_html=True,
    )

    fig = go.Figure()
    added_classes = set()

    for cd in chrom_list:
        rt       = cd["rt"]
        tic      = cd["tic"]
        fname    = cd["file"]
        cls      = cd["class"]
        peaks_rt = cd.get("peaks_rt", [])

        color      = class_colors.get(cls, "#318CE7")
        show_leg   = cls not in added_classes
        added_classes.add(cls)

        # Auto-convert seconds to minutes if RT values are large
        use_min  = rt.max() > 300
        rt_disp  = rt / 60.0 if use_min else rt
        rt_label = "RT (min)" if use_min else "RT (s)"

        # TIC line
        fig.add_trace(go.Scatter(
            x=rt_disp, y=tic,
            mode="lines",
            name=cls,
            line=dict(color=color, width=1.5),
            opacity=0.8,
            showlegend=show_leg,
            hovertemplate=(
                f"<b>{fname}</b><br>RT: %{{x:.2f}}<br>TIC: %{{y:.2e}}<extra></extra>"
            ),
        ))

        # Peak apex triangles
        if peaks_rt:
            pr_disp = [p / 60.0 if use_min else p for p in peaks_rt]
            pr_tic  = [float(tic[np.argmin(np.abs(rt - p))]) for p in peaks_rt]
            fig.add_trace(go.Scatter(
                x=pr_disp, y=pr_tic,
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color=color,
                            line=dict(color="white", width=1.2)),
                showlegend=False,
                hovertemplate=(
                    f"<b>Peak [{fname}]</b><br>RT: %{{x:.2f}}<br>"
                    "Intensity: %{y:.2e}<extra></extra>"
                ),
            ))

    fig.update_layout(
        height=230,
        margin=dict(l=4, r=4, t=30, b=28),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=10, color="#1e3a5f", family="Arial"),
        title=dict(
            text="TIC — detected peaks (▲)",
            font=dict(size=11, color="#1e3a5f"), x=0.01,
        ),
        xaxis=dict(
            title=rt_label, title_font=dict(size=10), tickfont=dict(size=9),
            gridcolor="#f1f5f9", linecolor="#cbd5e1",
        ),
        yaxis=dict(
            title="Intensity", title_font=dict(size=10), tickfont=dict(size=9),
            gridcolor="#f1f5f9", linecolor="#cbd5e1", tickformat=".2e",
        ),
        legend=dict(
            font=dict(size=9), bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#e2e8f0", borderwidth=1,
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        ),
        hovermode="x unified",
    )

    st.sidebar.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False, "scrollZoom": False})

    total_peaks = sum(len(cd.get("peaks_rt", [])) for cd in chrom_list)
    auto_dists  = [cd.get("auto_distance", 0) for cd in chrom_list if cd.get("auto_distance", 0) > 0]
    dist_note   = f"  ·  auto dist = {int(np.median(auto_dists))} scans" if auto_dists else ""
    st.sidebar.caption(
        f"▲ {total_peaks} peak(s) across {len(chrom_list)} file(s){dist_note}"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  UNCHANGED HELPERS  (full compatibility with the rest of Profiler)
# ─────────────────────────────────────────────────────────────────────────────

def save_uploaded_file(uploaded_file, destination: str) -> str:
    with open(destination, "wb") as f:
        f.write(uploaded_file.read())
    return destination


def add_group():
    st.session_state["file_groups"].append({
        "class_name": f"Class {len(st.session_state['file_groups']) + 1}",
        "files": []
    })
    st.rerun()


def lazy_import(module_name: str):
    import importlib
    return importlib.import_module(module_name)


def get_data(source_name):
    mapping = {
        "Raw Data"    : st.session_state.get("final_data", st.session_state.get("data")),
        "Preprocessed": st.session_state.get("preprocessed_data"),
        "Oversampled" : st.session_state.get("oversampled_data"),
        "Undersampled": st.session_state.get("undersampled_data"),
    }
    return mapping.get(source_name, None)


def safe_load_data(data_source):
    if data_source == "None":
        return None
    session_key = f"cached_data_{data_source}"
    if session_key not in st.session_state:
        with st.spinner("Loading selected data source..."):
            st.session_state[session_key] = get_data(data_source)
    return st.session_state[session_key]


def get_data_for_source(source_name):
    return get_data(source_name)


def finalize_data_load(df, source_label):
    """Add required columns, coerce feature columns to numeric, store in session state."""
    if df is None:
        return
    for col in ["File", "RT", "Sum"]:
        if col not in df.columns:
            df[col] = "Unknown" if col == "File" else 0
    _meta_excl   = {"Class", "ID", "File", "RT", "Sum", "Original_index"}
    feature_cols = [c for c in df.columns
                    if c not in _meta_excl and not str(c).endswith("_meta")]
    if feature_cols:
        df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    st.session_state["data"] = df
    if "Class" in df.columns:
        st.session_state["class_renaming"] = {cls: cls for cls in df["Class"].unique()}
        st.session_state["class_colors"] = {
            cls: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
            for i, cls in enumerate(df["Class"].unique())
        }
    st.success(f"{source_label} data processed successfully!")
