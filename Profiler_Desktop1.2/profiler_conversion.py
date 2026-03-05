"""
profiler_conversion.py  –  Desktop version
Part of the Profiler project (Profiler_Desktop_Gui).

Key differences vs the web version
────────────────────────────────────
• Bruker .d files are **directories**, not single files.
  On the desktop the user can provide them in two ways:
    1. Already extracted: the `raw_dir` folder contains one or more *.d
       sub-folders (e.g.  raw_dir/sample1.d/,  raw_dir/sample2.d/).
    2. Zipped: the user uploads *.d.zip (or a single big ZIP that contains
       several .d folders). The GUI extracts them into raw_dir first; this
       module then finds the resulting .d sub-folders automatically.

• msconvert is called **locally** (no Docker) when it is on the PATH or in
  the standard ProteoWizard install locations.
  Docker fallback is kept for environments where only Docker is available.

• All other instrument types (Thermo .raw, Waters .raw) work exactly as
  before (Docker or local msconvert).

Author: Yanis Zirem
"""

import os
import shutil
import subprocess
import zipfile

import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# Helper: locate a local msconvert binary
# ─────────────────────────────────────────────────────────────────────────────

def _find_local_msconvert():
    """
    Return the absolute path of a local msconvert executable, or None.
    Checked in order:
      1. System PATH  (works on Linux / macOS / Windows if added to PATH)
      2. Typical Windows ProteoWizard install directories
    """
    # 1 — PATH
    found = shutil.which("msconvert")
    if found:
        return found

    # 2 — Common Windows install locations
    win_candidates = [
        r"C:\Program Files\ProteoWizard\msconvert.exe",
        r"C:\Program Files (x86)\ProteoWizard\msconvert.exe",
    ]
    for p in win_candidates:
        if os.path.isfile(p):
            return p

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Helper: scan raw_dir for .d folders AND unzip any .d.zip / .zip inside it
# ─────────────────────────────────────────────────────────────────────────────

def _collect_bruker_d_folders(raw_files_path):
    """
    Return the names of all *.d sub-folders found inside raw_files_path.

    Also handles the case where the user uploaded a ZIP that contains a .d
    folder at the top level or nested one level deep (e.g. sample.d.zip ->
    unzip -> sample.d/).  ZIPs are extracted in-place and then removed.
    """
    # First pass: extract any remaining ZIPs that might contain .d folders
    for entry in list(os.listdir(raw_files_path)):
        if entry.lower().endswith(".zip"):
            zip_path = os.path.join(raw_files_path, entry)
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(raw_files_path)
                os.remove(zip_path)
            except Exception as e:
                st.warning(f"⚠️ Could not extract {entry}: {e}")

    # Second pass: collect all *.d directories
    d_folders = []
    for entry in os.listdir(raw_files_path):
        full = os.path.join(raw_files_path, entry)
        if entry.lower().endswith(".d") and os.path.isdir(full):
            d_folders.append(entry)

    return d_folders


# ─────────────────────────────────────────────────────────────────────────────
# Main conversion function
# ─────────────────────────────────────────────────────────────────────────────

def convert_raw_to_mzml(
    raw_files_path,
    output_dir,
    file_type,
    mass_range=None,
    peak_picking=False,
    lock_mass=None,
    output_format="mzML",
):
    """
    Convert RAW / .d files to mzML (or other ProteoWizard-supported formats).

    Parameters
    ----------
    raw_files_path : str
        Folder containing the input files / sub-folders.
    output_dir : str
        Destination folder for converted files.
    file_type : str
        One of "thermo", "waters", "bruker".
    mass_range : list | tuple | None
        e.g. [600, 1000]  ->  --filter "mzWindow [600,1000]"
    peak_picking : bool
        Apply centroiding filter.
    lock_mass : float | None
        Waters lock-mass m/z value.
    output_format : str
        "mzML" (default), "mzXML", "mz5", "mzDB", ...
    """

    # ── Validate inputs ──────────────────────────────────────────────────────
    if not os.path.isdir(raw_files_path):
        st.error("❌ Invalid input folder path.")
        return

    if file_type not in ("thermo", "waters", "bruker"):
        st.error(f"❌ Unsupported file type: {file_type}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # ── Collect files/folders to convert ────────────────────────────────────
    if file_type == "bruker":
        # .d entries are DIRECTORIES — use dedicated collector that also
        # handles ZIPs that may still be sitting in raw_files_path
        raw_files = _collect_bruker_d_folders(raw_files_path)
        if not raw_files:
            st.error(
                "❌ No Bruker .d folder found in the input directory.\n\n"
                "**How to provide Bruker data on the desktop:**\n"
                "- **Option A** — ZIP each .d folder and upload the ZIP "
                "(e.g. drag `sample1.d.zip` into the uploader). "
                "The GUI will extract it automatically.\n"
                "- **Option B** — Place the already-extracted .d folders "
                "directly inside the input directory."
            )
            return
    else:
        # Thermo / Waters: .raw entries are directories on Windows
        raw_files = [
            f for f in os.listdir(raw_files_path)
            if f.lower().endswith(".raw")
            and os.path.isdir(os.path.join(raw_files_path, f))
        ]
        if not raw_files:
            st.error("❌ No .raw folder found in the input directory.")
            return

    # ── Locate msconvert ─────────────────────────────────────────────────────
    local_msconvert = _find_local_msconvert()
    use_docker = local_msconvert is None

    if use_docker:
        st.info(
            "ℹ️ msconvert not found locally — falling back to Docker. "
            "For faster conversions, install ProteoWizard and make sure "
            "`msconvert` is on your PATH."
        )
    else:
        st.info(f"ℹ️ Using local msconvert: `{local_msconvert}`")

    # ── Build extra filter arguments ─────────────────────────────────────────
    def _extra_filters():
        args = []
        if file_type == "waters" and lock_mass:
            args += ["--filter", f"lockmassRefiner mz={lock_mass} tol=1.0"]
        if mass_range:
            args += ["--filter", f"mzWindow {mass_range}"]
        if peak_picking:
            args += ["--filter", "peakPicking true 1-"]
        return args

    # ── Convert each file/folder ─────────────────────────────────────────────
    st.write(f"🔄 Converting **{len(raw_files)}** {file_type} file(s) → {output_format} …")
    progress_bar = st.progress(0)

    for i, raw_file in enumerate(raw_files):
        raw_file_path = os.path.join(raw_files_path, raw_file)

        # Build output filename: strip the original extension, add output fmt
        stem = raw_file
        for ext in (".d", ".raw", ".RAW"):
            if stem.lower().endswith(ext.lower()):
                stem = stem[: -len(ext)]
                break
        out_filename = f"{stem}.{output_format.lower()}"

        try:
            if use_docker:
                # ── Docker path (unchanged from web version) ─────────────
                cmd = [
                    "sudo", "docker", "run",
                    "-v", f"{raw_files_path}:/data",
                    "-v", f"{output_dir}:/out",
                    "chambm/pwiz-skyline-i-agree-to-the-vendor-licenses",
                    "wine", "msconvert",
                    f"/data/{raw_file}",
                    f"--{output_format}",
                    "--outfile", out_filename,
                    "-o", "/out",
                ]
            else:
                # ── Local msconvert path (desktop) ────────────────────────
                # Pass the full local path — no Docker volume mapping needed
                cmd = [
                    local_msconvert,
                    raw_file_path,
                    f"--{output_format}",
                    "--outfile", out_filename,
                    "-o", output_dir,
                ]

            cmd += _extra_filters()

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if result.returncode != 0:
                st.error(
                    f"❌ Error converting **{raw_file}**:\n```\n{result.stderr.strip()}\n```"
                )
            else:
                st.success(f"✅ {raw_file}  →  {out_filename}")

        except FileNotFoundError:
            st.error(
                f"❌ Could not launch msconvert for **{raw_file}**. "
                "Make sure ProteoWizard is installed (or Docker is running)."
            )
        except Exception as exc:
            st.error(f"❌ Unexpected error with **{raw_file}**: {exc}")

        progress_bar.progress((i + 1) / len(raw_files))
