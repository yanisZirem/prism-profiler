"""
Software Name: Profiler – Desktop Edition
Module: main.py  (entry point)
Author: Yanis Zirem
Email : yanis.zirem@univ-lille.fr
Last Updated: 23/04/2026
Version: 1.3.1

Usage:
    Direct  →  python app/main.py
    Via bat  →  run_profiler.bat   (recommended on Windows)

Behaviour:
    Each launch opens a NEW independent session on a free port (8501, 8502, …).
    Multiple instances can run simultaneously without interfering with each other.

License: IDDN.FR.001.300044.000.S6.C7.2025.0009.3123010
© 2025 PRISM U1192 Laboratory – INSERM / CHU de Lille / Université de Lille
"""

import os
import sys
import time
import socket
import subprocess
import webbrowser
import urllib.request
import urllib.error
from pathlib import Path

# ── Silence TensorFlow / CUDA warnings BEFORE spawning the Streamlit process ─
# These env-vars are inherited by the child process, so TF never prints the
# "Could not load dynamic library 'cudart64_110.dll'" family of warnings
# on machines without a CUDA-capable GPU.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")        # 0=DEBUG 1=INFO 2=WARNING 3=ERROR
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")       # avoids a separate oneDNN info line
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")        # tell TF there is no GPU → skip CUDA probe
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")  # suppress allocator warnings

# Keras / TF mixed-precision: force float32 so the mixed_float16 slow-GPU
# warning is never triggered on CPU-only machines.
os.environ.setdefault("TF_KERAS_DEFAULT_DTYPE", "float32")

# --- Ensure project root is on sys.path so all app.* imports resolve ---------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# --- Resolve paths robustly (PyInstaller bundle or plain Python) --------------

def _resource_path(relative: str) -> str:
    """Return absolute path, works both in-source and PyInstaller bundle."""
    if hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)
    else:
        base = Path(__file__).resolve().parent
    return str(base / relative)


# --- Port helpers -------------------------------------------------------------

PORT_RANGE_START = 8501
PORT_RANGE_END   = 8600   # scan up to 100 ports before giving up


def _port_in_use(port: int) -> bool:
    """
    Return True if something is already listening on *port*.
    Uses a connect() attempt — more reliable than bind() on Windows
    where TIME_WAIT sockets can give false negatives.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        try:
            s.connect(("127.0.0.1", port))
            return True   # connection succeeded -> port is occupied
        except (ConnectionRefusedError, socket.timeout, OSError):
            return False  # nothing listening -> port is free


def _find_free_port(start: int = PORT_RANGE_START,
                    end: int   = PORT_RANGE_END) -> int:
    """Return the first free TCP port in [start, end], or raise RuntimeError."""
    for port in range(start, end + 1):
        if not _port_in_use(port):
            return port
    raise RuntimeError(
        f"No free port found between {start} and {end}. "
        "Close some Profiler sessions and try again."
    )


# --- Streamlit launch ---------------------------------------------------------

GUI_ENTRYPOINT = _resource_path("gui/Profiler_Desktop_Gui.py")


def _run_streamlit(port: int) -> subprocess.Popen:
    """Spawn a Streamlit instance on *port* and return the process handle."""
    creationflags = 0
    if sys.platform.startswith("win"):
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        GUI_ENTRYPOINT,
        f"--server.port={port}",
        "--server.headless=true",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
        "--server.maxUploadSize=10240",
    ]

    # Pass the current environment (which already contains TF_CPP_MIN_LOG_LEVEL
    # and CUDA_VISIBLE_DEVICES set above) to the child process.
    return subprocess.Popen(cmd, creationflags=creationflags, env=os.environ.copy())


def _wait_for_streamlit(url: str, proc: subprocess.Popen,
                        timeout: int = 90) -> bool:
    """
    Poll until Streamlit responds HTTP 200, then return True.
    Returns False if the process crashes or the timeout expires.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        # Abort early if Streamlit already crashed
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.8)
    return False


# --- Main --------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting Profiler Offline ...")
    print(f"    GUI entrypoint : {GUI_ENTRYPOINT}")

    if not os.path.isfile(GUI_ENTRYPOINT):
        print(f"GUI file not found: {GUI_ENTRYPOINT}")
        sys.exit(1)

    # Find a genuinely free port
    try:
        port = _find_free_port()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print(f"    Port           : {port}")

    # Launch Streamlit
    streamlit_proc = _run_streamlit(port)

    # Small pause so Streamlit has time to bind the socket
    time.sleep(1.5)

    # Detect immediate crash (e.g. port was stolen between find and launch)
    if streamlit_proc.poll() is not None:
        print(f"Streamlit exited immediately (return code {streamlit_proc.returncode}).")
        print("Try launching again — a new free port will be selected automatically.")
        sys.exit(1)

    url = f"http://localhost:{port}"

    try:
        ready = _wait_for_streamlit(url, streamlit_proc)
        if not ready:
            print("Streamlit did not become ready within the timeout.")
            streamlit_proc.terminate()
            sys.exit(1)

        print("Profiler is ready.")
        webbrowser.open(url)
        print(f"Profiler running at {url}  (Ctrl+C to stop)")
        streamlit_proc.wait()

    except KeyboardInterrupt:
        print("\nShutdown requested by user.")
    finally:
        streamlit_proc.terminate()
        try:
            streamlit_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            streamlit_proc.kill()
        print("Profiler closed cleanly.")
