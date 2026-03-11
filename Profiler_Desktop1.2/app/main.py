"""
Software Name: Profiler – Desktop Edition
Module: main.py  (entry point)
Author: Yanis Zirem
Email : yanis.zirem@univ-lille.fr
Last Updated: 05/03/2026
Version: 1.2.0

Usage:
    Direct  →  python app/main.py
    Via bat  →  run_profiler.bat   (recommended on Windows)

License: IDDN.FR.001.300044.000.S6.C7.2025.0009.3123010
© 2025 PRISM U1192 Laboratory – INSERM / CHU de Lille / Université de Lille
"""

import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path

# ─── Ensure project root is on sys.path so all app.* imports resolve ─────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ─── Resolve paths robustly (PyInstaller bundle or plain Python) ──────────────

def _resource_path(relative: str) -> str:
    """Return absolute path, works both in-source and PyInstaller bundle."""
    if hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)
    else:
        base = Path(__file__).resolve().parent
    return str(base / relative)


# ─── Streamlit launch ─────────────────────────────────────────────────────────

STREAMLIT_PORT = 8501
GUI_ENTRYPOINT = _resource_path("gui/Profiler_Desktop_Gui.py")


def _run_streamlit() -> subprocess.Popen:
    """Spawn Streamlit in a child process and return the handle."""
    creationflags = 0
    if sys.platform.startswith("win"):
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        GUI_ENTRYPOINT,
        f"--server.port={STREAMLIT_PORT}",
        "--server.headless=true",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
        "--server.maxUploadSize=10240",
    ]

    return subprocess.Popen(cmd, creationflags=creationflags)


def _wait_for_streamlit(url: str = f"http://localhost:{STREAMLIT_PORT}",
                        timeout: int = 90) -> None:
    """Poll until Streamlit is responsive, then return."""
    import urllib.request
    import urllib.error

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    print("✅  Profiler is ready.")
                    return
        except Exception:
            time.sleep(1)

    print("❌  Streamlit did not start within the timeout. Check the logs.")
    sys.exit(1)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🔵  Starting Profiler Desktop …")
    print(f"    GUI entrypoint : {GUI_ENTRYPOINT}")

    if not os.path.isfile(GUI_ENTRYPOINT):
        print(f"❌  GUI file not found: {GUI_ENTRYPOINT}")
        sys.exit(1)

    streamlit_proc = _run_streamlit()

    try:
        url = f"http://localhost:{STREAMLIT_PORT}"
        _wait_for_streamlit(url)
        webbrowser.open(url)
        print(f"🟢  Profiler running at {url}  (Ctrl+C to stop)")
        streamlit_proc.wait()          # block until user closes window
    except KeyboardInterrupt:
        print("\n🟥  Shutdown requested by user.")
    finally:
        streamlit_proc.terminate()
        try:
            streamlit_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            streamlit_proc.kill()
        print("✅  Profiler closed cleanly.")
