import os
import sys
import time
import subprocess
import requests
import webbrowser

def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))

def run_streamlit():
    """Launch Streamlit in a separate process."""
    gui_path = resource_path("Profiler_Desktop_Gui.py")

    # Windows-only creation flag
    creationflags = 0
    if sys.platform.startswith("win"):
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            gui_path,
            "--server.port=8501",
            "--server.headless=true",
            "--server.enableCORS=false",
            "--server.enableXsrfProtection=false",
            "--server.maxUploadSize=10240"
        ],
        creationflags=creationflags,
    )
    return process

def wait_for_streamlit(url="http://localhost:8501", timeout=60):
    """Wait until Streamlit is ready before opening the browser."""
    start = time.time()
    while True:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                print("✅ Profiler Desktop application is ready!")
                break
        except:
            if time.time() - start > timeout:
                print("❌ Error: Streamlit did not start.")
                sys.exit(1)
            time.sleep(1)

if __name__ == "__main__":
    streamlit_process = run_streamlit()

    try:
        wait_for_streamlit()
        webbrowser.open("http://localhost:8501")
        print("🟢 Profiler Desktop application running (Ctrl+C to exit)")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🟥 Exit requested by user.")
    finally:
        streamlit_process.terminate()
        streamlit_process.wait()
        print("✅ Profiler closed cleanly.")