

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
    """Lance Streamlit dans un processus séparé."""
    gui_path = resource_path("Profiler_Desktop_Gui.py")
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
            "--server.maxUploadSize=10240"  # increase the limit as needed (by default in Porfiler 10GB)
        ],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,  # Windows uniquement
    )
    return process

def wait_for_streamlit(url="http://localhost:8501", timeout=60):
    """Wait until Streamlit is ready before opening the browser."""
    start_time = time.time()
    while True:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print("✅ Streamlit is ready !")
                break
        except Exception as e:
            if time.time() - start_time > timeout:
                print(f"❌ Error: Unable to start Streamlit ({e})")
                sys.exit(1)
            time.sleep(1)

if __name__ == "__main__":
    # Lancer Streamlit dans un processus séparé
    streamlit_process = run_streamlit()

    try:
        wait_for_streamlit()

        webbrowser.open("http://localhost:8501")

        # Maintenir le programme actif tant que Streamlit tourne
        print("🟢 Porfiler Desktop application running (Ctrl+C to exit)")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🟥 Closure requested by the user.")
    finally:
        # Tuer le processus Streamlit proprement
        streamlit_process.terminate()
        streamlit_process.wait()
        print("✅ Profiler closed cleanly.")
