
import subprocess
import sys
import os

def main():
    """Launch the Streamlit app"""
    print("Starting Streamlit Interface...")
    print("URL: http://localhost:8501")
    print("-" * 50)

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Streamlit interface stopped.")
    except Exception as e:
        print(f" Error starting Streamlit: {e}")

if __name__ == "__main__":
    main()
