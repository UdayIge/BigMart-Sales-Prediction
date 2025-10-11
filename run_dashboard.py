"""
Dashboard Runner Script
Author: Data Analysis Project
Description: Simple script to run the Streamlit dashboard
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit dashboard"""
    print("🚀 Starting BigMart Sales Prediction Dashboard...")
    print("📊 Dashboard will open in your default browser")
    print("🔗 URL: http://localhost:8501")
    print("\nPress Ctrl+C to stop the dashboard")
    
    try:
        # Run streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error running dashboard: {str(e)}")
        print("Make sure Streamlit is installed: pip install streamlit")

if __name__ == "__main__":
    main()
