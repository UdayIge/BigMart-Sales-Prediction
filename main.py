"""
Simplified Main Runner for BigMart Sales Prediction Project
Author: Data Analysis Project
Description: Simplified pipeline that works with basic packages
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_banner():
    """Print project banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║            🛒 BigMart Sales Prediction (Simple)             ║
    ║                                                              ║
    ║              Basic Data Analysis & ML Pipeline                ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_requirements():
    """Check if required files exist"""
    required_files = [
        'data/Train.csv',
        'data/Test.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all required files are present before running the pipeline.")
        return False
    
    print("✅ All required files found!")
    return True

def create_directories():
    """Create necessary directories"""
    directories = ['models', 'results']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")

def run_step(step_name, script_path, description):
    """Run a pipeline step"""
    print(f"\n{'='*60}")
    print(f"🚀 {step_name}")
    print(f"📝 {description}")
    print(f"{'='*60}")
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {step_name} completed successfully!")
            if result.stdout:
                print("Output:")
                print(result.stdout)
        else:
            print(f"❌ {step_name} failed!")
            if result.stderr:
                print("Error:")
                print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running {step_name}: {str(e)}")
        return False
    
    return True

def main():
    """Main function to run the simplified pipeline"""
    print_banner()
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Pipeline steps
    steps = [
        {
            'name': 'Data Preprocessing',
            'script': 'src/data_preprocessing.py',
            'description': 'Cleaning and preparing data for analysis'
        },
        {
            'name': 'Exploratory Data Analysis',
            'script': 'src/eda.py',
            'description': 'Analyzing data patterns and creating visualizations'
        },
        {
            'name': 'Simple Model Training',
            'script': 'src/simple_model_training.py',
            'description': 'Training basic machine learning models'
        }
    ]
    
    # Run each step
    for step in steps:
        success = run_step(step['name'], step['script'], step['description'])
        if not success:
            print(f"\n❌ Pipeline stopped at {step['name']}")
            print("Please fix the errors and run again.")
            sys.exit(1)
        
        # Small delay between steps
        time.sleep(2)
    
    # Final message
    print(f"\n{'='*60}")
    print("🎉 SIMPLIFIED PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print("\n📊 Next Steps:")
    print("1. View results in the 'results/' directory")
    print("2. Check trained models in the 'models/' directory")
    print("3. Run the interactive dashboard:")
    print("   streamlit run dashboard/app.py")
    print("\n📁 Generated Files:")
    print("   - Processed data: data/processed_*.csv")
    print("   - Model results: results/model_performance.csv")
    print("   - Visualizations: results/*.png")
    print("   - Trained models: models/*.pkl")
    print("\n🚀 Happy Analyzing!")

if __name__ == "__main__":
    main()
