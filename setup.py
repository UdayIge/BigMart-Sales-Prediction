"""
Setup Script for BigMart Sales Prediction Project
Author: Data Analysis Project
Description: Automated setup and installation script
"""

import os
import sys
import subprocess
import platform

def print_banner():
    """Print setup banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║            🛒 BigMart Sales Prediction Setup                 ║
    ║                                                              ║
    ║              Automated Project Configuration                  ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                     check=True)
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        'data',
        'src', 
        'dashboard',
        'models',
        'results',
        'notebooks'
    ]
    
    print("\n📁 Creating project directories...")
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"   Created: {directory}/")
        else:
            print(f"   Exists: {directory}/")

def check_data_files():
    """Check if data files exist"""
    data_files = ['data/Train.csv', 'data/Test.csv']
    missing_files = []
    
    for file in data_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("\n⚠️  Missing data files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n📥 Please download the BigMart Sales Prediction dataset from Kaggle:")
        print("   https://www.kaggle.com/competitions/bigmart-sales-data")
        print("   Place Train.csv and Test.csv in the data/ directory")
        return False
    
    print("\n✅ Data files found!")
    return True

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
env/
ENV/

# Jupyter Notebook
.ipynb_checkpoints

# Data files
data/*.csv
!data/Train.csv
!data/Test.csv

# Model files
models/*.pkl

# Results
results/*.png
results/*.html
results/*.csv

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    print("📝 Created .gitignore file")

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Setup failed during package installation")
        sys.exit(1)
    
    # Check data files
    data_available = check_data_files()
    
    # Create .gitignore
    create_gitignore()
    
    # Final message
    print(f"\n{'='*60}")
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    
    print("\n📋 Next Steps:")
    print("1. Ensure data files are in the data/ directory")
    print("2. Run the complete pipeline:")
    print("   python main.py")
    print("3. Or run individual components:")
    print("   python src/data_preprocessing.py")
    print("   python src/eda.py")
    print("   python src/model_training.py")
    print("4. Launch the dashboard:")
    print("   python run_dashboard.py")
    print("   or")
    print("   streamlit run dashboard/app.py")
    
    print("\n📚 Documentation:")
    print("   Read README.md for detailed information")
    
    print("\n🚀 Happy Analyzing!")
    
    if not data_available:
        print("\n⚠️  Remember to add your data files before running the analysis!")

if __name__ == "__main__":
    main()
