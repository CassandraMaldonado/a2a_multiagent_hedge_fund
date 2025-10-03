import subprocess
import sys
import os
import importlib.util

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"Python version: {sys.version.split()[0]}")
    return True

def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    
    try:
        spec = importlib.util.find_spec(import_name)
        return spec is not None
    except ImportError:
        return False

def install_requirements():
    """Install required packages"""
    
    # Essential packages
    essential_packages = [
        ("streamlit", "streamlit"),
        ("pandas", "pandas"),
        ("numpy", "numpy"), 
        ("plotly", "plotly"),
        ("yfinance", "yfinance"),
        ("nest-asyncio", "nest_asyncio"),
        ("matplotlib", "matplotlib"),
        ("scipy", "scipy")
    ]
    
    print("🔍 Checking essential packages...")
    
    missing_packages = []
    for package, import_name in essential_packages:
        if check_package(import_name):
            print(f"✅ {package}")
        else:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Installing {len(missing_packages)} missing packages...")
        
        for package in missing_packages:
            print(f"Installing {package}...")
            if install_package(package):
                print(f"✅ {package} installed successfully")
            else:
                print(f"❌ Failed to install {package}")
                return False
    
    print("\n✅ All essential packages are installed!")
    return True

def check_files():
    """Check if required files exist"""
    required_files = [
        "Final_GENAI_V3.py",
        "streamlit_app.py"
    ]
    
    print("\n📁 Checking required files...")
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - Missing")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {missing_files}")
        print("Please ensure you have:")
        print("- Final_GENAI_V3.py (your AI model)")
        print("- streamlit_app.py (the Streamlit interface)")
        return False
    
    return True

def test_imports():
    """Test if key imports work"""
    print("\n🧪 Testing imports...")
    
    test_imports = [
        ("streamlit", "import streamlit as st"),
        ("pandas", "import pandas as pd"),
        ("numpy", "import numpy as np"),
        ("plotly", "import plotly.graph_objects as go"),
        ("yfinance", "import yfinance as yf"),
        ("nest_asyncio", "import nest_asyncio")
    ]
    
    for name, import_statement in test_imports:
        try:
            exec(import_statement)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            return False
    
    return True

def create_run_script():
    """Create a convenient run script"""
    run_script = '''#!/usr/bin/env python3
"""
Quick run script for AI Financial Forecasting App
"""
import subprocess
import sys

if __name__ == "__main__":
    print("🚀 Starting AI Financial Forecasting App...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    except KeyboardInterrupt:
        print("\\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
'''
    
    with open("run_app.py", "w") as f:
        f.write(run_script)
    
    print("✅ Created run_app.py")

def main():
    """Main setup process"""
    print("🤖 AI Financial Forecasting Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        return False
    
    # Check files
    if not check_files():
        print("❌ Missing required files")
        return False
    
    # Test imports
    if not test_imports():
        print("❌ Import test failed")
        return False
    
    # Create convenience script
    create_run_script()
    
    print("\n🎉 Setup Complete!")
    print("=" * 50)
    print("Ready to run! Use one of these commands:")
    print("• python run_app.py")
    print("• streamlit run streamlit_app.py")
    print("\nThe app will open in your browser automatically.")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)
