import streamlit as st

# Configure page first - this should always work
st.set_page_config(
    page_title="🤖 AI Financial Forecasting - Debug",
    page_icon="📈",
    layout="wide"
)

# Show basic content immediately
st.title("🤖 AI Financial Forecasting System")
st.write("✔ Streamlit is working!")

# Now try imports one by one and show status
st.markdown("## 🔍 System Diagnostic")

# Test basic imports
try:
    import pandas as pd
    st.success("✔ Pandas imported successfully")
except Exception as e:
    st.error(f"Pandas failed: {e}")

try:
    import numpy as np
    st.success("✔ NumPy imported successfully")
except Exception as e:
    st.error(f"NumPy failed: {e}")

try:
    from datetime import datetime
    st.success("✔ Datetime imported successfully")
except Exception as e:
    st.error(f"Datetime failed: {e}")

try:
    import json
    st.success("✔ JSON imported successfully")
except Exception as e:
    st.error(f"JSON failed: {e}")

# Test optional imports
st.markdown("### Optional Dependencies")

try:
    import asyncio
    st.success("✔ Asyncio available")
except Exception as e:
    st.warning(f"Asyncio issue: {e}")

try:
    import nest_asyncio
    st.success("✔ Nest-asyncio available")
except Exception as e:
    st.warning(f"Nest-asyncio not available: {e}")

try:
    import plotly.graph_objects as go
    st.success("✔ Plotly available")
except Exception as e:
    st.warning(f"Plotly not available: {e}")

try:
    import yfinance as yf
    st.success("✅ YFinance available")
except Exception as e:
    st.warning(f"YFinance not available: {e}")

# Test final_vCM import
st.markdown("### AI Pipeline Import Test")

try:
    # Try to import the pipeline file
    import final_vCM
    st.success("✅ final_vCM.py imported successfully!")
    
    # Test if key classes are available
    if hasattr(final_vCM, 'MarketDataAgent'):
        st.success("✅ MarketDataAgent class found")
    else:
        st.error("❌ MarketDataAgent class not found")
        
    if hasattr(final_vCM, 'FinancialGoal'):
        st.success("✅ FinancialGoal class found")
    else:
        st.error("❌ FinancialGoal class not found")
        
    if hasattr(final_vCM, 'run_complete_analysis'):
        st.success("✅ run_complete_analysis function found")
    else:
        st.error("❌ run_complete_analysis function not found")
        
except ImportError as e:
    st.error(f"❌ Cannot import final_vCM.py: {e}")
    st.code(str(e))
except SyntaxError as e:
    st.error(f"❌ Syntax error in final_vCM.py: {e}")
    st.code(f"Line {e.lineno}: {e.text}")
except Exception as e:
    st.error(f"❌ Unexpected error importing final_vCM.py: {e}")
    st.code(str(e))

# Show environment info
st.markdown("### Environment Information")
import sys
st.write(f"**Python Version:** {sys.version}")
st.write(f"**Streamlit Version:** {st.__version__}")

# Simple test functionality
st.markdown("### Basic Functionality Test")

if st.button("🧪 Test Basic Function"):
    st.success("✅ Button clicks work!")
    st.balloons()

# File listing
st.markdown("### File Directory")
try:
    import os
    files = os.listdir('.')
    st.write("**Files in current directory:**")
    for file in files:
        if file.endswith('.py'):
            st.write(f"📄 {file}")
except Exception as e:
    st.error(f"Cannot list files: {e}")

# Debug info
with st.expander("🔧 Debug Information"):
    st.write("This minimal app helps identify what's causing issues.")
    st.write("If you see this page, Streamlit itself is working.")
    st.write("Check the import results above to see what's failing.")
    
    if st.button("🔄 Refresh Page"):
        st.experimental_rerun()
