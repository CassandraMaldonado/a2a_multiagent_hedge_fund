"""
AI Financial Forecasting System - Streamlit Dashboard (Robust Version)
This module imports all functionality from Final_GENAI_V3 and creates a beautiful web interface
with graceful handling of missing dependencies
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Configure Streamlit page first
st.set_page_config(
    page_title="AI Financial Forecasting System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Handle optional imports gracefully
PLOTLY_AVAILABLE = False
ASYNCIO_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    st.warning("⚠️ Plotly not available. Charts will be basic.")

try:
    import asyncio
    import nest_asyncio
    nest_asyncio.apply()
    ASYNCIO_AVAILABLE = True
except ImportError:
    st.warning("⚠️ Async functionality limited.")

# Import from your Final_GENAI_V3.py file
NOTEBOOK_IMPORTED = False
try:
    from Final_GENAI_V3 import *
    NOTEBOOK_IMPORTED = True
    st.success("✅ Successfully imported from Final_GENAI_V3.py")
except ImportError as e:
    st.error(f"⚠️ Could not import from Final_GENAI_V3.py: {str(e)}")
    st.error("Please ensure Final_GENAI_V3.py is in the same directory.")
except Exception as e:
    st.error(f"⚠️ Error importing: {str(e)}")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1E88E5, #43A047);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1E88E5;
        margin-bottom: 1rem;
    }
    
    .success-card {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

def run_pipeline_safely(symbol, openai_key=None, financial_goal=None):
    """
    Safe pipeline execution with multiple fallbacks
    """
    try:
        # Method 1: Try existing pipeline functions
        if NOTEBOOK_IMPORTED:
            if 'run_pipeline' in globals():
                if ASYNCIO_AVAILABLE:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(
                            run_pipeline(symbol=symbol, openai_api_key=openai_key, financial_goal=financial_goal)
                        )
                    finally:
                        loop.close()
                else:
                    st.error("Async not available - pipeline requires async support")
                    return None
            
            elif 'run_complete_analysis' in globals():
                return run_complete_analysis(symbol)
            
            elif 'run_with_financial_planning' in globals() and financial_goal:
                return run_with_financial_planning(
                    symbol=symbol,
                    target_amount=financial_goal.target_amount,
                    current_amount=financial_goal.current_amount,
                    monthly_contribution=financial_goal.monthly_contribution,
                    time_horizon_years=financial_goal.time_horizon_years,
                    age=financial_goal.age,
                    risk_tolerance=financial_goal.risk_tolerance
                )
        
        # Method 2: Try manual agent execution
        return run_manual_analysis(symbol, openai_key, financial_goal)
        
    except Exception as e:
        st.error(f"Pipeline error: {str(e)}")
        return run_basic_analysis(symbol)

def run_manual_analysis(symbol, openai_key=None, financial_goal=None):
    """
    Manual analysis using individual agents
    """
    try:
        if not NOTEBOOK_IMPORTED:
            return run_basic_analysis(symbol)
        
        state = {}
        
        # Try to run agents individually
        if 'MarketDataAgent' in globals():
            st.write("📊 Fetching market data...")
            agent = MarketDataAgent()
            if ASYNCIO_AVAILABLE:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    state = loop.run_until_complete(agent.process(state, symbol=symbol))
                finally:
                    loop.close()
            else:
                # Try sync version if available
                state = agent.process(state, symbol=symbol)
        
        if 'RiskAgent' in globals() and state.get('market_data'):
            st.write("⚠️ Analyzing risk...")
            agent = RiskAgent()
            if ASYNCIO_AVAILABLE:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    state = loop.run_until_complete(agent.process(state))
                finally:
                    loop.close()
        
        # Add more agents as available...
        
        return state if state else run_basic_analysis(symbol)
        
    except Exception as e:
        st.error(f"Manual analysis failed: {str(e)}")
        return run_basic_analysis(symbol)

def run_basic_analysis(symbol):
    """
    Basic analysis using just yfinance
    """
    try:
        import yfinance as yf
        
        st.write("📊 Running basic market analysis...")
        
        # Get basic market data
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1y")
        
        if data.empty:
            st.error(f"No data found for {symbol}")
            return None
        
        # Basic calculations
        current_price = float(data['Close'].iloc[-1])
        returns = data['Close'].pct_change()
        
        # Create simple result structure
        result = {
            'symbol': symbol,
            'analysis_type': 'basic',
            'market_data': {
                'current_price': current_price,
                'return_1d': float(returns.iloc[-1]) if len(returns) > 0 else 0.0,
                'return_5d': float(returns.tail(5).mean()) if len(returns) >= 5 else 0.0,
                'volatility_20d': float(returns.tail(20).std() * np.sqrt(252)) if len(returns) >= 20 else 0.0,
                'trend': 'bullish' if returns.tail(5).mean() > 0 else 'bearish',
                'high_52w': float(data['High'].max()),
                'low_52w': float(data['Low'].min()),
                'volume_avg': float(data['Volume'].mean()),
                'price_data': data['Close'].to_dict(),
                'volume_data': data['Volume'].to_dict()
            }
        }
        
        return result
        
    except Exception as e:
        st.error(f"Basic analysis failed: {str(e)}")
        return None

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Financial Forecasting System</h1>', unsafe_allow_html=True)
    st.markdown("### Comprehensive Multi-Agent Financial Analysis Platform")
    
    # Show system status
    show_system_status()
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("## 🎛️ Configuration Panel")
        
        # Stock Symbol Input
        symbol = st.text_input(
            "📊 Stock Symbol",
            value="AAPL",
            help="Enter a valid stock ticker symbol"
        ).upper()
        
        # OpenAI API Key (Optional)
        openai_key = st.text_input(
            "🔑 OpenAI API Key (Optional)",
            type="password",
            help="For enhanced AI recommendations"
        )
        
        # Analysis Type
        analysis_type = st.selectbox(
            "🎯 Analysis Type",
            ["Stock Analysis Only", "Stock Analysis + Financial Planning"]
        )
        
        # Financial Planning Parameters
        financial_goal = None
        if analysis_type == "Stock Analysis + Financial Planning" and NOTEBOOK_IMPORTED:
            st.markdown("### 💰 Financial Goal Settings")
            
            try:
                target_amount = st.number_input("Target Amount ($)", min_value=1000, value=1000000, step=10000)
                current_amount = st.number_input("Current Savings ($)", min_value=0, value=50000, step=1000)
                monthly_contribution = st.number_input("Monthly Contribution ($)", min_value=0, value=2000, step=100)
                time_horizon = st.slider("Time Horizon (Years)", min_value=1, max_value=50, value=25)
                age = st.slider("Current Age", min_value=18, max_value=80, value=35)
                annual_income = st.number_input("Annual Income ($)", min_value=20000, value=120000, step=5000)
                risk_tolerance = st.selectbox("Risk Tolerance", ["conservative", "moderate", "aggressive"], index=1)
                
                # Create financial goal if class is available
                if 'FinancialGoal' in globals():
                    financial_goal = FinancialGoal(
                        target_amount=float(target_amount),
                        current_amount=float(current_amount),
                        monthly_contribution=float(monthly_contribution),
                        time_horizon_years=int(time_horizon),
                        risk_tolerance=risk_tolerance,
                        age=int(age),
                        annual_income=float(annual_income),
                        goal_type="retirement"
                    )
            except Exception as e:
                st.error(f"Error creating financial goal: {e}")
        
        # Run Analysis Button
        run_analysis = st.button("🚀 Run Analysis", type="primary")
    
    # Main Content
    if run_analysis:
        if not symbol:
            st.error("Please enter a valid stock symbol.")
            return
        
        with st.spinner(f"🔍 Analyzing {symbol}..."):
            results = run_pipeline_safely(symbol, openai_key, financial_goal)
            
            if results:
                display_results(results)
            else:
                st.error("❌ Analysis failed. Please check your setup.")
    else:
        display_welcome_screen()

def show_system_status():
    """Show system status and capabilities"""
    
    with st.expander("🔧 System Status", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Core Libraries:**")
            st.write("✅ Streamlit" if True else "❌ Streamlit")
            st.write("✅ Pandas" if True else "❌ Pandas")
            st.write("✅ NumPy" if True else "❌ NumPy")
        
        with col2:
            st.write("**Optional Libraries:**")
            st.write("✅ Plotly" if PLOTLY_AVAILABLE else "❌ Plotly")
            st.write("✅ Asyncio" if ASYNCIO_AVAILABLE else "❌ Asyncio")
            
        with col3:
            st.write("**AI Model:**")
            st.write("✅ Final_GENAI_V3" if NOTEBOOK_IMPORTED else "❌ Final_GENAI_V3")
            
        if not NOTEBOOK_IMPORTED:
            st.warning("⚠️ AI model not loaded. Only basic analysis available.")

def display_welcome_screen():
    """Display welcome screen"""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h2>🎯 Welcome to AI Financial Forecasting</h2>
            <p style="font-size: 1.2rem; color: #666;">
                Get comprehensive financial analysis powered by AI
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Features
    st.markdown("### 🌟 Available Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Market Analysis</h4>
            <ul>
                <li>Real-time stock data</li>
                <li>Price trends</li>
                <li>Volatility analysis</li>
                <li>52-week ranges</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if NOTEBOOK_IMPORTED:
            st.markdown("""
            <div class="success-card">
                <h4>🤖 AI Features</h4>
                <ul>
                    <li>Multi-agent analysis</li>
                    <li>Risk assessment</li>
                    <li>Price forecasting</li>
                    <li>Smart recommendations</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <h4>🤖 AI Features</h4>
                <ul>
                    <li>Basic analysis only</li>
                    <li>Install AI model for full features</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>💰 Financial Planning</h4>
            <ul>
                <li>Goal-based planning</li>
                <li>Asset allocation</li>
                <li>Risk assessment</li>
                <li>Success probability</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def display_results(results):
    """Display analysis results"""
    
    if not results:
        st.error("❌ No results to display")
        return
    
    symbol = results.get('symbol', 'UNKNOWN')
    analysis_type = results.get('analysis_type', 'full')
    
    st.markdown(f"## 📈 Analysis Results for {symbol}")
    
    if analysis_type == 'basic':
        display_basic_results(results)
    else:
        display_full_results(results)

def display_basic_results(results):
    """Display basic analysis results"""
    
    market_data = results.get('market_data', {})
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = market_data.get('current_price', 0)
        return_1d = market_data.get('return_1d', 0)
        st.metric("Current Price", f"${current_price:.2f}", f"{return_1d:.2%}")
    
    with col2:
        volatility = market_data.get('volatility_20d', 0)
        st.metric("Volatility (20d)", f"{volatility:.1%}")
    
    with col3:
        high_52w = market_data.get('high_52w', 0)
        st.metric("52W High", f"${high_52w:.2f}")
    
    with col4:
        low_52w = market_data.get('low_52w', 0)
        st.metric("52W Low", f"${low_52w:.2f}")
    
    # Basic chart if Plotly available
    if PLOTLY_AVAILABLE and market_data.get('price_data'):
        st.markdown("### 📊 Price Chart")
        
        price_data = market_data['price_data']
        dates = list(price_data.keys())
        prices = list(price_data.values())
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode='lines',
            name='Price',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            title=f"{results['symbol']} Price Chart",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Basic insights
    st.markdown("### 💡 Key Insights")
    
    trend = market_data.get('trend', 'neutral')
    return_5d = market_data.get('return_5d', 0)
    
    if trend == 'bullish':
        st.success(f"📈 **Bullish Trend**: 5-day average return is {return_5d:.2%}")
    elif trend == 'bearish':
        st.error(f"📉 **Bearish Trend**: 5-day average return is {return_5d:.2%}")
    else:
        st.info(f"⚖️ **Neutral Trend**: Mixed signals in recent price action")
    
    if volatility > 0.3:
        st.warning(f"⚠️ **High Volatility**: {volatility:.1%} indicates significant price swings")
    elif volatility < 0.15:
        st.info(f"😌 **Low Volatility**: {volatility:.1%} indicates stable price action")

def display_full_results(results):
    """Display full AI analysis results"""
    
    # This would use the same display functions from the original app
    # but with error handling for missing data
    
    if results.get('market_data'):
        st.markdown("### 📊 Market Analysis")
        market_data = results['market_data']
        
        if hasattr(market_data, 'current_price'):
            st.metric("Current Price", f"${market_data.current_price:.2f}")
        elif isinstance(market_data, dict):
            st.metric("Current Price", f"${market_data.get('current_price', 0):.2f}")
    
    if results.get('risk_metrics'):
        st.markdown("### ⚠️ Risk Analysis")
        risk_metrics = results['risk_metrics']
        
        if hasattr(risk_metrics, 'sharpe_ratio'):
            st.metric("Sharpe Ratio", f"{risk_metrics.sharpe_ratio:.2f}")
    
    if results.get('recommendation'):
        st.markdown("### 🧠 AI Recommendation")
        rec = results['recommendation']
        
        if hasattr(rec, 'action'):
            if rec.action == "BUY":
                st.success(f"🚀 **{rec.action}** - Confidence: {rec.confidence:.1%}")
            elif rec.action == "SELL":
                st.error(f"📉 **{rec.action}** - Confidence: {rec.confidence:.1%}")
            else:
                st.warning(f"⚖️ **{rec.action}** - Confidence: {rec.confidence:.1%}")

# Create download function
def create_download_link(results):
    """Create download link for results"""
    try:
        json_string = json.dumps(results, indent=2, default=str)
        st.download_button(
            label="📥 Download Results",
            data=json_string,
            file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    except Exception as e:
        st.error(f"Download failed: {str(e)}")

if __name__ == "__main__":
    main()
