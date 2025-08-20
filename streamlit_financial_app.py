"""
AI Financial Forecasting System - Streamlit Dashboard
Simple wrapper around final_vCM.py with beautiful UI
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Configure Streamlit page first
st.set_page_config(
    page_title="🤖 AI Financial Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize status variables
ASYNCIO_AVAILABLE = False
NEST_ASYNCIO_AVAILABLE = False
PLOTLY_AVAILABLE = False
PIPELINE_LOADED = False

# Try importing asyncio
try:
    import asyncio
    ASYNCIO_AVAILABLE = True
except ImportError:
    st.warning("⚠️ asyncio not available")

# Try importing nest_asyncio
try:
    import nest_asyncio
    nest_asyncio.apply()
    NEST_ASYNCIO_AVAILABLE = True
except ImportError:
    st.warning("⚠️ nest_asyncio not available. Some async features may be limited.")
    st.info("💡 Install with: `pip install nest-asyncio`")

# Try importing Plotly for charts
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    st.warning("⚠️ Plotly not available. Charts will be basic.")
    st.info("💡 Install with: `pip install plotly`")

# Try importing your final_vCM.py
try:
    from final_vCM import *
    PIPELINE_LOADED = True
    st.success("✅ AI Pipeline Successfully Loaded")
except ImportError as e:
    st.error(f"❌ Could not import final_vCM.py: {e}")
    st.error("Please ensure final_vCM.py is in the same directory")
    PIPELINE_LOADED = False
except Exception as e:
    st.error(f"❌ Error loading AI pipeline: {e}")
    PIPELINE_LOADED = False

# Show installation instructions if needed
if not PIPELINE_LOADED or not NEST_ASYNCIO_AVAILABLE or not PLOTLY_AVAILABLE:
    with st.expander("📦 Installation Instructions", expanded=True):
        st.markdown("### Required Dependencies")
        
        missing_deps = []
        if not NEST_ASYNCIO_AVAILABLE:
            missing_deps.append("nest-asyncio")
        if not PLOTLY_AVAILABLE:
            missing_deps.append("plotly")
        
        if missing_deps:
            st.code(f"pip install {' '.join(missing_deps)}")
        
        if not PIPELINE_LOADED:
            st.markdown("### AI Pipeline Setup")
            st.markdown("1. Ensure `final_vCM.py` is in the same directory as this app")
            st.markdown("2. Install required dependencies for the pipeline:")
            st.code("""pip install yfinance pandas numpy matplotlib plotly fredapi newsapi-python textblob praw nest-asyncio""")
        
        st.markdown("3. Restart the Streamlit app after installation")

# Stop execution if critical components are missing
if not PIPELINE_LOADED:
    st.error("🚫 Cannot proceed without AI pipeline. Please follow setup instructions above.")
    st.stop()

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1E88E5, #43A047, #FF9800);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .success-container {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .warning-container {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .info-container {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

def run_analysis_pipeline(symbol, openai_key=None, financial_goal=None):
    """Run the complete AI analysis pipeline"""
    try:
        if not PIPELINE_LOADED:
            st.error("❌ AI Pipeline not loaded")
            return None
        
        # Try async pipeline if both asyncio and nest_asyncio are available
        if ASYNCIO_AVAILABLE and NEST_ASYNCIO_AVAILABLE:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    result = loop.run_until_complete(
                        run_pipeline_with_real_apis(
                            symbol=symbol,
                            openai_api_key=openai_key,
                            financial_goal=financial_goal
                        )
                    )
                    return result
                finally:
                    loop.close()
            except Exception as e:
                st.warning(f"Async pipeline failed: {e}. Trying sync version...")
        
        # Fallback to sync version
        try:
            if 'run_complete_analysis' in globals():
                return run_complete_analysis(symbol)
            else:
                st.error("❌ No analysis functions available")
                return None
        except Exception as e:
            st.error(f"Sync analysis failed: {e}")
            return None
            
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        return None

def main():
    """Main Streamlit Application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Financial Forecasting System</h1>', unsafe_allow_html=True)
    st.markdown("### 🚀 Powered by Multi-Agent AI Architecture")
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Stock Symbol
        symbol = st.text_input(
            "📈 Stock Symbol",
            value="AAPL",
            help="Enter any valid stock ticker (e.g., AAPL, TSLA, MSFT, GOOGL)"
        ).upper()
        
        # API Keys Section
        with st.expander("🔑 API Keys (Optional)", expanded=False):
            openai_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="For GPT-4 powered recommendations"
            )
            
            fred_key = st.text_input(
                "FRED API Key", 
                type="password",
                help="For real macroeconomic data"
            )
            
            news_key = st.text_input(
                "News API Key",
                type="password", 
                help="For sentiment analysis"
            )
            
            reddit_id = st.text_input(
                "Reddit Client ID",
                type="password",
                help="For social media sentiment"
            )
        
        # Analysis Options
        st.markdown("### 🎯 Analysis Options")
        
        include_planning = st.checkbox(
            "💰 Include Financial Planning",
            value=False,
            help="Add personalized financial goal analysis"
        )
        
        forecast_days = st.slider(
            "📅 Forecast Horizon (Days)",
            min_value=1, max_value=30, value=5
        )
        
        # Financial Planning Section
        financial_goal = None
        if include_planning:
            st.markdown("### 💼 Financial Goal Setup")
            
            goal_type = st.selectbox(
                "🎯 Goal Type",
                ["retirement", "house", "education", "general"]
            )
            
            target_amount = st.number_input(
                "💰 Target Amount ($)",
                min_value=1000,
                value=1000000,
                step=10000,
                format="%d"
            )
            
            current_amount = st.number_input(
                "🏦 Current Savings ($)",
                min_value=0,
                value=50000,
                step=1000,
                format="%d"
            )
            
            monthly_contribution = st.number_input(
                "📅 Monthly Contribution ($)",
                min_value=0,
                value=2000,
                step=100,
                format="%d"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("👤 Age", min_value=18, max_value=80, value=35)
                time_horizon = st.number_input("⏰ Years", min_value=1, max_value=50, value=25)
            
            with col2:
                annual_income = st.number_input("💼 Income ($)", min_value=20000, value=120000, step=5000, format="%d")
                risk_tolerance = st.selectbox("⚖️ Risk Level", ["conservative", "moderate", "aggressive"], index=1)
            
            # Create the financial goal
            if 'FinancialGoal' in globals():
                financial_goal = FinancialGoal(
                    target_amount=float(target_amount),
                    current_amount=float(current_amount),
                    monthly_contribution=float(monthly_contribution),
                    time_horizon_years=int(time_horizon),
                    risk_tolerance=risk_tolerance,
                    age=int(age),
                    annual_income=float(annual_income),
                    goal_type=goal_type
                )
                
                # Quick calculation preview
                monthly_needed = (target_amount - current_amount) / (time_horizon * 12)
                if monthly_contribution >= monthly_needed:
                    st.success(f"✅ On track! Need ${monthly_needed:,.0f}/month")
                else:
                    shortfall = monthly_needed - monthly_contribution
                    st.warning(f"⚠️ Increase by ${shortfall:,.0f}/month")
            else:
                st.error("FinancialGoal class not available")
        
        # Run Analysis Button
        st.markdown("---")
        run_analysis = st.button(
            "🚀 Run Complete Analysis",
            type="primary",
            use_container_width=True
        )
    
    # Main Content Area
    if run_analysis and symbol:
        # Set environment variables for API keys
        if fred_key:
            import os
            os.environ['FRED_API_KEY'] = fred_key
        if news_key:
            import os
            os.environ['NEWS_API_KEY'] = news_key
        if reddit_id:
            import os
            os.environ['REDDIT_CLIENT_ID'] = reddit_id
        
        # Progress indicator
        progress_container = st.container()
        with progress_container:
            st.markdown(f"### 🔍 Analyzing {symbol}...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate progress updates
            steps = [
                "📊 Fetching market data...",
                "⚠️ Analyzing risk metrics...", 
                "🔮 Generating forecasts...",
                "🌍 Macro economic analysis...",
                "😊 Sentiment analysis...",
                "🧠 AI recommendations...",
                "💰 Financial planning..." if financial_goal else "✅ Finalizing..."
            ]
            
            for i, step in enumerate(steps):
                status_text.text(step)
                progress_bar.progress((i + 1) * 100 // len(steps))
                
        # Run the analysis
        with st.spinner("🤖 AI agents working..."):
            results = run_analysis_pipeline(symbol, openai_key, financial_goal)
        
        # Clear progress
        progress_container.empty()
        
        if results:
            display_results(results)
        else:
            st.error("❌ Analysis failed. Please check your inputs and try again.")
            
    elif run_analysis and not symbol:
        st.error("❌ Please enter a stock symbol")
        
    else:
        # Welcome Screen
        display_welcome_screen()

def display_welcome_screen():
    """Display the welcome screen"""
    
    # Hero section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="info-container">
            <h2 style="text-align: center; margin-bottom: 1rem;">🎯 AI-Powered Financial Analysis</h2>
            <p style="text-align: center; font-size: 1.2rem;">
                Get comprehensive investment insights using advanced AI agents
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature highlights
    st.markdown("### 🌟 What You'll Get")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <h3>📊 Market Analysis</h3>
            <ul style="text-align: left; margin-top: 1rem;">
                <li>Real-time price data</li>
                <li>Technical indicators</li>
                <li>Trend analysis</li>
                <li>Support/resistance levels</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <h3>🤖 AI Insights</h3>
            <ul style="text-align: left; margin-top: 1rem;">
                <li>Multi-model forecasts</li>
                <li>Risk assessment</li>
                <li>Sentiment analysis</li>
                <li>Smart recommendations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-container">
            <h3>💰 Financial Planning</h3>
            <ul style="text-align: left; margin-top: 1rem;">
                <li>Goal-based planning</li>
                <li>Asset allocation</li>
                <li>Monte Carlo simulation</li>
                <li>Success probability</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Sample analysis preview
    st.markdown("### 📸 Sample Output Preview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.code("""
🤖 AI RECOMMENDATION for AAPL:
   Action: BUY
   Confidence: 82.3%
   Position Size: 18.5%
   Risk Level: MEDIUM
   Entry Price: $185.64
   Stop Loss: $170.79
   Take Profit: $232.05
   Risk/Reward: 2.1
        """, language="")
    
    with col2:
        st.code("""
💰 FINANCIAL PLAN RESULTS:
   Goal Achievement: 94.7%
   Projected Value: $1,147,832
   Plan Sharpe Ratio: 1.42
   Expected Volatility: 11.8%
   Success Probability: 89.2%
   Required Monthly: $1,847
        """, language="")

def display_results(results):
    """Display the analysis results in a beautiful format"""
    
    symbol = results.get('symbol', 'UNKNOWN')
    
    # Header
    st.markdown(f"## 📈 Complete Analysis for **{symbol}**")
    st.markdown("---")
    
    # Key Metrics Row
    display_key_metrics_row(results)
    
    # Main Results Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Market Data", "⚠️ Risk Analysis", "🔮 Forecasts", 
        "🧠 AI Recommendation", "💰 Financial Plan"
    ])
    
    with tab1:
        display_market_data(results)
    
    with tab2:
        display_risk_analysis(results)
    
    with tab3:
        display_forecasts(results)
    
    with tab4:
        display_ai_recommendation(results)
    
    with tab5:
        display_financial_plan(results)
    
    # Charts section
    if PLOTLY_AVAILABLE:
        st.markdown("---")
        display_charts(results)
    
    # Download section
    st.markdown("---")
    display_download_section(results)

def display_key_metrics_row(results):
    """Display key metrics in a row"""
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Extract data safely
    market_data = results.get('market_data')
    if market_data:
        current_price = getattr(market_data, 'current_price', 0)
        return_1d = getattr(market_data, 'return_1d', 0)
        volatility = getattr(market_data, 'volatility_20d', 0)
        trend = getattr(market_data, 'trend', 'neutral')
        rsi = getattr(market_data, 'rsi', 50)
    else:
        current_price = return_1d = volatility = rsi = 0
        trend = 'neutral'
    
    with col1:
        st.metric("💰 Price", f"${current_price:.2f}", f"{return_1d:.2%}")
    
    with col2:
        st.metric("📊 Volatility", f"{volatility:.1%}")
    
    with col3:
        trend_emoji = {
            "strongly_bullish": "🚀", "bullish": "📈", "neutral": "⚖️",
            "bearish": "📉", "strongly_bearish": "💥"
        }.get(trend, "⚖️")
        st.metric("📈 Trend", f"{trend_emoji} {trend.replace('_', ' ').title()}")
    
    with col4:
        st.metric("📊 RSI", f"{rsi:.1f}")
    
    with col5:
        # AI Recommendation
        recommendation = results.get('recommendation')
        if recommendation:
            action = getattr(recommendation, 'action', 'HOLD')
            confidence = getattr(recommendation, 'confidence', 0.5)
            action_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(action, "🟡")
            st.metric("🧠 AI Rec", f"{action_emoji} {action}", f"{confidence:.1%}")
        else:
            st.metric("🧠 AI Rec", "Analyzing...")

def display_market_data(results):
    """Display market data analysis"""
    
    market_data = results.get('market_data')
    if not market_data:
        st.warning("⚠️ No market data available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Price Metrics")
        st.write(f"**Current Price:** ${getattr(market_data, 'current_price', 0):.2f}")
        st.write(f"**1-Day Return:** {getattr(market_data, 'return_1d', 0):.2%}")
        st.write(f"**5-Day Return:** {getattr(market_data, 'return_5d', 0):.2%}")
        st.write(f"**20-Day Return:** {getattr(market_data, 'return_20d', 0):.2%}")
        st.write(f"**Support Level:** ${getattr(market_data, 'support_level', 0):.2f}")
        st.write(f"**Resistance Level:** ${getattr(market_data, 'resistance_level', 0):.2f}")
    
    with col2:
        st.markdown("#### 🎯 Technical Indicators")
        st.write(f"**RSI:** {getattr(market_data, 'rsi', 50):.1f}")
        st.write(f"**Trend:** {getattr(market_data, 'trend', 'neutral').replace('_', ' ').title()}")
        st.write(f"**MACD Signal:** {getattr(market_data, 'macd_signal', 'neutral').title()}")
        st.write(f"**Bollinger Position:** {getattr(market_data, 'bollinger_position', 'middle').replace('_', ' ').title()}")
        st.write(f"**Volume Trend:** {getattr(market_data, 'volume_trend', 'neutral').title()}")
        st.write(f"**20-Day Volatility:** {getattr(market_data, 'volatility_20d', 0):.1%}")

def display_risk_analysis(results):
    """Display risk analysis"""
    
    risk_metrics = results.get('risk_metrics')
    if not risk_metrics:
        st.warning("⚠️ No risk analysis available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⚠️ Risk Metrics")
        st.write(f"**Portfolio Volatility:** {getattr(risk_metrics, 'portfolio_volatility', 0):.1%}")
        st.write(f"**Sharpe Ratio:** {getattr(risk_metrics, 'sharpe_ratio', 0):.2f}")
        st.write(f"**Sortino Ratio:** {getattr(risk_metrics, 'sortino_ratio', 0):.2f}")
        st.write(f"**Calmar Ratio:** {getattr(risk_metrics, 'calmar_ratio', 0):.2f}")
        st.write(f"**Maximum Drawdown:** {getattr(risk_metrics, 'maximum_drawdown', 0):.1%}")
    
    with col2:
        st.markdown("#### 📉 Value at Risk")
        st.write(f"**VaR (5%):** {getattr(risk_metrics, 'value_at_risk_5pct', 0):.1%}")
        st.write(f"**VaR (1%):** {getattr(risk_metrics, 'value_at_risk_1pct', 0):.1%}")
        st.write(f"**Expected Shortfall:** {getattr(risk_metrics, 'expected_shortfall', 0):.1%}")
        st.write(f"**GARCH Volatility:** {getattr(risk_metrics, 'garch_volatility', 0):.1%}")

def display_forecasts(results):
    """Display forecast analysis"""
    
    forecast_data = results.get('forecast_data')
    if not forecast_data:
        st.warning("⚠️ No forecast data available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔮 Price Forecasts")
        st.write(f"**ARIMA Forecast:** ${getattr(forecast_data, 'arima_forecast', 0):.2f}")
        st.write(f"**Prophet Forecast:** ${getattr(forecast_data, 'prophet_forecast', 0):.2f}")
        st.write(f"**LSTM Forecast:** ${getattr(forecast_data, 'lstm_forecast', 0):.2f}")
        st.write(f"**Ensemble Forecast:** ${getattr(forecast_data, 'ensemble_forecast', 0):.2f}")
        
        # Calculate price change
        current_price_data = results.get('market_data')
        if current_price_data and hasattr(current_price_data, 'current_price'):
            current = current_price_data.current_price
            ensemble = getattr(forecast_data, 'ensemble_forecast', current)
            change = ((ensemble - current) / current) * 100
            st.write(f"**Expected Change:** {change:+.1f}%")
    
    with col2:
        st.markdown("#### 📊 Forecast Metrics")
        st.write(f"**Confidence:** {getattr(forecast_data, 'forecast_confidence', 0.5):.1%}")
        st.write(f"**Upside Probability:** {getattr(forecast_data, 'upside_probability', 0.5):.1%}")
        st.write(f"**Downside Risk:** {getattr(forecast_data, 'downside_risk', 0.5):.1%}")
        st.write(f"**Volatility Forecast:** {getattr(forecast_data, 'volatility_forecast', 0.2):.1%}")
        st.write(f"**Accuracy Score:** {getattr(forecast_data, 'forecast_accuracy_score', 0):.1%}")

def display_ai_recommendation(results):
    """Display AI recommendation"""
    
    recommendation = results.get('recommendation')
    if not recommendation:
        st.warning("⚠️ No AI recommendation available")
        return
    
    # Main recommendation box
    action = getattr(recommendation, 'action', 'HOLD')
    confidence = getattr(recommendation, 'confidence', 0.5)
    
    if action == "BUY":
        st.markdown(f"""
        <div class="success-container">
            <h2>🟢 BUY RECOMMENDATION</h2>
            <h3>Confidence: {confidence:.1%}</h3>
        </div>
        """, unsafe_allow_html=True)
    elif action == "SELL":
        st.markdown(f"""
        <div class="warning-container">
            <h2>🔴 SELL RECOMMENDATION</h2>
            <h3>Confidence: {confidence:.1%}</h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="info-container">
            <h2>🟡 HOLD RECOMMENDATION</h2>
            <h3>Confidence: {confidence:.1%}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Position Details")
        st.write(f"**Position Size:** {getattr(recommendation, 'position_size', 0):.1%}")
        st.write(f"**Entry Price:** ${getattr(recommendation, 'entry_price', 0):.2f}")
        st.write(f"**Stop Loss:** ${getattr(recommendation, 'stop_loss', 0):.2f}")
        st.write(f"**Take Profit:** ${getattr(recommendation, 'take_profit', 0):.2f}")
        st.write(f"**Risk Level:** {getattr(recommendation, 'risk_level', 'MEDIUM')}")
        st.write(f"**Time Horizon:** {getattr(recommendation, 'time_horizon', 'MEDIUM')}")
    
    with col2:
        st.markdown("#### 📊 Risk/Reward Analysis")
        st.write(f"**Risk/Reward Ratio:** {getattr(recommendation, 'risk_reward_ratio', 1.0):.2f}")
        st.write(f"**Success Probability:** {getattr(recommendation, 'probability_of_success', 0.5):.1%}")
        st.write(f"**Max Drawdown Est:** {getattr(recommendation, 'maximum_drawdown_estimate', 0.15):.1%}")
    
    # Detailed reasoning
    if hasattr(recommendation, 'detailed_reasoning'):
        st.markdown("#### 🧠 AI Reasoning")
        st.text_area("", getattr(recommendation, 'detailed_reasoning', ''), height=100, disabled=True)

def display_financial_plan(results):
    """Display financial planning results"""
    
    financial_plan = results.get('financial_plan')
    if not financial_plan:
        st.info("💡 Enable financial planning in the sidebar to see personalized recommendations")
        return
    
    # Success probability
    success_prob = getattr(financial_plan, 'success_probability', 0)
    projected_value = getattr(financial_plan, 'projected_value', 0)
    
    if success_prob > 0.8:
        st.markdown(f"""
        <div class="success-container">
            <h2>🎯 EXCELLENT PLAN</h2>
            <h3>Success Probability: {success_prob:.1%}</h3>
            <p>Projected Value: ${projected_value:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    elif success_prob > 0.6:
        st.markdown(f"""
        <div class="info-container">
            <h2>👍 GOOD PLAN</h2>
            <h3>Success Probability: {success_prob:.1%}</h3>
            <p>Projected Value: ${projected_value:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="warning-container">
            <h2>⚠️ NEEDS ADJUSTMENT</h2>
            <h3>Success Probability: {success_prob:.1%}</h3>
            <p>Projected Value: ${projected_value:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Plan metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Plan Metrics")
        st.write(f"**Plan Sharpe Ratio:** {getattr(financial_plan, 'plan_sharpe_ratio', 0):.2f}")
        st.write(f"**Plan Volatility:** {getattr(financial_plan, 'plan_volatility', 0):.1%}")
        st.write(f"**Max Drawdown:** {getattr(financial_plan, 'plan_max_drawdown', 0):.1%}")
        st.write(f"**Required Monthly:** ${getattr(financial_plan, 'required_monthly', 0):,.0f}")
    
    with col2:
        st.markdown("#### 🎯 Asset Allocation")
        allocation = getattr(financial_plan, 'asset_allocation', {})
        for asset, percent in allocation.items():
            st.write(f"**{asset.replace('_', ' ').title()}:** {percent:.1%}")
    
    # Recommendations
    recommendations = getattr(financial_plan, 'recommendations', [])
    if recommendations:
        st.markdown("#### 💡 Key Recommendations")
        for i, rec in enumerate(recommendations[:5], 1):
            st.write(f"{i}. {rec}")

def display_charts(results):
    """Display interactive charts"""
    
    st.markdown("### 📊 Interactive Charts")
    
    # Market data chart
    market_data = results.get('market_data')
    if market_data and hasattr(market_data, 'prices'):
        fig = go.Figure()
        
        prices = market_data.prices
        fig.add_trace(go.Scatter(
            x=prices.index,
            y=prices.values,
            mode='lines',
            name='Price',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Add support/resistance lines
        if hasattr(market_data, 'support_level'):
            fig.add_hline(y=market_data.support_level, 
                         line_dash="dash", line_color="green",
                         annotation_text="Support")
        
        if hasattr(market_data, 'resistance_level'):
            fig.add_hline(y=market_data.resistance_level, 
                         line_dash="dash", line_color="red",
                         annotation_text="Resistance")
        
        fig.update_layout(
            title=f"{results.get('symbol', 'Stock')} Price Chart",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk metrics chart
    risk_metrics = results.get('risk_metrics')
    if risk_metrics and hasattr(risk_metrics, 'rolling_volatility') and not risk_metrics.rolling_volatility.empty:
        fig2 = go.Figure()
        
        rolling_vol = risk_metrics.rolling_volatility
        fig2.add_trace(go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol.values * 100,  # Convert to percentage
            mode='lines',
            name='Rolling Volatility',
            line=dict(color='#ff7f0e', width=2)
        ))
        
        fig2.update_layout(
            title="Rolling 20-Day Volatility",
            xaxis_title="Date",
            yaxis_title="Volatility (%)",
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # Forecast comparison chart
    forecast_data = results.get('forecast_data')
    if forecast_data:
        forecasts = {
            'ARIMA': getattr(forecast_data, 'arima_forecast', 0),
            'Prophet': getattr(forecast_data, 'prophet_forecast', 0),
            'LSTM': getattr(forecast_data, 'lstm_forecast', 0),
            'Ensemble': getattr(forecast_data, 'ensemble_forecast', 0)
        }
        
        fig3 = go.Figure(data=[
            go.Bar(x=list(forecasts.keys()), y=list(forecasts.values()))
        ])
        
        fig3.update_layout(
            title="Price Forecast Comparison",
            xaxis_title="Model",
            yaxis_title="Predicted Price ($)",
            height=400
        )
        
        st.plotly_chart(fig3, use_container_width=True)

def display_download_section(results):
    """Display download options"""
    
    st.markdown("### 📥 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # JSON download
        try:
            # Convert results to JSON-serializable format
            json_data = {}
            for key, value in results.items():
                if hasattr(value, '__dict__'):
                    json_data[key] = {attr: getattr(value, attr) for attr in dir(value) 
                                    if not attr.startswith('_') and not callable(getattr(value, attr))}
                else:
                    json_data[key] = value
            
            json_string = json.dumps(json_data, indent=2, default=str)
            st.download_button(
                label="📄 Download JSON",
                data=json_string,
                file_name=f"{results.get('symbol', 'analysis')}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        except Exception as e:
            st.error(f"JSON export failed: {e}")
    
    with col2:
        # Summary report
        try:
            summary = create_summary_report(results)
            st.download_button(
                label="📋 Download Summary",
                data=summary,
                file_name=f"{results.get('symbol', 'analysis')}_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        except Exception as e:
            st.error(f"Summary export failed: {e}")
    
    with col3:
        # CSV data
        try:
            market_data = results.get('market_data')
            if market_data and hasattr(market_data, 'prices'):
                csv_data = pd.DataFrame({
                    'Date': market_data.prices.index,
                    'Price': market_data.prices.values
                })
                csv_string = csv_data.to_csv(index=False)
                st.download_button(
                    label="📊 Download CSV",
                    data=csv_string,
                    file_name=f"{results.get('symbol', 'data')}_prices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"CSV export failed: {e}")

def create_summary_report(results):
    """Create a text summary report"""
    
    symbol = results.get('symbol', 'UNKNOWN')
    report = f"""
AI FINANCIAL ANALYSIS REPORT
============================
Symbol: {symbol}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MARKET DATA SUMMARY
------------------
"""
    
    market_data = results.get('market_data')
    if market_data:
        report += f"""Current Price: ${getattr(market_data, 'current_price', 0):.2f}
1-Day Return: {getattr(market_data, 'return_1d', 0):.2%}
20-Day Volatility: {getattr(market_data, 'volatility_20d', 0):.1%}
Trend: {getattr(market_data, 'trend', 'neutral').replace('_', ' ').title()}
RSI: {getattr(market_data, 'rsi', 50):.1f}
Support Level: ${getattr(market_data, 'support_level', 0):.2f}
Resistance Level: ${getattr(market_data, 'resistance_level', 0):.2f}

"""
    
    risk_metrics = results.get('risk_metrics')
    if risk_metrics:
        report += f"""RISK ANALYSIS
-------------
Portfolio Volatility: {getattr(risk_metrics, 'portfolio_volatility', 0):.1%}
Sharpe Ratio: {getattr(risk_metrics, 'sharpe_ratio', 0):.2f}
Maximum Drawdown: {getattr(risk_metrics, 'maximum_drawdown', 0):.1%}
VaR (5%): {getattr(risk_metrics, 'value_at_risk_5pct', 0):.1%}

"""
    
    forecast_data = results.get('forecast_data')
    if forecast_data:
        report += f"""FORECASTS
---------
Ensemble Forecast: ${getattr(forecast_data, 'ensemble_forecast', 0):.2f}
Confidence: {getattr(forecast_data, 'forecast_confidence', 0.5):.1%}
Upside Probability: {getattr(forecast_data, 'upside_probability', 0.5):.1%}

"""
    
    recommendation = results.get('recommendation')
    if recommendation:
        report += f"""AI RECOMMENDATION
-----------------
Action: {getattr(recommendation, 'action', 'HOLD')}
Confidence: {getattr(recommendation, 'confidence', 0.5):.1%}
Position Size: {getattr(recommendation, 'position_size', 0):.1%}
Risk Level: {getattr(recommendation, 'risk_level', 'MEDIUM')}
Risk/Reward Ratio: {getattr(recommendation, 'risk_reward_ratio', 1.0):.2f}

"""
    
    financial_plan = results.get('financial_plan')
    if financial_plan:
        report += f"""FINANCIAL PLANNING
------------------
Success Probability: {getattr(financial_plan, 'success_probability', 0):.1%}
Projected Value: ${getattr(financial_plan, 'projected_value', 0):,.0f}
Plan Sharpe Ratio: {getattr(financial_plan, 'plan_sharpe_ratio', 0):.2f}
Required Monthly: ${getattr(financial_plan, 'required_monthly', 0):,.0f}

"""
    
    report += """
DISCLAIMER
----------
This analysis is for informational purposes only and should not be considered as financial advice.
Always consult with a qualified financial advisor before making investment decisions.
Past performance does not guarantee future results.
"""
    
    return report

# Show system status on startup
if PIPELINE_LOADED:
    st.sidebar.success("🚀 AI Pipeline Ready")
    st.sidebar.markdown("**Available Agents:**")
    agents = [
        "📊 Market Data Agent",
        "⚠️ Risk Analysis Agent", 
        "🔮 Forecasting Agent",
        "🌍 Macro Economic Agent",
        "😊 Sentiment Agent",
        "🧠 AI Strategist Agent",
        "💰 Financial Planner Agent"
    ]
    for agent in agents:
        st.sidebar.markdown(f"✅ {agent}")
    
    # Show async status
    if NEST_ASYNCIO_AVAILABLE:
        st.sidebar.success("🔄 Async Support: Available")
    else:
        st.sidebar.warning("🔄 Async Support: Limited")
        
    # Show plotting status
    if PLOTLY_AVAILABLE:
        st.sidebar.success("📊 Interactive Charts: Available")
    else:
        st.sidebar.warning("📊 Interactive Charts: Basic only")
        
else:
    st.sidebar.error("❌ Pipeline Not Loaded")
    st.sidebar.markdown("**Setup Required:**")
    st.sidebar.markdown("1. Install dependencies")
    st.sidebar.markdown("2. Add final_vCM.py file")
    st.sidebar.markdown("3. Restart app")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🤖 Powered by Multi-Agent AI Architecture | Built with Streamlit</p>
    <p>⚠️ For educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
