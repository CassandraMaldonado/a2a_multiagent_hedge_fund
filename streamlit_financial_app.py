"""
AI Financial Forecasting System - Streamlit Dashboard
This module imports all functionality from Final_GENAI_V3 and creates a beautiful web interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
import nest_asyncio
from datetime import datetime, timedelta
import json

# Configure Streamlit page
st.set_page_config(
    page_title="AI Financial Forecasting System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply nest_asyncio for Streamlit compatibility
nest_asyncio.apply()

# Import all classes and functions from your Final_GENAI_V3.py file
try:
    from Final_GENAI_V3 import (
        # Data structures
        MarketData, ForecastData, MacroData, SentimentData, RiskMetrics,
        PersonalizedRecommendation, FinancialGoal, FinancialPlanResult,
        
        # Agents
        MarketDataAgent, RiskAgent, ForecastingAgent, MacroEconomicAgent,
        SentimentAgent, StrategistAgent, FinancialPlannerAgent,
        
        # Pipeline functions
        run_pipeline, run_pipeline_sync, create_dashboard_visualizations,
        print_pipeline_summary
    )
    NOTEBOOK_IMPORTED = True
    print("✅ Successfully imported from Final_GENAI_V3.py")
except ImportError as e:
    st.error(f"⚠️ Could not import from Final_GENAI_V3.py: {str(e)}")
    st.error("Please ensure Final_GENAI_V3.py is in the same directory as this Streamlit app.")
    NOTEBOOK_IMPORTED = False
except Exception as e:
    st.error(f"⚠️ Error importing from Final_GENAI_V3.py: {str(e)}")
    NOTEBOOK_IMPORTED = False

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
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #424242;
        margin-bottom: 1rem;
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
    
    .warning-card {
        background: linear-gradient(135deg, #fff3cd 0%, #fce4a6 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
    }
    
    .danger-card {
        background: linear-gradient(135deg, #f8d7da 0%, #f1b2b7 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #dc3545;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Financial Forecasting System</h1>', unsafe_allow_html=True)
    st.markdown("### Comprehensive Multi-Agent Financial Analysis Platform")
    
    if not NOTEBOOK_IMPORTED:
        st.error("❌ Cannot run analysis without importing the core functions.")
        st.info("Please ensure Final_GENAI_V3.py is in the same directory and contains all required functions.")
        st.code("""
# Required functions in Final_GENAI_V3.py:
- run_pipeline_sync() or run_pipeline()
- All agent classes (MarketDataAgent, RiskAgent, etc.)
- All data classes (MarketData, ForecastData, etc.)
        """)
        st.stop()
    
    # Sidebar Configuration
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1E88E5/white?text=AI+Finance", width=300)
        st.markdown("## 🎛️ Configuration Panel")
        
        # Stock Symbol Input
        symbol = st.text_input(
            "📊 Stock Symbol",
            value="AAPL",
            help="Enter a valid stock ticker symbol (e.g., AAPL, TSLA, MSFT)"
        ).upper()
        
        # OpenAI API Key (Optional)
        openai_key = st.text_input(
            "🔑 OpenAI API Key (Optional)",
            type="password",
            help="Enter your OpenAI API key for enhanced AI recommendations"
        )
        
        # Analysis Type
        analysis_type = st.selectbox(
            "🎯 Analysis Type",
            ["Stock Analysis Only", "Stock Analysis + Financial Planning"],
            help="Choose whether to include financial planning analysis"
        )
        
        # Financial Planning Parameters (if selected)
        if analysis_type == "Stock Analysis + Financial Planning":
            st.markdown("### 💰 Financial Goal Settings")
            
            goal_type = st.selectbox(
                "Goal Type",
                ["retirement", "house", "education", "general"]
            )
            
            target_amount = st.number_input(
                "Target Amount ($)",
                min_value=1000,
                max_value=10000000,
                value=1000000,
                step=10000,
                format="%d"
            )
            
            current_amount = st.number_input(
                "Current Savings ($)",
                min_value=0,
                max_value=5000000,
                value=50000,
                step=1000,
                format="%d"
            )
            
            monthly_contribution = st.number_input(
                "Monthly Contribution ($)",
                min_value=0,
                max_value=50000,
                value=2000,
                step=100,
                format="%d"
            )
            
            time_horizon = st.slider(
                "Time Horizon (Years)",
                min_value=1,
                max_value=50,
                value=25,
                help="How many years until you need this money?"
            )
            
            age = st.slider(
                "Current Age",
                min_value=18,
                max_value=80,
                value=35
            )
            
            annual_income = st.number_input(
                "Annual Income ($)",
                min_value=20000,
                max_value=1000000,
                value=120000,
                step=5000,
                format="%d"
            )
            
            risk_tolerance = st.selectbox(
                "Risk Tolerance",
                ["conservative", "moderate", "aggressive"],
                index=1
            )
        
        # Run Analysis Button
        run_analysis = st.button(
            "🚀 Run AI Analysis",
            help="Click to start the comprehensive AI financial analysis",
            type="primary"
        )
    
    # Main Content Area
    if run_analysis:
        if not symbol:
            st.error("Please enter a valid stock symbol.")
            return
        
        # Create financial goal if planning is selected
        financial_goal = None
        if analysis_type == "Stock Analysis + Financial Planning":
            try:
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
            except Exception as e:
                st.error(f"Error creating financial goal: {e}")
                return
        
        # Run the analysis
        with st.spinner(f"🔍 Analyzing {symbol}... This may take a moment."):
            try:
                # Run pipeline synchronously using asyncio.run()
                results = run_analysis_pipeline(symbol, openai_key, financial_goal)
                
                # Display results
                if results:
                    display_analysis_results(results)
                    
                    # Add download option
                    st.markdown("---")
                    create_download_link(results)
                else:
                    st.error("❌ No results returned from analysis")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.exception(e)
                
                # Show debug info
                with st.expander("🔧 Debug Information"):
                    st.write("Available functions in Final_GENAI_V3:")
                    try:
                        import Final_GENAI_V3
                        functions = [name for name in dir(Final_GENAI_V3) if not name.startswith('_')]
                        st.write(functions)
                    except:
                        st.write("Could not inspect Final_GENAI_V3 module")
    
    else:
        # Welcome screen
        display_welcome_screen()

def run_analysis_pipeline(symbol, openai_key, financial_goal):
    """
    Synchronous wrapper for running the analysis pipeline
    """
    try:
        # Apply nest_asyncio to handle event loop in Streamlit
        nest_asyncio.apply()
        
        # Try different pipeline functions in order of preference
        if 'run_pipeline_sync' in globals():
            # Use async version with asyncio.run
            return asyncio.run(run_pipeline_sync(
                symbol=symbol,
                openai_api_key=openai_key if openai_key else None,
                financial_goal=financial_goal
            ))
        elif 'run_pipeline' in globals():
            # Use regular pipeline function
            return asyncio.run(run_pipeline(
                symbol=symbol,
                openai_api_key=openai_key if openai_key else None,
                financial_goal=financial_goal
            ))
        else:
            # Fallback: run manual pipeline
            return asyncio.run(run_manual_pipeline_sync(symbol, openai_key, financial_goal))
            
    except Exception as e:
        st.error(f"Pipeline execution error: {str(e)}")
        return None

def run_manual_pipeline_sync(symbol, openai_key, financial_goal):
    """
    Synchronous version of manual pipeline execution
    """
    async def _async_manual_pipeline():
        state = {}
        
        try:
            # Initialize agents
            market_agent = MarketDataAgent()
            risk_agent = RiskAgent() 
            forecast_agent = ForecastingAgent()
            macro_agent = MacroEconomicAgent()
            sentiment_agent = SentimentAgent()
            strategist_agent = StrategistAgent(api_key=openai_key)
            
            # Run agents in sequence
            st.write("📊 Fetching market data...")
            state = await market_agent.process(state, symbol=symbol, period="1y")
            
            st.write("⚠️ Analyzing risk metrics...")
            state = await risk_agent.process(state)
            
            st.write("🔮 Generating forecasts...")
            state = await forecast_agent.process(state, forecast_horizon=5)
            
            st.write("🌍 Analyzing macro environment...")
            state = await macro_agent.process(state)
            
            st.write("💭 Analyzing sentiment...")
            state = await sentiment_agent.process(state)
            
            st.write("🧠 Generating AI recommendation...")
            state = await strategist_agent.process(state)
            
            # Add financial planning if goal provided
            if financial_goal:
                st.write("💰 Creating financial plan...")
                planner_agent = FinancialPlannerAgent()
                state = await planner_agent.process(state, financial_goal)
            
            return state
            
        except Exception as e:
            st.error(f"Manual pipeline failed: {str(e)}")
            return {}
    
    return asyncio.run(_async_manual_pipeline())

async def run_manual_pipeline(symbol, openai_key, financial_goal):
async def run_manual_pipeline(symbol, openai_key, financial_goal):
    """
    Manual pipeline execution as fallback if main pipeline functions aren't available
    """
    state = {}
    
    try:
        # Initialize agents
        market_agent = MarketDataAgent()
        risk_agent = RiskAgent() 
        forecast_agent = ForecastingAgent()
        macro_agent = MacroEconomicAgent()
        sentiment_agent = SentimentAgent()
        strategist_agent = StrategistAgent(api_key=openai_key)
        
        # Run agents in sequence
        st.write("📊 Fetching market data...")
        state = await market_agent.process(state, symbol=symbol, period="1y")
        
        st.write("⚠️ Analyzing risk metrics...")
        state = await risk_agent.process(state)
        
        st.write("🔮 Generating forecasts...")
        state = await forecast_agent.process(state, forecast_horizon=5)
        
        st.write("🌍 Analyzing macro environment...")
        state = await macro_agent.process(state)
        
        st.write("💭 Analyzing sentiment...")
        state = await sentiment_agent.process(state)
        
        st.write("🧠 Generating AI recommendation...")
        state = await strategist_agent.process(state)
        
        # Add financial planning if goal provided
        if financial_goal:
            st.write("💰 Creating financial plan...")
            planner_agent = FinancialPlannerAgent()
            state = await planner_agent.process(state, financial_goal)
        
        return state
        
    except Exception as e:
        st.error(f"Manual pipeline failed: {str(e)}")
        return {}

def check_data_availability(results):
    """Check what data is available in results"""
    available_data = []
    
    if results.get('market_data'):
        available_data.append("📊 Market Data")
    if results.get('risk_metrics'):
        available_data.append("⚠️ Risk Analysis")
    if results.get('forecast_data'):
        available_data.append("🔮 Forecasting")
    if results.get('macro_data'):
        available_data.append("🌍 Macro Economics")
    if results.get('sentiment_data'):
        available_data.append("💭 Sentiment Analysis")
    if results.get('recommendation'):
        available_data.append("🧠 AI Recommendation")
    if results.get('financial_plan'):
        available_data.append("💰 Financial Planning")
    
    return available_data

def display_welcome_screen():
    """Display welcome screen with instructions"""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h2>🎯 Welcome to AI Financial Forecasting</h2>
            <p style="font-size: 1.2rem; color: #666;">
                Get comprehensive AI-powered financial analysis using our multi-agent system
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature highlights
    st.markdown("### 🌟 Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Market Analysis</h4>
            <ul>
                <li>Real-time market data</li>
                <li>Technical indicators</li>
                <li>Price forecasting</li>
                <li>Risk metrics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>🤖 AI Insights</h4>
            <ul>
                <li>Multi-agent analysis</li>
                <li>Sentiment analysis</li>
                <li>Macro economic factors</li>
                <li>Smart recommendations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>💰 Financial Planning</h4>
            <ul>
                <li>Goal-based planning</li>
                <li>Monte Carlo simulations</li>
                <li>Risk-adjusted returns</li>
                <li>Asset allocation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Instructions
    st.markdown("### 🚀 Getting Started")
    st.markdown("""
    1. **Enter a stock symbol** in the sidebar (e.g., AAPL, TSLA, MSFT)
    2. **Choose analysis type** - stock only or include financial planning
    3. **Configure parameters** based on your needs
    4. **Click 'Run AI Analysis'** to start the comprehensive analysis
    5. **Review results** in our interactive dashboard
    """)

def display_analysis_results(results):
    """Display comprehensive analysis results"""
    
    if not results:
        st.error("❌ No results to display.")
        return
    
    # Check what data is available
    available_data = check_data_availability(results)
    
    if not available_data:
        st.warning("⚠️ No analysis data was generated. Please check your Final_GENAI_V3.py file.")
        return
    
    symbol = results.get('symbol', 'UNKNOWN')
    
    # Header with symbol and available data
    st.markdown(f"## 📈 Analysis Results for {symbol}")
    
    # Show what data is available
    st.success(f"✅ Available Analysis: {' • '.join(available_data)}")
    
    # Create tabs for different sections (only for available data)
    tab_names = []
    tab_keys = []
    
    if results.get('market_data') or results.get('risk_metrics') or results.get('forecast_data'):
        tab_names.append("📊 Overview")
        tab_keys.append("overview")
    
    if results.get('market_data'):
        tab_names.append("💰 Market Data")
        tab_keys.append("market")
    
    if results.get('risk_metrics'):
        tab_names.append("⚠️ Risk Analysis")
        tab_keys.append("risk")
    
    if results.get('forecast_data'):
        tab_names.append("🔮 Forecasting")
        tab_keys.append("forecast")
    
    if results.get('recommendation'):
        tab_names.append("🧠 AI Recommendation")
        tab_keys.append("recommendation")
    
    if results.get('financial_plan'):
        tab_names.append("💡 Financial Planning")
        tab_keys.append("planning")
    
    if results.get('macro_data') or results.get('sentiment_data'):
        tab_names.append("🌍 Macro & Sentiment")
        tab_keys.append("macro_sentiment")
    
    if not tab_names:
        st.error("No valid data found for display.")
        return
    
    tabs = st.tabs(tab_names)
    
    # Display each tab
    for i, (tab_name, tab_key) in enumerate(zip(tab_names, tab_keys)):
        with tabs[i]:
            if tab_key == "overview":
                display_overview(results)
            elif tab_key == "market":
                display_market_data(results)
            elif tab_key == "risk":
                display_risk_analysis(results)
            elif tab_key == "forecast":
                display_forecasting(results)
            elif tab_key == "recommendation":
                display_ai_recommendation(results)
            elif tab_key == "planning":
                display_financial_planning(results)
            elif tab_key == "macro_sentiment":
                display_macro_sentiment(results)

def display_overview(results):
    """Display overview dashboard"""
    
    symbol = results.get('symbol', 'UNKNOWN')
    market_data = results.get('market_data')
    risk_metrics = results.get('risk_metrics')
    forecast_data = results.get('forecast_data')
    recommendation = results.get('recommendation')
    
    # Key metrics at the top
    if market_data:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current Price",
                f"${market_data.current_price:.2f}",
                f"{market_data.return_1d:.2%}"
            )
        
        with col2:
            trend_color = "🟢" if market_data.trend in ["bullish", "strongly_bullish"] else "🔴" if market_data.trend in ["bearish", "strongly_bearish"] else "🟡"
            st.metric("Trend", f"{trend_color} {market_data.trend.replace('_', ' ').title()}")
        
        with col3:
            rsi_color = "🔴" if market_data.rsi > 70 else "🟢" if market_data.rsi < 30 else "🟡"
            st.metric("RSI", f"{rsi_color} {market_data.rsi:.1f}")
        
        with col4:
            if risk_metrics:
                st.metric("Sharpe Ratio", f"{risk_metrics.sharpe_ratio:.2f}")
    
    # Create comprehensive chart
    if market_data and hasattr(market_data, 'prices') and len(market_data.prices) > 0:
        fig = create_overview_chart(results)
        st.plotly_chart(fig, use_container_width=True)
    
    # Quick insights
    st.markdown("### 🎯 Quick Insights")
    
    insights = []
    
    if market_data:
        if market_data.rsi > 70:
            insights.append("⚠️ **Overbought**: RSI indicates potential selling pressure")
        elif market_data.rsi < 30:
            insights.append("✅ **Oversold**: RSI suggests potential buying opportunity")
        
        if market_data.trend in ["strongly_bullish", "bullish"]:
            insights.append("📈 **Bullish Trend**: Strong upward momentum detected")
        elif market_data.trend in ["strongly_bearish", "bearish"]:
            insights.append("📉 **Bearish Trend**: Downward pressure observed")
    
    if forecast_data:
        price_change = ((forecast_data.ensemble_forecast - market_data.current_price) / market_data.current_price) * 100
        if price_change > 5:
            insights.append(f"🚀 **Strong Upside**: Forecast shows {price_change:.1f}% potential gain")
        elif price_change < -5:
            insights.append(f"⚠️ **Downside Risk**: Forecast shows {price_change:.1f}% potential decline")
    
    if recommendation:
        confidence_emoji = "🎯" if recommendation.confidence > 0.8 else "⚖️" if recommendation.confidence > 0.6 else "❓"
        insights.append(f"{confidence_emoji} **AI Recommendation**: {recommendation.action} with {recommendation.confidence:.0%} confidence")
    
    for insight in insights:
        st.markdown(f"- {insight}")

def create_overview_chart(results):
    """Create comprehensive overview chart"""
    
    market_data = results.get('market_data')
    forecast_data = results.get('forecast_data')
    
    if not market_data or not hasattr(market_data, 'prices'):
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Price Chart', 'Volume', 'RSI', 'Forecast Comparison'],
        specs=[[{"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.1
    )
    
    prices = market_data.prices
    
    # Price chart with support/resistance
    fig.add_trace(
        go.Scatter(
            x=prices.index,
            y=prices.values,
            name="Price",
            line=dict(color="#1f77b4", width=2)
        ),
        row=1, col=1
    )
    
    # Support and resistance lines
    fig.add_hline(
        y=market_data.support_level,
        line_dash="dash",
        line_color="green",
        annotation_text="Support",
        row=1, col=1
    )
    
    fig.add_hline(
        y=market_data.resistance_level,
        line_dash="dash",
        line_color="red",
        annotation_text="Resistance",
        row=1, col=1
    )
    
    # Volume chart
    if hasattr(market_data, 'volume') and len(market_data.volume) > 0:
        fig.add_trace(
            go.Bar(
                x=market_data.volume.index,
                y=market_data.volume.values,
                name="Volume",
                marker_color="rgba(31, 119, 180, 0.6)"
            ),
            row=1, col=2
        )
    
    # RSI
    if len(prices) >= 14:
        rsi_values = [market_data.rsi] * len(prices.tail(30))
        fig.add_trace(
            go.Scatter(
                x=prices.tail(30).index,
                y=rsi_values,
                name="RSI",
                line=dict(color="purple")
            ),
            row=2, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # Forecast comparison
    if forecast_data:
        forecasts = ['ARIMA', 'Prophet', 'LSTM', 'Ensemble']
        values = [
            forecast_data.arima_forecast,
            forecast_data.prophet_forecast,
            forecast_data.lstm_forecast,
            forecast_data.ensemble_forecast
        ]
        
        fig.add_trace(
            go.Bar(
                x=forecasts,
                y=values,
                name="Forecasts",
                marker_color=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        height=600,
        title_text=f"Comprehensive Analysis Dashboard",
        showlegend=True
    )
    
    return fig

def display_market_data(results):
    """Display detailed market data analysis"""
    
    market_data = results.get('market_data')
    
    if not market_data:
        st.warning("No market data available.")
        return
    
    # Current metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Price Metrics</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("Current Price", f"${market_data.current_price:.2f}")
        st.metric("Support Level", f"${market_data.support_level:.2f}")
        st.metric("Resistance Level", f"${market_data.resistance_level:.2f}")
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>📈 Returns</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("1-Day Return", f"{market_data.return_1d:.2%}")
        st.metric("5-Day Return", f"{market_data.return_5d:.2%}")
        st.metric("20-Day Return", f"{market_data.return_20d:.2%}")
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Technical Indicators</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("RSI", f"{market_data.rsi:.1f}")
        st.metric("Volatility (20d)", f"{market_data.volatility_20d:.1%}")
        st.metric("Trend", market_data.trend.replace('_', ' ').title())
    
    # Technical analysis details
    st.markdown("### 🔍 Technical Analysis Details")
    
    tech_details = {
        "MACD Signal": market_data.macd_signal,
        "Bollinger Position": market_data.bollinger_position,
        "Volume Trend": market_data.volume_trend
    }
    
    for indicator, value in tech_details.items():
        col1, col2 = st.columns([1, 3])
        with col1:
            st.write(f"**{indicator}:**")
        with col2:
            color = "success" if value in ["bullish", "increasing", "above_upper"] else "warning" if value == "neutral" else "danger"
            st.success(value.replace('_', ' ').title()) if color == "success" else st.warning(value.replace('_', ' ').title()) if color == "warning" else st.error(value.replace('_', ' ').title())

def display_risk_analysis(results):
    """Display comprehensive risk analysis"""
    
    risk_metrics = results.get('risk_metrics')
    
    if not risk_metrics:
        st.warning("No risk metrics available.")
        return
    
    # Risk overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Portfolio Volatility", f"{risk_metrics.portfolio_volatility:.1%}")
    
    with col2:
        sharpe_color = "🟢" if risk_metrics.sharpe_ratio > 1 else "🟡" if risk_metrics.sharpe_ratio > 0 else "🔴"
        st.metric("Sharpe Ratio", f"{sharpe_color} {risk_metrics.sharpe_ratio:.2f}")
    
    with col3:
        st.metric("Max Drawdown", f"{risk_metrics.maximum_drawdown:.1%}")
    
    with col4:
        st.metric("VaR (5%)", f"{risk_metrics.value_at_risk_5pct:.1%}")
    
    # Risk details
    st.markdown("### ⚠️ Detailed Risk Metrics")
    
    risk_data = {
        "Expected Shortfall": f"{risk_metrics.expected_shortfall:.1%}",
        "Sortino Ratio": f"{risk_metrics.sortino_ratio:.2f}",
        "Calmar Ratio": f"{risk_metrics.calmar_ratio:.2f}",
        "GARCH Volatility": f"{risk_metrics.garch_volatility:.1%}"
    }
    
    for metric, value in risk_data.items():
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write(f"**{metric}:**")
        with col2:
            st.write(value)
    
    # Risk interpretation
    st.markdown("### 📋 Risk Assessment")
    
    risk_level = "Low"
    risk_color = "success"
    
    if risk_metrics.portfolio_volatility > 0.4:
        risk_level = "High"
        risk_color = "danger"
    elif risk_metrics.portfolio_volatility > 0.25:
        risk_level = "Medium"
        risk_color = "warning"
    
    if risk_color == "success":
        st.success(f"✅ **{risk_level} Risk**: Portfolio shows manageable risk levels with volatility of {risk_metrics.portfolio_volatility:.1%}")
    elif risk_color == "warning":
        st.warning(f"⚠️ **{risk_level} Risk**: Moderate volatility detected. Consider risk management strategies.")
    else:
        st.error(f"🚨 **{risk_level} Risk**: High volatility indicates significant risk. Careful position sizing recommended.")

def display_forecasting(results):
    """Display forecasting analysis"""
    
    forecast_data = results.get('forecast_data')
    market_data = results.get('market_data')
    
    if not forecast_data:
        st.warning("No forecast data available.")
        return
    
    current_price = market_data.current_price if market_data else 100
    
    # Forecast overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        price_change = ((forecast_data.ensemble_forecast - current_price) / current_price) * 100
        st.metric(
            "Ensemble Forecast",
            f"${forecast_data.ensemble_forecast:.2f}",
            f"{price_change:+.1f}%"
        )
    
    with col2:
        confidence_color = "🟢" if forecast_data.forecast_confidence > 0.8 else "🟡" if forecast_data.forecast_confidence > 0.6 else "🔴"
        st.metric("Confidence", f"{confidence_color} {forecast_data.forecast_confidence:.1%}")
    
    with col3:
        st.metric("Upside Probability", f"{forecast_data.upside_probability:.1%}")
    
    # Individual forecasts comparison
    st.markdown("### 🔮 Model Comparison")
    
    forecast_comparison = {
        "ARIMA": forecast_data.arima_forecast,
        "Prophet": forecast_data.prophet_forecast,
        "LSTM": forecast_data.lstm_forecast,
        "Ensemble": forecast_data.ensemble_forecast
    }
    
    fig = go.Figure()
    
    models = list(forecast_comparison.keys())
    values = list(forecast_comparison.values())
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    fig.add_trace(go.Bar(
        x=models,
        y=values,
        marker_color=colors,
        text=[f"${v:.2f}" for v in values],
        textposition='auto'
    ))
    
    fig.add_hline(
        y=current_price,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Current: ${current_price:.2f}"
    )
    
    fig.update_layout(
        title="Forecast Model Comparison",
        xaxis_title="Model",
        yaxis_title="Price ($)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast details
    st.markdown("### 📊 Forecast Analysis")
    
    forecast_details = {
        "Volatility Forecast": f"{forecast_data.volatility_forecast:.1%}",
        "Downside Risk": f"{forecast_data.downside_risk:.1%}",
        "Prediction Interval": f"${forecast_data.prediction_interval[0]:.2f} - ${forecast_data.prediction_interval[1]:.2f}",
        "Forecast Horizon": f"{forecast_data.forecast_horizon_days} days"
    }
    
    for detail, value in forecast_details.items():
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write(f"**{detail}:**")
        with col2:
            st.write(value)

def display_ai_recommendation(results):
    """Display AI recommendation"""
    
    recommendation = results.get('recommendation')
    
    if not recommendation:
        st.warning("No AI recommendation available.")
        return
    
    # Main recommendation
    action_color = {
        "BUY": "success",
        "SELL": "danger", 
        "HOLD": "warning"
    }
    
    action_emoji = {
        "BUY": "🚀",
        "SELL": "📉",
        "HOLD": "⚖️"
    }
    
    color = action_color.get(recommendation.action, "info")
    emoji = action_emoji.get(recommendation.action, "📊")
    
    if color == "success":
        st.success(f"{emoji} **{recommendation.action}** Recommendation with {recommendation.confidence:.0%} confidence")
    elif color == "danger":
        st.error(f"{emoji} **{recommendation.action}** Recommendation with {recommendation.confidence:.0%} confidence")
    else:
        st.warning(f"{emoji} **{recommendation.action}** Recommendation with {recommendation.confidence:.0%} confidence")
    
    # Recommendation details
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Position Details</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("Position Size", f"{recommendation.position_size:.1%}")
        st.metric("Risk Level", recommendation.risk_level)
        st.metric("Time Horizon", recommendation.time_horizon)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>💰 Price Targets</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("Entry Price", f"${recommendation.entry_price:.2f}")
        st.metric("Stop Loss", f"${recommendation.stop_loss:.2f}")
        st.metric("Take Profit", f"${recommendation.take_profit:.2f}")
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>📈 Risk Metrics</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("Risk/Reward Ratio", f"{recommendation.risk_reward_ratio:.2f}")
        st.metric("Success Probability", f"{recommendation.probability_of_success:.1%}")
        st.metric("Max Drawdown Est.", f"{recommendation.maximum_drawdown_estimate:.1%}")
    
    # Detailed reasoning
    if hasattr(recommendation, 'detailed_reasoning') and recommendation.detailed_reasoning:
        st.markdown("### 🧠 AI Reasoning")
        st.info(recommendation.detailed_reasoning)
    
    # Risk and opportunity factors
    col1, col2 = st.columns(2)
    
    with col1:
        if hasattr(recommendation, 'key_risk_factors') and recommendation.key_risk_factors:
            st.markdown("### ⚠️ Key Risk Factors")
            for risk in recommendation.key_risk_factors:
                st.markdown(f"- 🔴 {risk}")
    
    with col2:
        if hasattr(recommendation, 'key_opportunity_factors') and recommendation.key_opportunity_factors:
            st.markdown("### ✅ Key Opportunities")
            for opportunity in recommendation.key_opportunity_factors:
                st.markdown(f"- 🟢 {opportunity}")
    
    # Alternative scenarios
    if hasattr(recommendation, 'alternative_scenarios') and recommendation.alternative_scenarios:
        st.markdown("### 🎭 Alternative Scenarios")
        
        for scenario, description in recommendation.alternative_scenarios.items():
            scenario_name = scenario.replace('_', ' ').title()
            if 'bull' in scenario.lower():
                st.success(f"**{scenario_name}**: {description}")
            elif 'bear' in scenario.lower():
                st.error(f"**{scenario_name}**: {description}")
            else:
                st.info(f"**{scenario_name}**: {description}")

def display_financial_planning(results):
    """Display financial planning analysis"""
    
    financial_plan = results.get('financial_plan')
    
    if not financial_plan:
        st.info("💡 No financial planning data available. Select 'Stock Analysis + Financial Planning' to see comprehensive financial planning analysis.")
        return
    
    # Goal achievement overview
    success_rate = financial_plan.success_probability
    
    if success_rate >= 0.9:
        st.success(f"🎉 **Excellent!** {success_rate:.0%} probability of achieving your financial goal")
    elif success_rate >= 0.7:
        st.warning(f"⚖️ **Good** {success_rate:.0%} probability of achieving your goal with current plan")
    else:
        st.error(f"⚠️ **Challenging** Only {success_rate:.0%} probability with current parameters")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Target Amount",
            f"${financial_plan.goal.target_amount:,.0f}"
        )
    
    with col2:
        shortfall = financial_plan.goal.target_amount - financial_plan.projected_value
        st.metric(
            "Projected Value",
            f"${financial_plan.projected_value:,.0f}",
            f"{shortfall:+,.0f}" if shortfall != 0 else "✅ Goal Met"
        )
    
    with col3:
        st.metric(
            "Plan Sharpe Ratio",
            f"{financial_plan.plan_sharpe_ratio:.2f}"
        )
    
    with col4:
        st.metric(
            "Expected Volatility",
            f"{financial_plan.plan_volatility:.1%}"
        )
    
    # Asset allocation chart
    st.markdown("### 📊 Recommended Asset Allocation")
    
    allocation = financial_plan.asset_allocation
    labels = [asset.replace('_', ' ').title() for asset in allocation.keys()]
    values = list(allocation.values())
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        textinfo='label+percent',
        textposition='auto',
        marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    )])
    
    fig.update_layout(
        title="Portfolio Asset Allocation",
        height=400,
        annotations=[dict(text=f'{financial_plan.goal.risk_tolerance.title()}<br>Risk Profile', 
                         x=0.5, y=0.5, font_size=16, showarrow=False)]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly breakdown
    st.markdown("### 💳 Monthly Investment Breakdown")
    
    monthly_breakdown = financial_plan.monthly_breakdown
    total_monthly = monthly_breakdown.get('total_monthly', 0)
    
    breakdown_data = []
    for key, value in monthly_breakdown.items():
        if key.endswith('_monthly') and key != 'total_monthly':
            asset_name = key.replace('_monthly', '').replace('_', ' ').title()
            breakdown_data.append({
                'Asset Class': asset_name,
                'Monthly Amount': f"${value:.0f}",
                'Percentage': f"{(value/total_monthly)*100:.1f}%" if total_monthly > 0 else "0%"
            })
    
    if breakdown_data:
        df_breakdown = pd.DataFrame(breakdown_data)
        st.dataframe(df_breakdown, use_container_width=True)
    
    # Monte Carlo results
    if hasattr(financial_plan, 'monte_carlo_results') and financial_plan.monte_carlo_results:
        st.markdown("### 🎲 Monte Carlo Simulation Results")
        
        mc_results = financial_plan.monte_carlo_results
        
        # Percentile chart
        percentiles = ['5th', '25th', '50th', '75th', '95th']
        values = [
            mc_results.get('percentile_5', 0),
            mc_results.get('percentile_25', 0),
            mc_results.get('percentile_50', 0),
            mc_results.get('percentile_75', 0),
            mc_results.get('percentile_95', 0)
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=percentiles,
            y=values,
            mode='lines+markers',
            name='Portfolio Value Scenarios',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))
        
        # Add target line
        fig.add_hline(
            y=financial_plan.goal.target_amount,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Target: ${financial_plan.goal.target_amount:,.0f}"
        )
        
        fig.update_layout(
            title="Monte Carlo Simulation - Portfolio Value Percentiles",
            xaxis_title="Percentile",
            yaxis_title="Portfolio Value ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Monte Carlo metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Success Rate", f"{mc_results.get('success_rate', 0):.1%}")
        
        with col2:
            st.metric("Average Outcome", f"${mc_results.get('mean', 0):,.0f}")
        
        with col3:
            st.metric("Worst Case (5%)", f"${mc_results.get('percentile_5', 0):,.0f}")
    
    # Tax optimization
    if hasattr(financial_plan, 'tax_optimization') and financial_plan.tax_optimization:
        st.markdown("### 💰 Tax Optimization Strategy")
        
        tax_opt = financial_plan.tax_optimization
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Annual Contributions:**")
            st.write(f"• 401(k): ${tax_opt.get('401k_annual', 0):,.0f}")
            st.write(f"• IRA: ${tax_opt.get('ira_annual', 0):,.0f}")
            st.write(f"• Taxable: ${tax_opt.get('taxable_annual', 0):,.0f}")
        
        with col2:
            st.markdown("**Tax Benefits:**")
            st.write(f"• Annual Tax Savings: ${tax_opt.get('tax_savings', 0):,.0f}")
            st.write(f"• Marginal Tax Rate: {tax_opt.get('marginal_rate', 0):.1%}")
    
    # Recommendations
    st.markdown("### 💡 Personalized Recommendations")
    
    if financial_plan.recommendations:
        for i, rec in enumerate(financial_plan.recommendations, 1):
            # Parse emoji and content
            if rec.startswith('✅'):
                st.success(f"{i}. {rec}")
            elif rec.startswith('⚠️') or rec.startswith('📊'):
                st.warning(f"{i}. {rec}")
            elif rec.startswith('🎯') or rec.startswith('📈'):
                st.info(f"{i}. {rec}")
            else:
                st.markdown(f"{i}. {rec}")

def display_macro_sentiment(results):
    """Display macro economic and sentiment analysis"""
    
    macro_data = results.get('macro_data')
    sentiment_data = results.get('sentiment_data')
    
    if macro_data:
        st.markdown("### 🌍 Macro Economic Environment")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("GDP Growth", f"{macro_data.gdp_growth:.1f}%")
            st.metric("Inflation Rate", f"{macro_data.inflation_rate:.1f}%")
        
        with col2:
            st.metric("Fed Funds Rate", f"{macro_data.federal_funds_rate:.2f}%")
            st.metric("VIX", f"{macro_data.vix:.1f}")
        
        with col3:
            sentiment_color = "🟢" if macro_data.market_sentiment == "bullish" else "🔴" if macro_data.market_sentiment == "bearish" else "🟡"
            st.metric("Market Sentiment", f"{sentiment_color} {macro_data.market_sentiment.title()}")
            st.metric("Yield Curve", f"{macro_data.yield_curve_slope:.2f}")
    
    if sentiment_data:
        st.markdown("### 💭 Market Sentiment Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            overall_color = "🟢" if sentiment_data.overall_sentiment > 0.1 else "🔴" if sentiment_data.overall_sentiment < -0.1 else "🟡"
            st.metric("Overall Sentiment", f"{overall_color} {sentiment_data.sentiment_trend.title()}")
        
        with col2:
            st.metric("Fear & Greed Index", f"{sentiment_data.fear_greed_index:.0f}")
        
        with col3:
            st.metric("Analyst Trend", sentiment_data.analyst_rating_trend.title())
        
        # Key topics
        if sentiment_data.key_topics:
            st.markdown("**Key Discussion Topics:**")
            topics_str = " • ".join([topic.replace('_', ' ').title() for topic in sentiment_data.key_topics])
            st.info(topics_str)

# Helper function to create download link for results
def create_download_link(results):
    """Create download link for analysis results"""
    
    # Convert results to JSON-serializable format
    download_data = {}
    
    for key, value in results.items():
        if hasattr(value, '__dict__'):
            # Convert dataclass to dict, excluding pandas objects
            download_data[key] = {
                'type': value.__class__.__name__,
                'data': {k: v for k, v in value.__dict__.items() 
                        if not isinstance(v, (pd.Series, pd.DataFrame))}
            }
        else:
            download_data[key] = value
    
    # Create JSON string
    json_string = json.dumps(download_data, indent=2, default=str)
    
    # Create download button
    st.download_button(
        label="📥 Download Analysis Results",
        data=json_string,
        file_name=f"financial_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

if __name__ == "__main__":
    main()
