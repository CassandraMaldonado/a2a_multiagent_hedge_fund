"""
AI FINANCIAL FORECASTING SYSTEM - STREAMLIT APP
Complete web interface for the multi-agent financial analysis pipeline
"""

import streamlit as st
import asyncio
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Import your existing classes and functions from Final_GENAI_V3.py
# Note: In production, you would import these from your module
# from Final_GENAI_V3 import (
#     MarketDataAgent, RiskAgent, ForecastingAgent, MacroEconomicAgent,
#     SentimentAgent, StrategistAgent, FinancialPlannerAgent,
#     FinancialGoal, run_pipeline
# )

# Page configuration
st.set_page_config(
    page_title="AI Financial Forecasting System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .recommendation-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .risk-warning {
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
    }
    .success-box {
        background-color: #d4edda;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False
if 'selected_symbol' not in st.session_state:
    st.session_state.selected_symbol = "AAPL"

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🤖 AI Financial Forecasting System")
    st.markdown("### Multi-Agent Investment Analysis & Planning Platform")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Stock Selection
        st.subheader("📊 Stock Analysis")
        
        # Popular stocks for quick selection
        popular_stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "JNJ"]
        
        col1, col2 = st.columns(2)
        with col1:
            quick_select = st.selectbox(
                "Quick Select",
                ["Custom"] + popular_stocks,
                help="Select from popular stocks or choose Custom to enter your own"
            )
        
        with col2:
            if quick_select == "Custom":
                symbol = st.text_input("Enter Symbol", value="AAPL", max_chars=10).upper()
            else:
                symbol = quick_select
                st.text_input("Selected Symbol", value=symbol, disabled=True)
        
        st.session_state.selected_symbol = symbol
        
        # Analysis Period
        period = st.selectbox(
            "Analysis Period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3,
            help="Historical data period for analysis"
        )
        
        # OpenAI Configuration (Optional)
        st.subheader("🧠 AI Enhancement")
        use_gpt = st.checkbox("Enable GPT-4 Analysis", value=False)
        
        if use_gpt:
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Optional: Provide API key for enhanced AI recommendations"
            )
        else:
            api_key = None
        
        # Financial Planning
        st.subheader("💰 Financial Planning")
        enable_planning = st.checkbox("Enable Financial Planning", value=False)
        
        if enable_planning:
            with st.expander("Goal Configuration", expanded=True):
                goal_type = st.selectbox(
                    "Goal Type",
                    ["retirement", "house", "education", "general"],
                    help="Type of financial goal"
                )
                
                target_amount = st.number_input(
                    "Target Amount ($)",
                    min_value=10000,
                    max_value=10000000,
                    value=1000000,
                    step=50000,
                    help="Total amount needed for your goal"
                )
                
                current_amount = st.number_input(
                    "Current Savings ($)",
                    min_value=0,
                    max_value=10000000,
                    value=50000,
                    step=5000,
                    help="Amount already saved"
                )
                
                monthly_contribution = st.number_input(
                    "Monthly Contribution ($)",
                    min_value=0,
                    max_value=50000,
                    value=2000,
                    step=100,
                    help="Amount you can save monthly"
                )
                
                time_horizon = st.slider(
                    "Time Horizon (Years)",
                    min_value=1,
                    max_value=40,
                    value=25,
                    help="Years until you need the money"
                )
                
                risk_tolerance = st.select_slider(
                    "Risk Tolerance",
                    options=["conservative", "moderate", "aggressive"],
                    value="moderate",
                    help="Your comfort level with investment risk"
                )
                
                age = st.number_input(
                    "Your Age",
                    min_value=18,
                    max_value=100,
                    value=35,
                    help="Your current age"
                )
                
                annual_income = st.number_input(
                    "Annual Income ($)",
                    min_value=0,
                    max_value=1000000,
                    value=120000,
                    step=5000,
                    help="Your annual income for tax planning"
                )
        
        # Run Analysis Button
        st.markdown("---")
        run_analysis = st.button(
            "🚀 Run Analysis",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.analysis_running
        )
    
    # Main Content Area
    if run_analysis:
        run_analysis_pipeline(
            symbol=symbol,
            period=period,
            api_key=api_key,
            enable_planning=enable_planning,
            goal_params={
                'goal_type': goal_type if enable_planning else None,
                'target_amount': target_amount if enable_planning else None,
                'current_amount': current_amount if enable_planning else None,
                'monthly_contribution': monthly_contribution if enable_planning else None,
                'time_horizon_years': time_horizon if enable_planning else None,
                'risk_tolerance': risk_tolerance if enable_planning else None,
                'age': age if enable_planning else None,
                'annual_income': annual_income if enable_planning else None
            } if enable_planning else None
        )
    
    # Display Results
    if st.session_state.analysis_results:
        display_results(st.session_state.analysis_results)
    else:
        display_welcome_screen()

def display_welcome_screen():
    """Display welcome screen with instructions"""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <h2>Welcome to AI Financial Forecasting System</h2>
            <p style='font-size: 18px; color: #666;'>
                Configure your analysis parameters in the sidebar and click "Run Analysis" to begin.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature highlights
        st.markdown("### 🎯 Key Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **📊 Market Analysis**
            - Real-time stock data
            - Technical indicators
            - Price forecasting
            """)
            
            st.success("""
            **🛡️ Risk Assessment**
            - Portfolio volatility
            - Sharpe ratio calculation
            - GARCH modeling
            """)
        
        with col2:
            st.warning("""
            **🌍 Macro Analysis**
            - Economic indicators
            - Market sentiment
            - News analysis
            """)
            
            st.error("""
            **💰 Financial Planning**
            - Goal-based planning
            - Monte Carlo simulation
            - Tax optimization
            """)

def run_analysis_pipeline(symbol, period, api_key, enable_planning, goal_params):
    """Run the complete analysis pipeline"""
    
    st.session_state.analysis_running = True
    
    # Create placeholder for progress
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    try:
        with st.spinner(f"Analyzing {symbol}..."):
            # Progress tracking
            progress_bar = progress_placeholder.progress(0)
            
            # Phase 1: Initialize
            status_placeholder.info("🔧 Initializing agents...")
            progress_bar.progress(10)
            
            # Import and create goal if planning enabled
            from Final_GENAI_V3 import FinancialGoal
            
            financial_goal = None
            if enable_planning and goal_params:
                financial_goal = FinancialGoal(
                    target_amount=goal_params['target_amount'],
                    current_amount=goal_params['current_amount'],
                    monthly_contribution=goal_params['monthly_contribution'],
                    time_horizon_years=goal_params['time_horizon_years'],
                    risk_tolerance=goal_params['risk_tolerance'],
                    age=goal_params['age'],
                    annual_income=goal_params['annual_income'],
                    goal_type=goal_params['goal_type']
                )
            
            # Phase 2: Run pipeline
            status_placeholder.info("📊 Collecting market data...")
            progress_bar.progress(30)
            
            # Import and run the pipeline
            from Final_GENAI_V3 import run_pipeline
            
            # Run async pipeline
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            status_placeholder.info("🔮 Running analysis pipeline...")
            progress_bar.progress(50)
            
            results = loop.run_until_complete(
                run_pipeline(
                    symbol=symbol,
                    openai_api_key=api_key,
                    financial_goal=financial_goal
                )
            )
            
            progress_bar.progress(90)
            
            # Store results
            st.session_state.analysis_results = results
            
            progress_bar.progress(100)
            status_placeholder.success(f"✅ Analysis complete for {symbol}!")
            
            # Clear progress indicators after a moment
            import time
            time.sleep(1)
            progress_placeholder.empty()
            status_placeholder.empty()
            
    except Exception as e:
        status_placeholder.error(f"❌ Error during analysis: {str(e)}")
        st.error("Please check your configuration and try again.")
    
    finally:
        st.session_state.analysis_running = False
        st.rerun()

def display_results(results):
    """Display comprehensive analysis results"""
    
    if not results:
        st.warning("No analysis results available.")
        return
    
    # Create tabs for different sections
    tabs = st.tabs([
        "📊 Overview",
        "📈 Market Data",
        "⚠️ Risk Analysis",
        "🔮 Forecasting",
        "🌍 Macro & Sentiment",
        "🤖 AI Recommendation",
        "💰 Financial Plan",
        "📉 Visualizations"
    ])
    
    # Tab 1: Overview
    with tabs[0]:
        display_overview(results)
    
    # Tab 2: Market Data
    with tabs[1]:
        display_market_data(results)
    
    # Tab 3: Risk Analysis
    with tabs[2]:
        display_risk_analysis(results)
    
    # Tab 4: Forecasting
    with tabs[3]:
        display_forecasting(results)
    
    # Tab 5: Macro & Sentiment
    with tabs[4]:
        display_macro_sentiment(results)
    
    # Tab 6: AI Recommendation
    with tabs[5]:
        display_ai_recommendation(results)
    
    # Tab 7: Financial Plan
    with tabs[6]:
        display_financial_plan(results)
    
    # Tab 8: Visualizations
    with tabs[7]:
        display_visualizations(results)

def display_overview(results):
    """Display overview dashboard"""
    
    st.header("Executive Summary")
    
    symbol = results.get('symbol', 'Unknown')
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    if 'market_data' in results and results['market_data']:
        md = results['market_data']
        with col1:
            st.metric(
                "Current Price",
                f"${md.current_price:.2f}",
                f"{md.return_1d:.2%}",
                delta_color="normal"
            )
        
        with col2:
            st.metric(
                "RSI",
                f"{md.rsi:.1f}",
                help="Relative Strength Index"
            )
    
    if 'risk_metrics' in results and results['risk_metrics']:
        rm = results['risk_metrics']
        with col3:
            st.metric(
                "Sharpe Ratio",
                f"{rm.sharpe_ratio:.2f}",
                help="Risk-adjusted return metric"
            )
        
        with col4:
            st.metric(
                "Max Drawdown",
                f"{rm.maximum_drawdown:.1%}",
                delta_color="inverse"
            )
    
    # Recommendation Summary
    if 'recommendation' in results and results['recommendation']:
        rec = results['recommendation']
        st.markdown("---")
        st.subheader("🎯 Investment Recommendation")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Color code based on recommendation
            color = "green" if rec.action == "BUY" else "red" if rec.action == "SELL" else "orange"
            st.markdown(f"""
            <div class='recommendation-box'>
                <h3 style='color: {color};'>{rec.action}</h3>
                <p><strong>Confidence:</strong> {rec.confidence:.1%}</p>
                <p><strong>Risk Level:</strong> {rec.risk_level}</p>
                <p><strong>Time Horizon:</strong> {rec.time_horizon}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Entry Price", f"${rec.entry_price:.2f}")
            st.metric("Stop Loss", f"${rec.stop_loss:.2f}")
        
        with col3:
            st.metric("Take Profit", f"${rec.take_profit:.2f}")
            st.metric("Risk/Reward", f"{rec.risk_reward_ratio:.2f}")

def display_market_data(results):
    """Display detailed market data analysis"""
    
    st.header("Market Data Analysis")
    
    if 'market_data' not in results or not results['market_data']:
        st.warning("No market data available")
        return
    
    md = results['market_data']
    
    # Technical Indicators
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Price Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['Current Price', '1-Day Return', '5-Day Return', '20-Day Return', 
                      'Support Level', 'Resistance Level'],
            'Value': [
                f"${md.current_price:.2f}",
                f"{md.return_1d:.2%}",
                f"{md.return_5d:.2%}",
                f"{md.return_20d:.2%}",
                f"${md.support_level:.2f}",
                f"${md.resistance_level:.2f}"
            ]
        })
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.subheader("📈 Technical Indicators")
        indicators_df = pd.DataFrame({
            'Indicator': ['RSI', 'Trend', 'MACD Signal', 'Bollinger Position', 
                         'Volume Trend', '20-Day Volatility'],
            'Value': [
                f"{md.rsi:.1f}",
                md.trend,
                md.macd_signal,
                md.bollinger_position,
                md.volume_trend,
                f"{md.volatility_20d:.1%}"
            ]
        })
        st.dataframe(indicators_df, hide_index=True, use_container_width=True)
    
    # Price Chart
    if hasattr(md, 'prices') and len(md.prices) > 0:
        st.subheader("📉 Price History")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=md.prices.index,
            y=md.prices.values,
            mode='lines',
            name='Price',
            line=dict(color='blue', width=2)
        ))
        
        # Add support and resistance lines
        fig.add_hline(y=md.support_level, line_dash="dash", 
                     line_color="green", annotation_text="Support")
        fig.add_hline(y=md.resistance_level, line_dash="dash", 
                     line_color="red", annotation_text="Resistance")
        
        fig.update_layout(
            title=f"{results.get('symbol', 'Stock')} Price Chart",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_risk_analysis(results):
    """Display risk analysis results"""
    
    st.header("Risk Analysis")
    
    if 'risk_metrics' not in results or not results['risk_metrics']:
        st.warning("No risk metrics available")
        return
    
    rm = results['risk_metrics']
    
    # Risk Metrics Overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Portfolio Volatility", f"{rm.portfolio_volatility:.1%}")
        st.metric("VaR (5%)", f"{rm.value_at_risk_5pct:.2%}")
    
    with col2:
        st.metric("Sharpe Ratio", f"{rm.sharpe_ratio:.2f}")
        st.metric("Expected Shortfall", f"{rm.expected_shortfall:.2%}")
    
    with col3:
        st.metric("Sortino Ratio", f"{rm.sortino_ratio:.2f}")
        st.metric("GARCH Volatility", f"{rm.garch_volatility:.1%}")
    
    # Detailed Risk Table
    st.subheader("📊 Detailed Risk Metrics")
    
    risk_data = {
        'Metric': [
            'Maximum Drawdown',
            'Value at Risk (1%)',
            'Calmar Ratio',
            'Current GARCH Volatility'
        ],
        'Value': [
            f"{rm.maximum_drawdown:.2%}",
            f"{rm.value_at_risk_1pct:.2%}",
            f"{rm.calmar_ratio:.2f}",
            f"{rm.garch_volatility:.2%}"
        ],
        'Interpretation': [
            "Worst peak-to-trough decline",
            "1% chance of losing more than this",
            "Return per unit of max drawdown",
            "Current volatility estimate"
        ]
    }
    
    st.dataframe(pd.DataFrame(risk_data), hide_index=True, use_container_width=True)
    
    # GARCH Forecast
    if rm.garch_forecast:
        st.subheader("📈 Volatility Forecast (GARCH)")
        
        days = list(range(1, len(rm.garch_forecast) + 1))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=days,
            y=[v * 100 for v in rm.garch_forecast],
            mode='lines+markers',
            name='Forecasted Volatility',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title="5-Day Volatility Forecast",
            xaxis_title="Days Ahead",
            yaxis_title="Volatility (%)",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_forecasting(results):
    """Display forecasting results"""
    
    st.header("Price Forecasting")
    
    if 'forecast_data' not in results or not results['forecast_data']:
        st.warning("No forecast data available")
        return
    
    fd = results['forecast_data']
    current_price = results.get('market_data', {}).current_price if 'market_data' in results else 100
    
    # Forecast Summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Forecast Summary")
        
        price_change = ((fd.ensemble_forecast - current_price) / current_price) * 100
        
        st.metric(
            "Ensemble Forecast",
            f"${fd.ensemble_forecast:.2f}",
            f"{price_change:+.1f}%",
            delta_color="normal"
        )
        
        st.metric(
            "Confidence",
            f"{fd.forecast_confidence:.1%}"
        )
        
        st.metric(
            "Upside Probability",
            f"{fd.upside_probability:.1%}"
        )
    
    with col2:
        st.subheader("📈 Model Forecasts")
        
        forecasts_df = pd.DataFrame({
            'Model': ['ARIMA', 'Prophet', 'LSTM', 'Ensemble'],
            'Forecast': [
                f"${fd.arima_forecast:.2f}",
                f"${fd.prophet_forecast:.2f}",
                f"${fd.lstm_forecast:.2f}",
                f"${fd.ensemble_forecast:.2f}"
            ]
        })
        
        st.dataframe(forecasts_df, hide_index=True, use_container_width=True)
    
    # Forecast Comparison Chart
    st.subheader("🔮 Forecast Comparison")
    
    fig = go.Figure()
    
    models = ['Current', 'ARIMA', 'Prophet', 'LSTM', 'Ensemble']
    values = [current_price, fd.arima_forecast, fd.prophet_forecast, 
             fd.lstm_forecast, fd.ensemble_forecast]
    
    colors = ['gray', 'blue', 'green', 'orange', 'red']
    
    fig.add_trace(go.Bar(
        x=models,
        y=values,
        marker_color=colors,
        text=[f"${v:.2f}" for v in values],
        textposition='auto'
    ))
    
    # Add prediction interval
    fig.add_hline(y=fd.prediction_interval[0], line_dash="dash", 
                 line_color="gray", annotation_text="Lower Bound")
    fig.add_hline(y=fd.prediction_interval[1], line_dash="dash", 
                 line_color="gray", annotation_text="Upper Bound")
    
    fig.update_layout(
        title="Price Forecasts by Model",
        xaxis_title="Model",
        yaxis_title="Price ($)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_macro_sentiment(results):
    """Display macro economic and sentiment analysis"""
    
    st.header("Macro Economic & Sentiment Analysis")
    
    col1, col2 = st.columns(2)
    
    # Macro Economic Data
    with col1:
        st.subheader("🌍 Macro Economic Indicators")
        
        if 'macro_data' in results and results['macro_data']:
            macro = results['macro_data']
            
            macro_df = pd.DataFrame({
                'Indicator': [
                    'GDP Growth',
                    'Inflation Rate',
                    'Unemployment',
                    'Fed Funds Rate',
                    'VIX',
                    'Dollar Index',
                    'Yield Curve Slope',
                    'Market Sentiment'
                ],
                'Value': [
                    f"{macro.gdp_growth:.1f}%",
                    f"{macro.inflation_rate:.1f}%",
                    f"{macro.unemployment_rate:.1f}%",
                    f"{macro.federal_funds_rate:.2f}%",
                    f"{macro.vix:.1f}",
                    f"{macro.dollar_index:.1f}",
                    f"{macro.yield_curve_slope:.2f}",
                    macro.market_sentiment
                ]
            })
            
            st.dataframe(macro_df, hide_index=True, use_container_width=True)
        else:
            st.info("No macro economic data available")
    
    # Sentiment Analysis
    with col2:
        st.subheader("💭 Sentiment Analysis")
        
        if 'sentiment_data' in results and results['sentiment_data']:
            sentiment = results['sentiment_data']
            
            # Sentiment gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sentiment.fear_greed_index,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Fear & Greed Index"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "red"},
                        {'range': [25, 50], 'color': "orange"},
                        {'range': [50, 75], 'color': "yellow"},
                        {'range': [75, 100], 'color': "green"}
                    ]
                }
            ))
            
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
            
            # Sentiment details
            sentiment_df = pd.DataFrame({
                'Source': ['News', 'Social Media', 'Overall'],
                'Sentiment': [
                    f"{sentiment.news_sentiment:.2f}",
                    f"{sentiment.social_media_sentiment:.2f}",
                    f"{sentiment.overall_sentiment:.2f}"
                ],
                'Trend': [
                    sentiment.sentiment_trend,
                    sentiment.analyst_rating_trend,
                    sentiment.sentiment_trend
                ]
            })
            
            st.dataframe(sentiment_df, hide_index=True, use_container_width=True)
            
            # Key topics
            if sentiment.key_topics:
                st.write("**Key Topics:**", ", ".join(sentiment.key_topics[:5]))
        else:
            st.info("No sentiment data available")

def display_ai_recommendation(results):
    """Display AI recommendation details"""
    
    st.header("🤖 AI Strategic Recommendation")
    
    if 'recommendation' not in results or not results['recommendation']:
        st.warning("No AI recommendation available")
        return
    
    rec = results['recommendation']
    
    # Main Recommendation
    color_map = {"BUY": "green", "SELL": "red", "HOLD": "orange"}
    action_color = color_map.get(rec.action, "gray")
    
    st.markdown(f"""
    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; 
                border-left: 5px solid {action_color};'>
        <h2 style='color: {action_color};'>{rec.action} Recommendation</h2>
        <p><strong>Confidence Level:</strong> {rec.confidence:.1%}</p>
        <p><strong>Position Size:</strong> {rec.position_size:.1%} of portfolio</p>
        <p><strong>Risk Level:</strong> {rec.risk_level}</p>
        <p><strong>Time Horizon:</strong> {rec.time_horizon}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Trading Parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Entry Price", f"${rec.entry_price:.2f}")
    with col2:
        st.metric("Stop Loss", f"${rec.stop_loss:.2f}")
    with col3:
        st.metric("Take Profit", f"${rec.take_profit:.2f}")
    
    # Risk Metrics
    st.subheader("📊 Risk Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Risk/Reward Ratio", f"{rec.risk_reward_ratio:.2f}")
    with col2:
        st.metric("Success Probability", f"{rec.probability_of_success:.1%}")
    with col3:
        st.metric("Max Drawdown Est.", f"{rec.maximum_drawdown_estimate:.1%}")
    
    # Detailed Analysis
    st.subheader("📝 Detailed Analysis")
    
    if hasattr(rec, 'detailed_reasoning') and rec.detailed_reasoning:
        st.info(rec.detailed_reasoning)
    
    # Risk and Opportunity Factors
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚠️ Key Risk Factors")
        if hasattr(rec, 'key_risk_factors') and rec.key_risk_factors:
            for risk in rec.key_risk_factors:
                st.markdown(f"• {risk}")
        else:
            st.write("No specific risk factors identified")
    
    with col2:
        st.subheader("✅ Key Opportunity Factors")
        if hasattr(rec, 'key_opportunity_factors') and rec.key_opportunity_factors:
            for opp in rec.key_opportunity_factors:
                st.markdown(f"• {opp}")
        else:
            st.write("No specific opportunity factors identified")
    
    # Alternative Scenarios
    if hasattr(rec, 'alternative_scenarios') and rec.alternative_scenarios:
        st.subheader("🔄 Alternative Scenarios")
        
        for scenario, description in rec.alternative_scenarios.items():
            scenario_title = scenario.replace('_', ' ').title()
            st.write(f"**{scenario_title}:** {description}")
    
    # Portfolio Impact
    if hasattr(rec, 'portfolio_impact') and rec.portfolio_impact:
        st.subheader("💼 Portfolio Impact")
        st.write(rec.portfolio_impact)
    
    # Market Timing
    if hasattr(rec, 'market_timing_analysis') and rec.market_timing_analysis:
        st.subheader("⏰ Market Timing Analysis")
        st.write(rec.market_timing_analysis)

def display_financial_plan(results):
    """Display financial planning results"""
    
    st.header("💰 Financial Planning Analysis")
    
    if 'financial_plan' not in results or not results['financial_plan']:
        st.info("No financial planning data available. Enable Financial Planning in the sidebar to see this analysis.")
        return
    
    plan = results['financial_plan']
    goal = plan.goal
    
    # Goal Summary
    st.subheader("🎯 Financial Goal Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Target Amount", f"${goal.target_amount:,.0f}")
        st.metric("Current Savings", f"${goal.current_amount:,.0f}")
    
    with col2:
        st.metric("Monthly Contribution", f"${goal.monthly_contribution:,.0f}")
        st.metric("Time Horizon", f"{goal.time_horizon_years} years")
    
    with col3:
        st.metric("Risk Tolerance", goal.risk_tolerance.title())
        st.metric("Goal Type", goal.goal_type.title())
    
    # Plan Results
    st.subheader("📊 Plan Analysis Results")
    
    # Success indicator
    if plan.is_achievable:
        st.success(f"✅ Goal is achievable! Projected value: ${plan.projected_value:,.0f}")
    else:
        additional_needed = plan.required_monthly - goal.monthly_contribution
        st.warning(f"⚠️ Additional ${additional_needed:,.0f}/month needed to reach goal")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Success Probability",
            f"{plan.success_probability:.1%}",
            help="Probability of achieving your financial goal"
        )
    
    with col2:
        st.metric(
            "Projected Value",
            f"${plan.projected_value:,.0f}",
            f"${plan.projected_value - goal.target_amount:,.0f}",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            "Plan Sharpe Ratio",
            f"{plan.plan_sharpe_ratio:.2f}",
            help="Risk-adjusted return of the plan"
        )
    
    with col4:
        st.metric(
            "Expected Volatility",
            f"{plan.plan_volatility:.1%}",
            help="Expected portfolio volatility"
        )
    
    # Asset Allocation
    st.subheader("🎯 Recommended Asset Allocation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Pie chart
        labels = [asset.replace('_', ' ').title() for asset in plan.asset_allocation.keys()]
        values = list(plan.asset_allocation.values())
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        )])
        
        fig.update_layout(
            title="Portfolio Allocation",
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Allocation table
        allocation_df = pd.DataFrame({
            'Asset Class': labels,
            'Allocation': [f"{v:.1%}" for v in values]
        })
        st.dataframe(allocation_df, hide_index=True, use_container_width=True)
    
    # Monte Carlo Results
    if 'monte_carlo_results' in plan.__dict__ and plan.monte_carlo_results:
        st.subheader("📈 Monte Carlo Simulation Results")
        
        mc = plan.monte_carlo_results
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create percentile chart
            percentiles = ['5th', '25th', '50th', '75th', '95th']
            values = [
                mc.get('percentile_5', 0),
                mc.get('percentile_25', 0),
                mc.get('percentile_50', 0),
                mc.get('percentile_75', 0),
                mc.get('percentile_95', 0)
            ]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=percentiles,
                y=values,
                mode='lines+markers',
                name='Projected Value',
                line=dict(color='blue', width=2),
                fill='tonexty',
                fillcolor='rgba(31, 119, 180, 0.2)'
            ))
            
            # Add target line
            fig.add_hline(
                y=goal.target_amount,
                line_dash="dash",
                line_color="red",
                annotation_text="Target Amount"
            )
            
            fig.update_layout(
                title="Monte Carlo Simulation Percentiles",
                xaxis_title="Percentile",
                yaxis_title="Projected Value ($)",
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric("Success Rate", f"{mc.get('success_rate', 0):.1%}")
            st.metric("Mean Outcome", f"${mc.get('mean', 0):,.0f}")
            st.metric("Median Outcome", f"${mc.get('median', 0):,.0f}")
            st.metric("Worst Drawdown", f"{mc.get('worst_drawdown', 0):.1%}")
    
    # Tax Optimization
    if 'tax_optimization' in plan.__dict__ and plan.tax_optimization:
        st.subheader("🏦 Tax Optimization Strategy")
        
        tax = plan.tax_optimization
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("401(k) Monthly", f"${tax.get('401k_monthly', 0):,.0f}")
        with col2:
            st.metric("IRA Monthly", f"${tax.get('ira_monthly', 0):,.0f}")
        with col3:
            st.metric("Annual Tax Savings", f"${tax.get('tax_savings', 0):,.0f}")
    
    # Recommendations
    if plan.recommendations:
        st.subheader("💡 Personalized Recommendations")
        
        for i, rec in enumerate(plan.recommendations, 1):
            st.write(f"{rec}")

def display_visualizations(results):
    """Display comprehensive visualizations"""
    
    st.header("📊 Advanced Visualizations")
    
    # Check for required data
    if not results or 'market_data' not in results:
        st.warning("No data available for visualization")
        return
    
    # Technical Analysis Chart
    if 'market_data' in results and hasattr(results['market_data'], 'prices'):
        st.subheader("📈 Technical Analysis")
        
        md = results['market_data']
        prices = md.prices
        
        # Create candlestick chart with indicators
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=('Price & Indicators', 'Volume')
        )
        
        # Price line
        fig.add_trace(
            go.Scatter(
                x=prices.index,
                y=prices.values,
                name='Price',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Moving averages
        if len(prices) >= 20:
            ma20 = prices.rolling(window=20).mean()
            fig.add_trace(
                go.Scatter(
                    x=ma20.index,
                    y=ma20.values,
                    name='MA20',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
        
        if len(prices) >= 50:
            ma50 = prices.rolling(window=50).mean()
            fig.add_trace(
                go.Scatter(
                    x=ma50.index,
                    y=ma50.values,
                    name='MA50',
                    line=dict(color='red', width=1)
                ),
                row=1, col=1
            )
        
        # Support and resistance
        fig.add_hline(
            y=md.support_level,
            line_dash="dash",
            line_color="green",
            annotation_text="Support",
            row=1, col=1
        )
        fig.add_hline(
            y=md.resistance_level,
            line_dash="dash",
            line_color="red",
            annotation_text="Resistance",
            row=1, col=1
        )
        
        # Volume
        if hasattr(md, 'volume') and len(md.volume) > 0:
            fig.add_trace(
                go.Bar(
                    x=md.volume.index,
                    y=md.volume.values,
                    name='Volume',
                    marker_color='lightblue'
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title=f"{results.get('symbol', 'Stock')} Technical Analysis",
            xaxis_title="Date",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk-Return Scatter Plot
    if 'risk_metrics' in results and 'forecast_data' in results:
        st.subheader("⚖️ Risk-Return Analysis")
        
        rm = results['risk_metrics']
        fd = results['forecast_data']
        
        # Calculate expected return
        current_price = results.get('market_data', {}).current_price if 'market_data' in results else 100
        expected_return = ((fd.ensemble_forecast - current_price) / current_price) * 100
        
        fig = go.Figure()
        
        # Add current position
        fig.add_trace(go.Scatter(
            x=[rm.portfolio_volatility * 100],
            y=[expected_return],
            mode='markers+text',
            name='Current Position',
            marker=dict(size=20, color='red', symbol='star'),
            text=['Current'],
            textposition='top center'
        ))
        
        # Add efficient frontier representation
        volatilities = np.linspace(5, 40, 50)
        returns = volatilities * 0.4 - 2  # Simplified efficient frontier
        
        fig.add_trace(go.Scatter(
            x=volatilities,
            y=returns,
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='blue', dash='dash')
        ))
        
        fig.update_layout(
            title="Risk-Return Profile",
            xaxis_title="Volatility (%)",
            yaxis_title="Expected Return (%)",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation Heatmap (if multiple assets)
    st.subheader("🔥 Market Indicators Heatmap")
    
    # Create sample correlation matrix for demonstration
    indicators = ['Price', 'RSI', 'Volume', 'Volatility', 'Sentiment']
    correlations = np.random.rand(5, 5)
    correlations = (correlations + correlations.T) / 2  # Make symmetric
    np.fill_diagonal(correlations, 1)  # Set diagonal to 1
    
    fig = go.Figure(data=go.Heatmap(
        z=correlations,
        x=indicators,
        y=indicators,
        colorscale='RdBu',
        zmid=0,
        text=correlations.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Indicator Correlation Matrix",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Helper function to run async code in Streamlit
def run_async(coro):
    """Helper to run async coroutines in Streamlit"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

# Download results functionality
def create_download_report(results):
    """Create downloadable report from results"""
    
    report = []
    report.append("AI FINANCIAL FORECASTING REPORT")
    report.append("=" * 50)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Symbol: {results.get('symbol', 'N/A')}")
    report.append("")
    
    # Add sections based on available data
    if 'market_data' in results and results['market_data']:
        md = results['market_data']
        report.append("MARKET DATA")
        report.append("-" * 30)
        report.append(f"Current Price: ${md.current_price:.2f}")
        report.append(f"Trend: {md.trend}")
        report.append(f"RSI: {md.rsi:.1f}")
        report.append("")
    
    if 'recommendation' in results and results['recommendation']:
        rec = results['recommendation']
        report.append("AI RECOMMENDATION")
        report.append("-" * 30)
        report.append(f"Action: {rec.action}")
        report.append(f"Confidence: {rec.confidence:.1%}")
        report.append(f"Risk Level: {rec.risk_level}")
        report.append("")
    
    return "\n".join(report)

# Add download button in sidebar
def add_download_button(results):
    """Add download button for results"""
    
    if results:
        report = create_download_report(results)
        
        st.sidebar.markdown("---")
        st.sidebar.download_button(
            label="📥 Download Report",
            data=report,
            file_name=f"financial_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

# Main execution
if __name__ == "__main__":
    main()
    
    # Add download button if results exist
    if st.session_state.analysis_results:
        add_download_button(st.session_state.analysis_results)