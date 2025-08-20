import streamlit as st
import asyncio
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Page config
st.set_page_config(
    page_title="AI Financial Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import your financial system
try:
    # Import directly from Final_pls module (adjust path as needed)
    from Final_pls import (
        MarketDataAgent, RiskAgent, ForecastingAgent, MacroEconomicAgent,
        SentimentAgent, StrategistAgent, FinancialPlannerAgent,
        FinancialGoal, run_pipeline_with_real_apis, print_pipeline_summary,
        create_dashboard_visualizations
    )
    SYSTEM_AVAILABLE = True
except ImportError as e:
    st.error(f"Could not import the financial system: {e}")
    st.error("Please ensure Final_pls.py is in the same directory or in your Python path.")
    SYSTEM_AVAILABLE = False

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .recommendation-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .risk-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False

# Helper functions
def run_analysis_sync(symbol, use_financial_planning, financial_params, openai_key=None):
    """Run the analysis synchronously with proper error handling"""
    try:
        # Create new event loop for this analysis
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            financial_goal = None
            if use_financial_planning:
                financial_goal = FinancialGoal(**financial_params)
            
            # Set OpenAI key if provided
            api_key = openai_key if openai_key else None
            
            result = loop.run_until_complete(
                run_pipeline_with_real_apis(
                    symbol=symbol, 
                    openai_api_key=api_key,
                    financial_goal=financial_goal
                )
            )
            return result
        finally:
            loop.close()
            
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        st.error("This may be due to API limits, network issues, or missing dependencies.")
        return None

def create_price_chart(market_data):
    """Create price chart with technical indicators"""
    if not market_data or not hasattr(market_data, 'prices'):
        return None
    
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=market_data.prices.index,
        y=market_data.prices.values,
        mode='lines',
        name='Price',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Support and resistance levels
    fig.add_hline(
        y=market_data.support_level,
        line_dash="dash",
        line_color="green",
        annotation_text="Support"
    )
    fig.add_hline(
        y=market_data.resistance_level,
        line_dash="dash",
        line_color="red",
        annotation_text="Resistance"
    )
    
    fig.update_layout(
        title=f"{market_data.symbol} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        showlegend=True,
        height=400
    )
    
    return fig

def create_risk_dashboard(risk_metrics):
    """Create risk metrics dashboard"""
    if not risk_metrics:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Sharpe Ratio', 'Volatility (%)', 'Max Drawdown (%)', 'VaR 5% (%)'],
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}]]
    )
    
    # Sharpe Ratio gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=risk_metrics.sharpe_ratio,
        title={'text': "Sharpe Ratio"},
        gauge={'axis': {'range': [-2, 3]},
               'bar': {'color': "darkblue"},
               'steps': [{'range': [-2, 0], 'color': "lightgray"},
                        {'range': [0, 1], 'color': "yellow"},
                        {'range': [1, 3], 'color': "green"}],
               'threshold': {'line': {'color': "red", 'width': 4},
                           'thickness': 0.75, 'value': 1.5}},
        domain={'x': [0, 1], 'y': [0, 1]}
    ), row=1, col=1)
    
    # Volatility gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=risk_metrics.portfolio_volatility * 100,
        title={'text': "Volatility (%)"},
        gauge={'axis': {'range': [0, 50]},
               'bar': {'color': "orange"},
               'steps': [{'range': [0, 15], 'color': "green"},
                        {'range': [15, 25], 'color': "yellow"},
                        {'range': [25, 50], 'color': "red"}]},
        domain={'x': [0, 1], 'y': [0, 1]}
    ), row=1, col=2)
    
    # Max Drawdown gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=abs(risk_metrics.maximum_drawdown) * 100,
        title={'text': "Max Drawdown (%)"},
        gauge={'axis': {'range': [0, 50]},
               'bar': {'color': "red"},
               'steps': [{'range': [0, 10], 'color': "green"},
                        {'range': [10, 20], 'color': "yellow"},
                        {'range': [20, 50], 'color': "red"}]},
        domain={'x': [0, 1], 'y': [0, 1]}
    ), row=2, col=1)
    
    # VaR gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=abs(risk_metrics.value_at_risk_5pct) * 100,
        title={'text': "VaR 5% (%)"},
        gauge={'axis': {'range': [0, 20]},
               'bar': {'color': "purple"},
               'steps': [{'range': [0, 5], 'color': "green"},
                        {'range': [5, 10], 'color': "yellow"},
                        {'range': [10, 20], 'color': "red"}]},
        domain={'x': [0, 1], 'y': [0, 1]}
    ), row=2, col=2)
    
    fig.update_layout(height=500, title_text="Risk Metrics Dashboard")
    return fig

def create_forecast_comparison(forecast_data, current_price):
    """Create forecast comparison chart"""
    if not forecast_data:
        return None
    
    forecasts = {
        'ARIMA': forecast_data.arima_forecast,
        'Prophet': forecast_data.prophet_forecast,
        'LSTM': forecast_data.lstm_forecast,
        'Ensemble': forecast_data.ensemble_forecast
    }
    
    # Calculate percentage changes
    pct_changes = {k: ((v - current_price) / current_price) * 100 for k, v in forecasts.items()}
    
    fig = go.Figure()
    
    # Add bars for each forecast
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#1f77b4']
    for i, (name, value) in enumerate(pct_changes.items()):
        fig.add_trace(go.Bar(
            x=[name],
            y=[value],
            name=name,
            marker_color=colors[i],
            text=f'{value:+.1f}%',
            textposition='auto'
        ))
    
    fig.update_layout(
        title="Price Forecast Comparison",
        xaxis_title="Forecast Method",
        yaxis_title="Expected Price Change (%)",
        showlegend=False,
        height=400
    )
    
    return fig

def create_sentiment_pie(sentiment_data):
    """Create sentiment analysis pie chart"""
    if not sentiment_data:
        return None
    
    # Convert sentiment to categories
    overall = sentiment_data.overall_sentiment
    if overall > 0.1:
        values = [60 + overall*20, 30, 10 - overall*10]
    elif overall < -0.1:
        values = [10 + overall*10, 30, 60 - overall*20]
    else:
        values = [40, 40, 20]
    
    # Ensure non-negative values
    values = [max(0, v) for v in values]
    
    fig = go.Figure(data=[go.Pie(
        labels=['Positive', 'Neutral', 'Negative'],
        values=values,
        hole=.3,
        marker_colors=['#28a745', '#ffc107', '#dc3545']
    )])
    
    fig.update_layout(
        title="Market Sentiment Analysis",
        annotations=[dict(text='Sentiment', x=0.5, y=0.5, font_size=20, showarrow=False)],
        height=400
    )
    
    return fig

def create_asset_allocation_chart(allocation_dict):
    """Create asset allocation pie chart"""
    if not allocation_dict:
        return None
    
    labels = [k.replace('_', ' ').title() for k in allocation_dict.keys()]
    values = [v * 100 for v in allocation_dict.values()]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.3
    )])
    
    fig.update_layout(
        title="Recommended Asset Allocation",
        height=400
    )
    
    return fig

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">AI Financial Forecasting System</div>', unsafe_allow_html=True)
    
    if not SYSTEM_AVAILABLE:
        st.error("Financial analysis system is not available. Please check the import configuration.")
        st.info("Make sure you have installed all required packages: yfinance, pandas, numpy, plotly, etc.")
        return
    
    # Sidebar
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.header("Analysis Configuration")
    
    # Stock symbol input
    symbol = st.sidebar.text_input(
        "Stock Symbol", 
        value="AAPL", 
        help="Enter a valid stock ticker symbol (e.g., AAPL, TSLA, MSFT)"
    ).upper().strip()
    
    # OpenAI API Key
    openai_key = st.sidebar.text_input(
        "OpenAI API Key (Optional)", 
        type="password", 
        help="For enhanced AI recommendations using GPT-4"
    )
    
    # Financial planning toggle
    use_financial_planning = st.sidebar.checkbox(
        "Include Financial Planning", 
        value=False,
        help="Enable comprehensive financial planning analysis"
    )
    
    financial_params = {}
    if use_financial_planning:
        st.sidebar.subheader("Financial Planning Parameters")
        financial_params = {
            'target_amount': st.sidebar.number_input(
                "Target Amount ($)", 
                value=1000000, 
                min_value=1000,
                step=10000,
                help="Your financial goal amount"
            ),
            'current_amount': st.sidebar.number_input(
                "Current Amount ($)", 
                value=50000, 
                min_value=0,
                step=1000,
                help="Your current savings/investment amount"
            ),
            'monthly_contribution': st.sidebar.number_input(
                "Monthly Contribution ($)", 
                value=2000, 
                min_value=0,
                step=100,
                help="Monthly amount you plan to invest"
            ),
            'time_horizon_years': st.sidebar.slider(
                "Time Horizon (years)", 
                min_value=1, 
                max_value=50, 
                value=25,
                help="Number of years until you need the money"
            ),
            'age': st.sidebar.slider(
                "Age", 
                min_value=18, 
                max_value=100, 
                value=35,
                help="Your current age"
            ),
            'annual_income': st.sidebar.number_input(
                "Annual Income ($)", 
                value=120000, 
                min_value=0,
                step=5000,
                help="Your current annual income"
            ),
            'risk_tolerance': st.sidebar.selectbox(
                "Risk Tolerance", 
                ["conservative", "moderate", "aggressive"], 
                index=1,
                help="Your investment risk preference"
            ),
            'goal_type': st.sidebar.selectbox(
                "Goal Type", 
                ["retirement", "house", "education", "general"], 
                index=0,
                help="The purpose of your financial goal"
            ),
            'tax_rate': st.sidebar.slider(
                "Tax Rate", 
                min_value=0.0, 
                max_value=0.5, 
                value=0.22, 
                step=0.01,
                help="Your marginal tax rate"
            )
        }
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Run Analysis button
    if st.sidebar.button("🚀 Run Analysis", type="primary", disabled=st.session_state.analysis_running):
        if symbol:
            st.session_state.analysis_running = True
            
            # Clear previous results
            st.session_state.analysis_results = None
            
            # Create progress container
            progress_container = st.container()
            
            with progress_container:
                with st.spinner(f"Running comprehensive analysis for {symbol}..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Update progress
                    status_text.text("Initializing agents...")
                    progress_bar.progress(10)
                    
                    status_text.text("Fetching market data...")
                    progress_bar.progress(30)
                    
                    status_text.text("Analyzing risk metrics...")
                    progress_bar.progress(50)
                    
                    status_text.text("Generating forecasts...")
                    progress_bar.progress(70)
                    
                    status_text.text("Creating AI recommendation...")
                    progress_bar.progress(90)
                    
                    # Run the actual analysis
                    results = run_analysis_sync(symbol, use_financial_planning, financial_params, openai_key)
                    
                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")
                    
                    st.session_state.analysis_results = results
                    st.session_state.analysis_running = False
                    
                    # Clear progress indicators after a short delay
                    import time
                    time.sleep(1)
                    progress_container.empty()
                    
                    if results:
                        st.success("Analysis completed successfully!")
                        st.rerun()  # Refresh to show results
                    else:
                        st.error("Analysis failed. Please check the logs and try again.")
        else:
            st.error("Please enter a valid stock symbol.")
    
    # Display results
    if st.session_state.analysis_results:
        display_results(st.session_state.analysis_results)
    else:
        # Show instructions when no results
        st.info("👈 Configure your analysis parameters in the sidebar and click 'Run Analysis' to get started.")
        
        # Show example configuration
        with st.expander("📚 Example Configurations"):
            st.markdown("""
            **Basic Stock Analysis:**
            - Symbol: AAPL, TSLA, MSFT, etc.
            - Leave financial planning disabled
            
            **Retirement Planning:**
            - Enable financial planning
            - Set target amount: $1,000,000
            - Current amount: $50,000
            - Monthly contribution: $2,000
            - Time horizon: 25 years
            - Risk tolerance: Moderate
            
            **House Down Payment:**
            - Goal type: House
            - Target amount: $100,000
            - Time horizon: 5 years
            - Risk tolerance: Conservative
            """)

def display_results(results):
    """Display analysis results with improved error handling"""
    if not results:
        st.warning("No results to display.")
        return
    
    try:
        # Summary metrics at the top
        if 'market_data' in results and results['market_data']:
            md = results['market_data']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                change_color = "normal" if abs(md.return_1d) < 0.02 else ("inverse" if md.return_1d < 0 else "normal")
                st.metric("Current Price", f"${md.current_price:.2f}", f"{md.return_1d:.2%}", delta_color=change_color)
            with col2:
                st.metric("Trend", md.trend.replace('_', ' ').title(), "")
            with col3:
                rsi_status = "Oversold" if md.rsi < 30 else "Overbought" if md.rsi > 70 else "Normal"
                st.metric("RSI", f"{md.rsi:.1f}", rsi_status)
            with col4:
                st.metric("Volatility (20d)", f"{md.volatility_20d:.1%}", "")
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Market Analysis", 
            "⚠️ Risk Metrics", 
            "🔮 Forecasting", 
            "🧠 AI Recommendation", 
            "💰 Financial Planning"
        ])
        
        with tab1:
            display_market_analysis(results)
        
        with tab2:
            display_risk_analysis(results)
        
        with tab3:
            display_forecasting_analysis(results)
        
        with tab4:
            display_ai_recommendation(results)
        
        with tab5:
            display_financial_planning(results)
            
    except Exception as e:
        st.error(f"Error displaying results: {str(e)}")
        st.error("Some data may be incomplete or corrupted.")

def display_market_analysis(results):
    """Display market analysis tab"""
    st.subheader("Market Data & Technical Analysis")
    
    if 'market_data' in results and results['market_data']:
        md = results['market_data']
        
        # Price chart
        price_fig = create_price_chart(md)
        if price_fig:
            st.plotly_chart(price_fig, use_container_width=True)
        
        # Technical indicators in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.write("**Technical Indicators**")
            st.write(f"• RSI: {md.rsi:.1f} {'(Oversold)' if md.rsi < 30 else '(Overbought)' if md.rsi > 70 else '(Normal)'}")
            st.write(f"• MACD Signal: {md.macd_signal.title()}")
            st.write(f"• Bollinger Position: {md.bollinger_position.replace('_', ' ').title()}")
            st.write(f"• Volume Trend: {md.volume_trend.title()}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.write("**Support & Resistance Levels**")
            st.write(f"• Support Level: ${md.support_level:.2f}")
            st.write(f"• Resistance Level: ${md.resistance_level:.2f}")
            st.write(f"• 5-Day Return: {md.return_5d:.2%}")
            st.write(f"• 20-Day Return: {md.return_20d:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Additional market insights
        st.subheader("Market Insights")
        insights = []
        
        if md.rsi < 30:
            insights.append("📉 Stock appears oversold based on RSI, potential buying opportunity")
        elif md.rsi > 70:
            insights.append("📈 Stock appears overbought based on RSI, consider taking profits")
        
        if md.trend in ["bullish", "strongly_bullish"]:
            insights.append("🚀 Strong upward trend detected")
        elif md.trend in ["bearish", "strongly_bearish"]:
            insights.append("📉 Downward trend detected, exercise caution")
        
        if md.volume_trend == "increasing":
            insights.append("📊 Increasing volume supports current price movement")
        
        if insights:
            for insight in insights:
                st.info(insight)
    else:
        st.warning("Market data not available.")

def display_risk_analysis(results):
    """Display risk analysis tab"""
    st.subheader("Risk Analysis")
    
    if 'risk_metrics' in results and results['risk_metrics']:
        rm = results['risk_metrics']
        
        # Risk dashboard
        risk_fig = create_risk_dashboard(rm)
        if risk_fig:
            st.plotly_chart(risk_fig, use_container_width=True)
        
        # Risk metrics in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="risk-box">', unsafe_allow_html=True)
            st.write("**Portfolio Risk Metrics**")
            st.write(f"• Volatility: {rm.portfolio_volatility:.2%}")
            st.write(f"• Sharpe Ratio: {rm.sharpe_ratio:.3f}")
            st.write(f"• Sortino Ratio: {rm.sortino_ratio:.3f}")
            st.write(f"• Calmar Ratio: {rm.calmar_ratio:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="risk-box">', unsafe_allow_html=True)
            st.write("**Downside Risk Measures**")
            st.write(f"• VaR (5%): {rm.value_at_risk_5pct:.2%}")
            st.write(f"• VaR (1%): {rm.value_at_risk_1pct:.2%}")
            st.write(f"• Expected Shortfall: {rm.expected_shortfall:.2%}")
            st.write(f"• Maximum Drawdown: {rm.maximum_drawdown:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Risk interpretation
        st.subheader("Risk Interpretation")
        risk_insights = []
        
        if rm.sharpe_ratio > 1.5:
            risk_insights.append("✅ Excellent risk-adjusted returns (Sharpe > 1.5)")
        elif rm.sharpe_ratio > 1.0:
            risk_insights.append("👍 Good risk-adjusted returns (Sharpe > 1.0)")
        elif rm.sharpe_ratio < 0:
            risk_insights.append("⚠️ Poor risk-adjusted returns (negative Sharpe ratio)")
        
        if rm.portfolio_volatility > 0.3:
            risk_insights.append("⚠️ High volatility detected (>30%)")
        elif rm.portfolio_volatility < 0.15:
            risk_insights.append("✅ Low volatility, stable investment")
        
        if abs(rm.maximum_drawdown) > 0.2:
            risk_insights.append("⚠️ Significant historical drawdowns (>20%)")
        
        for insight in risk_insights:
            st.info(insight)
    else:
        st.warning("Risk metrics not available.")

def display_forecasting_analysis(results):
    """Display forecasting analysis tab"""
    st.subheader("Forecasting Analysis")
    
    if 'forecast_data' in results and results['forecast_data']:
        fd = results['forecast_data']
        current_price = results.get('market_data', {}).current_price if 'market_data' in results else 100
        
        # Forecast comparison chart
        forecast_fig = create_forecast_comparison(fd, current_price)
        if forecast_fig:
            st.plotly_chart(forecast_fig, use_container_width=True)
        
        # Forecast details in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.write("**Forecast Results**")
            st.write(f"• Ensemble Forecast: ${fd.ensemble_forecast:.2f}")
            price_change = ((fd.ensemble_forecast - current_price) / current_price) * 100
            st.write(f"• Expected Change: {price_change:+.1f}%")
            st.write(f"• Confidence: {fd.forecast_confidence:.1%}")
            st.write(f"• Upside Probability: {fd.upside_probability:.1%}")
            st.write(f"• Downside Risk: {fd.downside_risk:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.write("**Individual Model Forecasts**")
            st.write(f"• ARIMA: ${fd.arima_forecast:.2f}")
            st.write(f"• Prophet: ${fd.prophet_forecast:.2f}")
            st.write(f"• LSTM: ${fd.lstm_forecast:.2f}")
            st.write(f"• Volatility Forecast: {fd.volatility_forecast:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Forecast insights
        st.subheader("Forecast Insights")
        forecast_insights = []
        
        if fd.forecast_confidence > 0.7:
            forecast_insights.append("✅ High confidence in forecast accuracy")
        elif fd.forecast_confidence < 0.4:
            forecast_insights.append("⚠️ Low confidence in forecast, high uncertainty")
        
        if fd.upside_probability > 0.6:
            forecast_insights.append("📈 Higher probability of price increase")
        elif fd.upside_probability < 0.4:
            forecast_insights.append("📉 Higher probability of price decrease")
        
        price_change = ((fd.ensemble_forecast - current_price) / current_price) * 100
        if abs(price_change) > 10:
            forecast_insights.append(f"⚠️ Significant price movement expected ({price_change:+.1f}%)")
        
        for insight in forecast_insights:
            st.info(insight)
    else:
        st.warning("Forecast data not available.")

def display_ai_recommendation(results):
    """Display AI recommendation tab"""
    st.subheader("AI-Powered Investment Recommendation")
    
    if 'recommendation' in results and results['recommendation']:
        rec = results['recommendation']
        
        # Main recommendation display
        action_color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
        st.markdown(f'<div class="recommendation-box">', unsafe_allow_html=True)
        st.markdown(f"## {action_color.get(rec.action, '⚪')} **{rec.action}** Recommendation")
        
        # Key metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Confidence", f"{rec.confidence:.1%}")
        with col2:
            st.metric("Position Size", f"{rec.position_size:.1%}")
        with col3:
            st.metric("Risk Level", rec.risk_level)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed recommendation info
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Entry & Exit Strategy**")
            st.write(f"• Entry Price: ${rec.entry_price:.2f}")
            st.write(f"• Stop Loss: ${rec.stop_loss:.2f}")
            st.write(f"• Take Profit: ${rec.take_profit:.2f}")
            st.write(f"• Risk/Reward Ratio: {rec.risk_reward_ratio:.2f}")
            st.write(f"• Time Horizon: {rec.time_horizon}")
        
        with col2:
            st.write("**Success Metrics**")
            st.write(f"• Probability of Success: {rec.probability_of_success:.1%}")
            st.write(f"• Max Drawdown Estimate: {rec.maximum_drawdown_estimate:.1%}")
            
            # Risk/reward assessment
            if rec.risk_reward_ratio > 2:
                st.success("Favorable risk/reward ratio")
            elif rec.risk_reward_ratio < 1:
                st.warning("Unfavorable risk/reward ratio")
        
        # Detailed reasoning
        if hasattr(rec, 'detailed_reasoning') and rec.detailed_reasoning:
            st.subheader("AI Analysis Reasoning")
            st.write(rec.detailed_reasoning)
        
        # Risk and opportunity factors
        if hasattr(rec, 'key_risk_factors') and rec.key_risk_factors:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Key Risk Factors**")
                for factor in rec.key_risk_factors[:3]:
                    st.write(f"• {factor}")
            
            with col2:
                if hasattr(rec, 'key_opportunity_factors') and rec.key_opportunity_factors:
                    st.write("**Key Opportunities**")
                    for factor in rec.key_opportunity_factors[:3]:
                        st.write(f"• {factor}")
    
    # Sentiment and macro analysis
    col1, col2 = st.columns(2)
    
    with col1:
        if 'sentiment_data' in results and results['sentiment_data']:
            sd = results['sentiment_data']
            st.subheader("Market Sentiment")
            
            sentiment_fig = create_sentiment_pie(sd)
            if sentiment_fig:
                st.plotly_chart(sentiment_fig, use_container_width=True)
            
            st.write(f"**Overall Sentiment:** {sd.sentiment_trend.title()}")
            st.write(f"**News Sentiment:** {sd.news_sentiment:.2f}")
            st.write(f"**Social Media Sentiment:** {sd.social_media_sentiment:.2f}")
            st.write(f"**Fear & Greed Index:** {sd.fear_greed_index:.0f}")
            if sd.key_topics:
                st.write(f"**Key Topics:** {', '.join(sd.key_topics[:3])}")
    
    with col2:
        if 'macro_data' in results and results['macro_data']:
            st.subheader("Macroeconomic Environment")
            macro = results['macro_data']
            
            st.metric("GDP Growth", f"{macro.gdp_growth:.1f}%")
            st.metric("Inflation Rate", f"{macro.inflation_rate:.1f}%")
            st.metric("Unemployment", f"{macro.unemployment_rate:.1f}%")
            st.metric("Fed Funds Rate", f"{macro.federal_funds_rate:.2f}%")
            st.metric("VIX", f"{macro.vix:.1f}")

def display_financial_planning(results):
    """Display financial planning tab"""
    st.subheader("Financial Planning Analysis")
    
    if 'financial_plan' in results and results['financial_plan']:
        plan = results['financial_plan']
        
        # Plan summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Success Probability", f"{plan.success_probability:.1%}")
        with col2:
            st.metric("Projected Value", f"${plan.projected_value:,.0f}")
        with col3:
            st.metric("Target Amount", f"${plan.goal.target_amount:,.0f}")
        with col4:
            success_color = "normal" if plan.success_probability > 0.8 else "inverse"
            gap = plan.goal.target_amount - plan.projected_value
            st.metric("Gap", f"${gap:,.0f}", delta_color=success_color)
        
        # Asset allocation visualization
        st.subheader("Recommended Asset Allocation")
        allocation_fig = create_asset_allocation_chart(plan.asset_allocation)
        if allocation_fig:
            st.plotly_chart(allocation_fig, use_container_width=True)
        
        # Detailed allocation table
        allocation_df = pd.DataFrame.from_dict(plan.asset_allocation, orient='index', columns=['Allocation %'])
        allocation_df['Allocation %'] = allocation_df['Allocation %'] * 100
        allocation_df['Monthly ] = allocation_df['Allocation %'] / 100 * plan.goal.monthly_contribution
        allocation_df.index = [idx.replace('_', ' ').title() for idx in allocation_df.index]
        st.dataframe(allocation_df, use_container_width=True)
        
        # Plan metrics
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Plan Risk Metrics")
            st.write(f"**Plan Sharpe Ratio:** {plan.plan_sharpe_ratio:.2f}")
            st.write(f"**Expected Volatility:** {plan.plan_volatility:.1%}")
            st.write(f"**Est. Max Drawdown:** {plan.plan_max_drawdown:.1%}")
        
        with col2:
            st.subheader("Tax Optimization")
            if plan.tax_optimization:
                for key, value in plan.tax_optimization.items():
                    if 'monthly' in key:
                        st.write(f"**{key.replace('_', ' ').title()}:** ${value:,.0f}")
        
        # Monte Carlo results
        if plan.monte_carlo_results:
            st.subheader("Monte Carlo Simulation Results")
            mc = plan.monte_carlo_results
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Expected Value", f"${mc.get('mean', 0):,.0f}")
            with col2:
                st.metric("Success Rate", f"{mc.get('success_rate', 0):.1%}")
            with col3:
                st.metric("90th Percentile", f"${mc.get('percentile_90', 0):,.0f}")
        
        # Recommendations
        st.subheader("Planning Recommendations")
        for i, rec in enumerate(plan.recommendations, 1):
            st.write(f"{i}. {rec}")
        
        # Plan insights
        st.subheader("Plan Insights")
        insights = []
        
        if plan.success_probability > 0.8:
            insights.append("Your financial goal appears highly achievable with the current plan")
        elif plan.success_probability < 0.5:
            insights.append("Consider increasing contributions or extending timeline to improve success probability")
        
        if plan.plan_sharpe_ratio > 1.0:
            insights.append("Excellent risk-adjusted returns expected from the recommended allocation")
        
        if plan.goal.age < 35:
            insights.append("Young investor advantage: time is your greatest asset for wealth building")
        elif plan.goal.age > 50:
            insights.append("Consider focusing on risk management while maintaining growth potential")
        
        for insight in insights:
            st.info(insight)
    else:
        st.info("Financial planning analysis not available. Enable it in the sidebar to see detailed planning results.")
        
        # Show what financial planning includes
        with st.expander("What's included in Financial Planning?"):
            st.markdown("""
            - **Goal Achievement Analysis**: Probability of reaching your target
            - **Asset Allocation Optimization**: Risk-adjusted portfolio recommendations
            - **Monte Carlo Simulation**: Statistical modeling of potential outcomes
            - **Tax Optimization**: Strategies to minimize tax burden
            - **Risk Assessment**: Plan-level risk metrics and drawdown estimates
            - **Actionable Recommendations**: Specific steps to improve your plan
            """)

if __name__ == "__main__":
    main()
