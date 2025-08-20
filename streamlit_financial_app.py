"""
SIMPLIFIED AI FINANCIAL FORECASTING SYSTEM - STREAMLIT APP
Basic version without complex dependencies to avoid syntax errors
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Configure page
st.set_page_config(
    page_title="AI Financial Forecasting System",
    page_icon="📈",
    layout="wide"
)

# Basic data structures
@dataclass
class MarketData:
    symbol: str
    current_price: float
    prices: pd.Series
    rsi: float = 50.0
    trend: str = "neutral"
    return_1d: float = 0.0
    volatility_20d: float = 0.0
    support_level: float = 0.0
    resistance_level: float = 0.0

@dataclass
class ForecastData:
    ensemble_forecast: float = 0.0
    forecast_confidence: float = 0.5
    upside_probability: float = 0.5

@dataclass
class RiskMetrics:
    portfolio_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    maximum_drawdown: float = 0.0
    value_at_risk_5pct: float = 0.0

@dataclass
class Recommendation:
    action: str
    confidence: float
    risk_level: str
    entry_price: float
    stop_loss: float
    take_profit: float

# Simple market data agent
class SimpleMarketDataAgent:
    def __init__(self):
        self.name = "MarketDataAgent"

    def process(self, symbol: str = "AAPL") -> MarketData:
        try:
            st.write(f"📊 Fetching market data for {symbol}...")
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1y")
            
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            current_price = float(data['Close'].iloc[-1])
            prices = data['Close']
            
            # Calculate basic metrics
            returns = prices.pct_change()
            return_1d = float(returns.iloc[-1]) if len(returns) > 0 else 0.0
            volatility_20d = float(returns.tail(20).std() * np.sqrt(252)) if len(returns) >= 20 else 0.0
            
            # RSI calculation
            rsi = self._calculate_rsi(prices)
            
            # Trend analysis
            trend = self._analyze_trend(prices)
            
            # Support/Resistance
            high_20 = prices.tail(20).max()
            low_20 = prices.tail(20).min()
            support_level = float(low_20 * 1.02)
            resistance_level = float(high_20 * 0.98)
            
            return MarketData(
                symbol=symbol,
                current_price=current_price,
                prices=prices,
                rsi=rsi,
                trend=trend,
                return_1d=return_1d,
                volatility_20d=volatility_20d,
                support_level=support_level,
                resistance_level=resistance_level
            )
            
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0

    def _analyze_trend(self, prices: pd.Series) -> str:
        if len(prices) < 20:
            return "neutral"
        
        ma_5 = prices.tail(5).mean()
        ma_20 = prices.tail(20).mean()
        
        if ma_5 > ma_20 * 1.02:
            return "bullish"
        elif ma_5 < ma_20 * 0.98:
            return "bearish"
        else:
            return "neutral"

# Simple forecasting agent
class SimpleForecastingAgent:
    def __init__(self):
        self.name = "ForecastingAgent"

    def process(self, market_data: MarketData) -> ForecastData:
        try:
            st.write(f"🔮 Generating price forecast...")
            
            prices = market_data.prices
            current_price = market_data.current_price
            
            # Simple trend-based forecast
            if len(prices) >= 20:
                ma_5 = prices.tail(5).mean()
                ma_20 = prices.tail(20).mean()
                trend_factor = (ma_5 - ma_20) / ma_20
                forecast = current_price * (1 + trend_factor * 0.3)
            else:
                forecast = current_price * 1.01
            
            # Confidence based on volatility
            volatility = market_data.volatility_20d
            confidence = max(0.3, min(0.9, 1.0 - volatility))
            
            # Upside probability
            expected_return = (forecast - current_price) / current_price
            upside_probability = max(0.1, min(0.9, 0.5 + expected_return))
            
            return ForecastData(
                ensemble_forecast=forecast,
                forecast_confidence=confidence,
                upside_probability=upside_probability
            )
            
        except Exception as e:
            st.error(f"Error in forecasting: {e}")
            return ForecastData()

# Simple risk agent
class SimpleRiskAgent:
    def __init__(self):
        self.name = "RiskAgent"

    def process(self, market_data: MarketData) -> RiskMetrics:
        try:
            st.write(f"⚠️ Calculating risk metrics...")
            
            prices = market_data.prices
            returns = prices.pct_change().dropna()
            
            if len(returns) < 30:
                return RiskMetrics()
            
            # Basic risk metrics
            portfolio_volatility = float(returns.std() * np.sqrt(252))
            var_5pct = float(np.percentile(returns, 5))
            
            # Sharpe ratio
            excess_returns = returns.mean() * 252 - 0.02  # 2% risk-free rate
            sharpe_ratio = float(excess_returns / portfolio_volatility) if portfolio_volatility > 0 else 0.0
            
            # Maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            maximum_drawdown = float(drawdown.min())
            
            return RiskMetrics(
                portfolio_volatility=portfolio_volatility,
                sharpe_ratio=sharpe_ratio,
                maximum_drawdown=maximum_drawdown,
                value_at_risk_5pct=var_5pct
            )
            
        except Exception as e:
            st.error(f"Error in risk calculation: {e}")
            return RiskMetrics()

# Simple strategist
class SimpleStrategistAgent:
    def __init__(self):
        self.name = "StrategistAgent"

    def process(self, market_data: MarketData, forecast_data: ForecastData, risk_metrics: RiskMetrics) -> Recommendation:
        try:
            st.write(f"🤖 Generating recommendation...")
            
            # Simple scoring system
            score = 0.0
            
            # Technical score
            if market_data.trend == "bullish":
                score += 0.3
            elif market_data.trend == "bearish":
                score -= 0.3
            
            if market_data.rsi < 30:
                score += 0.2  # Oversold
            elif market_data.rsi > 70:
                score -= 0.2  # Overbought
            
            # Forecast score
            expected_return = (forecast_data.ensemble_forecast - market_data.current_price) / market_data.current_price
            score += expected_return * forecast_data.forecast_confidence
            
            # Risk adjustment
            if risk_metrics.sharpe_ratio > 1.0:
                score += 0.1
            elif risk_metrics.sharpe_ratio < 0:
                score -= 0.2
            
            # Generate recommendation
            if score > 0.3:
                action = "BUY"
                confidence = min(0.9, 0.6 + score * 0.5)
                risk_level = "MEDIUM"
            elif score < -0.3:
                action = "SELL"
                confidence = min(0.9, 0.6 + abs(score) * 0.5)
                risk_level = "HIGH"
            else:
                action = "HOLD"
                confidence = 0.6
                risk_level = "LOW"
            
            current_price = market_data.current_price
            
            return Recommendation(
                action=action,
                confidence=confidence,
                risk_level=risk_level,
                entry_price=current_price,
                stop_loss=current_price * (0.92 if action == "BUY" else 1.08),
                take_profit=current_price * (1.15 if action == "BUY" else 0.85)
            )
            
        except Exception as e:
            st.error(f"Error generating recommendation: {e}")
            return Recommendation("HOLD", 0.5, "MEDIUM", 100, 95, 105)

# Main application
def main():
    st.title("🤖 AI Financial Forecasting System")
    st.markdown("### Simplified Multi-Agent Investment Analysis")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Stock selection
        popular_stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "JPM"]
        
        symbol_option = st.selectbox("Select Stock", ["Custom"] + popular_stocks)
        
        if symbol_option == "Custom":
            symbol = st.text_input("Enter Symbol", value="AAPL").upper()
        else:
            symbol = symbol_option
        
        # Run analysis button
        if st.button("🚀 Run Analysis", type="primary"):
            run_analysis(symbol)

def run_analysis(symbol):
    """Run the simplified analysis pipeline"""
    
    # Initialize agents
    market_agent = SimpleMarketDataAgent()
    forecast_agent = SimpleForecastingAgent()
    risk_agent = SimpleRiskAgent()
    strategist_agent = SimpleStrategistAgent()
    
    # Create progress bar
    progress = st.progress(0)
    
    try:
        # Step 1: Market data
        progress.progress(25)
        market_data = market_agent.process(symbol)
        
        if not market_data:
            st.error("Failed to fetch market data")
            return
        
        # Step 2: Forecasting
        progress.progress(50)
        forecast_data = forecast_agent.process(market_data)
        
        # Step 3: Risk analysis
        progress.progress(75)
        risk_metrics = risk_agent.process(market_data)
        
        # Step 4: Generate recommendation
        progress.progress(100)
        recommendation = strategist_agent.process(market_data, forecast_data, risk_metrics)
        
        # Clear progress bar
        progress.empty()
        
        # Display results
        display_results(market_data, forecast_data, risk_metrics, recommendation)
        
    except Exception as e:
        st.error(f"Analysis failed: {e}")

def display_results(market_data, forecast_data, risk_metrics, recommendation):
    """Display analysis results"""
    
    st.success("✅ Analysis Complete!")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Market Data", "⚠️ Risk Analysis", "🤖 Recommendation"])
    
    with tab1:
        st.header("Executive Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${market_data.current_price:.2f}", f"{market_data.return_1d:.2%}")
        
        with col2:
            st.metric("RSI", f"{market_data.rsi:.1f}")
        
        with col3:
            st.metric("Sharpe Ratio", f"{risk_metrics.sharpe_ratio:.2f}")
        
        with col4:
            st.metric("Forecast", f"${forecast_data.ensemble_forecast:.2f}")
        
        # Recommendation summary
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            color = "green" if recommendation.action == "BUY" else "red" if recommendation.action == "SELL" else "orange"
            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid {color};'>
                <h3 style='color: {color};'>{recommendation.action} Recommendation</h3>
                <p><strong>Confidence:</strong> {recommendation.confidence:.1%}</p>
                <p><strong>Risk Level:</strong> {recommendation.risk_level}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Entry Price", f"${recommendation.entry_price:.2f}")
            st.metric("Stop Loss", f"${recommendation.stop_loss:.2f}")
            st.metric("Take Profit", f"${recommendation.take_profit:.2f}")
    
    with tab2:
        st.header("Market Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Key Metrics")
            metrics_data = {
                'Metric': ['Current Price', '1-Day Return', 'Volatility', 'Trend', 'RSI'],
                'Value': [
                    f"${market_data.current_price:.2f}",
                    f"{market_data.return_1d:.2%}",
                    f"{market_data.volatility_20d:.1%}",
                    market_data.trend.title(),
                    f"{market_data.rsi:.1f}"
                ]
            }
            st.dataframe(pd.DataFrame(metrics_data), hide_index=True)
        
        with col2:
            st.subheader("📈 Support & Resistance")
            levels_data = {
                'Level': ['Support', 'Current', 'Resistance'],
                'Price': [
                    f"${market_data.support_level:.2f}",
                    f"${market_data.current_price:.2f}",
                    f"${market_data.resistance_level:.2f}"
                ]
            }
            st.dataframe(pd.DataFrame(levels_data), hide_index=True)
        
        # Price chart
        if len(market_data.prices) > 0:
            st.subheader("📉 Price Chart")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=market_data.prices.index,
                y=market_data.prices.values,
                mode='lines',
                name='Price',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_hline(y=market_data.support_level, line_dash="dash", 
                         line_color="green", annotation_text="Support")
            fig.add_hline(y=market_data.resistance_level, line_dash="dash", 
                         line_color="red", annotation_text="Resistance")
            
            fig.update_layout(
                title=f"{market_data.symbol} Price History",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Portfolio Volatility", f"{risk_metrics.portfolio_volatility:.1%}")
            st.metric("Sharpe Ratio", f"{risk_metrics.sharpe_ratio:.2f}")
        
        with col2:
            st.metric("Maximum Drawdown", f"{risk_metrics.maximum_drawdown:.1%}")
            st.metric("VaR (5%)", f"{risk_metrics.value_at_risk_5pct:.2%}")
        
        # Risk interpretation
        st.subheader("📊 Risk Assessment")
        
        if risk_metrics.sharpe_ratio > 1.0:
            st.success("✅ Good risk-adjusted returns (Sharpe > 1.0)")
        elif risk_metrics.sharpe_ratio > 0.5:
            st.warning("⚠️ Moderate risk-adjusted returns")
        else:
            st.error("❌ Poor risk-adjusted returns")
        
        if abs(risk_metrics.maximum_drawdown) > 0.2:
            st.warning("⚠️ High drawdown risk detected")
        else:
            st.success("✅ Acceptable drawdown levels")
    
    with tab4:
        st.header("AI Recommendation")
        
        # Main recommendation
        color = "green" if recommendation.action == "BUY" else "red" if recommendation.action == "SELL" else "orange"
        
        st.markdown(f"""
        <div style='background-color: #f8f9fa; padding: 30px; border-radius: 15px; border-left: 8px solid {color}; margin: 20px 0;'>
            <h2 style='color: {color}; margin-bottom: 15px;'>{recommendation.action} RECOMMENDATION</h2>
            <div style='font-size: 18px; line-height: 1.6;'>
                <p><strong>Confidence Level:</strong> {recommendation.confidence:.1%}</p>
                <p><strong>Risk Assessment:</strong> {recommendation.risk_level}</p>
                <p><strong>Entry Price:</strong> ${recommendation.entry_price:.2f}</p>
                <p><strong>Stop Loss:</strong> ${recommendation.stop_loss:.2f}</p>
                <p><strong>Take Profit:</strong> ${recommendation.take_profit:.2f}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Forecast details
        st.subheader("🔮 Price Forecast")
        
        col1, col2 = st.columns(2)
        
        with col1:
            price_change = ((forecast_data.ensemble_forecast - market_data.current_price) / market_data.current_price) * 100
            st.metric("Forecasted Price", f"${forecast_data.ensemble_forecast:.2f}", f"{price_change:+.1f}%")
        
        with col2:
            st.metric("Forecast Confidence", f"{forecast_data.forecast_confidence:.1%}")
        
        # Action plan
        st.subheader("📋 Action Plan")
        
        if recommendation.action == "BUY":
            st.success("""
            **Recommended Actions:**
            - Consider buying at current levels
            - Set stop loss to limit downside risk
            - Monitor for take profit levels
            - Review position regularly
            """)
        elif recommendation.action == "SELL":
            st.error("""
            **Recommended Actions:**
            - Consider reducing position
            - Monitor for oversold conditions
            - Wait for better entry points
            - Maintain risk management
            """)
        else:
            st.info("""
            **Recommended Actions:**
            - Hold current position
            - Monitor market conditions
            - Wait for clearer signals
            - Maintain portfolio balance
            """)

if __name__ == "__main__":
    main()
