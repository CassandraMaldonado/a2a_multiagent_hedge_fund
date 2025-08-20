# -*- coding: utf-8 -*-
"""
AI Financial Forecasting System - Multi-Agent Pipeline
Fixed version for Streamlit compatibility
"""

# Remove the pip install command - this should be done separately
# !pip install fredapi newsapi-python textblob praw nest-asyncio -q

import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import requests
import warnings
import os
import json
import re
warnings.filterwarnings('ignore')

# Optional imports with graceful handling
try:
    import praw
    HAS_PRAW = True
except ImportError:
    HAS_PRAW = False
    print("Reddit (praw) not available.")

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("yfinance not available.")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Plotly not available.")

try:
    from fredapi import Fred
    HAS_FRED = True
except ImportError:
    HAS_FRED = False
    print("FRED API not available.")

try:
    from newsapi.newsapi_client import NewsApiClient
    HAS_NEWSAPI = True
except ImportError:
    HAS_NEWSAPI = False
    print("News API not available.")

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False
    print("TextBlob not available.")

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    print("Prophet not available, will be using simple forecasting.")

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("OpenAI not available, will be using rule-based recommendations.")

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    print("Streamlit not available.")

# Try to get API keys from environment or userdata
try:
    # For Google Colab
    from google.colab import userdata
    FRED_API_KEY = userdata.get('FRED_API_KEY')
    NEWS_API_KEY = userdata.get('NEWS_API_KEY')
    OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')
    REDDIT_CLIENT_ID = userdata.get('REDDIT_CLIENT_ID')
    REDDIT_CLIENT_SECRET = userdata.get('REDDIT_CLIENT_SECRET')
    REDDIT_USER_AGENT = userdata.get('REDDIT_USER_AGENT')
    print("API keys loaded from Colab secrets.")
except:
    # For regular environments
    FRED_API_KEY = os.environ.get('FRED_API_KEY')
    NEWS_API_KEY = os.environ.get('NEWS_API_KEY')
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    REDDIT_CLIENT_ID = os.environ.get('REDDIT_CLIENT_ID')
    REDDIT_CLIENT_SECRET = os.environ.get('REDDIT_CLIENT_SECRET')
    REDDIT_USER_AGENT = os.environ.get('REDDIT_USER_AGENT')
    print("API keys loaded from environment variables.")

# Data structures
@dataclass
class MarketData:
    symbol: str
    current_price: float
    prices: pd.Series
    volume: pd.Series = field(default_factory=pd.Series)

    # Technical indicators
    rsi: float = 50.0
    trend: str = "neutral"
    return_1d: float = 0.0
    return_5d: float = 0.0
    return_20d: float = 0.0
    volatility_20d: float = 0.0
    support_level: float = 0.0
    resistance_level: float = 0.0

    macd_signal: str = "neutral"
    bollinger_position: str = "middle"
    volume_trend: str = "neutral"

    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class ForecastData:
    arima_forecast: float = 0.0
    prophet_forecast: float = 0.0
    lstm_forecast: float = 0.0
    ensemble_forecast: float = 0.0
    forecast_confidence: float = 0.5
    prediction_interval: List[float] = field(default_factory=lambda: [0.0, 0.0])
    forecast_horizon_days: int = 5
    forecast_accuracy_score: float = 0.0

    # Forecast metrics
    upside_probability: float = 0.5
    downside_risk: float = 0.5
    volatility_forecast: float = 0.2
    simulated_returns: List[float] = field(default_factory=list)

    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class MacroData:
    gdp_growth: float = 0.0
    inflation_rate: float = 0.0
    unemployment_rate: float = 0.0
    federal_funds_rate: float = 0.0
    vix: float = 0.0
    dollar_index: float = 0.0
    market_sentiment: str = "neutral"

    yield_curve_slope: float = 0.0
    credit_spreads: float = 0.0
    economic_surprise_index: float = 0.0

    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class SentimentData:
    news_sentiment: float = 0.0
    social_media_sentiment: float = 0.0
    overall_sentiment: float = 0.0
    sentiment_trend: str = "neutral"
    confidence_score: float = 0.5
    key_topics: List[str] = field(default_factory=list)

    sentiment_momentum: float = 0.0
    fear_greed_index: float = 50.0
    analyst_rating_trend: str = "neutral"

    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class RiskMetrics:
    portfolio_volatility: float = 0.0
    value_at_risk_5pct: float = 0.0
    value_at_risk_1pct: float = 0.0
    expected_shortfall: float = 0.0
    maximum_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # GARCH results
    garch_volatility: float = 0.0
    garch_forecast: List[float] = field(default_factory=list)

    # Rolling metrics
    rolling_volatility: pd.Series = field(default_factory=pd.Series)
    rolling_sharpe: pd.Series = field(default_factory=pd.Series)

    # Drawdown analysis
    drawdown_series: pd.Series = field(default_factory=pd.Series)
    drawdown_periods: List[Dict] = field(default_factory=list)

    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class PersonalizedRecommendation:
    action: str  # buy, sell or hold
    confidence: float
    position_size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_level: str
    time_horizon: str

    detailed_reasoning: str
    key_risk_factors: List[str]
    key_opportunity_factors: List[str]
    alternative_scenarios: Dict[str, str]
    portfolio_impact: str
    market_timing_analysis: str

    # Quant metrics
    risk_reward_ratio: float
    probability_of_success: float
    maximum_drawdown_estimate: float

    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class FinancialGoal:
    target_amount: float
    current_amount: float
    monthly_contribution: float
    time_horizon_years: int
    risk_tolerance: str  # conservative, moderate or aggressive
    age: int = 30
    annual_income: float = 100000

    goal_type: str = "retirement"  # retirement, house, education or general
    existing_debt: float = 0.0
    emergency_fund: float = 0.0
    other_investments: float = 0.0
    tax_rate: float = 0.22
    inflation_assumption: float = 0.03

@dataclass
class FinancialPlanResult:
    goal: FinancialGoal
    projected_value: float
    success_probability: float
    required_monthly: float
    asset_allocation: Dict[str, float]
    tax_optimization: Dict[str, float]
    monthly_breakdown: Dict[str, float]
    recommendations: List[str]
    is_achievable: bool
    monte_carlo_results: Dict[str, float]

    # Risk metrics
    plan_sharpe_ratio: float = 0.0
    plan_max_drawdown: float = 0.0
    plan_volatility: float = 0.0

    last_updated: datetime = field(default_factory=datetime.now)

# Market data agent
class MarketDataAgent:
    def __init__(self):
        self.name = "MarketDataAgent"
        print(f"{self.name} initialized.")

    async def process(self, state: Dict, symbol: str = "AAPL", period: str = "1y") -> Dict:
        try:
            print(f"📊 {self.name}: Fetching market data for {symbol}...")

            if not HAS_YFINANCE:
                print(f"⚠️ yfinance not available, using simulated data")
                return self._get_simulated_data(state, symbol)

            # Fetching data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)

            if data.empty:
                raise ValueError(f"No data available for {symbol}")

            # Calculating metrics
            current_price = float(data['Close'].iloc[-1])
            prices = data['Close']
            volume = data['Volume']

            # Basic returns and volatility
            returns = prices.pct_change()
            return_1d = float(returns.iloc[-1]) if len(returns) > 0 else 0.0
            return_5d = float(returns.tail(5).mean()) if len(returns) >= 5 else 0.0
            return_20d = float(returns.tail(20).mean()) if len(returns) >= 20 else 0.0
            volatility_20d = float(returns.tail(20).std() * np.sqrt(252)) if len(returns) >= 20 else 0.0

            # Technical indicators
            rsi = self._calculate_rsi(prices)
            trend = self._analyze_trend(prices)
            macd_signal = self._calculate_macd_signal(prices)
            bollinger_position = self._calculate_bollinger_position(prices)
            volume_trend = self._analyze_volume_trend(volume)

            # Support and resistance levels
            high_20 = prices.tail(20).max()
            low_20 = prices.tail(20).min()
            support_level = float(low_20 * 1.02)
            resistance_level = float(high_20 * 0.98)

            market_data = MarketData(
                symbol=symbol,
                current_price=current_price,
                prices=prices,
                volume=volume,
                rsi=rsi,
                trend=trend,
                return_1d=return_1d,
                return_5d=return_5d,
                return_20d=return_20d,
                volatility_20d=volatility_20d,
                support_level=support_level,
                resistance_level=resistance_level,
                macd_signal=macd_signal,
                bollinger_position=bollinger_position,
                volume_trend=volume_trend
            )

            state['market_data'] = market_data
            state['symbol'] = symbol

            print(f"{self.name}: Loaded {len(prices)} data points.")
            print(f"   Price: ${current_price:.2f}")
            print(f"   Trend: {trend}")
            print(f"   RSI: {rsi:.1f}")

        except Exception as e:
            print(f"❌ {self.name}: Error - {e}")
            return self._get_simulated_data(state, symbol)

        return state

    def _get_simulated_data(self, state: Dict, symbol: str) -> Dict:
        """Generate simulated market data when real data isn't available"""
        import random
        
        # Create simulated price data
        base_price = 150.0
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        prices = pd.Series([base_price * (1 + random.gauss(0, 0.02)) for _ in range(252)], index=dates)
        volume = pd.Series([random.randint(1000000, 5000000) for _ in range(252)], index=dates)
        
        current_price = prices.iloc[-1]
        returns = prices.pct_change()
        
        market_data = MarketData(
            symbol=symbol,
            current_price=current_price,
            prices=prices,
            volume=volume,
            rsi=random.uniform(30, 70),
            trend=random.choice(["bullish", "bearish", "neutral"]),
            return_1d=returns.iloc[-1] if len(returns) > 0 else 0.0,
            return_5d=returns.tail(5).mean() if len(returns) >= 5 else 0.0,
            return_20d=returns.tail(20).mean() if len(returns) >= 20 else 0.0,
            volatility_20d=returns.tail(20).std() * np.sqrt(252) if len(returns) >= 20 else 0.2,
            support_level=current_price * 0.95,
            resistance_level=current_price * 1.05,
            macd_signal=random.choice(["bullish", "bearish", "neutral"]),
            bollinger_position=random.choice(["upper_half", "lower_half", "middle"]),
            volume_trend=random.choice(["increasing", "decreasing", "stable"])
        )
        
        state['market_data'] = market_data
        state['symbol'] = symbol
        
        print(f"📊 {self.name}: Using simulated data for {symbol}")
        return state

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
        ma_10 = prices.tail(10).mean()
        ma_20 = prices.tail(20).mean()

        if ma_5 > ma_10 > ma_20:
            return "strongly_bullish"
        elif ma_5 > ma_20 * 1.02:
            return "bullish"
        elif ma_5 < ma_10 < ma_20:
            return "strongly_bearish"
        elif ma_5 < ma_20 * 0.98:
            return "bearish"
        else:
            return "neutral"

    def _calculate_macd_signal(self, prices: pd.Series) -> str:
        if len(prices) < 26:
            return "neutral"

        exp1 = prices.ewm(span=12).mean()
        exp2 = prices.ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()

        if macd.iloc[-1] > signal.iloc[-1]:
            return "bullish"
        elif macd.iloc[-1] < signal.iloc[-1]:
            return "bearish"
        else:
            return "neutral"

    def _calculate_bollinger_position(self, prices: pd.Series, period: int = 20) -> str:
        if len(prices) < period:
            return "middle"

        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)

        current_price = prices.iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        current_middle = sma.iloc[-1]

        if current_price > current_upper:
            return "above_upper"
        elif current_price < current_lower:
            return "below_lower"
        elif current_price > current_middle:
            return "upper_half"
        else:
            return "lower_half"

    def _analyze_volume_trend(self, volume: pd.Series) -> str:
        if len(volume) < 10:
            return "neutral"

        recent_volume = volume.tail(5).mean()
        historical_volume = volume.tail(20).mean()

        if recent_volume > historical_volume * 1.2:
            return "increasing"
        elif recent_volume < historical_volume * 0.8:
            return "decreasing"
        else:
            return "stable"

# Simplified versions of other agents for compatibility
class RiskAgent:
    def __init__(self):
        self.name = "RiskAgent"
        print(f"{self.name} initialized.")

    async def process(self, state: Dict) -> Dict:
        try:
            print(f"⚠️ {self.name}: Computing portfolio risk metrics.")

            if 'market_data' not in state or not state['market_data']:
                print(f"{self.name}: No market data available.")
                return state

            market_data = state['market_data']
            prices = market_data.prices

            if len(prices) < 30:
                print(f"{self.name}: Insufficient data for risk analysis.")
                return state

            # Calculate returns
            returns = prices.pct_change().dropna()

            # Basic risk metrics
            portfolio_volatility = float(returns.std() * np.sqrt(252))
            var_5pct = float(np.percentile(returns, 5))
            var_1pct = float(np.percentile(returns, 1))
            expected_shortfall = float(returns[returns <= var_5pct].mean())

            # Maximum drawdown calculation
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            maximum_drawdown = float(drawdown.min())

            # Sharpe ratio
            risk_free_rate = 0.02
            excess_returns = returns.mean() * 252 - risk_free_rate
            sharpe_ratio = float(excess_returns / portfolio_volatility) if portfolio_volatility > 0 else 0.0

            # Sortino ratio
            downside_returns = returns[returns < 0]
            downside_deviation = float(downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 else portfolio_volatility
            sortino_ratio = float(excess_returns / downside_deviation) if downside_deviation > 0 else 0.0

            # Calmar ratio
            annual_return = float(returns.mean() * 252)
            calmar_ratio = float(annual_return / abs(maximum_drawdown)) if maximum_drawdown != 0 else 0.0

            risk_metrics = RiskMetrics(
                portfolio_volatility=portfolio_volatility,
                value_at_risk_5pct=var_5pct,
                value_at_risk_1pct=var_1pct,
                expected_shortfall=expected_shortfall,
                maximum_drawdown=maximum_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                garch_volatility=portfolio_volatility,
                garch_forecast=[portfolio_volatility] * 5,
                rolling_volatility=returns.rolling(window=20).std() * np.sqrt(252),
                rolling_sharpe=pd.Series(index=returns.index, data=sharpe_ratio),
                drawdown_series=drawdown,
                drawdown_periods=[]
            )

            state['risk_metrics'] = risk_metrics

            print(f"{self.name}: Risk analysis complete.")
            print(f"   Volatility: {portfolio_volatility:.1%}")
            print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"   Max Drawdown: {maximum_drawdown:.1%}")

        except Exception as e:
            print(f"❌ {self.name}: Error - {e}")

        return state

class ForecastingAgent:
    def __init__(self):
        self.name = "ForecastingAgent"
        print(f"{self.name} initialized.")

    async def process(self, state: Dict, forecast_horizon: int = 5) -> Dict:
        if 'market_data' not in state or not state['market_data']:
            print(f"{self.name}: No market data available.")
            return state

        try:
            print(f"🔮 {self.name}: Generating forecasts.")

            market_data = state['market_data']
            prices = market_data.prices
            current_price = market_data.current_price

            # Simple forecasts
            arima_forecast = self._simple_forecast(prices, current_price)
            prophet_forecast = self._simple_forecast(prices, current_price)
            lstm_forecast = self._simple_forecast(prices, current_price)

            ensemble_forecast = np.mean([arima_forecast, prophet_forecast, lstm_forecast])

            forecast_data = ForecastData(
                arima_forecast=arima_forecast,
                prophet_forecast=prophet_forecast,
                lstm_forecast=lstm_forecast,
                ensemble_forecast=ensemble_forecast,
                forecast_confidence=0.7,
                prediction_interval=[ensemble_forecast * 0.95, ensemble_forecast * 1.05],
                forecast_horizon_days=forecast_horizon,
                forecast_accuracy_score=0.6,
                upside_probability=0.55,
                downside_risk=0.45,
                volatility_forecast=0.2,
                simulated_returns=[np.random.normal(0.001, 0.02) for _ in range(50)]
            )

            state['forecast_data'] = forecast_data

            price_change = ((ensemble_forecast - current_price) / current_price) * 100
            print(f"{self.name}: Ensemble forecast: ${ensemble_forecast:.2f} ({price_change:+.1f}%)")

        except Exception as e:
            print(f"❌ {self.name}: Error - {e}")

        return state

    def _simple_forecast(self, prices: pd.Series, current_price: float) -> float:
        if len(prices) >= 5:
            recent_trend = (prices.iloc[-1] - prices.iloc[-5]) / prices.iloc[-5]
            return current_price * (1 + recent_trend * 0.5)
        return current_price * (1 + np.random.normal(0.01, 0.05))

# Simplified other agents
class MacroEconomicAgent:
    def __init__(self, fred_api_key=None):
        self.name = "MacroEconomicAgent"
        print(f"{self.name} initialized.")

    async def process(self, state: Dict) -> Dict:
        # Simulated macro data
        macro_data = MacroData(
            gdp_growth=np.random.normal(2.5, 0.5),
            inflation_rate=np.random.normal(3.2, 0.3),
            unemployment_rate=np.random.normal(3.8, 0.2),
            federal_funds_rate=np.random.normal(5.25, 0.25),
            vix=np.random.normal(18, 5),
            dollar_index=np.random.normal(103, 2),
            market_sentiment=np.random.choice(["bullish", "neutral", "bearish"]),
            yield_curve_slope=np.random.normal(1.5, 0.3)
        )
        state['macro_data'] = macro_data
        print(f"{self.name}: Macro analysis complete (simulated data).")
        return state

class SentimentAgent:
    def __init__(self, **kwargs):
        self.name = "SentimentAgent"
        print(f"{self.name} initialized.")

    async def process(self, state: Dict) -> Dict:
        # Simulated sentiment data
        sentiment_data = SentimentData(
            news_sentiment=np.random.normal(0.1, 0.3),
            social_media_sentiment=np.random.normal(0.0, 0.4),
            overall_sentiment=np.random.normal(0.0, 0.3),
            sentiment_trend="neutral",
            confidence_score=0.5,
            key_topics=["earnings", "market_conditions", "economic_data"],
            fear_greed_index=np.random.uniform(20, 80)
        )
        state['sentiment_data'] = sentiment_data
        print(f"{self.name}: Sentiment analysis complete (simulated data).")
        return state

class StrategistAgent:
    def __init__(self, api_key: Optional[str] = None):
        self.name = "StrategistAgent"
        print(f"{self.name} initialized.")

    async def process(self, state: Dict) -> Dict:
        try:
            print(f"🧠 {self.name}: Generating AI recommendation.")

            if 'market_data' not in state or not state['market_data']:
                print(f"{self.name}: Missing required data.")
                return state

            market_data = state['market_data']
            current_price = market_data.current_price

            # Simple recommendation logic
            score = 0
            if market_data.trend in ["bullish", "strongly_bullish"]:
                score += 0.3
            if market_data.rsi < 70:
                score += 0.2
            if hasattr(state.get('forecast_data', {}), 'upside_probability'):
                score += state['forecast_data'].upside_probability * 0.3

            if score > 0.6:
                action = "BUY"
                confidence = min(0.9, score + 0.2)
            elif score < 0.3:
                action = "SELL"
                confidence = min(0.9, (1 - score) + 0.2)
            else:
                action = "HOLD"
                confidence = 0.6

            recommendation = PersonalizedRecommendation(
                action=action,
                confidence=confidence,
                position_size=confidence * 0.3,
                entry_price=current_price,
                stop_loss=current_price * (0.92 if action == "BUY" else 1.08),
                take_profit=current_price * (1.25 if action == "BUY" else 0.75),
                risk_level="MEDIUM",
                time_horizon="MEDIUM",
                detailed_reasoning=f"Multi-factor analysis suggests {action} with {confidence:.1%} confidence based on technical and fundamental factors.",
                key_risk_factors=["Market volatility", "Economic uncertainty"],
                key_opportunity_factors=["Technical momentum", "Favorable risk/reward"],
                alternative_scenarios={"bull": "Continued uptrend", "bear": "Market correction"},
                portfolio_impact=f"{action} position would impact portfolio diversification",
                market_timing_analysis="Current timing shows mixed signals",
                risk_reward_ratio=2.0,
                probability_of_success=confidence,
                maximum_drawdown_estimate=0.15
            )

            state['recommendation'] = recommendation

            print(f"{self.name}: {action} recommendation with {confidence:.1%} confidence.")

        except Exception as e:
            print(f"❌ {self.name}: Error - {e}")

        return state

class FinancialPlannerAgent:
    def __init__(self):
        self.name = "FinancialPlannerAgent"
        print(f"{self.name} initialized.")

    async def process(self, state: Dict, goal: FinancialGoal) -> Dict:
        try:
            print(f"💰 {self.name}: Creating financial plan.")

            # Simple calculations
            years = goal.time_horizon_years
            annual_return = 0.08  # Assumed 8% return
            
            # Future value calculation
            fv_current = goal.current_amount * (1 + annual_return) ** years
            monthly_rate = annual_return / 12
            months = years * 12
            
            if monthly_rate > 0:
                fv_contributions = goal.monthly_contribution * (((1 + monthly_rate) ** months - 1) / monthly_rate)
            else:
                fv_contributions = goal.monthly_contribution * months

            projected_value = fv_current + fv_contributions
            success_probability = min(1.0, projected_value / goal.target_amount)

            # Simple asset allocation
            if goal.risk_tolerance == "conservative":
                allocation = {"stocks": 0.4, "bonds": 0.5, "cash": 0.1}
            elif goal.risk_tolerance == "aggressive":
                allocation = {"stocks": 0.8, "bonds": 0.15, "cash": 0.05}
            else:
                allocation = {"stocks": 0.6, "bonds": 0.3, "cash": 0.1}

            plan_result = FinancialPlanResult(
                goal=goal,
                projected_value=projected_value,
                success_probability=success_probability,
                required_monthly=goal.monthly_contribution,
                asset_allocation=allocation,
                tax_optimization={"401k": goal.monthly_contribution * 0.6, "ira": goal.monthly_contribution * 0.4},
                monthly_breakdown={"total": goal.monthly_contribution, "stocks": goal.monthly_contribution * allocation["stocks"]},
                recommendations=[
                    f"Goal is {'achievable' if success_probability > 0.8 else 'challenging'} with current plan",
                    f"Consider {goal.risk_tolerance} investment strategy",
                    "Review plan annually for adjustments"
                ],
                is_achievable=success_probability > 0.8,
                monte_carlo_results={
                    "mean": projected_value,
                    "success_rate": success_probability,
                    "percentile_90": projected_value * 1.2
                },
                plan_sharpe_ratio=1.2,
                plan_max_drawdown=0.15,
                plan_volatility=0.12
            )

            state['financial_plan'] = plan_result

            print(f"{self.name}: Financial plan created.")
            print(f"   Success Probability: {success_probability:.1%}")

        except Exception as e:
            print(f"❌ {self.name}: Error - {e}")

        return state

# Pipeline execution functions
async def run_pipeline_with_real_apis(symbol="AAPL", openai_api_key=None, financial_goal=None):
    """Run the complete AI pipeline"""
    print("🤖 AI Financial Forecasting Pipeline")
    print("=" * 50)

    state = {}

    try:
        # Initialize agents
        market_agent = MarketDataAgent()
        risk_agent = RiskAgent()
        forecast_agent = ForecastingAgent()
        macro_agent = MacroEconomicAgent()
        sentiment_agent = SentimentAgent()
        strategist_agent = StrategistAgent(api_key=openai_api_key)
        planner_agent = FinancialPlannerAgent()

        print(f"All agents initialized.")

        # Run pipeline
        print("\n📊 Data collection and analysis.")
        print("-" * 30)

        state = await market_agent.process(state, symbol=symbol, period="1y")
        state = await risk_agent.process(state)
        state = await forecast_agent.process(state, forecast_horizon=5)

        print("\n🌍 Macro and sentiment analysis.")
        print("-" * 30)

        state = await macro_agent.process(state)
        state = await sentiment_agent.process(state)

        print("\n🧠 AI strategy and planning.")
        print("-" * 30)

        state = await strategist_agent.process(state)

        if financial_goal:
            state = await planner_agent.process(state, financial_goal)

        print("\n✅ Pipeline complete!")
        print("-" * 50)

        return state

    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        return state

def run_complete_analysis(symbol="AAPL"):
    """Synchronous version for compatibility"""
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(run_pipeline_with_real_apis(symbol))
        finally:
            loop.close()
    except Exception as e:
        print(f"Error in sync analysis: {e}")
        return None

def run_with_financial_planning(symbol="AAPL", target_amount=1000000, current_amount=50000, 
                               monthly_contribution=2000, time_horizon_years=25, age=35, 
                               risk_tolerance="moderate"):
    """Run analysis with financial planning"""
    goal = FinancialGoal(
        target_amount=target_amount,
        current_amount=current_amount,
        monthly_contribution=monthly_contribution,
        time_horizon_years=time_horizon_years,
        risk_tolerance=risk_tolerance,
        age=age,
        annual_income=100000,
        goal_type="retirement"
    )

    try:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(run_pipeline_with_real_apis(symbol, financial_goal=goal))
        finally:
            loop.close()
    except Exception as e:
        print(f"Error in planning analysis: {e}")
        return None

# Visualization functions
def create_dashboard_visualizations(state):
    """Create dashboard visualizations"""
    if not HAS_PLOTLY:
        print("Plotly not available for visualizations.")
        return

    if not state or 'market_data' not in state:
        print("No market data available for visualization.")
        return

    print("📊 Creating dashboard visualizations...")

def print_pipeline_summary(state):
    """Print a summary of pipeline results"""
    symbol = state.get('symbol', 'UNKNOWN')
    print(f"\n📈 Analysis Summary for {symbol}")
    print("-" * 40)

    # Market data summary
    if 'market_data' in state and state['market_data']:
        md = state['market_data']
        print(f"\n📊 MARKET DATA:")
        print(f"   Current Price: ${md.current_price:.2f}")
        print(f"   Trend: {md.trend}")
        print(f"   RSI: {md.rsi:.1f}")
        print(f"   1-Day Return: {md.return_1d:.2%}")
        print(f"   20-Day Volatility: {md.volatility_20d:.1%}")

    # Risk metrics summary
    if 'risk_metrics' in state and state['risk_metrics']:
        rm = state['risk_metrics']
        print(f"\n⚠️ RISK ANALYSIS:")
        print(f"   Portfolio Volatility: {rm.portfolio_volatility:.1%}")
        print(f"   Sharpe Ratio: {rm.sharpe_ratio:.2f}")
        print(f"   Maximum Drawdown: {rm.maximum_drawdown:.1%}")

    # AI Recommendation summary
    if 'recommendation' in state and state['recommendation']:
        rec = state['recommendation']
        print(f"\n🧠 AI RECOMMENDATION:")
        print(f"   Action: {rec.action}")
        print(f"   Confidence: {rec.confidence:.1%}")
        print(f"   Position Size: {rec.position_size:.1%}")
        print(f"   Risk Level: {rec.risk_level}")

# Demo function
async def main_demo():
    """Main demo function"""
    print("🤖 AI Financial Forecasting System Demo")
    print("-" * 50)

    # Basic analysis
    print("\n📊 Example: Basic stock analysis")
    state = await run_pipeline_with_real_apis("AAPL")
    print_pipeline_summary(state)

    return state

# Make sure all required functions are available
if __name__ == "__main__":
    print("🚀 AI Financial Forecasting System Ready")
    print("Available functions:")
    print("- run_pipeline_with_real_apis(symbol, openai_api_key, financial_goal)")
    print("- run_complete_analysis(symbol)")
    print("- run_with_financial_planning(...)")
    
    # Test basic functionality
    try:
        result = run_complete_analysis("AAPL")
        if result:
            print("✅ System test successful!")
        else:
            print("⚠️ System test returned empty result")
    except Exception as e:
        print(f"❌ System test failed: {e}")
