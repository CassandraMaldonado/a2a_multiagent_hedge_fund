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

#!pip install fredapi newsapi-python textblob praw nest-asyncio -q

try:
    from google.colab import files as _colab_files, drive as _colab_drive  # type: ignore
    IN_COLAB = True
except Exception:
    _colab_files = None
    _colab_drive = None
    IN_COLAB = False

files = _colab_files
drive = _colab_drive

import asyncio
import praw
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from google.colab import userdata
import requests
from fredapi import Fred
from newsapi.newsapi_client import NewsApiClient  # News API
import nest_asyncio
from textblob import TextBlob
import tweepy
import warnings
import os
import json
import re
warnings.filterwarnings('ignore')

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

try:
    FRED_API_KEY = userdata.get('FRED_API_KEY')
    NEWS_API_KEY = userdata.get('NEWS_API_KEY')
    OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')

    # Reddit API.
    REDDIT_CLIENT_ID = userdata.get('REDDIT_CLIENT_ID')
    REDDIT_CLIENT_SECRET = userdata.get('REDDIT_CLIENT_SECRET')
    REDDIT_USER_AGENT = userdata.get('REDDIT_USER_AGENT')

    print("API keys loaded from Colab secrets.")
except Exception as e:
    print(f"Error loading API keys: {e}")
    FRED_API_KEY = None
    NEWS_API_KEY = None
    OPENAI_API_KEY = None
    REDDIT_CLIENT_ID = None
    REDDIT_CLIENT_SECRET = None
    REDDIT_USER_AGENT = None

# Data structures.

@dataclass
class MarketData:
    symbol: str
    current_price: float
    prices: pd.Series
    volume: pd.Series = field(default_factory=pd.Series)

    # Technical indicators.
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

    # Forecast metrics.
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

# Risk analysis results.
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

    # GARCH results.
    garch_volatility: float = 0.0
    garch_forecast: List[float] = field(default_factory=list)

    # Rolling metrics.
    rolling_volatility: pd.Series = field(default_factory=pd.Series)
    rolling_sharpe: pd.Series = field(default_factory=pd.Series)

    # Drawdown analysis.
    drawdown_series: pd.Series = field(default_factory=pd.Series)
    drawdown_periods: List[Dict] = field(default_factory=list)

    last_updated: datetime = field(default_factory=datetime.now)

# Recommendation with GPT reasoning.
@dataclass
class PersonalizedRecommendation:
    action: str  #buy, sell or hold.
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

    # Quant metrics.
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
    risk_tolerance: str  # conservative, moderate or aggressive.
    age: int = 30
    annual_income: float = 100000


    goal_type: str = "retirement"  # retirement,house,education or general.
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

    # Risk metrics.
    plan_sharpe_ratio: float = 0.0
    plan_max_drawdown: float = 0.0
    plan_volatility: float = 0.0

    last_updated: datetime = field(default_factory=datetime.now)

# Market data agent
class MarketDataAgent:

    def __init__(self):
        self.name = "MarketDataAgent"
        print(f"{self.name} initialized.")

# Fetches and processes the market data with technical analysis.
    async def process(self, state: Dict, symbol: str = "AAPL", period: str = "1y") -> Dict:

        try:
            print(f"📊 {self.name}: Fetching market data for {symbol}...")

            # Fetching data from Yahoo Finance.
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)

            if data.empty:
                raise ValueError(f"No data available for {symbol}")

            # Calculating metrics.
            current_price = float(data['Close'].iloc[-1])
            prices = data['Close']
            volume = data['Volume']

            # Basic returns and volatility.
            returns = prices.pct_change()
            return_1d = float(returns.iloc[-1]) if len(returns) > 0 else 0.0
            return_5d = float(returns.tail(5).mean()) if len(returns) >= 5 else 0.0
            return_20d = float(returns.tail(20).mean()) if len(returns) >= 20 else 0.0
            volatility_20d = float(returns.tail(20).std() * np.sqrt(252)) if len(returns) >= 20 else 0.0

            # Technical indicators.
            rsi = self._calculate_rsi(prices)
            trend = self._analyze_trend(prices)
            macd_signal = self._calculate_macd_signal(prices)
            bollinger_position = self._calculate_bollinger_position(prices)
            volume_trend = self._analyze_volume_trend(volume)

            # Support and resistance levels.
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
            print(f"X {self.name}: Error - {e}")

        return state

# RSI indicator.
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0

 # Analyzing price trends using multiple moving averages.
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

 # Calculating MACD signal.
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

# Calculating the position relative to Bollinger Bands.
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

# Risk analysis agent with GARCH and portfolio risk metrics.
class RiskAgent:

    def __init__(self):
        self.name = "RiskAgent"
        print(f"{self.name} initialized.")

# Risk metrics including GARCH and Sharpe ratio.
    async def process(self, state: Dict) -> Dict:

        try:
            print(f"{self.name}: Computing portfolio risk metrics.")

            if 'market_data' not in state or not state['market_data']:
                print(f"{self.name}: No market data available.")
                return state

            market_data = state['market_data']
            prices = market_data.prices

            if len(prices) < 30:
                print(f"{self.name}: Insufficient data for risk analysis.")
                return state

            # Calculate returns.
            returns = prices.pct_change().dropna()

            # Basic risk metrics.
            portfolio_volatility = float(returns.std() * np.sqrt(252))  # Annualized

            # Value at Risk, historical method.
            var_5pct = float(np.percentile(returns, 5))
            var_1pct = float(np.percentile(returns, 1))

            # Expected shortfall, Conditional VAR.
            expected_shortfall = float(returns[returns <= var_5pct].mean())

            # Maximum drawdown calculation.
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            maximum_drawdown = float(drawdown.min())

            # Sharpe ratio with a 2% risk-free rate.
            risk_free_rate = 0.02
            excess_returns = returns.mean() * 252 - risk_free_rate
            sharpe_ratio = float(excess_returns / portfolio_volatility) if portfolio_volatility > 0 else 0.0

            # Sortino ratio.
            downside_returns = returns[returns < 0]
            downside_deviation = float(downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 else portfolio_volatility
            sortino_ratio = float(excess_returns / downside_deviation) if downside_deviation > 0 else 0.0

            # Calmar ratio, return vs max drawdown.
            annual_return = float(returns.mean() * 252)
            calmar_ratio = float(annual_return / abs(maximum_drawdown)) if maximum_drawdown != 0 else 0.0

            # GARCH volatility.
            garch_volatility = self._calculate_garch_volatility(returns)
            garch_forecast = self._forecast_garch_volatility(returns, horizon=5)

            # Rolling metrics.
            rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
            rolling_sharpe = self._calculate_rolling_sharpe(returns, window=60)

            # Detailed drawdown analysis.
            drawdown_periods = self._analyze_drawdown_periods(drawdown)

            risk_metrics = RiskMetrics(
                portfolio_volatility=portfolio_volatility,
                value_at_risk_5pct=var_5pct,
                value_at_risk_1pct=var_1pct,
                expected_shortfall=expected_shortfall,
                maximum_drawdown=maximum_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                garch_volatility=garch_volatility,
                garch_forecast=garch_forecast,
                rolling_volatility=rolling_vol,
                rolling_sharpe=rolling_sharpe,
                drawdown_series=drawdown,
                drawdown_periods=drawdown_periods
            )

            state['risk_metrics'] = risk_metrics

            print(f"{self.name}: Risk analysis complete.")
            print(f"   Volatility: {portfolio_volatility:.1%}")
            print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"   Max Drawdown: {maximum_drawdown:.1%}")
            print(f"   VaR (5%): {var_5pct:.1%}")
            print(f"   GARCH Vol: {garch_volatility:.1%}")

        except Exception as e:
            print(f"X {self.name}: Error - {e}")

        return state

# Simplified GARCH(1,1) volatility calculation.
    def _calculate_garch_volatility(self, returns: pd.Series) -> float:
        try:
            # EWMA as GARCH approximation.
            lambda_param = 0.94

            # Exponentially weighted variance.
            squared_returns = returns ** 2
            ewma_variance = squared_returns.ewm(alpha=1-lambda_param).mean()

            # Returning annualized volatility.
            return float(np.sqrt(ewma_variance.iloc[-1] * 252))

        except Exception:
            # Fallback to rolling volatility.
            return float(returns.rolling(window=20).std().iloc[-1] * np.sqrt(252))

# Forecast GARCH volatility for next periods.
    def _forecast_garch_volatility(self, returns: pd.Series, horizon: int = 5) -> List[float]:
        try:
            current_vol = self._calculate_garch_volatility(returns)
            long_term_vol = float(returns.std() * np.sqrt(252))

            # Simple mean reversion forecast.
            forecasts = []
            decay_rate = 0.05

            for i in range(horizon):
                # Mean revert to long-term volatility.
                forecast_vol = current_vol * (1 - decay_rate * i) + long_term_vol * (decay_rate * i)
                forecasts.append(forecast_vol)

            return forecasts

        except Exception:
            # Fallback to constant volatility.
            vol = float(returns.std() * np.sqrt(252))
            return [vol] * horizon

# Calculating rolling Sharpe ratio.
    def _calculate_rolling_sharpe(self, returns: pd.Series, window: int = 60) -> pd.Series:
        try:
            risk_free_rate = 0.02 / 252

            rolling_mean = returns.rolling(window=window).mean()
            rolling_std = returns.rolling(window=window).std()

            rolling_sharpe = (rolling_mean - risk_free_rate) / rolling_std * np.sqrt(252)

            return rolling_sharpe

        except Exception:
            return pd.Series(index=returns.index, data=0.0)

    def _analyze_drawdown_periods(self, drawdown: pd.Series) -> List[Dict]:
        try:
            drawdown_periods = []

            # Finding drawdown periods.
            in_drawdown = drawdown < -0.01

            if not in_drawdown.any():
                return drawdown_periods

            # Finding the start and end of the drawdown periods.
            drawdown_changes = in_drawdown.diff()
            starts = drawdown_changes[drawdown_changes == True].index
            ends = drawdown_changes[drawdown_changes == False].index

            # Handling edge cases.
            if in_drawdown.iloc[0]:
                starts = [drawdown.index[0]] + list(starts)
            if in_drawdown.iloc[-1]:
                ends = list(ends) + [drawdown.index[-1]]

            # Analyzing each period.
            for start, end in zip(starts, ends):
                period_drawdown = drawdown.loc[start:end]
                max_dd = period_drawdown.min()
                duration = len(period_drawdown)

                drawdown_periods.append({
                    'start_date': start,
                    'end_date': end,
                    'duration_days': duration,
                    'max_drawdown': float(max_dd),
                    'recovery_time': 0
                })

            return drawdown_periods[:5]

        except Exception:
            return []

# Forecasting Agent with simulated returns for risk analysis.

class ForecastingAgent:

    def __init__(self):
        self.name = "ForecastingAgent"
        print(f"{self.name} initialized.")

# Generates price forecasts with simulated return sequences.
    async def process(self, state: Dict, forecast_horizon: int = 5) -> Dict:

        if 'market_data' not in state or not state['market_data']:
            print(f"{self.name}: No market data available.")
            return state

        try:
            print(f"{self.name}: Generating forecasts.")

            market_data = state['market_data']
            prices = market_data.prices
            current_price = market_data.current_price

            # Generating multiple forecasts.
            arima_forecast = self._arima_forecast(prices, current_price)
            prophet_forecast = self._prophet_forecast(prices, forecast_horizon)
            lstm_forecast = self._lstm_forecast(prices, current_price)

            # Ensemble forecast.
            ensemble_forecast = np.mean([arima_forecast, prophet_forecast, lstm_forecast])

            # Calculating confidence.
            forecast_std = np.std([arima_forecast, prophet_forecast, lstm_forecast])
            volatility = market_data.volatility_20d
            confidence = max(0.3, min(0.9, 1.0 - (forecast_std / current_price) - volatility * 0.5))

            # Prediction intervals.
            lower_bound = ensemble_forecast - 1.96 * forecast_std
            upper_bound = ensemble_forecast + 1.96 * forecast_std

            # Generating simulated return sequence for risk analysis.
            simulated_returns = self._generate_simulated_returns(prices, forecast_horizon)

            # Calculating probabilities.
            expected_return = (ensemble_forecast - current_price) / current_price
            upside_probability = max(0.1, min(0.9, 0.5 + expected_return))
            downside_risk = 1.0 - upside_probability

            # Forecasting volatility.
            volatility_forecast = self._forecast_volatility(prices)

            # Calculating the accuracy score.
            trend_consistency = self._calculate_trend_consistency(prices)
            accuracy_score = confidence * trend_consistency

            forecast_data = ForecastData(
                arima_forecast=arima_forecast,
                prophet_forecast=prophet_forecast,
                lstm_forecast=lstm_forecast,
                ensemble_forecast=ensemble_forecast,
                forecast_confidence=confidence,
                prediction_interval=[lower_bound, upper_bound],
                forecast_horizon_days=forecast_horizon,
                forecast_accuracy_score=accuracy_score,
                upside_probability=upside_probability,
                downside_risk=downside_risk,
                volatility_forecast=volatility_forecast,
                simulated_returns=simulated_returns
            )

            state['forecast_data'] = forecast_data

            price_change = ((ensemble_forecast - current_price) / current_price) * 100
            print(f"{self.name}: Ensemble forecast: ${ensemble_forecast:.2f} ({price_change:+.1f}%)")
            print(f"   Confidence: {confidence:.1%}")
            print(f"   Upside Probability: {upside_probability:.1%}")

        except Exception as e:
            print(f"X {self.name}: Error - {e}")

        return state

# ARIMA forecast using moving averages.
    def _arima_forecast(self, prices: pd.Series, current_price: float) -> float:
        if len(prices) >= 20:
            ma_5 = prices.tail(5).mean()
            ma_20 = prices.tail(20).mean()
            trend_factor = (ma_5 - ma_20) / ma_20

            momentum = (prices.iloc[-1] - prices.iloc[-5]) / prices.iloc[-5]

            return current_price * (1 + trend_factor * 0.5 + momentum * 0.3)
        return current_price * 1.01

    def _prophet_forecast(self, prices: pd.Series, horizon: int) -> float:
        if HAS_PROPHET and len(prices) >= 30:
            try:
                df = pd.DataFrame({
                    'ds': prices.index,
                    'y': prices.values
                })

                model = Prophet(
                    daily_seasonality=True,
                    yearly_seasonality=False,
                    weekly_seasonality=True,
                    changepoint_prior_scale=0.05
                )
                model.fit(df)

                future = model.make_future_dataframe(periods=horizon)
                forecast = model.predict(future)

                return float(forecast['yhat'].iloc[-1])
            except:
                pass

        # Fallback method, seasonal adjustment.
        if len(prices) >= 7:
            # Simple weekly seasonality.
            day_of_week = len(prices) % 7
            seasonal_factor = 1.0 + np.random.normal(0, 0.01)
            return prices.iloc[-1] * seasonal_factor * (1 + np.random.normal(0.02, 0.05))

        return prices.iloc[-1] * (1 + np.random.normal(0.02, 0.05))

# LSTM forecast using exponential smoothing.
    def _lstm_forecast(self, prices: pd.Series, current_price: float) -> float:
        if len(prices) >= 20:
            # Using multiple lookback windows for LSTM simulation.
            weights_short = np.exp(np.linspace(-2, 0, 5))
            weights_short = weights_short / weights_short.sum()

            weights_long = np.exp(np.linspace(-3, 0, 15))
            weights_long = weights_long / weights_long.sum()

            short_term = np.sum(prices.tail(5) * weights_short)
            long_term = np.sum(prices.tail(15) * weights_long)

            # Combining with volatility adjustment.
            volatility = prices.pct_change().tail(20).std()
            noise = np.random.normal(0, volatility * 0.1)

            return (short_term * 0.7 + long_term * 0.3) * (1 + noise)

        return current_price * (1 + np.random.normal(0.02, 0.05))

    def _generate_simulated_returns(self, prices: pd.Series, horizon: int) -> List[float]:
        returns = prices.pct_change().dropna()

        if len(returns) < 20:
            mean_return = 0.001
            std_return = 0.02
        else:
            mean_return = returns.mean()
            std_return = returns.std()

        # Generating random returns for the forecast horizon.
        simulated_returns = np.random.normal(mean_return, std_return, horizon * 50).tolist()

        return simulated_returns

# Forecast future volatility using GARCH approach.
    def _forecast_volatility(self, prices: pd.Series) -> float:
        returns = prices.pct_change().dropna()
        if len(returns) < 20:
            return 0.2

        # Simple volatility forecasting.
        recent_vol = returns.tail(10).std() * np.sqrt(252)
        long_term_vol = returns.std() * np.sqrt(252)

        # Weighted average with more weight on recent.
        forecast_vol = 0.7 * recent_vol + 0.3 * long_term_vol

        return min(1.0, max(0.05, forecast_vol))

    def _calculate_trend_consistency(self, prices: pd.Series) -> float:
        if len(prices) < 10:
            return 0.5

        # R-squared of linear regression for trend strength.
        x = np.arange(len(prices.tail(20)))
        y = prices.tail(20).values

        try:
            correlation = np.corrcoef(x, y)[0, 1]
            r_squared = correlation ** 2
            return min(1.0, r_squared + 0.3)
        except:
            return 0.5

# Macro Economic agent.

class MacroEconomicAgent:

    def __init__(self, fred_api_key=None):
        self.name = "MacroEconomicAgent"
        self.fred_api_key = fred_api_key or FRED_API_KEY

        if self.fred_api_key:
            self.fred = Fred(api_key=self.fred_api_key)
            print(f"{self.name} initialized with FRED API.")
        else:
            self.fred = None
            print(f"{self.name} initialized without FRED API will use simulated data.")

# Macro econ factors using real FRED data.
    async def process(self, state: Dict) -> Dict:

        try:
            print(f"{self.name}: Fetching real macro-economic data from FRED.")

            if self.fred:
                try:
                    # GDP growth rate.
                    gdp_data = self.fred.get_series('GDP', limit=2)
                    gdp_growth = ((gdp_data.iloc[-1] / gdp_data.iloc[-2]) - 1) * 100

                    # Inflation rate.
                    cpi_data = self.fred.get_series('CPIAUCSL', limit=13)
                    inflation_rate = ((cpi_data.iloc[-1] / cpi_data.iloc[-13]) - 1) * 100

                    # Unemployment rate.
                    unemployment_rate = self.fred.get_series('UNRATE', limit=1).iloc[-1]

                    # Federal funds rate
                    federal_funds_rate = self.fred.get_series('FEDFUNDS', limit=1).iloc[-1]

                    # Volatility Index.
                    try:
                        vix = self.fred.get_series('VIXCLS', limit=1).iloc[-1]
                    except:
                        vix = 20.0

                    # Dollar Index.
                    try:
                        dollar_index = self.fred.get_series('DTWEXBGS', limit=1).iloc[-1]
                    except:
                        dollar_index = 100.0

                    # Yield curve.
                    try:
                        ten_year = self.fred.get_series('GS10', limit=1).iloc[-1]
                        two_year = self.fred.get_series('GS2', limit=1).iloc[-1]
                        yield_curve_slope = ten_year - two_year
                    except:
                        yield_curve_slope = 1.0

                    # Market sentiment based on economic indicators.
                    if inflation_rate < 3 and unemployment_rate < 5 and yield_curve_slope > 0:
                        market_sentiment = "bullish"
                    elif inflation_rate > 5 or unemployment_rate > 7 or yield_curve_slope < -0.5:
                        market_sentiment = "bearish"
                    else:
                        market_sentiment = "neutral"

                    print(f"{self.name}: Real FRED data retrieved.")

                except Exception as e:
                    print(f"FRED API error: {e}, using simulated data.")
                    return self._get_simulated_data(state)

            else:
                print(f"No FRED API key, using simulated data.")
                return self._get_simulated_data(state)

            macro_data = MacroData(
                gdp_growth=float(gdp_growth),
                inflation_rate=float(inflation_rate),
                unemployment_rate=float(unemployment_rate),
                federal_funds_rate=float(federal_funds_rate),
                vix=float(vix),
                dollar_index=float(dollar_index),
                market_sentiment=market_sentiment,
                yield_curve_slope=float(yield_curve_slope),
                credit_spreads=1.2,
                economic_surprise_index=0.0
            )

            state['macro_data'] = macro_data

            print(f"{self.name}: Analysis complete.")
            print(f"   GDP Growth: {gdp_growth:.1f}%")
            print(f"   Inflation: {inflation_rate:.1f}%")
            print(f"   Unemployment: {unemployment_rate:.1f}%")
            print(f"   Fed Funds Rate: {federal_funds_rate:.2f}%")
            print(f"   VIX: {vix:.1f}")
            print(f"   Market Sentiment: {market_sentiment}")

        except Exception as e:
            print(f"X {self.name}: Error - {e}")
            return self._get_simulated_data(state)

        return state

    def _get_simulated_data(self, state):
        macro_data = MacroData(
            gdp_growth=np.random.normal(2.5, 0.5),
            inflation_rate=np.random.normal(3.2, 0.3),
            unemployment_rate=np.random.normal(3.8, 0.2),
            federal_funds_rate=np.random.normal(5.25, 0.25),
            vix=np.random.normal(18, 5),
            dollar_index=np.random.normal(103, 2),
            market_sentiment=np.random.choice(["bullish", "neutral", "bearish"], p=[0.3, 0.4, 0.3]),
            yield_curve_slope=np.random.normal(1.5, 0.3),
            credit_spreads=np.random.normal(1.2, 0.2),
            economic_surprise_index=np.random.normal(0.0, 0.5)
        )
        state['macro_data'] = macro_data
        return state

# Sentiment agent.

class SentimentAgent:

    def __init__(self, news_api_key=None, reddit_client_id=None, reddit_client_secret=None, reddit_user_agent=None):
        self.name = "SentimentAgent"
        self.news_api_key = news_api_key or NEWS_API_KEY
        self.reddit_client_id = reddit_client_id or REDDIT_CLIENT_ID
        self.reddit_client_secret = reddit_client_secret or REDDIT_CLIENT_SECRET
        self.reddit_user_agent = reddit_user_agent or REDDIT_USER_AGENT

        # Initializing News API.
        if self.news_api_key:
            self.newsapi = NewsApiClient(api_key=self.news_api_key)
            print(f"{self.name} initialized with News API.")
        else:
            self.newsapi = None
            print(f"{self.name} initialized without News API.")

        # Initializing Reddit API.
        if all([self.reddit_client_id, self.reddit_client_secret, self.reddit_user_agent]):
            try:
                self.reddit = praw.Reddit(
                    client_id=self.reddit_client_id,
                    client_secret=self.reddit_client_secret,
                    user_agent=self.reddit_user_agent
                )
                self.reddit.user.me()
                print(f"{self.name} initialized with Reddit API.")
            except Exception as e:
                print(f"Reddit API initialization failed: {e}")
                self.reddit = None
        else:
            self.reddit = None
            print(f"{self.name} initialized without Reddit API.")

    async def process(self, state: Dict) -> Dict:

        try:
            print(f"{self.name}: Analyzing real market sentiment...")

            symbol = state.get('symbol', 'UNKNOWN')
            company_name = self._get_company_name(symbol)

            # News sentiment.
            news_sentiment = await self._analyze_news_sentiment(symbol, company_name)

            # Reddit sentiment.
            reddit_sentiment = await self._analyze_reddit_sentiment(symbol, company_name)

            # Calculating overall sentiment.
            if news_sentiment is not None and reddit_sentiment is not None:
                overall_sentiment = (news_sentiment * 0.6 + reddit_sentiment * 0.4)
                confidence_score = 0.8
            elif news_sentiment is not None:
                overall_sentiment = news_sentiment
                confidence_score = 0.6
            elif reddit_sentiment is not None:
                overall_sentiment = reddit_sentiment
                confidence_score = 0.5
            else:
                # Fallback to simulated data.
                overall_sentiment = np.random.normal(0.0, 0.3)
                news_sentiment = np.random.normal(0.1, 0.3)
                reddit_sentiment = np.random.normal(0.0, 0.4)
                confidence_score = 0.3
                print(f"⚠️ Using simulated sentiment data")

            # Determining sentiment trend.
            if overall_sentiment > 0.2:
                sentiment_trend = "positive"
            elif overall_sentiment < -0.2:
                sentiment_trend = "negative"
            else:
                sentiment_trend = "neutral"

            # Generating the key topics from news and Reddit.
            key_topics = await self._extract_key_topics(symbol, company_name)

            # Calculating the sentiment momentum.
            sentiment_momentum = np.random.normal(0.0, 0.1)  # Would need historical data

            # Fear & Greed Index calculation.
            fear_greed_index = self._calculate_fear_greed_index(overall_sentiment)

            # Analyst rating trend.
            analyst_rating_trend = self._determine_analyst_trend(overall_sentiment)

            sentiment_data = SentimentData(
                news_sentiment=news_sentiment or 0.0,
                social_media_sentiment=reddit_sentiment or 0.0,
                overall_sentiment=overall_sentiment,
                sentiment_trend=sentiment_trend,
                confidence_score=confidence_score,
                key_topics=key_topics,
                sentiment_momentum=sentiment_momentum,
                fear_greed_index=fear_greed_index,
                analyst_rating_trend=analyst_rating_trend
            )

            state['sentiment_data'] = sentiment_data

            print(f"{self.name}: Analysis complete.")
            print(f"   Overall Sentiment: {sentiment_trend} ({overall_sentiment:.2f})")
            print(f"   News Sentiment: {news_sentiment:.2f}" if news_sentiment else "   News Sentiment: N/A")
            print(f"   Reddit Sentiment: {reddit_sentiment:.2f}" if reddit_sentiment else "   Reddit Sentiment: N/A")
            print(f"   Confidence: {confidence_score:.1%}")
            print(f"   Key Topics: {', '.join(key_topics[:3])}")

        except Exception as e:
            print(f"X {self.name}: Error - {e}")
            # Fallback to simulated data.
            return self._get_simulated_sentiment(state)

        return state

    async def _analyze_news_sentiment(self, symbol: str, company_name: str) -> float:
        if not self.newsapi:
            return None

        try:
            # Searching for recent news articles.
            articles = self.newsapi.get_everything(
                q=f"{symbol} OR {company_name}",
                language='en',
                sort_by='publishedAt',
                page_size=20,
                from_param=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            )

            if not articles['articles']:
                return None

            sentiment_scores = []

            for article in articles['articles']:
                title = article.get('title', '')
                description = article.get('description', '')

                if title and description:
                    text = f"{title}. {description}"
                    blob = TextBlob(text)
                    sentiment_scores.append(blob.sentiment.polarity)

            return np.mean(sentiment_scores) if sentiment_scores else 0.0

        except Exception as e:
            print(f"News API error: {e}")
            return None

    async def _analyze_reddit_sentiment(self, symbol: str, company_name: str) -> float:
        if not self.reddit:
            return None

        try:
            sentiment_scores = []

            # Searching relevant subreddits.
            subreddits = ['stocks', 'investing', 'SecurityAnalysis', 'ValueInvesting', 'StockMarket', 'wallstreetbets']

            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)

                    for submission in subreddit.search(f"{symbol}", sort='new', limit=10, time_filter='week'):
                        # Analying post title and selftext.
                        text_content = f"{submission.title} {submission.selftext}"
                        if len(text_content.strip()) > 10:
                            blob = TextBlob(text_content)
                            sentiment_scores.append(blob.sentiment.polarity)

                        # Analyzing top comments.
                        submission.comments.replace_more(limit=0)
                        for comment in submission.comments[:5]:
                            if hasattr(comment, 'body') and len(comment.body) > 20:
                                if comment.body not in ['[deleted]', '[removed]']:
                                    blob = TextBlob(comment.body)
                                    sentiment_scores.append(blob.sentiment.polarity)

                except Exception as e:
                    print(f"⚠️ Error with subreddit {subreddit_name}: {e}")
                    continue

            try:
                company_subreddit_names = {
                    'AAPL': 'apple',
                    'TSLA': 'teslamotors',
                    'MSFT': 'microsoft',
                    'GOOGL': 'google',
                    'AMZN': 'amazon',
                    'META': 'facebook',
                    'NVDA': 'nvidia'
                }

                if symbol.upper() in company_subreddit_names:
                    company_sub = self.reddit.subreddit(company_subreddit_names[symbol.upper()])

                    for submission in company_sub.hot(limit=15):
                        text_content = f"{submission.title} {submission.selftext}"
                        if len(text_content.strip()) > 10:
                            blob = TextBlob(text_content)
                            sentiment_scores.append(blob.sentiment.polarity)

            except Exception as e:
                print(f"Error with company subreddit: {e}")

            return np.mean(sentiment_scores) if sentiment_scores else 0.0

        except Exception as e:
            print(f"Reddit API error: {e}")
            return None

    async def _extract_key_topics(self, symbol: str, company_name: str) -> List[str]:
        topics_from_news = []
        topics_from_reddit = []

        # Extracting topics from news.
        if self.newsapi:
            try:
                articles = self.newsapi.get_everything(
                    q=f"{symbol} OR {company_name}",
                    language='en',
                    sort_by='publishedAt',
                    page_size=10
                )

                all_text = ""
                for article in articles['articles']:
                    title = article.get('title', '')
                    description = article.get('description', '')
                    all_text += f" {title} {description}"

                topics_from_news = self._extract_topics_from_text(all_text)

            except Exception as e:
                print(f"Topic extraction from news error: {e}")

        # Extracting topics from Reddit.
        if self.reddit:
            try:
                reddit_text = ""
                subreddit = self.reddit.subreddit('stocks+investing+StockMarket')

                for submission in subreddit.search(f"{symbol}", limit=20, time_filter='week'):
                    reddit_text += f" {submission.title} {submission.selftext}"

                topics_from_reddit = self._extract_topics_from_text(reddit_text)

            except Exception as e:
                print(f"Topic extraction from Reddit error: {e}")

        all_topics = list(set(topics_from_news + topics_from_reddit))
        return all_topics[:5] if all_topics else ["market_conditions", "earnings"]

    def _extract_topics_from_text(self, text: str) -> List[str]:
        """Extract topics from text using keyword matching"""
        keywords = {
            'earnings': ['earnings', 'revenue', 'profit', 'eps', 'quarterly', 'guidance'],
            'product_launch': ['launch', 'product', 'release', 'announce', 'unveil', 'debut'],
            'regulation': ['regulation', 'regulatory', 'sec', 'compliance', 'lawsuit', 'legal'],
            'competition': ['competitor', 'competition', 'market_share', 'rival', 'compete'],
            'fed_policy': ['fed', 'interest_rate', 'monetary', 'policy', 'powell', 'fomc'],
            'geopolitical': ['trade', 'tariff', 'china', 'war', 'sanctions', 'politics'],
            'merger_acquisition': ['merger', 'acquisition', 'buyout', 'takeover', 'deal'],
            'technology': ['ai', 'artificial_intelligence', 'innovation', 'tech', 'patent'],
            'market_conditions': ['market', 'volatility', 'correction', 'rally', 'bull', 'bear'],
            'economic_data': ['gdp', 'inflation', 'unemployment', 'jobs', 'economic', 'economy']
        }

        found_topics = []
        text_lower = text.lower()

        for topic, words in keywords.items():
            if any(word in text_lower for word in words):
                found_topics.append(topic)

        return found_topics

    def _get_company_name(self, symbol: str) -> str:
        company_map = {
            'AAPL': 'Apple',
            'TSLA': 'Tesla',
            'MSFT': 'Microsoft',
            'GOOGL': 'Google',
            'AMZN': 'Amazon',
            'META': 'Meta',
            'NVDA': 'Nvidia',
            'NFLX': 'Netflix',
            'AMD': 'AMD',
            'INTC': 'Intel'
        }
        return company_map.get(symbol.upper(), symbol)

# Calculating Fear & Greed Index based on sentiment.
    def _calculate_fear_greed_index(self, sentiment: float) -> float:
        base_index = 50  # Neutral
        sentiment_adjustment = sentiment * 30
        return max(0, min(100, base_index + sentiment_adjustment))

# Determining analyst rating trend based on sentiment.
    def _determine_analyst_trend(self, sentiment: float) -> str:
        if sentiment > 0.3:
            return "upgrade"
        elif sentiment < -0.3:
            return "downgrade"
        else:
            return "neutral"

    def _get_simulated_sentiment(self, state):
        sentiment_data = SentimentData(
            news_sentiment=np.random.normal(0.1, 0.3),
            social_media_sentiment=np.random.normal(0.0, 0.4),
            overall_sentiment=np.random.normal(0.0, 0.3),
            sentiment_trend="neutral",
            confidence_score=0.3,
            key_topics=["market_conditions", "earnings", "economic_data"],
            sentiment_momentum=np.random.normal(0.0, 0.2),
            fear_greed_index=np.random.uniform(20, 80),
            analyst_rating_trend="neutral"
        )
        state['sentiment_data'] = sentiment_data
        return state

# Strategist agent.

class StrategistAgent:

    def __init__(self, api_key: Optional[str] = None):
        self.name = "StrategistAgent"
        self.client = None
        self.has_openai = False

        if api_key and HAS_OPENAI:
            try:
                self.client = OpenAI(api_key=api_key)
                self.has_openai = True
                print(f"{self.name} initialized with GPT-4.")
            except Exception as e:
                print(f"GPT initialization failed: {e}")
                self.has_openai = False
        else:
            print(f"{self.name} initialized with rule-based logic.")

# Generating personalized recommendation using all agent data.
    async def process(self, state: Dict) -> Dict:

        try:
            print(f"{self.name}: Generating AI recommendation.")

            # Verifing we have the required data from other agents.
            required_data = ['market_data']
            missing_data = [key for key in required_data if key not in state or not state[key]]

            if missing_data:
                print(f"{self.name}: Missing required data: {missing_data}")
                return state

            # Generating a recommendation using GPT-4 or fallback logic.
            if self.has_openai and self.client:
                try:
                    recommendation = await self._generate_gpt_recommendation(state)
                    print(f"{self.name}: GPT-4 recommendation generated.")
                except Exception as e:
                    print(f"GPT-4 failed, using enhanced fallback: {e}")
                    recommendation = self._generate_enhanced_recommendation(state)
            else:
                recommendation = self._generate_enhanced_recommendation(state)

            state['recommendation'] = recommendation

            print(f"{self.name}: {recommendation.action} recommendation.")
            print(f"   Confidence: {recommendation.confidence:.1%}")
            print(f"   Position Size: {recommendation.position_size:.1%}")
            print(f"   Risk Level: {recommendation.risk_level}")
            print(f"   Risk/Reward: {recommendation.risk_reward_ratio:.2f}")

        except Exception as e:
            print(f"X {self.name}: Error - {e}")

        return state

# Generating recommendation using enhanced rule-based logic with all agent data.
    def _generate_enhanced_recommendation(self, state: Dict) -> PersonalizedRecommendation:

        # Get current price.
        current_price = 100.0
        if 'market_data' in state and state['market_data']:
            current_price = state['market_data'].current_price

        # Comprehensive scores using all available agent data.
        technical_score = self._calculate_technical_score(state)
        forecast_score = self._calculate_forecast_score(state)
        risk_score = self._calculate_risk_score(state)
        macro_score = self._calculate_macro_score(state)
        sentiment_score = self._calculate_sentiment_score(state)

        # Weighted overall score.
        overall_score = (
            technical_score * 0.25 +
            forecast_score * 0.25 +
            risk_score * 0.20 +
            macro_score * 0.15 +
            sentiment_score * 0.15
        )

        if overall_score > 0.6:
            action = "BUY"
            confidence = min(0.95, 0.6 + overall_score * 0.3)
            position_size = min(0.4, confidence * 0.5)
            risk_level = "LOW" if confidence > 0.8 else "MEDIUM"
            time_horizon = "MEDIUM"
        elif overall_score < -0.6:
            action = "SELL"
            confidence = min(0.95, 0.6 + abs(overall_score) * 0.3)
            position_size = min(0.3, confidence * 0.4)
            risk_level = "MEDIUM" if confidence > 0.7 else "HIGH"
            time_horizon = "SHORT"
        else:
            action = "HOLD"
            confidence = 0.6 + abs(overall_score) * 0.2
            position_size = 0.0
            risk_level = "LOW"
            time_horizon = "LONG"

        detailed_reasoning = self._generate_detailed_reasoning(state, overall_score)

        try:
            risk_factors, opportunity_factors = self._identify_risk_opportunity_factors(state)
        except:
            risk_factors = ["High volatility environment", "Technical overbought conditions"]
            opportunity_factors = ["Strong trend momentum", "Forecast confidence"]

        try:
            alternative_scenarios = self._generate_alternative_scenarios(state)
        except:
            alternative_scenarios = {
                "bull_case": "Technical momentum continues with improving fundamentals",
                "bear_case": "Risk factors materialize leading to correction",
                "base_case": "Mixed signals result in sideways price action"
            }

        try:
            portfolio_impact = self._analyze_portfolio_impact(state, action, position_size)
        except:
            portfolio_impact = f"{action} position of {position_size:.1%} would impact portfolio risk profile"

        try:
            market_timing_analysis = self._analyze_market_timing(state)
        except:
            market_timing_analysis = "Mixed timing signals suggest cautious approach"

        # Calculating the risk metrics.
        risk_reward_ratio = self._calculate_risk_reward_ratio(current_price, action, state)
        probability_of_success = min(0.9, confidence + 0.1)
        max_drawdown_estimate = self._estimate_maximum_drawdown(state)

        return PersonalizedRecommendation(
            action=action,
            confidence=confidence,
            position_size=position_size,
            entry_price=current_price,
            stop_loss=current_price * (0.92 if action == "BUY" else 1.08),
            take_profit=current_price * (1.25 if action == "BUY" else 0.75),
            risk_level=risk_level,
            time_horizon=time_horizon,
            detailed_reasoning=detailed_reasoning,
            key_risk_factors=risk_factors,
            key_opportunity_factors=opportunity_factors,
            alternative_scenarios=alternative_scenarios,
            portfolio_impact=portfolio_impact,
            market_timing_analysis=market_timing_analysis,
            risk_reward_ratio=risk_reward_ratio,
            probability_of_success=probability_of_success,
            maximum_drawdown_estimate=max_drawdown_estimate
        )


# Calculating technical analysis score from market data.
    def _calculate_technical_score(self, state: Dict) -> float:
        if 'market_data' not in state or not state['market_data']:
            return 0.0

        md = state['market_data']
        score = 0.0

        # Trend analysis.
        trend_scores = {
            "strongly_bullish": 1.0, "bullish": 0.5, "neutral": 0.0,
            "bearish": -0.5, "strongly_bearish": -1.0
        }
        score += trend_scores.get(md.trend, 0.0) * 0.4

        # RSI analysis.
        if md.rsi < 30:
            score += 0.3  # Oversold = bullish.
        elif md.rsi > 70:
            score -= 0.3  # Overbought = bearish.
        elif 40 <= md.rsi <= 60:
            score += 0.1  # Neutral zone = slightly positive.

        # MACD signal.
        macd_scores = {"bullish": 0.2, "bearish": -0.2, "neutral": 0.0}
        score += macd_scores.get(md.macd_signal, 0.0)

        # Volume trend.
        volume_scores = {"increasing": 0.1, "stable": 0.0, "decreasing": -0.05}
        score += volume_scores.get(md.volume_trend, 0.0)

        return max(-1.0, min(1.0, score))

    def _calculate_forecast_score(self, state: Dict) -> float:
        if 'forecast_data' not in state or not state['forecast_data']:
            return 0.0

        fd = state['forecast_data']
        current_price = state.get('market_data', {}).current_price if 'market_data' in state else 100

        # Expected price change.
        price_change = (fd.ensemble_forecast - current_price) / current_price
        confidence_weighted_change = price_change * fd.forecast_confidence
        upside_bonus = (fd.upside_probability - 0.5) * 0.5
        volatility_penalty = -fd.volatility_forecast * 0.2

        score = confidence_weighted_change + upside_bonus + volatility_penalty
        return max(-1.0, min(1.0, score * 2))

# Calculating risk-adjusted score from risk metrics.
    def _calculate_risk_score(self, state: Dict) -> float:
        if 'risk_metrics' not in state or not state['risk_metrics']:
            return 0.0

        rm = state['risk_metrics']
        score = 0.0

        # Sharpe ratio component.
        if rm.sharpe_ratio > 1.5:
            score += 0.5
        elif rm.sharpe_ratio > 1.0:
            score += 0.25
        elif rm.sharpe_ratio < 0:
            score -= 0.5

        # Volatility component.
        if rm.portfolio_volatility > 0.4:
            score -= 0.3
        elif rm.portfolio_volatility < 0.15:
            score += 0.15

        # Drawdown component.
        if rm.maximum_drawdown < -0.3:
            score -= 0.2
        elif rm.maximum_drawdown > -0.1:
            score += 0.1

        return max(-1.0, min(1.0, score))

# Calculating macroecon score.
    def _calculate_macro_score(self, state: Dict) -> float:
        if 'macro_data' not in state or not state['macro_data']:
            return 0.0

        md = state['macro_data']
        score = 0.0

        # Market sentiment.
        sentiment_scores = {"bullish": 0.5, "neutral": 0.0, "bearish": -0.5}
        score += sentiment_scores.get(md.market_sentiment, 0.0)

        # VIX analysis.
        if md.vix < 15:
            score += 0.3
        elif md.vix > 30:
            score -= 0.3

        # Economic indicators.
        if md.gdp_growth > 3.0:
            score += 0.1
        elif md.gdp_growth < 1.0:
            score -= 0.1

        return max(-1.0, min(1.0, score))

# Calculating the sentiment score.
    def _calculate_sentiment_score(self, state: Dict) -> float:
        if 'sentiment_data' not in state or not state['sentiment_data']:
            return 0.0

        sd = state['sentiment_data']
        sentiment_score = sd.overall_sentiment * sd.confidence_score
        fear_greed_factor = (sd.fear_greed_index - 50) / 100
        analyst_scores = {"upgrade": 0.2, "neutral": 0.0, "downgrade": -0.2}
        analyst_score = analyst_scores.get(sd.analyst_rating_trend, 0.0)

        total_score = sentiment_score * 0.6 + fear_greed_factor * 0.3 + analyst_score * 0.1
        return max(-1.0, min(1.0, total_score))

    def _generate_detailed_reasoning(self, state: Dict, overall_score: float) -> str:
        reasoning = f"Multi-agent analysis yields overall score of {overall_score:.2f}.\n\n"

        if 'market_data' in state and state['market_data']:
            md = state['market_data']
            reasoning += f"TECHNICAL: {md.trend} trend, RSI {md.rsi:.1f}, {md.macd_signal} MACD.\n"

        if 'risk_metrics' in state and state['risk_metrics']:
            rm = state['risk_metrics']
            reasoning += f"RISK: Sharpe {rm.sharpe_ratio:.2f}, volatility {rm.portfolio_volatility:.1%}.\n"

        if 'forecast_data' in state and state['forecast_data']:
            fd = state['forecast_data']
            reasoning += f"FORECAST: ${fd.ensemble_forecast:.2f} target, {fd.forecast_confidence:.1%} confidence.\n"

        return reasoning

# Identifying key risk and opportunity factors.
    def _identify_risk_opportunity_factors(self, state: Dict) -> tuple:
        risk_factors = []
        opportunity_factors = []

        if 'market_data' in state and state['market_data']:
            md = state['market_data']
            if md.rsi > 70:
                risk_factors.append("Overbought conditions (RSI > 70).")
            elif md.rsi < 30:
                opportunity_factors.append("Oversold conditions.")

        if 'risk_metrics' in state and state['risk_metrics']:
            rm = state['risk_metrics']
            if rm.portfolio_volatility > 0.3:
                risk_factors.append("High volatility.")
            if rm.sharpe_ratio > 1.0:
                opportunity_factors.append("Strong risk-adjusted returns.")

        return risk_factors[:3], opportunity_factors[:3]

    def _generate_alternative_scenarios(self, state: Dict) -> dict:
        return {
            "bull_case": "Technical momentum and positive catalysts drive upside.",
            "bear_case": "Risk factors materialize leading to correction.",
            "base_case": "Mixed signals result in sideways action."
        }

# Analyze impact on overall portfolio.
    def _analyze_portfolio_impact(self, state: Dict, action: str, position_size: float) -> str:
        return f"{action} position of {position_size:.1%} would impact portfolio risk and diversification."

# Analyze market timing considerations.
    def _analyze_market_timing(self, state: Dict) -> str:
        timing_factors = []

        if 'market_data' in state and state['market_data']:
            md = state['market_data']
            if md.rsi > 70:
                timing_factors.append("overbought conditions suggest caution")
            elif md.rsi < 30:
                timing_factors.append("oversold conditions favor entry")

        if timing_factors:
            return "Timing analysis: " + "; ".join(timing_factors)
        else:
            return "Mixed timing signals suggest neutral approach"

    def _calculate_risk_reward_ratio(self, current_price: float, action: str, state: Dict) -> float:
        """Calculate risk/reward ratio for the recommendation"""
        if action == "HOLD":
            return 1.0

        # Reward/risk ratio.
        base_ratio = 2.0

        # Based on volatility.
        if 'risk_metrics' in state and state['risk_metrics']:
            volatility = state['risk_metrics'].portfolio_volatility
            if volatility > 0.3:  # High volatility
                base_ratio = 1.5
            elif volatility < 0.15:  # Low volatility
                base_ratio = 2.5

        return base_ratio

    def _estimate_maximum_drawdown(self, state: Dict) -> float:
        """Estimate maximum drawdown for the position"""
        if 'risk_metrics' in state and state['risk_metrics']:
            return abs(state['risk_metrics'].maximum_drawdown)
        return 0.15

print("Strategist Agent ready.")

# Financial planner agent with Sharpe ratio and risk-adjusted planning.
class FinancialPlannerAgent:

    def __init__(self):
        self.name = "FinancialPlannerAgent"

        # Asset allocation strategies.
        self.strategies = {
            "conservative": {
                "us_stocks": 0.30, "international_stocks": 0.10,
                "bonds": 0.55, "cash": 0.05
            },
            "moderate": {
                "us_stocks": 0.50, "international_stocks": 0.20,
                "bonds": 0.25, "cash": 0.05
            },
            "aggressive": {
                "us_stocks": 0.60, "international_stocks": 0.25,
                "bonds": 0.10, "cash": 0.05
            }
        }

        # Expected returns and volatilities.
        self.asset_assumptions = {
            "us_stocks": {"return": 0.10, "volatility": 0.16},
            "international_stocks": {"return": 0.08, "volatility": 0.18},
            "bonds": {"return": 0.04, "volatility": 0.06},
            "cash": {"return": 0.02, "volatility": 0.01}
        }

        print(f"{self.name} initialized.")

# Comprehensive financial plan with risk-adjusted metrics.
    async def process(self, state: Dict, goal: FinancialGoal) -> Dict:

        try:
            print(f"{self.name}: Creating financial plan.")

            # Calculating projections.
            projections = self._calculate_projections(goal)

            # Optimizing asset allocation.
            asset_allocation = self._optimize_allocation(goal)

            # Calculating plan level risk metrics.
            plan_sharpe = self._calculate_plan_sharpe(asset_allocation)
            plan_volatility = self._calculate_plan_volatility(asset_allocation)
            plan_drawdown = self._estimate_plan_drawdown(asset_allocation)

            # Tax optimization.
            tax_optimization = self._optimize_taxes(goal)

            # Monthly breakdown.
            monthly_breakdown = self._calculate_monthly_breakdown(goal, asset_allocation)

            # Monte Carlo simulation with risk metrics.
            monte_carlo = self._run_monte_carlo(goal, asset_allocation, 1000)

            # Generating actionable recommendations.
            recommendations = self._generate_recommendations(goal, projections, monte_carlo)

            plan_result = FinancialPlanResult(
                goal=goal,
                projected_value=projections['projected_value'],
                success_probability=projections['success_probability'],
                required_monthly=projections['required_monthly'],
                asset_allocation=asset_allocation,
                tax_optimization=tax_optimization,
                monthly_breakdown=monthly_breakdown,
                recommendations=recommendations,
                is_achievable=projections['is_achievable'],
                monte_carlo_results=monte_carlo,
                plan_sharpe_ratio=plan_sharpe,
                plan_max_drawdown=plan_drawdown,
                plan_volatility=plan_volatility
            )

            state['financial_plan'] = plan_result

            print(f"{self.name}: Financial plan created.")
            print(f"   Success Probability: {projections['success_probability']:.1%}")
            print(f"   Plan Sharpe Ratio: {plan_sharpe:.2f}")
            print(f"   Expected Volatility: {plan_volatility:.1%}")
            print(f"   Est. Max Drawdown: {plan_drawdown:.1%}")

        except Exception as e:
            print(f"X {self.name}: Error - {e}")

        return state

# Financial projections with compound growth.
    def _calculate_projections(self, goal: FinancialGoal) -> Dict[str, float]:

        # Getting the portfolio expected return based on risk tolerance.
        allocation = self.strategies[goal.risk_tolerance]
        portfolio_return = sum(
            allocation[asset] * self.asset_assumptions[asset]["return"]
            for asset in allocation
        )

        # Future value calculations.
        years = goal.time_horizon_years
        monthly_rate = portfolio_return / 12
        months = years * 12

        # Future value of current amount.
        fv_current = goal.current_amount * (1 + portfolio_return) ** years

        # Future value of monthly contributions.
        if monthly_rate > 0:
            fv_contributions = goal.monthly_contribution * (
                ((1 + monthly_rate) ** months - 1) / monthly_rate
            )
        else:
            fv_contributions = goal.monthly_contribution * months

        # Total projected value.
        projected_value = fv_current + fv_contributions

        # Success analysis.
        success_probability = min(1.0, projected_value / goal.target_amount)
        gap = goal.target_amount - projected_value

        # Required additional monthly contribution if not achievable.
        if gap > 0 and monthly_rate > 0:
            required_monthly = gap / (((1 + monthly_rate) ** months - 1) / monthly_rate)
        else:
            required_monthly = 0

        return {
            'projected_value': projected_value,
            'success_probability': success_probability,
            'required_monthly': required_monthly,
            'is_achievable': gap <= 0,
            'portfolio_return': portfolio_return
        }

    def _optimize_allocation(self, goal: FinancialGoal) -> Dict[str, float]:

        base_allocation = self.strategies[goal.risk_tolerance].copy()

        # Age adjustments.
        target_equity_ratio = max(0.3, min(0.9, (100 - goal.age) / 100))
        current_equity_ratio = base_allocation["us_stocks"] + base_allocation["international_stocks"]

        equity_adjustment = target_equity_ratio - current_equity_ratio

        if abs(equity_adjustment) > 0.05:
            if equity_adjustment > 0:
                base_allocation["us_stocks"] += equity_adjustment * 0.7
                base_allocation["international_stocks"] += equity_adjustment * 0.3
                base_allocation["bonds"] -= equity_adjustment * 0.8
                base_allocation["cash"] -= equity_adjustment * 0.2
            else:
                reduction = abs(equity_adjustment)
                base_allocation["us_stocks"] -= reduction * 0.7
                base_allocation["international_stocks"] -= reduction * 0.3
                base_allocation["bonds"] += reduction * 0.8
                base_allocation["cash"] += reduction * 0.2

        # Time horizon adjustments.
        if goal.time_horizon_years > 20:
            base_allocation["us_stocks"] += 0.05
            base_allocation["bonds"] -= 0.05
        elif goal.time_horizon_years < 5:
            base_allocation["us_stocks"] -= 0.10
            base_allocation["bonds"] += 0.07
            base_allocation["cash"] += 0.03

        # Goal type adjustments.
        if goal.goal_type == "house":
            base_allocation["cash"] += 0.10
            base_allocation["bonds"] += 0.05
            base_allocation["us_stocks"] -= 0.15
        elif goal.goal_type == "education":
            if goal.time_horizon_years < 10:
                base_allocation["bonds"] += 0.10
                base_allocation["us_stocks"] -= 0.10

        # Ensure all allocations are non-negative and normalize.
        for key in base_allocation:
            base_allocation[key] = max(0, base_allocation[key])

        # Normalize to sum to 1.
        total = sum(base_allocation.values())
        if total > 0:
            base_allocation = {k: v/total for k, v in base_allocation.items()}

        return base_allocation

# Calculating the expected Sharpe ratio for the financial plan.
    def _calculate_plan_sharpe(self, allocation: Dict[str, float]) -> float:

        # Portfolio expected return.
        portfolio_return = sum(
            allocation[asset] * self.asset_assumptions[asset]["return"]
            for asset in allocation
        )

        # Portfolio volatility.
        portfolio_variance = sum(
            (allocation[asset] ** 2) * (self.asset_assumptions[asset]["volatility"] ** 2)
            for asset in allocation
        )
        portfolio_volatility = np.sqrt(portfolio_variance)

        # Sharpe ratio calculation.
        risk_free_rate = 0.02
        if portfolio_volatility > 0:
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        else:
            sharpe_ratio = 0.0

        return sharpe_ratio

    def _calculate_plan_volatility(self, allocation: Dict[str, float]) -> float:

        portfolio_variance = sum(
            (allocation[asset] ** 2) * (self.asset_assumptions[asset]["volatility"] ** 2)
            for asset in allocation
        )

        return np.sqrt(portfolio_variance)

# Estimating the maximum drawdown for the financial plan.
    def _estimate_plan_drawdown(self, allocation: Dict[str, float]) -> float:

        # Simple estimation based on portfolio volatility.
        portfolio_volatility = self._calculate_plan_volatility(allocation)

        estimated_drawdown = portfolio_volatility * 2.5

        # Cap between bounds.
        return min(0.6, max(0.05, estimated_drawdown))

# Optimizing the tax account allocation.
    def _optimize_taxes(self, goal: FinancialGoal) -> Dict[str, float]:

        annual_contribution = goal.monthly_contribution * 12

        # 2024 contribution limits with catch-up provisions.
        max_401k = 23000 + (7500 if goal.age >= 50 else 0)
        max_ira = 7000 + (1000 if goal.age >= 50 else 0)

        # Income-based phase-outs.
        if goal.annual_income > 120000:
            max_ira = max(0, max_ira - (goal.annual_income - 120000) * 0.1)

        # Maximize tax advantages.
        remaining_contribution = annual_contribution

        # Maximize 401(k) first.
        optimal_401k = min(max_401k, remaining_contribution * 0.6)
        remaining_contribution -= optimal_401k

        # Fill IRA.
        optimal_ira = min(max_ira, remaining_contribution)
        remaining_contribution -= optimal_ira

        # Additional 401k.
        additional_401k = min(max_401k - optimal_401k, remaining_contribution)
        optimal_401k += additional_401k
        remaining_contribution -= additional_401k

        optimal_taxable = remaining_contribution

        # Calculating the tax savings.
        tax_deductible = optimal_401k + optimal_ira
        tax_savings = tax_deductible * goal.tax_rate

        return {
            "401k_annual": optimal_401k,
            "401k_monthly": optimal_401k / 12,
            "ira_annual": optimal_ira,
            "ira_monthly": optimal_ira / 12,
            "taxable_annual": optimal_taxable,
            "taxable_monthly": optimal_taxable / 12,
            "tax_savings": tax_savings,
            "marginal_rate": goal.tax_rate
        }

# Calculating monthly contribution breakdown by asset class.
    def _calculate_monthly_breakdown(self, goal: FinancialGoal,
                                   allocation: Dict[str, float]) -> Dict[str, float]:

        monthly = goal.monthly_contribution

        breakdown = {
            "total_monthly": monthly
        }

        # Adding monthly allocation for each asset class.
        for asset, percentage in allocation.items():
            breakdown[f"{asset}_monthly"] = monthly * percentage

        return breakdown

 # Monte Carlo simulation with risk metrics.
    def _run_monte_carlo(self, goal: FinancialGoal, allocation: Dict[str, float],
                        n_simulations: int = 1000) -> Dict[str, float]:

        # Portfolio characteristics.
        portfolio_return = sum(
            allocation[asset] * self.asset_assumptions[asset]["return"]
            for asset in allocation
        )
        portfolio_volatility = self._calculate_plan_volatility(allocation)

        results = []
        drawdowns = []
        sharpe_ratios = []

        for _ in range(n_simulations):
            value = goal.current_amount
            peak_value = value
            max_drawdown = 0
            annual_returns = []

            for year in range(goal.time_horizon_years):
                # Annual contributions.
                value += goal.monthly_contribution * 12

                # Applying random annual return (normal distribution).
                annual_return = np.random.normal(portfolio_return, portfolio_volatility)
                annual_returns.append(annual_return)

                # Applying market stress scenarios.
                if np.random.random() < 0.1:
                    stress_factor = np.random.uniform(0.7, 0.9)
                    annual_return *= stress_factor

                value *= (1 + annual_return)

                # Tracking drawdown.
                if value > peak_value:
                    peak_value = value
                else:
                    current_drawdown = (peak_value - value) / peak_value
                    max_drawdown = max(max_drawdown, current_drawdown)

                # Ensuring value doesn't go negative.
                value = max(0, value)

            # Calculating Sharpe ratio for this simulation.
            if len(annual_returns) > 0:
                avg_return = np.mean(annual_returns)
                return_std = np.std(annual_returns)
                sim_sharpe = (avg_return - 0.02) / return_std if return_std > 0 else 0
                sharpe_ratios.append(sim_sharpe)

            results.append(value)
            drawdowns.append(max_drawdown)

        results = np.array(results)
        drawdowns = np.array(drawdowns)
        sharpe_ratios = np.array(sharpe_ratios)

        return {
            "mean": np.mean(results),
            "median": np.percentile(results, 50),
            "percentile_5": np.percentile(results, 5),
            "percentile_10": np.percentile(results, 10),
            "percentile_25": np.percentile(results, 25),
            "percentile_75": np.percentile(results, 75),
            "percentile_90": np.percentile(results, 90),
            "percentile_95": np.percentile(results, 95),
            "success_rate": np.mean(results >= goal.target_amount),
            "average_drawdown": np.mean(drawdowns),
            "worst_drawdown": np.max(drawdowns),
            "average_sharpe": np.mean(sharpe_ratios),
            "sharpe_std": np.std(sharpe_ratios)
        }

    def _generate_recommendations(self, goal: FinancialGoal, projections: Dict,
                                monte_carlo: Dict) -> List[str]:

        recommendations = []

        # Goal achievability assessment.
        if projections['is_achievable']:
            recommendations.append(f"Goal is achievable! Projected: ${projections['projected_value']:,.0f}.")
        else:
            shortfall = projections['required_monthly']
            recommendations.append(f"Increase monthly contribution by ${shortfall:,.0f} to reach goal.")

        # Monte Carlo insights.
        success_rate = monte_carlo['success_rate']
        if success_rate > 0.8:
            recommendations.append("High probability of success based on Monte Carlo analysis.")
        elif success_rate < 0.5:
            recommendations.append("Consider extending timeline or increasing contributions.")

        # Risk based recommendations.
        avg_sharpe = monte_carlo.get('average_sharpe', 0)
        if avg_sharpe > 1.0:
            recommendations.append(f"Excellent risk-adjusted returns expected (Sharpe: {avg_sharpe:.2f}).")
        elif avg_sharpe < 0.5:
            recommendations.append("Consider adjusting allocation for better risk-adjusted returns.")

        # Age recommendations.
        if goal.age < 35:
            recommendations.append("Young investor advantage, time is your greatest asset.")
        elif goal.age > 45:
            recommendations.append("Focus on risk management while maintaining growth.")

        # Time horizon recommendations.
        if goal.time_horizon_years > 15:
            recommendations.append("Long timeline allows for growth strategy.")
        elif goal.time_horizon_years < 5:
            recommendations.append("Short timeline requires conservative approach.")

        # Risk tolerance recommendations.
        if goal.risk_tolerance == "conservative" and goal.time_horizon_years > 10:
            recommendations.append("Consider moderate risk for better long-term growth potential.")

        # Practical action items.
        recommendations.append("Rebalance portfolio quarterly to maintain target allocation.")
        recommendations.append("Review and adjust plan annually or after major life changes.")

        return recommendations[:8]

# Pipeline and execution

async def run_pipeline_with_real_apis(symbol="AAPL", openai_api_key=None, financial_goal=None):

    print("AI Financial Forecasting pipeline.")
    print("=" * 60)

    state = {}

    print("\n Initializing agents with APIs...")
    market_agent = MarketDataAgent()
    risk_agent = RiskAgent()
    forecast_agent = ForecastingAgent()
    macro_agent = MacroEconomicAgent(fred_api_key=FRED_API_KEY)
    sentiment_agent = SentimentAgent(
        news_api_key=NEWS_API_KEY,
        reddit_client_id=REDDIT_CLIENT_ID,
        reddit_client_secret=REDDIT_CLIENT_SECRET,
        reddit_user_agent=REDDIT_USER_AGENT
    )
    strategist_agent = StrategistAgent(api_key=OPENAI_API_KEY)
    planner_agent = FinancialPlannerAgent()

    print(f"All agents initialized with APIs.")

    try:
        # Run the pipeline with real data
        print("Data collection and analysis.")
        print("-" * 50)

        state = await market_agent.process(state, symbol=symbol, period="1y")
        state = await risk_agent.process(state)
        state = await forecast_agent.process(state, forecast_horizon=5)

        print("\n Real macro and sentiment analysis.")
        print("-" * 50)

        state = await macro_agent.process(state)
        state = await sentiment_agent.process(state)

        print("\n AI strategy and planning.")
        print("-" * 50)

        state = await strategist_agent.process(state)

        if financial_goal:
            state = await planner_agent.process(state, financial_goal)

        print("\n Results Summary.")
        print("-" * 60)

        print_pipeline_summary(state)


        return state

    except Exception as e:
        print(f"\n X Pipeline error: {e}")
        print("=" * 60)
        return state

def print_pipeline_summary(state):

    symbol = state.get('symbol', 'UNKNOWN')
    print(f"\n Analysis summary for {symbol}.")
    print("-" * 40)

    # Market data summary.
    if 'market_data' in state and state['market_data']:
        md = state['market_data']
        print(f"\n MARKET DATA:")
        print(f"   Current Price: ${md.current_price:.2f}")
        print(f"   Trend: {md.trend}")
        print(f"   RSI: {md.rsi:.1f}")
        print(f"   1-Day Return: {md.return_1d:.2%}")
        print(f"   20-Day Volatility: {md.volatility_20d:.1%}")
        print(f"   Support Level: ${md.support_level:.2f}")
        print(f"   Resistance Level: ${md.resistance_level:.2f}")

    # Risk metrics summary.
    if 'risk_metrics' in state and state['risk_metrics']:
        rm = state['risk_metrics']
        print(f"\n  RISK ANALYSIS:")
        print(f"   Portfolio Volatility: {rm.portfolio_volatility:.1%}")
        print(f"   Sharpe Ratio: {rm.sharpe_ratio:.2f}")
        print(f"   Maximum Drawdown: {rm.maximum_drawdown:.1%}")
        print(f"   VaR (5%): {rm.value_at_risk_5pct:.1%}")
        print(f"   Expected Shortfall: {rm.expected_shortfall:.1%}")
        print(f"   GARCH Volatility: {rm.garch_volatility:.1%}")

    # Forecast summary.
    if 'forecast_data' in state and state['forecast_data']:
        fd = state['forecast_data']
        current_price = state.get('market_data', {}).current_price if 'market_data' in state else 100
        price_change = ((fd.ensemble_forecast - current_price) / current_price) * 100

        print(f"\n FORECASTING:")
        print(f"   Ensemble Forecast: ${fd.ensemble_forecast:.2f} ({price_change:+.1f}%)")
        print(f"   Forecast Confidence: {fd.forecast_confidence:.1%}")
        print(f"   Upside Probability: {fd.upside_probability:.1%}")
        print(f"   Downside Risk: {fd.downside_risk:.1%}")
        print(f"   Volatility Forecast: {fd.volatility_forecast:.1%}")
        print(f"   Prediction Interval: ${fd.prediction_interval[0]:.2f} - ${fd.prediction_interval[1]:.2f}")

    # Macro Economic summary.
    if 'macro_data' in state and state['macro_data']:
        macro = state['macro_data']
        print(f"\n MACRO ECONOMICS:")
        print(f"   Market Sentiment: {macro.market_sentiment}")
        print(f"   GDP Growth: {macro.gdp_growth:.1f}%")
        print(f"   Inflation Rate: {macro.inflation_rate:.1f}%")
        print(f"   Unemployment Rate: {macro.unemployment_rate:.1f}%")
        print(f"   Fed Funds Rate: {macro.federal_funds_rate:.2f}%")
        print(f"   VIX: {macro.vix:.1f}")
        print(f"   Yield Curve Slope: {macro.yield_curve_slope:.2f}")

    # Sentiment summary.
    if 'sentiment_data' in state and state['sentiment_data']:
        sentiment = state['sentiment_data']
        print(f"\n SENTIMENT ANALYSIS:")
        print(f"   Overall Sentiment: {sentiment.sentiment_trend} ({sentiment.overall_sentiment:.2f})")
        print(f"   News Sentiment: {sentiment.news_sentiment:.2f}")
        print(f"   Social Media Sentiment: {sentiment.social_media_sentiment:.2f}")
        print(f"   Fear & Greed Index: {sentiment.fear_greed_index:.0f}")
        print(f"   Analyst Rating Trend: {sentiment.analyst_rating_trend}")
        print(f"   Key Topics: {', '.join(sentiment.key_topics[:3])}")

    # AI Recommendation summary.
    if 'recommendation' in state and state['recommendation']:
        rec = state['recommendation']
        print(f"\n AI RECOMMENDATION:")
        print(f"   Action: {rec.action}")
        print(f"   Confidence: {rec.confidence:.1%}")
        print(f"   Position Size: {rec.position_size:.1%}")
        print(f"   Risk Level: {rec.risk_level}")
        print(f"   Time Horizon: {rec.time_horizon}")
        print(f"   Risk/Reward Ratio: {rec.risk_reward_ratio:.2f}")
        print(f"   Probability of Success: {rec.probability_of_success:.1%}")
        print(f"   Entry Price: ${rec.entry_price:.2f}")
        print(f"   Stop Loss: ${rec.stop_loss:.2f}")
        print(f"   Take Profit: ${rec.take_profit:.2f}")

        if hasattr(rec, 'detailed_reasoning') and rec.detailed_reasoning:
            print(f"\n    Key Reasoning:")
            reasoning_lines = rec.detailed_reasoning.split('\n')[:3]  # First 3 lines
            for line in reasoning_lines:
                if line.strip():
                    print(f"      {line.strip()}")

    # Financial planning summary.
    if 'financial_plan' in state and state['financial_plan']:
        plan = state['financial_plan']
        print(f"\n FINANCIAL PLANNING:")
        print(f"   Goal Achievement Probability: {plan.success_probability:.1%}")
        print(f"   Projected Value: ${plan.projected_value:,.0f}")
        print(f"   Target Amount: ${plan.goal.target_amount:,.0f}")
        print(f"   Plan Sharpe Ratio: {plan.plan_sharpe_ratio:.2f}")
        print(f"   Expected Volatility: {plan.plan_volatility:.1%}")
        print(f"   Estimated Max Drawdown: {plan.plan_max_drawdown:.1%}")

        # Asset allocation.
        print(f"\n   Asset Allocation:")
        for asset, allocation in plan.asset_allocation.items():
            print(f"      {asset.replace('_', ' ').title()}: {allocation:.1%}")

        # Top recommendations.
        print(f"\n   Top Recommendations:")
        for i, rec in enumerate(plan.recommendations[:3], 1):
            print(f"      {i}. {rec}")

# Visuals of the results.
def create_dashboard_visualizations(state):

    if not state or 'market_data' not in state:
        print("No market data available for visualization.")
        return

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Price Chart with Technical Indicators',
            'Risk Metrics Dashboard',
            'Forecast Comparison',
            'Sentiment Analysis',
            'Portfolio Allocation',
            'Monte Carlo Results'
        ],
        specs=[
            [{"secondary_y": True}, {"type": "indicator"}],
            [{"type": "bar"}, {"type": "pie"}],
            [{"type": "scatter"}, {"type": "histogram"}]
        ]
    )

    market_data = state['market_data']
    prices = market_data.prices

    # Price chart with indicators.
    fig.add_trace(
        go.Scatter(
            x=prices.index,
            y=prices.values,
            name="Price",
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )

    # Adding support and resistance levels.
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

    # Risk metrics gauge.
    if 'risk_metrics' in state:
        rm = state['risk_metrics']
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=rm.sharpe_ratio,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Sharpe Ratio"},
                gauge={
                    'axis': {'range': [-2, 3]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-2, 0], 'color': "lightgray"},
                        {'range': [0, 1], 'color': "yellow"},
                        {'range': [1, 3], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 2.0
                    }
                }
            ),
            row=1, col=2
        )

    # Forecast comparison.
    if 'forecast_data' in state:
        fd = state['forecast_data']
        forecasts = ['ARIMA', 'Prophet', 'LSTM', 'Ensemble']
        values = [fd.arima_forecast, fd.prophet_forecast, fd.lstm_forecast, fd.ensemble_forecast]

        fig.add_trace(
            go.Bar(x=forecasts, y=values, name="Forecasts"),
            row=2, col=1
        )

    # Sentiment pie chart.
    if 'sentiment_data' in state:
        sentiment = state['sentiment_data']
        labels = ['Positive', 'Neutral', 'Negative']

        overall = sentiment.overall_sentiment
        if overall > 0.1:
            values = [60 + overall*20, 30, 10 - overall*10]
        elif overall < -0.1:
            values = [10 + overall*10, 30, 60 - overall*20]
        else:
            values = [40, 40, 20]

        fig.add_trace(
            go.Pie(labels=labels, values=values, name="Sentiment"),
            row=2, col=2
        )

    # Portfolio allocation.
    if 'financial_plan' in state:
        plan = state['financial_plan']
        assets = list(plan.asset_allocation.keys())
        allocations = list(plan.asset_allocation.values())

        fig.add_trace(
            go.Bar(
                x=assets,
                y=[a*100 for a in allocations],
                name="Allocation %"
            ),
            row=3, col=1
        )

    # Monte Carlo histogram.
    if 'financial_plan' in state and 'monte_carlo_results' in state['financial_plan']:
        mc = state['financial_plan'].monte_carlo_results
        percentiles = ['5th', '25th', '50th', '75th', '95th']
        values = [
            mc['percentile_5'], mc['percentile_25'], mc['percentile_50'],
            mc['percentile_75'], mc['percentile_95']
        ]

        fig.add_trace(
            go.Scatter(
                x=percentiles,
                y=values,
                mode='lines+markers',
                name="Monte Carlo Percentiles"
            ),
            row=3, col=2
        )

    fig.update_layout(
        height=900,
        title_text="AI Financial Forecasting Dashboard",
        showlegend=True
    )

    fig.show()

    print("Dashboard visualization created.")

# Main demo.
async def main_demo():

    print("AI Financial Forecasting system.")
    print("-" * 70)

    # Basic stock analysis.
    print("\n EXAMPLE 1: Basic stock analysis.")
    print("-" * 50)

    state = await run_pipeline_with_real_apis(
        symbol="AAPL",
        financial_goal=None
    )

    # Example 2: Stock analysis with financial planning
    print("\n EXAMPLE 2: Stock analysis with financial planning.")
    print("-" * 50)

    # Create a financial goal.
    retirement_goal = FinancialGoal(
        target_amount=1000000,
        current_amount=50000,
        monthly_contribution=2000,
        time_horizon_years=25,
        risk_tolerance="moderate",
        age=35,
        annual_income=120000,
        goal_type="retirement"
    )

    state_with_planning = await run_pipeline_with_real_apis(
        symbol="TSLA",
        financial_goal=retirement_goal
    )


    print("\n Dashboard Visuals.")
    create_dashboard_visualizations(state_with_planning)

    print("\n Demo completed.")
    print("-" * 70)

    return state_with_planning

def save_pipeline_results(state, filename="pipeline_results.json"):

    # Convert state to JSON format.
    serializable_state = {}

    for key, value in state.items():
        if hasattr(value, '__dict__'):
            serializable_state[key] = {
                'type': value.__class__.__name__,
                'data': {k: v for k, v in value.__dict__.items()
                        if not isinstance(v, (pd.Series, pd.DataFrame))}
            }
        else:
            serializable_state[key] = value

    with open(filename, 'w') as f:
        json.dump(serializable_state, f, indent=2, default=str)

    print(f"Pipeline results saved to {filename}.")

def load_pipeline_results(filename="pipeline_results.json"):

    with open(filename, 'r') as f:
        return json.load(f)

def run_complete_analysis(symbol="AAPL", openai_api_key=None):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.run(run_pipeline_with_real_apis(symbol, openai_api_key=openai_api_key))
        else:
            return asyncio.run(run_pipeline_with_real_apis(symbol, openai_api_key=openai_api_key))
    except ImportError:
        try:
            loop = asyncio.get_event_loop()
            task = loop.create_task(run_pipeline_with_real_apis(symbol, openai_api_key=openai_api_key))
            return loop.run_until_complete(task)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(run_pipeline_with_real_apis(symbol, openai_api_key=openai_api_key))
            finally:
                loop.close()

def run_with_financial_planning(symbol="AAPL", target_amount=1000000,
                               current_amount=50000, monthly_contribution=2000,
                               time_horizon_years=25, age=35, risk_tolerance="moderate"):
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
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.run(run_pipeline_with_real_apis(symbol, financial_goal=goal))
        else:
            return asyncio.run(run_pipeline_with_real_apis(symbol, financial_goal=goal))
    except ImportError:
        try:
            loop = asyncio.get_event_loop()
            task = loop.create_task(run_pipeline_with_real_apis(symbol, financial_goal=goal))
            return loop.run_until_complete(task)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(run_pipeline_with_real_apis(symbol, financial_goal=goal))
            finally:
                loop.close()


async def run_pipeline_sync(symbol="AAPL", openai_api_key=None, financial_goal=None):
    return await run_pipeline_with_real_apis(symbol, openai_api_key, financial_goal)

if __name__ == "__main__":
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            print("Detected notebook environment.")
            print("Running quick AAPL analysis...")

            # For notebooks, use await syntax instead
            print("Use this code in a notebook cell:")
            print("```python")
            print("# Basic analysis")
            print("results = await run_pipeline_sync('AAPL')")
            print("")
            print("# With financial planning")
            print("goal = FinancialGoal(target_amount=1000000, current_amount=50000,")
            print("                    monthly_contribution=2000, time_horizon_years=25,")
            print("                    risk_tolerance='moderate', age=35)")
            print("results = await run_pipeline_sync('TSLA', financial_goal=goal)")
            print("```")

        else:
            print("Running quick AAPL analysis...")
            results = run_complete_analysis("AAPL")

            print("\n Running TSLA analysis with retirement planning...")
            results_with_planning = run_with_financial_planning(
                symbol="TSLA",
                target_amount=2000000,
                current_amount=100000,
                monthly_contribution=3000,
                time_horizon_years=20,
                age=40,
                risk_tolerance="aggressive"
            )

            save_pipeline_results(results_with_planning, "tsla_retirement_analysis.json")
            print("\n Analysis complete.")

    except ImportError:
        print("Running quick AAPL analysis...")
        results = run_complete_analysis("AAPL")

        print("\n Running TSLA analysis with retirement planning...")
        results_with_planning = run_with_financial_planning(
            symbol="TSLA",
            target_amount=2000000,
            current_amount=100000,
            monthly_contribution=3000,
            time_horizon_years=20,
            age=40,
            risk_tolerance="aggressive"
        )

        save_pipeline_results(results_with_planning, "tsla_retirement_analysis.json")
        print("\n Analysis complete.")

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
            from legacy.Final_GENAI_V3 import FinancialGoal
            
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
            from legacy.Final_GENAI_V3 import run_pipeline
            
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
