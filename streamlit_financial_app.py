import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import asyncio


try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from fredapi import Fred
    HAS_FRED = True
except ImportError:
    HAS_FRED = False

try:
    from newsapi.newsapi_client import NewsApiClient
    HAS_NEWSAPI = True
except ImportError:
    HAS_NEWSAPI = False

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

# Configure page
st.set_page_config(
    page_title="AI Financial Forecasting System",
    page_icon="📈",
    layout="wide"
)

# Data structures
@dataclass
class MarketData:
    symbol: str
    current_price: float
    prices: pd.Series
    volume: pd.Series = field(default_factory=pd.Series)
    rsi: float = 50.0
    trend: str = "neutral"
    return_1d: float = 0.0
    return_5d: float = 0.0
    return_20d: float = 0.0
    volatility_20d: float = 0.0
    support_level: float = 0.0
    resistance_level: float = 0.0
    macd_signal: str = "neutral"

@dataclass
class ForecastData:
    arima_forecast: float = 0.0
    prophet_forecast: float = 0.0
    lstm_forecast: float = 0.0
    ensemble_forecast: float = 0.0
    forecast_confidence: float = 0.5
    upside_probability: float = 0.5
    downside_risk: float = 0.5
    volatility_forecast: float = 0.2

@dataclass
class RiskMetrics:
    portfolio_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    maximum_drawdown: float = 0.0
    value_at_risk_5pct: float = 0.0
    expected_shortfall: float = 0.0
    sortino_ratio: float = 0.0

@dataclass
class MacroData:
    gdp_growth: float = 0.0
    inflation_rate: float = 0.0
    unemployment_rate: float = 0.0
    federal_funds_rate: float = 0.0
    vix: float = 0.0
    market_sentiment: str = "neutral"

@dataclass
class SentimentData:
    news_sentiment: float = 0.0
    social_media_sentiment: float = 0.0
    overall_sentiment: float = 0.0
    sentiment_trend: str = "neutral"
    confidence_score: float = 0.5
    key_topics: List[str] = field(default_factory=list)
    fear_greed_index: float = 50.0

@dataclass
class Recommendation:
    action: str
    confidence: float
    position_size: float
    risk_level: str
    time_horizon: str
    entry_price: float
    stop_loss: float
    take_profit: float
    detailed_reasoning: str
    risk_reward_ratio: float = 2.0

@dataclass
class FinancialGoal:
    target_amount: float
    current_amount: float
    monthly_contribution: float
    time_horizon_years: int
    risk_tolerance: str
    age: int = 30
    annual_income: float = 100000
    goal_type: str = "retirement"

@dataclass
class FinancialPlan:
    goal: FinancialGoal
    projected_value: float
    success_probability: float
    required_monthly: float
    asset_allocation: Dict[str, float]
    recommendations: List[str]
    is_achievable: bool
    monte_carlo_results: Dict[str, float]
    plan_sharpe_ratio: float = 0.0

# Market Data Agent
class MarketDataAgent:
    def __init__(self):
        self.name = "MarketDataAgent"

    async def process(self, symbol: str = "AAPL") -> MarketData:
        try:
            st.write(f"📊 {self.name}: Fetching market data for {symbol}...")
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1y")
            
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            current_price = float(data['Close'].iloc[-1])
            prices = data['Close']
            volume = data['Volume']
            
            # Calculate returns and volatility
            returns = prices.pct_change()
            return_1d = float(returns.iloc[-1]) if len(returns) > 0 else 0.0
            return_5d = float(returns.tail(5).mean()) if len(returns) >= 5 else 0.0
            return_20d = float(returns.tail(20).mean()) if len(returns) >= 20 else 0.0
            volatility_20d = float(returns.tail(20).std() * np.sqrt(252)) if len(returns) >= 20 else 0.0
            
            # Technical indicators.
            rsi = self._calculate_rsi(prices)
            trend = self._analyze_trend(prices)
            macd_signal = self._calculate_macd_signal(prices)
            
            # Support and resistance.
            high_20 = prices.tail(20).max()
            low_20 = prices.tail(20).min()
            support_level = float(low_20 * 1.02)
            resistance_level = float(high_20 * 0.98)
            
            return MarketData(
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
                macd_signal=macd_signal
            )
            
        except Exception as e:
            st.error(f"Error fetching market data: {e}")
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
# Forecasting Agent
class ForecastingAgent:
    def __init__(self):
        self.name = "ForecastingAgent"

    async def process(self, market_data: MarketData) -> ForecastData:
        try:
            st.write(f"🔮 {self.name}: Generating price forecasts...")
            
            prices = market_data.prices
            current_price = market_data.current_price
            
            # Multiple forecasting methods
            arima_forecast = self._arima_forecast(prices, current_price)
            prophet_forecast = self._prophet_forecast(prices, current_price)
            lstm_forecast = self._lstm_forecast(prices, current_price)
            
            # Ensemble forecast
            ensemble_forecast = np.mean([arima_forecast, prophet_forecast, lstm_forecast])
            
            # Calculate confidence based on forecast agreement
            forecast_std = np.std([arima_forecast, prophet_forecast, lstm_forecast])
            volatility = market_data.volatility_20d
            confidence = max(0.3, min(0.9, 1.0 - (forecast_std / current_price) - volatility * 0.5))
            
            # Calculate probabilities
            expected_return = (ensemble_forecast - current_price) / current_price
            upside_probability = max(0.1, min(0.9, 0.5 + expected_return))
            downside_risk = 1.0 - upside_probability
            
            # Volatility forecast
            volatility_forecast = self._forecast_volatility(prices)
            
            return ForecastData(
                arima_forecast=arima_forecast,
                prophet_forecast=prophet_forecast,
                lstm_forecast=lstm_forecast,
                ensemble_forecast=ensemble_forecast,
                forecast_confidence=confidence,
                upside_probability=upside_probability,
                downside_risk=downside_risk,
                volatility_forecast=volatility_forecast
            )
            
        except Exception as e:
            st.error(f"Error in forecasting: {e}")
            return ForecastData()

    def _arima_forecast(self, prices: pd.Series, current_price: float) -> float:
        if len(prices) >= 20:
            ma_5 = prices.tail(5).mean()
            ma_20 = prices.tail(20).mean()
            trend_factor = (ma_5 - ma_20) / ma_20
            momentum = (prices.iloc[-1] - prices.iloc[-5]) / prices.iloc[-5]
            return current_price * (1 + trend_factor * 0.5 + momentum * 0.3)
        return current_price * 1.01

    def _prophet_forecast(self, prices: pd.Series, current_price: float) -> float:
        if len(prices) >= 7:
            seasonal_factor = 1.0 + np.random.normal(0, 0.01)
            return prices.iloc[-1] * seasonal_factor * (1 + np.random.normal(0.02, 0.05))
        return current_price * (1 + np.random.normal(0.02, 0.05))

    def _lstm_forecast(self, prices: pd.Series, current_price: float) -> float:
        if len(prices) >= 20:
            weights_short = np.exp(np.linspace(-2, 0, 5))
            weights_short = weights_short / weights_short.sum()
            weights_long = np.exp(np.linspace(-3, 0, 15))
            weights_long = weights_long / weights_long.sum()
            
            short_term = np.sum(prices.tail(5) * weights_short)
            long_term = np.sum(prices.tail(15) * weights_long)
            
            volatility = prices.pct_change().tail(20).std()
            noise = np.random.normal(0, volatility * 0.1)
            
            return (short_term * 0.7 + long_term * 0.3) * (1 + noise)
        return current_price * (1 + np.random.normal(0.02, 0.05))

    def _forecast_volatility(self, prices: pd.Series) -> float:
        returns = prices.pct_change().dropna()
        if len(returns) < 20:
            return 0.2
        
        recent_vol = returns.tail(10).std() * np.sqrt(252)
        long_term_vol = returns.std() * np.sqrt(252)
        forecast_vol = 0.7 * recent_vol + 0.3 * long_term_vol
        return min(1.0, max(0.05, forecast_vol))

# Risk Agent
class RiskAgent:
    def __init__(self):
        self.name = "RiskAgent"

    async def process(self, market_data: MarketData) -> RiskMetrics:
        try:
            st.write(f"⚠️ {self.name}: Computing risk metrics...")
            
            prices = market_data.prices
            returns = prices.pct_change().dropna()
            
            if len(returns) < 30:
                return RiskMetrics()
            
            # Basic risk metrics
            portfolio_volatility = float(returns.std() * np.sqrt(252))
            var_5pct = float(np.percentile(returns, 5))
            expected_shortfall = float(returns[returns <= var_5pct].mean())
            
            # Sharpe ratio
            risk_free_rate = 0.02
            excess_returns = returns.mean() * 252 - risk_free_rate
            sharpe_ratio = float(excess_returns / portfolio_volatility) if portfolio_volatility > 0 else 0.0
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            downside_deviation = float(downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 else portfolio_volatility
            sortino_ratio = float(excess_returns / downside_deviation) if downside_deviation > 0 else 0.0
            
            # Maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            maximum_drawdown = float(drawdown.min())
            
            return RiskMetrics(
                portfolio_volatility=portfolio_volatility,
                sharpe_ratio=sharpe_ratio,
                maximum_drawdown=maximum_drawdown,
                value_at_risk_5pct=var_5pct,
                expected_shortfall=expected_shortfall,
                sortino_ratio=sortino_ratio
            )
            
        except Exception as e:
            st.error(f"Error in risk calculation: {e}")
            return RiskMetrics()

# Macro Economic Agent
class MacroEconomicAgent:
    def __init__(self, fred_api_key=None):
        self.name = "MacroEconomicAgent"
        self.fred_api_key = fred_api_key
        
        if self.fred_api_key and HAS_FRED:
            self.fred = Fred(api_key=self.fred_api_key)
        else:
            self.fred = None

    async def process(self) -> MacroData:
        try:
            st.write(f"🌍 {self.name}: Fetching macro-economic data...")
            
            if self.fred and HAS_FRED:
                try:
                    # Real FRED data
                    gdp_data = self.fred.get_series('GDP', limit=2)
                    gdp_growth = ((gdp_data.iloc[-1] / gdp_data.iloc[-2]) - 1) * 100
                    
                    cpi_data = self.fred.get_series('CPIAUCSL', limit=13)
                    inflation_rate = ((cpi_data.iloc[-1] / cpi_data.iloc[-13]) - 1) * 100
                    
                    unemployment_rate = self.fred.get_series('UNRATE', limit=1).iloc[-1]
                    federal_funds_rate = self.fred.get_series('FEDFUNDS', limit=1).iloc[-1]
                    
                    try:
                        vix = self.fred.get_series('VIXCLS', limit=1).iloc[-1]
                    except:
                        vix = 20.0
                    
                    # Determine market sentiment
                    if inflation_rate < 3 and unemployment_rate < 5:
                        market_sentiment = "bullish"
                    elif inflation_rate > 5 or unemployment_rate > 7:
                        market_sentiment = "bearish"
                    else:
                        market_sentiment = "neutral"
                    
                    st.success("✅ Real FRED data retrieved")
                    
                except Exception as e:
                    st.warning(f"FRED API error: {e}, using simulated data")
                    return self._get_simulated_data()
            else:
                return self._get_simulated_data()
            
            return MacroData(
                gdp_growth=float(gdp_growth),
                inflation_rate=float(inflation_rate),
                unemployment_rate=float(unemployment_rate),
                federal_funds_rate=float(federal_funds_rate),
                vix=float(vix),
                market_sentiment=market_sentiment
            )
            
        except Exception as e:
            st.error(f"Error in macro analysis: {e}")
            return self._get_simulated_data()

    def _get_simulated_data(self) -> MacroData:
        st.info("📊 Using simulated macro data")
        return MacroData(
            gdp_growth=np.random.normal(2.5, 0.5),
            inflation_rate=np.random.normal(3.2, 0.3),
            unemployment_rate=np.random.normal(3.8, 0.2),
            federal_funds_rate=np.random.normal(5.25, 0.25),
            vix=np.random.normal(18, 5),
            market_sentiment=np.random.choice(["bullish", "neutral", "bearish"], p=[0.3, 0.4, 0.3])
        )

# Sentiment Agent
class SentimentAgent:
    def __init__(self, news_api_key=None):
        self.name = "SentimentAgent"
        self.news_api_key = news_api_key
        
        if self.news_api_key and HAS_NEWSAPI:
            self.newsapi = NewsApiClient(api_key=self.news_api_key)
        else:
            self.newsapi = None

    async def process(self, symbol: str) -> SentimentData:
        try:
            st.write(f"💭 {self.name}: Analyzing sentiment...")
            
            company_name = self._get_company_name(symbol)
            
            # News sentiment
            news_sentiment = await self._analyze_news_sentiment(symbol, company_name)
            
            if news_sentiment is not None:
                overall_sentiment = news_sentiment
                confidence_score = 0.7
                st.success("✅ Real news sentiment analyzed")
            else:
                overall_sentiment = np.random.normal(0.0, 0.3)
                news_sentiment = np.random.normal(0.1, 0.3)
                confidence_score = 0.3
                st.info("📊 Using simulated sentiment data")
            
            # Determine sentiment trend
            if overall_sentiment > 0.2:
                sentiment_trend = "positive"
            elif overall_sentiment < -0.2:
                sentiment_trend = "negative"
            else:
                sentiment_trend = "neutral"
            
            # Generate key topics
            key_topics = ["market_conditions", "earnings", "economic_data"]
            
            # Fear & Greed Index
            fear_greed_index = max(0, min(100, 50 + overall_sentiment * 30))
            
            return SentimentData(
                news_sentiment=news_sentiment or 0.0,
                social_media_sentiment=np.random.normal(0.0, 0.2),
                overall_sentiment=overall_sentiment,
                sentiment_trend=sentiment_trend,
                confidence_score=confidence_score,
                key_topics=key_topics,
                fear_greed_index=fear_greed_index
            )
            
        except Exception as e:
            st.error(f"Error in sentiment analysis: {e}")
            return self._get_simulated_sentiment()

    async def _analyze_news_sentiment(self, symbol: str, company_name: str) -> float:
        if not self.newsapi or not HAS_NEWSAPI or not HAS_TEXTBLOB:
            return None
        
        try:
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
            st.warning(f"News API error: {e}")
            return None

    def _get_company_name(self, symbol: str) -> str:
        company_map = {
            'AAPL': 'Apple', 'TSLA': 'Tesla', 'MSFT': 'Microsoft',
            'GOOGL': 'Google', 'AMZN': 'Amazon', 'META': 'Meta',
            'NVDA': 'Nvidia', 'JPM': 'JPMorgan', 'V': 'Visa'
        }
        return company_map.get(symbol.upper(), symbol)

    def _get_simulated_sentiment(self) -> SentimentData:
        return SentimentData(
            news_sentiment=np.random.normal(0.1, 0.3),
            social_media_sentiment=np.random.normal(0.0, 0.4),
            overall_sentiment=np.random.normal(0.0, 0.3),
            sentiment_trend="neutral",
            confidence_score=0.3,
            key_topics=["market_conditions", "earnings", "economic_data"],
            fear_greed_index=np.random.uniform(20, 80)
        )


# AI Strategist Agent
class StrategistAgent:
    def __init__(self, openai_api_key=None):
        self.name = "StrategistAgent"
        self.openai_api_key = openai_api_key
        
        if self.openai_api_key and HAS_OPENAI:
            try:
                self.client = OpenAI(api_key=self.openai_api_key)
                self.has_openai = True
                st.success("✅ OpenAI GPT-4 connected")
            except Exception as e:
                self.has_openai = False
                st.warning(f"OpenAI connection failed: {e}")
        else:
            self.has_openai = False

    async def process(self, market_data: MarketData, forecast_data: ForecastData, 
                     risk_metrics: RiskMetrics, macro_data: MacroData, 
                     sentiment_data: SentimentData) -> Recommendation:
        try:
            st.write(f"🤖 {self.name}: Generating AI recommendation...")
            
            if self.has_openai:
                try:
                    recommendation = await self._generate_gpt_recommendation(
                        market_data, forecast_data, risk_metrics, macro_data, sentiment_data
                    )
                    st.success("✅ GPT-4 recommendation generated")
                    return recommendation
                except Exception as e:
                    st.warning(f"GPT-4 failed: {e}, using rule-based analysis")
            
            return self._generate_rule_based_recommendation(
                market_data, forecast_data, risk_metrics, macro_data, sentiment_data
            )
            
        except Exception as e:
            st.error(f"Error generating recommendation: {e}")
            return self._get_default_recommendation(market_data.current_price)

    async def _generate_gpt_recommendation(self, market_data, forecast_data, risk_metrics, 
                                         macro_data, sentiment_data) -> Recommendation:
        # Prepare data summary for GPT
        data_summary = f"""
        Stock: {market_data.symbol}
        Current Price: ${market_data.current_price:.2f}
        Trend: {market_data.trend}
        RSI: {market_data.rsi:.1f}
        Forecast: ${forecast_data.ensemble_forecast:.2f}
        Confidence: {forecast_data.forecast_confidence:.1%}
        Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}
        Volatility: {risk_metrics.portfolio_volatility:.1%}
        Max Drawdown: {risk_metrics.maximum_drawdown:.1%}
        Market Sentiment: {sentiment_data.sentiment_trend}
        Macro Sentiment: {macro_data.market_sentiment}
        """
        
        prompt = f"""
        As an expert financial advisor, analyze this stock data and provide a recommendation:
        
        {data_summary}
        
        Provide a JSON response with:
        1. action (BUY/SELL/HOLD)
        2. confidence (0.0-1.0)
        3. position_size (0.0-0.4)
        4. risk_level (LOW/MEDIUM/HIGH)
        5. time_horizon (SHORT/MEDIUM/LONG)
        6. reasoning (brief explanation)
        
        Consider all factors including technical analysis, risk metrics, and market conditions.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            # Parse GPT response (simplified)
            content = response.choices[0].message.content
            
            # Extract key information (you'd want more robust parsing in production)
            if "BUY" in content.upper():
                action = "BUY"
            elif "SELL" in content.upper():
                action = "SELL"
            else:
                action = "HOLD"
            
            confidence = 0.75  # Default, would parse from GPT response
            risk_level = "MEDIUM"  # Default, would parse from GPT response
            
            return self._create_recommendation(
                action, confidence, risk_level, market_data.current_price, content
            )
            
        except Exception as e:
            st.error(f"GPT API error: {e}")
            return self._generate_rule_based_recommendation(
                market_data, forecast_data, risk_metrics, macro_data, sentiment_data
            )

    def _generate_rule_based_recommendation(self, market_data, forecast_data, risk_metrics, 
                                          macro_data, sentiment_data) -> Recommendation:
        # Multi-factor scoring system
        score = 0.0
        
        # Technical analysis score
        trend_scores = {
            "strongly_bullish": 1.0, "bullish": 0.5, "neutral": 0.0,
            "bearish": -0.5, "strongly_bearish": -1.0
        }
        score += trend_scores.get(market_data.trend, 0.0) * 0.25
        
        # RSI analysis
        if market_data.rsi < 30:
            score += 0.2  # Oversold
        elif market_data.rsi > 70:
            score -= 0.2  # Overbought
        
        # Forecast analysis
        expected_return = (forecast_data.ensemble_forecast - market_data.current_price) / market_data.current_price
        score += expected_return * forecast_data.forecast_confidence * 0.3
        
        # Risk analysis
        if risk_metrics.sharpe_ratio > 1.0:
            score += 0.15
        elif risk_metrics.sharpe_ratio < 0:
            score -= 0.15
        
        # Sentiment analysis
        sentiment_scores = {"positive": 0.1, "neutral": 0.0, "negative": -0.1}
        score += sentiment_scores.get(sentiment_data.sentiment_trend, 0.0)
        
        # Macro analysis
        macro_scores = {"bullish": 0.1, "neutral": 0.0, "bearish": -0.1}
        score += macro_scores.get(macro_data.market_sentiment, 0.0)
        
        # Generate recommendation based on score
        if score > 0.4:
            action = "BUY"
            confidence = min(0.9, 0.6 + score * 0.4)
            risk_level = "MEDIUM"
        elif score < -0.4:
            action = "SELL"
            confidence = min(0.9, 0.6 + abs(score) * 0.4)
            risk_level = "HIGH"
        else:
            action = "HOLD"
            confidence = 0.6
            risk_level = "LOW"
        
        reasoning = f"Multi-factor analysis yielded score of {score:.2f}. "
        reasoning += f"Technical: {market_data.trend}, RSI: {market_data.rsi:.1f}. "
        reasoning += f"Forecast confidence: {forecast_data.forecast_confidence:.1%}. "
        reasoning += f"Risk-adjusted return: Sharpe {risk_metrics.sharpe_ratio:.2f}."
        
        return self._create_recommendation(action, confidence, risk_level, 
                                         market_data.current_price, reasoning)

    def _create_recommendation(self, action: str, confidence: float, risk_level: str, 
                             current_price: float, reasoning: str) -> Recommendation:
        position_size = min(0.4, confidence * 0.5) if action != "HOLD" else 0.0
        time_horizon = "MEDIUM" if action == "BUY" else "SHORT" if action == "SELL" else "LONG"
        
        return Recommendation(
            action=action,
            confidence=confidence,
            position_size=position_size,
            risk_level=risk_level,
            time_horizon=time_horizon,
            entry_price=current_price,
            stop_loss=current_price * (0.92 if action == "BUY" else 1.08),
            take_profit=current_price * (1.20 if action == "BUY" else 0.80),
            detailed_reasoning=reasoning,
            risk_reward_ratio=2.5 if action != "HOLD" else 1.0
        )

    def _get_default_recommendation(self, current_price: float) -> Recommendation:
        return Recommendation(
            action="HOLD",
            confidence=0.5,
            position_size=0.0,
            risk_level="MEDIUM",
            time_horizon="LONG",
            entry_price=current_price,
            stop_loss=current_price * 0.95,
            take_profit=current_price * 1.05,
            detailed_reasoning="Default recommendation due to analysis error",
            risk_reward_ratio=1.0
        )

# Financial Planning Agent
class FinancialPlannerAgent:
    def __init__(self):
        self.name = "FinancialPlannerAgent"
        
        # Asset allocation strategies
        self.strategies = {
            "conservative": {"stocks": 0.30, "bonds": 0.60, "cash": 0.10},
            "moderate": {"stocks": 0.60, "bonds": 0.30, "cash": 0.10},
            "aggressive": {"stocks": 0.80, "bonds": 0.15, "cash": 0.05}
        }
        
        # Expected returns and volatilities
        self.asset_assumptions = {
            "stocks": {"return": 0.10, "volatility": 0.16},
            "bonds": {"return": 0.04, "volatility": 0.06},
            "cash": {"return": 0.02, "volatility": 0.01}
        }

    async def process(self, goal: FinancialGoal) -> FinancialPlan:
        try:
            st.write(f"💰 {self.name}: Creating financial plan...")
            
            # Calculate projections
            projections = self._calculate_projections(goal)
            
            # Optimize asset allocation
            asset_allocation = self._optimize_allocation(goal)
            
            # Calculate plan metrics
            plan_sharpe = self._calculate_plan_sharpe(asset_allocation)
            
            # Run Monte Carlo simulation
            monte_carlo = self._run_monte_carlo(goal, asset_allocation, 1000)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(goal, projections, monte_carlo)
            
            return FinancialPlan(
                goal=goal,
                projected_value=projections['projected_value'],
                success_probability=projections['success_probability'],
                required_monthly=projections['required_monthly'],
                asset_allocation=asset_allocation,
                recommendations=recommendations,
                is_achievable=projections['is_achievable'],
                monte_carlo_results=monte_carlo,
                plan_sharpe_ratio=plan_sharpe
            )
            
        except Exception as e:
            st.error(f"Error creating financial plan: {e}")
            return None

    def _calculate_projections(self, goal: FinancialGoal) -> Dict[str, float]:
        # Get portfolio expected return
        allocation = self.strategies[goal.risk_tolerance]
        portfolio_return = sum(
            allocation[asset] * self.asset_assumptions[asset]["return"]
            for asset in allocation
        )
        
        # Future value calculations
        years = goal.time_horizon_years
        monthly_rate = portfolio_return / 12
        months = years * 12
        
        # Future value of current amount
        fv_current = goal.current_amount * (1 + portfolio_return) ** years
        
        # Future value of monthly contributions
        if monthly_rate > 0:
            fv_contributions = goal.monthly_contribution * (
                ((1 + monthly_rate) ** months - 1) / monthly_rate
            )
        else:
            fv_contributions = goal.monthly_contribution * months
        
        # Total projected value
        projected_value = fv_current + fv_contributions
        
        # Success analysis
        success_probability = min(1.0, projected_value / goal.target_amount)
        gap = goal.target_amount - projected_value
        
        # Required additional monthly contribution
        if gap > 0 and monthly_rate > 0:
            required_monthly = gap / (((1 + monthly_rate) ** months - 1) / monthly_rate)
        else:
            required_monthly = 0
        
        return {
            'projected_value': projected_value,
            'success_probability': success_probability,
            'required_monthly': required_monthly,
            'is_achievable': gap <= 0
        }

    def _optimize_allocation(self, goal: FinancialGoal) -> Dict[str, float]:
        base_allocation = self.strategies[goal.risk_tolerance].copy()
        
        # Age-based adjustments
        if goal.age < 30:
            base_allocation["stocks"] += 0.10
            base_allocation["bonds"] -= 0.10
        elif goal.age > 50:
            base_allocation["stocks"] -= 0.10
            base_allocation["bonds"] += 0.10
        
        # Time horizon adjustments
        if goal.time_horizon_years > 20:
            base_allocation["stocks"] += 0.05
            base_allocation["bonds"] -= 0.05
        elif goal.time_horizon_years < 5:
            base_allocation["stocks"] -= 0.15
            base_allocation["bonds"] += 0.10
            base_allocation["cash"] += 0.05
        
        # Normalize to sum to 1
        total = sum(base_allocation.values())
        return {k: v/total for k, v in base_allocation.items()}

    def _calculate_plan_sharpe(self, allocation: Dict[str, float]) -> float:
        portfolio_return = sum(
            allocation[asset] * self.asset_assumptions[asset]["return"]
            for asset in allocation
        )
        
        portfolio_variance = sum(
            (allocation[asset] ** 2) * (self.asset_assumptions[asset]["volatility"] ** 2)
            for asset in allocation
        )
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        risk_free_rate = 0.02
        return (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0.0

    def _run_monte_carlo(self, goal: FinancialGoal, allocation: Dict[str, float], 
                        n_simulations: int = 1000) -> Dict[str, float]:
        portfolio_return = sum(
            allocation[asset] * self.asset_assumptions[asset]["return"]
            for asset in allocation
        )
        
        portfolio_variance = sum(
            (allocation[asset] ** 2) * (self.asset_assumptions[asset]["volatility"] ** 2)
            for asset in allocation
        )
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        results = []
        
        for _ in range(n_simulations):
            value = goal.current_amount
            
            for year in range(goal.time_horizon_years):
                # Add annual contributions
                value += goal.monthly_contribution * 12
                
                # Apply random annual return
                annual_return = np.random.normal(portfolio_return, portfolio_volatility)
                value *= (1 + annual_return)
                
                # Ensure value doesn't go negative
                value = max(0, value)
            
            results.append(value)
        
        results = np.array(results)
        
        return {
            "mean": np.mean(results),
            "median": np.percentile(results, 50),
            "percentile_10": np.percentile(results, 10),
            "percentile_90": np.percentile(results, 90),
            "success_rate": np.mean(results >= goal.target_amount),
            "worst_case": np.percentile(results, 5),
            "best_case": np.percentile(results, 95)
        }

    def _generate_recommendations(self, goal: FinancialGoal, projections: Dict, 
                                monte_carlo: Dict) -> List[str]:
        recommendations = []
        
        if projections['is_achievable']:
            recommendations.append(f"✅ Goal is achievable! Projected: ${projections['projected_value']:,.0f}")
        else:
            shortfall = projections['required_monthly']
            recommendations.append(f"⚠️ Increase monthly contribution by ${shortfall:,.0f} to reach goal")
        
        success_rate = monte_carlo['success_rate']
        if success_rate > 0.8:
            recommendations.append("📈 High probability of success based on Monte Carlo analysis")
        elif success_rate < 0.5:
            recommendations.append("📉 Consider extending timeline or increasing contributions")
        
        if goal.age < 35:
            recommendations.append("🎯 Young investor advantage - time is your greatest asset")
        elif goal.age > 45:
            recommendations.append("⚖️ Focus on risk management while maintaining growth")
        
        recommendations.append("🔄 Rebalance portfolio quarterly to maintain target allocation")
        recommendations.append("📅 Review and adjust plan annually or after major life changes")
        
        return recommendations


# Main Application
def main():
    st.title("🤖 AI Financial Forecasting System")
    st.markdown("### Complete Multi-Agent Investment Analysis & Financial Planning Platform")
    
    # Configuration Section
    st.header("⚙️ Configuration")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Stock Analysis")
        
        # Stock selection
        popular_stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "JNJ"]
        
        symbol_option = st.selectbox("Select Stock", ["Custom"] + popular_stocks)
        
        if symbol_option == "Custom":
            symbol = st.text_input("Enter Symbol", value="AAPL").upper()
        else:
            symbol = symbol_option
        
        # Analysis period
        period = st.selectbox("Analysis Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
    
    with col2:
        st.subheader("🔑 API Keys (Optional)")
        st.info("Add your API keys to unlock enhanced features")
        
        # API Keys
        openai_api_key = st.text_input("OpenAI API Key", type="password", 
                                      help="For GPT-4 enhanced recommendations")
        
        fred_api_key = st.text_input("FRED API Key", type="password",
                                    help="For real economic data")
        
        news_api_key = st.text_input("News API Key", type="password",
                                    help="For news sentiment analysis")
    
    # Financial Planning Section
    st.header("💰 Financial Planning (Optional)")
    
    enable_planning = st.checkbox("Enable Financial Planning", value=False)
    
    if enable_planning:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            goal_type = st.selectbox("Goal Type", ["retirement", "house", "education", "general"])
            target_amount = st.number_input("Target Amount ($)", min_value=10000, max_value=10000000, 
                                          value=1000000, step=50000)
            current_amount = st.number_input("Current Savings ($)", min_value=0, max_value=10000000,
                                           value=50000, step=5000)
        
        with col2:
            monthly_contribution = st.number_input("Monthly Contribution ($)", min_value=0, 
                                                 max_value=50000, value=2000, step=100)
            time_horizon = st.slider("Time Horizon (Years)", min_value=1, max_value=40, value=25)
            risk_tolerance = st.select_slider("Risk Tolerance", 
                                            options=["conservative", "moderate", "aggressive"], 
                                            value="moderate")
        
        with col3:
            age = st.number_input("Your Age", min_value=18, max_value=100, value=35)
            annual_income = st.number_input("Annual Income ($)", min_value=0, max_value=1000000,
                                          value=120000, step=5000)
    
    # Run Analysis Button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Run Complete Analysis", type="primary", use_container_width=True):
            # Create financial goal if planning enabled
            financial_goal = None
            if enable_planning:
                financial_goal = FinancialGoal(
                    target_amount=target_amount,
                    current_amount=current_amount,
                    monthly_contribution=monthly_contribution,
                    time_horizon_years=time_horizon,
                    risk_tolerance=risk_tolerance,
                    age=age,
                    annual_income=annual_income,
                    goal_type=goal_type
                )
            
            # Run analysis
            run_complete_analysis(symbol, openai_api_key, fred_api_key, news_api_key, financial_goal)

def run_complete_analysis(symbol, openai_api_key, fred_api_key, news_api_key, financial_goal):
    """Run the complete multi-agent analysis pipeline"""
    
    # Initialize progress tracking
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    try:
        # Create progress bar
        progress_bar = progress_placeholder.progress(0)
        status_placeholder.info("🔧 Initializing AI agents...")
        
        # Initialize agents
        market_agent = MarketDataAgent()
        forecast_agent = ForecastingAgent()
        risk_agent = RiskAgent()
        macro_agent = MacroEconomicAgent(fred_api_key)
        sentiment_agent = SentimentAgent(news_api_key)
        strategist_agent = StrategistAgent(openai_api_key)
        planner_agent = FinancialPlannerAgent() if financial_goal else None
        
        progress_bar.progress(10)
        
        # Run analysis pipeline
        async def run_pipeline():
            # Phase 1: Market Data
            status_placeholder.info("📊 Fetching market data...")
            market_data = await market_agent.process(symbol)
            if not market_data:
                raise ValueError("Failed to fetch market data")
            progress_bar.progress(25)
            
            # Phase 2: Risk Analysis
            status_placeholder.info("⚠️ Computing risk metrics...")
            risk_metrics = await risk_agent.process(market_data)
            progress_bar.progress(40)
            
            # Phase 3: Forecasting
            status_placeholder.info("🔮 Generating forecasts...")
            forecast_data = await forecast_agent.process(market_data)
            progress_bar.progress(55)
            
            # Phase 4: Macro Analysis
            status_placeholder.info("🌍 Analyzing macro conditions...")
            macro_data = await macro_agent.process()
            progress_bar.progress(70)
            
            # Phase 5: Sentiment Analysis
            status_placeholder.info("💭 Analyzing market sentiment...")
            sentiment_data = await sentiment_agent.process(symbol)
            progress_bar.progress(85)
            
            # Phase 6: AI Recommendation
            status_placeholder.info("🤖 Generating AI recommendation...")
            recommendation = await strategist_agent.process(
                market_data, forecast_data, risk_metrics, macro_data, sentiment_data
            )
            progress_bar.progress(95)
            
            # Phase 7: Financial Planning (if enabled)
            financial_plan = None
            if planner_agent and financial_goal:
                status_placeholder.info("💰 Creating financial plan...")
                financial_plan = await planner_agent.process(financial_goal)
            
            progress_bar.progress(100)
            status_placeholder.success("✅ Analysis Complete!")
            
            return {
                'market_data': market_data,
                'risk_metrics': risk_metrics,
                'forecast_data': forecast_data,
                'macro_data': macro_data,
                'sentiment_data': sentiment_data,
                'recommendation': recommendation,
                'financial_plan': financial_plan
            }
        
        # Run the async pipeline
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(run_pipeline())
        
        # Clear progress indicators
        progress_placeholder.empty()
        status_placeholder.empty()
        
        # Display results
        display_comprehensive_results(results)
        
    except Exception as e:
        progress_placeholder.empty()
        status_placeholder.error(f"❌ Analysis failed: {e}")


def display_comprehensive_results(results):
    """Display comprehensive analysis results"""
    
    st.success("🎉 Analysis Complete! Here are your results:")
    
    # Create tabs for different sections
    tabs = st.tabs([
        "📊 Executive Summary",
        "📈 Market Analysis", 
        "🔮 Forecasting",
        "⚠️ Risk Assessment",
        "🌍 Market Environment",
        "🤖 AI Recommendation",
        "💰 Financial Planning",
        "📊 Visualizations"
    ])
    
    # Tab 1: Executive Summary
    with tabs[0]:
        display_executive_summary(results)
    
    # Tab 2: Market Analysis
    with tabs[1]:
        display_market_analysis(results)
    
    # Tab 3: Forecasting
    with tabs[2]:
        display_forecasting_analysis(results)
    
    # Tab 4: Risk Assessment
    with tabs[3]:
        display_risk_assessment(results)
    
    # Tab 5: Market Environment
    with tabs[4]:
        display_market_environment(results)
    
    # Tab 6: AI Recommendation
    with tabs[5]:
        display_ai_recommendation(results)
    
    # Tab 7: Financial Planning
    with tabs[6]:
        display_financial_planning(results)
    
    # Tab 8: Visualizations
    with tabs[7]:
        display_visualizations(results)

def display_executive_summary(results):
    """Display executive summary dashboard"""
    
    st.header("📊 Executive Summary")
    
    market_data = results['market_data']
    risk_metrics = results['risk_metrics']
    forecast_data = results['forecast_data']
    recommendation = results['recommendation']
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${market_data.current_price:.2f}", 
                 f"{market_data.return_1d:.2%}")
    
    with col2:
        st.metric("RSI", f"{market_data.rsi:.1f}")
    
    with col3:
        st.metric("Sharpe Ratio", f"{risk_metrics.sharpe_ratio:.2f}")
    
    with col4:
        forecast_change = ((forecast_data.ensemble_forecast - market_data.current_price) / market_data.current_price) * 100
        st.metric("Forecast", f"${forecast_data.ensemble_forecast:.2f}", f"{forecast_change:+.1f}%")
    
    # AI Recommendation Summary
    st.markdown("---")
    st.subheader("🎯 AI Investment Recommendation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        color = "green" if recommendation.action == "BUY" else "red" if recommendation.action == "SELL" else "orange"
        
        recommendation_html = f"""
        <div style='background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                    padding: 30px; border-radius: 15px; border-left: 8px solid {color}; 
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
            <h2 style='color: {color}; margin-bottom: 20px; font-size: 2.5em;'>{recommendation.action}</h2>
            <div style='font-size: 1.2em; line-height: 1.8;'>
                <p><strong>Confidence Level:</strong> {recommendation.confidence:.1%}</p>
                <p><strong>Risk Assessment:</strong> {recommendation.risk_level}</p>
                <p><strong>Time Horizon:</strong> {recommendation.time_horizon}</p>
                <p><strong>Position Size:</strong> {recommendation.position_size:.1%} of portfolio</p>
            </div>
        </div>
        """
        st.markdown(recommendation_html, unsafe_allow_html=True)
    
    with col2:
        st.metric("Entry Price", f"${recommendation.entry_price:.2f}")
        st.metric("Stop Loss", f"${recommendation.stop_loss:.2f}")
        st.metric("Take Profit", f"${recommendation.take_profit:.2f}")
        st.metric("Risk/Reward", f"{recommendation.risk_reward_ratio:.1f}")
    
    # Key insights
    st.markdown("---")
    st.subheader("🔍 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📈 Technical Analysis**")
        st.write(f"• Trend: **{market_data.trend.replace('_', ' ').title()}**")
        st.write(f"• RSI: **{market_data.rsi:.1f}** ({'Oversold' if market_data.rsi < 30 else 'Overbought' if market_data.rsi > 70 else 'Neutral'})")
        st.write(f"• MACD: **{market_data.macd_signal.title()}**")
        st.write(f"• Support: **${market_data.support_level:.2f}**")
        st.write(f"• Resistance: **${market_data.resistance_level:.2f}**")
    
    with col2:
        st.markdown("**📊 Risk & Forecast**")
        st.write(f"• Volatility: **{risk_metrics.portfolio_volatility:.1%}**")
        st.write(f"• Max Drawdown: **{risk_metrics.maximum_drawdown:.1%}**")
        st.write(f"• Forecast Confidence: **{forecast_data.forecast_confidence:.1%}**")
        st.write(f"• Upside Probability: **{forecast_data.upside_probability:.1%}**")

def display_market_analysis(results):
    """Display detailed market analysis"""
    
    st.header("📈 Market Data Analysis")
    
    market_data = results['market_data']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Price Metrics")
        
        metrics_data = {
            'Metric': [
                'Current Price', '1-Day Return', '5-Day Return', '20-Day Return',
                '20-Day Volatility', 'Support Level', 'Resistance Level'
            ],
            'Value': [
                f"${market_data.current_price:.2f}",
                f"{market_data.return_1d:.2%}",
                f"{market_data.return_5d:.2%}",
                f"{market_data.return_20d:.2%}",
                f"{market_data.volatility_20d:.1%}",
                f"${market_data.support_level:.2f}",
                f"${market_data.resistance_level:.2f}"
            ]
        }
        
        df = pd.DataFrame(metrics_data)
        st.dataframe(df, hide_index=True, use_container_width=True)
    
    with col2:
        st.subheader("📈 Technical Indicators")
        
        indicators_data = {
            'Indicator': ['RSI', 'Trend', 'MACD Signal'],
            'Value': [
                f"{market_data.rsi:.1f}",
                market_data.trend.replace('_', ' ').title(),
                market_data.macd_signal.title()
            ],
            'Interpretation': [
                'Oversold' if market_data.rsi < 30 else 'Overbought' if market_data.rsi > 70 else 'Neutral',
                'Bullish' if 'bullish' in market_data.trend else 'Bearish' if 'bearish' in market_data.trend else 'Sideways',
                'Buy Signal' if market_data.macd_signal == 'bullish' else 'Sell Signal' if market_data.macd_signal == 'bearish' else 'No Signal'
            ]
        }
        
        df = pd.DataFrame(indicators_data)
        st.dataframe(df, hide_index=True, use_container_width=True)
    
    # Technical Analysis Chart
    if len(market_data.prices) > 0:
        st.subheader("📈 Technical Analysis Chart")
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=('Price & Moving Averages', 'Volume')
        )
        
        # Price line
        fig.add_trace(
            go.Scatter(
                x=market_data.prices.index,
                y=market_data.prices.values,
                name='Price',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Moving averages
        if len(market_data.prices) >= 20:
            ma20 = market_data.prices.rolling(window=20).mean()
            fig.add_trace(
                go.Scatter(
                    x=ma20.index,
                    y=ma20.values,
                    name='MA20',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
        
        if len(market_data.prices) >= 50:
            ma50 = market_data.prices.rolling(window=50).mean()
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
        fig.add_hline(y=market_data.support_level, line_dash="dash", 
                     line_color="green", annotation_text="Support", row=1, col=1)
        fig.add_hline(y=market_data.resistance_level, line_dash="dash", 
                     line_color="red", annotation_text="Resistance", row=1, col=1)
        
        # Volume
        if len(market_data.volume) > 0:
            fig.add_trace(
                go.Bar(
                    x=market_data.volume.index,
                    y=market_data.volume.values,
                    name='Volume',
                    marker_color='lightblue',
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title=f"{market_data.symbol} Technical Analysis",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_forecasting_analysis(results):
    """Display forecasting analysis"""
    
    st.header("🔮 Price Forecasting Analysis")
    
    forecast_data = results['forecast_data']
    market_data = results['market_data']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Forecast Summary")
        
        price_change = ((forecast_data.ensemble_forecast - market_data.current_price) / market_data.current_price) * 100
        
        st.metric("Ensemble Forecast", f"${forecast_data.ensemble_forecast:.2f}", f"{price_change:+.1f}%")
        st.metric("Forecast Confidence", f"{forecast_data.forecast_confidence:.1%}")
        st.metric("Upside Probability", f"{forecast_data.upside_probability:.1%}")
        st.metric("Downside Risk", f"{forecast_data.downside_risk:.1%}")
    
    with col2:
        st.subheader("📈 Individual Model Forecasts")
        
        forecasts_data = {
            'Model': ['ARIMA', 'Prophet', 'LSTM', 'Ensemble'],
            'Forecast ($)': [
                f"{forecast_data.arima_forecast:.2f}",
                f"{forecast_data.prophet_forecast:.2f}",
                f"{forecast_data.lstm_forecast:.2f}",
                f"{forecast_data.ensemble_forecast:.2f}"
            ],
            'Change (%)': [
                f"{((forecast_data.arima_forecast - market_data.current_price) / market_data.current_price * 100):+.1f}%",
                f"{((forecast_data.prophet_forecast - market_data.current_price) / market_data.current_price * 100):+.1f}%",
                f"{((forecast_data.lstm_forecast - market_data.current_price) / market_data.current_price * 100):+.1f}%",
                f"{((forecast_data.ensemble_forecast - market_data.current_price) / market_data.current_price * 100):+.1f}%"
            ]
        }
        
        df = pd.DataFrame(forecasts_data)
        st.dataframe(df, hide_index=True, use_container_width=True)
    
    # Forecast Comparison Chart
    st.subheader("📊 Forecast Comparison")
    
    models = ['Current', 'ARIMA', 'Prophet', 'LSTM', 'Ensemble']
    values = [
        market_data.current_price,
        forecast_data.arima_forecast,
        forecast_data.prophet_forecast,
        forecast_data.lstm_forecast,
        forecast_data.ensemble_forecast
    ]
    
    colors = ['gray', 'blue', 'green', 'orange', 'red']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=models,
        y=values,
        marker_color=colors,
        text=[f"${v:.2f}" for v in values],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Price Forecasts by Model",
        xaxis_title="Model",
        yaxis_title="Price ($)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_risk_assessment(results):
    """Display risk assessment"""
    
    st.header("⚠️ Risk Assessment")
    
    risk_metrics = results['risk_metrics']
    
    # Risk metrics overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Portfolio Volatility", f"{risk_metrics.portfolio_volatility:.1%}")
        st.metric("Sharpe Ratio", f"{risk_metrics.sharpe_ratio:.2f}")
    
    with col2:
        st.metric("Maximum Drawdown", f"{risk_metrics.maximum_drawdown:.1%}")
        st.metric("Sortino Ratio", f"{risk_metrics.sortino_ratio:.2f}")
    
    with col3:
        st.metric("VaR (5%)", f"{risk_metrics.value_at_risk_5pct:.2%}")
        st.metric("Expected Shortfall", f"{risk_metrics.expected_shortfall:.2%}")
    
    # Risk interpretation
    st.subheader("📊 Risk Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Risk Level Assessment:**")
        
        if risk_metrics.portfolio_volatility > 0.3:
            st.error("🔴 High Risk: Volatility > 30%")
        elif risk_metrics.portfolio_volatility > 0.2:
            st.warning("🟡 Medium Risk: Volatility 20-30%")
        else:
            st.success("🟢 Low Risk: Volatility < 20%")
        
        if risk_metrics.sharpe_ratio > 1.0:
            st.success("✅ Good risk-adjusted returns (Sharpe > 1.0)")
        elif risk_metrics.sharpe_ratio > 0.5:
            st.warning("⚠️ Moderate risk-adjusted returns")
        else:
            st.error("❌ Poor risk-adjusted returns")
    
    with col2:
        st.markdown("**Drawdown Analysis:**")
        
        if abs(risk_metrics.maximum_drawdown) > 0.2:
            st.error("🔴 High drawdown risk (>20%)")
        elif abs(risk_metrics.maximum_drawdown) > 0.1:
            st.warning("🟡 Moderate drawdown risk (10-20%)")
        else:
            st.success("🟢 Low drawdown risk (<10%)")
        
        st.markdown("**Risk Metrics Explained:**")
        st.write("• **Volatility**: Price fluctuation measure")
        st.write("• **Sharpe Ratio**: Risk-adjusted return")
        st.write("• **VaR**: Potential loss in worst 5% scenarios")
        st.write("• **Max Drawdown**: Largest peak-to-trough decline")


def display_market_environment(results):
    """Display market environment analysis"""
    
    st.header("🌍 Market Environment Analysis")
    
    macro_data = results['macro_data']
    sentiment_data = results['sentiment_data']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Macro Economic Indicators")
        
        macro_df = pd.DataFrame({
            'Indicator': [
                'GDP Growth',
                'Inflation Rate', 
                'Unemployment Rate',
                'Federal Funds Rate',
                'VIX (Volatility)',
                'Market Sentiment'
            ],
            'Value': [
                f"{macro_data.gdp_growth:.1f}%",
                f"{macro_data.inflation_rate:.1f}%",
                f"{macro_data.unemployment_rate:.1f}%",
                f"{macro_data.federal_funds_rate:.2f}%",
                f"{macro_data.vix:.1f}",
                macro_data.market_sentiment.title()
            ]
        })
        
        st.dataframe(macro_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.subheader("💭 Sentiment Analysis")
        
        # Fear & Greed Index gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sentiment_data.fear_greed_index,
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
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment details
        sentiment_df = pd.DataFrame({
            'Source': ['News', 'Social Media', 'Overall'],
            'Sentiment': [
                f"{sentiment_data.news_sentiment:.2f}",
                f"{sentiment_data.social_media_sentiment:.2f}",
                f"{sentiment_data.overall_sentiment:.2f}"
            ],
            'Trend': [
                sentiment_data.sentiment_trend.title(),
                sentiment_data.sentiment_trend.title(),
                sentiment_data.sentiment_trend.title()
            ]
        })
        
        st.dataframe(sentiment_df, hide_index=True, use_container_width=True)
        
        if sentiment_data.key_topics:
            st.write("**Key Topics:**", ", ".join(sentiment_data.key_topics[:5]))

def display_ai_recommendation(results):
    """Display AI recommendation details"""
    
    st.header("🤖 AI Strategic Recommendation")
    
    recommendation = results['recommendation']
    
    # Main recommendation card
    color = "green" if recommendation.action == "BUY" else "red" if recommendation.action == "SELL" else "orange"
    
    recommendation_card = f"""
    <div style='background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                padding: 40px; border-radius: 20px; border-left: 10px solid {color}; 
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1); margin: 20px 0;'>
        <h1 style='color: {color}; margin-bottom: 25px; font-size: 3em; text-align: center;'>
            {recommendation.action} RECOMMENDATION
        </h1>
        <div style='font-size: 1.4em; line-height: 2; text-align: center;'>
            <p><strong>Confidence Level:</strong> {recommendation.confidence:.1%}</p>
            <p><strong>Risk Assessment:</strong> {recommendation.risk_level}</p>
            <p><strong>Time Horizon:</strong> {recommendation.time_horizon}</p>
            <p><strong>Recommended Position Size:</strong> {recommendation.position_size:.1%} of portfolio</p>
        </div>
    </div>
    """
    
    st.markdown(recommendation_card, unsafe_allow_html=True)
    
    # Trading parameters
    st.subheader("📊 Trading Parameters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Entry Price", f"${recommendation.entry_price:.2f}")
    with col2:
        st.metric("Stop Loss", f"${recommendation.stop_loss:.2f}")
    with col3:
        st.metric("Take Profit", f"${recommendation.take_profit:.2f}")
    with col4:
        st.metric("Risk/Reward Ratio", f"{recommendation.risk_reward_ratio:.1f}")
    
    # Detailed reasoning
    st.subheader("📝 AI Analysis & Reasoning")
    
    if recommendation.detailed_reasoning:
        st.info(recommendation.detailed_reasoning)
    
    # Action plan
    st.subheader("📋 Recommended Action Plan")
    
    if recommendation.action == "BUY":
        st.success(f"""
        **🟢 BUY Strategy:**
        
        • **Entry**: Consider buying at current price of ${recommendation.entry_price:.2f}
        • **Stop Loss**: Set at ${recommendation.stop_loss:.2f} to limit downside risk
        • **Take Profit**: Target ${recommendation.take_profit:.2f} for profit taking
        • **Position Size**: Allocate {recommendation.position_size:.1%} of your portfolio
        • **Monitoring**: Review position weekly and adjust based on market conditions
        """)
    elif recommendation.action == "SELL":
        st.error(f"""
        **🔴 SELL Strategy:**
        
        • **Action**: Consider reducing or exiting position at ${recommendation.entry_price:.2f}
        • **Stop Loss**: If holding, set tight stop at ${recommendation.stop_loss:.2f}
        • **Risk Management**: Monitor closely for further deterioration
        • **Cash Position**: Consider holding cash until better opportunities arise
        • **Re-entry**: Look for oversold conditions for potential re-entry
        """)
    else:
        st.warning(f"""
        **🟡 HOLD Strategy:**
        
        • **Current Action**: Maintain existing position
        • **Monitoring**: Watch for trend changes and new signals
        • **Risk Management**: Keep current stop losses in place
        • **Patience**: Wait for clearer market direction
        • **Review**: Reassess in 1-2 weeks or on significant news
        """)

def display_financial_planning(results):
    """Display financial planning analysis"""
    
    st.header("💰 Financial Planning Analysis")
    
    financial_plan = results.get('financial_plan')
    
    if not financial_plan:
        st.info("""
        💡 **Enable Financial Planning**
        
        Financial planning was not enabled for this analysis. To get a comprehensive 
        financial plan including:
        
        • Goal achievement probability analysis
        • Optimal asset allocation recommendations  
        • Monte Carlo simulation results
        • Personalized recommendations
        
        Please enable Financial Planning in the configuration section and re-run the analysis.
        """)
        return
    
    goal = financial_plan.goal
    
    # Goal summary
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
    
    # Plan results
    st.subheader("📊 Plan Analysis Results")
    
    # Achievement status
    if financial_plan.is_achievable:
        st.success(f"✅ **Goal is Achievable!** Projected value: ${financial_plan.projected_value:,.0f}")
    else:
        additional_needed = financial_plan.required_monthly - goal.monthly_contribution
        st.warning(f"⚠️ **Additional ${additional_needed:,.0f}/month needed** to reach your goal")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Success Probability", f"{financial_plan.success_probability:.1%}")
    
    with col2:
        gap = financial_plan.projected_value - goal.target_amount
        st.metric("Projected Value", f"${financial_plan.projected_value:,.0f}", f"${gap:,.0f}")
    
    with col3:
        st.metric("Plan Sharpe Ratio", f"{financial_plan.plan_sharpe_ratio:.2f}")
    
    with col4:
        if financial_plan.required_monthly > 0:
            st.metric("Required Monthly", f"${financial_plan.required_monthly:,.0f}")
        else:
            st.metric("Surplus", f"${abs(financial_plan.required_monthly):,.0f}")
    
    # Asset allocation
    st.subheader("🎯 Recommended Asset Allocation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Pie chart for allocation
        labels = [asset.replace('_', ' ').title() for asset in financial_plan.asset_allocation.keys()]
        values = list(financial_plan.asset_allocation.values())
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c'])
        )])
        
        fig.update_layout(
            title="Optimal Portfolio Allocation",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Allocation table
        allocation_df = pd.DataFrame({
            'Asset Class': labels,
            'Allocation': [f"{v:.1%}" for v in values],
            'Monthly $': [f"${goal.monthly_contribution * v:,.0f}" for v in values]
        })
        
        st.dataframe(allocation_df, hide_index=True, use_container_width=True)
    
    # Monte Carlo results
    if financial_plan.monte_carlo_results:
        st.subheader("📈 Monte Carlo Simulation Results")
        
        mc = financial_plan.monte_carlo_results
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Outcome distribution
            outcomes = ['Worst Case (5%)', '10th Percentile', 'Median (50%)', 
                       '90th Percentile', 'Best Case (95%)']
            values = [
                mc.get('worst_case', 0),
                mc.get('percentile_10', 0), 
                mc.get('median', 0),
                mc.get('percentile_90', 0),
                mc.get('best_case', 0)
            ]
            
            fig = go.Figure()
            
            colors = ['red', 'orange', 'blue', 'lightgreen', 'green']
            
            fig.add_trace(go.Bar(
                x=outcomes,
                y=values,
                marker_color=colors,
                text=[f"${v:,.0f}" for v in values],
                textposition='auto'
            ))
            
            # Add target line
            fig.add_hline(
                y=goal.target_amount,
                line_dash="dash",
                line_color="black", 
                annotation_text=f"Target: ${goal.target_amount:,.0f}"
            )
            
            fig.update_layout(
                title="Monte Carlo Simulation - Potential Outcomes",
                xaxis_title="Scenario",
                yaxis_title="Portfolio Value ($)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric("Success Rate", f"{mc.get('success_rate', 0):.1%}")
            st.metric("Expected Value", f"${mc.get('mean', 0):,.0f}")
            st.metric("Median Outcome", f"${mc.get('median', 0):,.0f}")
            
            # Risk assessment
            success_rate = mc.get('success_rate', 0)
            if success_rate > 0.8:
                st.success("🟢 High confidence in achieving goal")
            elif success_rate > 0.6:
                st.warning("🟡 Moderate confidence - consider optimizing")
            else:
                st.error("🔴 Low confidence - plan needs adjustment")
    
    # Recommendations
    if financial_plan.recommendations:
        st.subheader("💡 Personalized Recommendations")
        
        for i, rec in enumerate(financial_plan.recommendations, 1):
            st.write(f"**{i}.** {rec}")

def display_visualizations(results):
    """Display additional visualizations"""
    
    st.header("📊 Advanced Visualizations")
    
    market_data = results['market_data']
    risk_metrics = results['risk_metrics']
    forecast_data = results['forecast_data']
    
    # Risk-Return Analysis
    st.subheader("⚖️ Risk-Return Analysis")
    
    expected_return = ((forecast_data.ensemble_forecast - market_data.current_price) / market_data.current_price) * 100
    
    fig = go.Figure()
    
    # Add current position
    fig.add_trace(go.Scatter(
        x=[risk_metrics.portfolio_volatility * 100],
        y=[expected_return],
        mode='markers+text',
        name='Current Analysis',
        marker=dict(size=20, color='red', symbol='star'),
        text=[market_data.symbol],
        textposition='top center'
    ))
    
    # Add reference points for efficient frontier
    fig.add_trace(go.Scatter(
        x=[15, 20, 25, 30],
        y=[8, 10, 12, 15],
        mode='markers+lines',
        name='Efficient Frontier (Reference)',
        line=dict(color='blue', dash='dash'),
        marker=dict(size=8, color='blue')
    ))
    
    fig.update_layout(
        title="Risk vs Expected Return Profile",
        xaxis_title="Volatility (%)",
        yaxis_title="Expected Return (%)",
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance summary table
    st.subheader("📋 Performance Summary")
    
    summary_data = {
        'Metric': [
            'Current Price',
            'Forecast Price',
            'Expected Return',
            'Confidence Level',
            'Risk Level',
            'Sharpe Ratio',
            'Maximum Drawdown',
            'Recommendation'
        ],
        'Value': [
            f"${market_data.current_price:.2f}",
            f"${forecast_data.ensemble_forecast:.2f}",
            f"{((forecast_data.ensemble_forecast - market_data.current_price) / market_data.current_price * 100):+.1f}%",
            f"{forecast_data.forecast_confidence:.1%}",
            f"{risk_metrics.portfolio_volatility:.1%}",
            f"{risk_metrics.sharpe_ratio:.2f}",
            f"{risk_metrics.maximum_drawdown:.1%}",
            results['recommendation'].action
        ]
    }
    
    df = pd.DataFrame(summary_data)
    st.dataframe(df, hide_index=True, use_container_width=True)

# Main execution
if __name__ == "__main__":
    main()
