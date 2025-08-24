# GenAI Financial Forecaster: A Personal Agentic AI Project

This is a personal project I built as part of my Master’s in Applied Data Science at the University of Chicago. The goal was to explore how agentic workflows and generative AI can be used to support financial decision-making — from market forecasting to personal investment planning.

While inspired by open-source repos like `ai-hedge-fund`, I wanted to better understand the inner workings by building my own version using modular agents, real APIs, and LLM reasoning.

---

## What It Does

This system simulates a simplified investment assistant that takes in:
- Real-time market data
- Sentiment from Reddit, News, Twitter
- Macroeconomic indicators (via FRED)
- Forecasts future price movements (via ARIMA, Prophet, LSTM)
- Assesses risk and market regime
- Uses an LLM to generate a final recommendation (buy/sell/hold)

There’s also an in-progress **FinancePlannerNode** that takes user goals (e.g., retirement target in 10 years) and outputs a personalized investment plan.

---

## Agentic Workflow

The project is built around modular agents that pass and update a shared `AgentState` — similar to how LangGraph or CrewAI works.

## Architecture
**Seven agents with a shared, typed Agent State:**
- **Market Data** -> OHLCV, returns, RSI, volatility, trend, support/resistance  
- **Sentiment** -> News headlines polarity & volume (heuristic fallback if no key)  
- **Macro Econ** -> FRED: GDP, inflation, unemployment, yield curve, VIX; macro “bull/neutral/bear”  
- **Forecasting** -> Prophet (if installed) + ARIMA-style smoothing fallback; **ensemble** with volatility-aware confidence  
- **Risk** → Volatility-aware down-weighting; drawdown/Sharpe/VAR style metrics  
- **Strategist** -> GPT-powered (or rule-based fallback) that outputs action, confidence, position size, risk level, horizon, and reasoning  
- **Financial Planner** -> Contributions/FV, Monte Carlo probability of reaching a target

| Agent | Role |
|-------|------|
| `MarketDataNode` | Fetch OHLCV price data |
| `SentimentNode` | Analyze financial sentiment using real Reddit/News |
| `MacroEconNode` | Pull macro indicators (GDP, inflation, rates) from FRED |
| `ForecastingNode` | Use ARIMA, Prophet, LSTM to predict prices |
| `RiskNode` | (In progress) Calculate Sharpe Ratio, volatility, drawdown |
| `StrategistAgent` | (In progress) Uses GPT to give a trading recommendation |
| `FinancePlannerNode` | (In progress) Suggests how much to invest monthly to hit goals |

---

## Real Integrations

This project connects to real APIs:

- [x] Yahoo Finance (via yfinance)
- [x] Binance
- [x] FRED (Federal Reserve)
- [x] NewsAPI / GNews
- [x] Reddit (via PRAW)
- [x] OpenAI GPT-4 (for reasoning)

---

## Streamlit Dashboard (Coming Soon)

A front-end dashboard is being developed using Streamlit to show:
- Forecasts vs actual prices
- Sentiment trends
- Macro regime
- Final LLM recommendation
- Personal investment plan

---

## Why I Built This

I wanted to go beyond traditional modeling and explore how AI agents can collaborate across domains — economics, NLP, time series — and build toward an assistant that feels more adaptive and insightful.

This is still a work-in-progress, but I’ve learned a lot about:
- Agent architecture
- API orchestration
- Prompt engineering
- Real-world data cleaning
- Model evaluation

---

## How to Run

1. Clone the repo
2. Add your API keys in a `.env` file or use Colab secrets
3. Run each agent step by step or orchestrate as a pipeline
4. Streamlit app coming soon
