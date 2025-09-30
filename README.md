# GenAI Multi-Agent Hedge Fund
_Agents for market forecasting, strategy and personal financial planning._

I built this project in the UChicago MS-ADS program to explore how agentic workflows + classical time-series models + LLM reasoning can work together for transparent investing and goal-based planning. It’s a 7-agent async pipeline that integrates Yahoo Finance, FRED and News APIs, then explains decisions in plain language.


## 
- **7 agents with shared state**: Market, Sentiment, Macro, Forecasting (Prophet + ARIMA fallback), Risk, Strategist (GPT or rules) and a Financial Planner. 
- **Ensemble forecasting** with volatility-aware confidence; robust to missing optional deps (graceful fallbacks). 
- **Personal planning**: contributions, projected growth, and Monte Carlo success probability. 
- **Results (AAPL, 30-day backtest)**: Prophet MAE **2.1** / RMSE **2.8**; Fallback MAE **3.4** / RMSE **4.7**; Ensemble MAE **2.0** / RMSE **2.6**. 
- **Live app demo** (Streamlit): https://a2amultiagenthedgefund-7jxenesgsrcamwahzhhbuf.streamlit.app


## Why this matters
Retail tools feel like black boxes—generic templates, unclear logic, and fragmented data. I wanted a system that **personalizes**, **justifies** its calls (buy/sell/hold & planning), and stays robust when APIs/LLMs aren’t available.


## Architecture
- **Market Data** -> OHLCV, returns, RSI, volatility, support/resistance  
- **Sentiment** -> News headline polarity + volume (heuristic fallback)  
- **Macro Econ** -> FRED indicators (GDP, CPI, unemployment, yield curve, VIX)  
- **Forecasting** -> Prophet + ARIMA fallback → ensemble w/ volatility-weighted confidence  
- **Risk** -> down-weights forecast confidence under high volatility  
- **Strategist** -> GPT-4 when key present; otherwise **transparent rule-based** schema  
- **Financial Planner** -> monthly contributions, projected FV, Monte Carlo success  
All agents read/write a common **Agent State** for modularity and traceability.


## Data & preprocessing
- **Markets**: Yahoo Finance OHLCV aligned to daily, handles missing values.
- **Macro**: FRED series aligned for comparability.
- **Text**: News headlines cleaned and polarity plus aggregate sentiment index.
- **Shared schema** enforces consistent inputs across agents.


## Evaluation & example output
**Backtest (AAPL, 30-day horizon):** Prophet MAE 2.1 / RMSE 2.8; Fallback MAE 3.4 / RMSE 4.7; **Ensemble** MAE 2.0 / RMSE 2.6. 

**One sample run (AAPL):** ensemble forecast ~$238.78 (+3.6%), confidence ~81.9%, with strategist = **HOLD (low risk, long horizon)** after weighing bullish technicals vs neutral/bearish macro-sentiment context. 

**Planner example:** 10-yr target projection with contribution plan + Monte Carlo success probability; also reports plan Sharpe and expected volatility.

