# GenAI Multi-Agent Hedge Fund
_Agents for market forecasting, strategy and personal financial planning._

I built this project in the UChicago MS-ADS program to explore how agentic workflows + classical time-series models + LLM reasoning can work together for transparent investing and goal-based planning. It’s a 7-agent async pipeline that integrates Yahoo Finance, FRED and News APIs, then explains decisions in plain language.


## Recruiter Snapshot (what I actually shipped)
- **7 agents with shared state**: Market, Sentiment, Macro, Forecasting (Prophet + ARIMA-style fallback), Risk, Strategist (GPT or rules), and a Financial Planner. :contentReference[oaicite:1]{index=1}  
- **Ensemble forecasting** with volatility-aware confidence; robust to missing optional deps (graceful fallbacks). :contentReference[oaicite:2]{index=2}  
- **Personal planning**: contributions, projected growth, and Monte Carlo success probability. :contentReference[oaicite:3]{index=3}  
- **Results (AAPL, 30-day backtest)**: Prophet MAE **2.1** / RMSE **2.8**; Fallback MAE **3.4** / RMSE **4.7**; **Ensemble** MAE **2.0** / RMSE **2.6**. :contentReference[oaicite:4]{index=4}  
- **Live app demo** (Streamlit): https://a2amultiagenthedgefund-7jxenesgsrcamwahzhhbuf.streamlit.app/ :contentReference[oaicite:5]{index=5}


## Why this matters (and what I wanted to learn)
Retail tools feel like black boxes—generic templates, unclear logic, and fragmented data. I wanted a system that **personalizes**, **justifies** its calls (buy/sell/hold & planning), and stays **robust** when APIs/LLMs aren’t available. :contentReference[oaicite:6]{index=6}


## Architecture (7 agents, one shared typed state)
- **Market Data** → OHLCV, returns, RSI, volatility, support/resistance  
- **Sentiment** → News headline polarity + volume (heuristic fallback)  
- **Macro Econ** → FRED indicators (GDP, CPI, unemployment, yield curve, VIX)  
- **Forecasting** → Prophet + ARIMA-style fallback → **ensemble** w/ volatility-weighted confidence  
- **Risk** → down-weights forecast confidence under high volatility  
- **Strategist** → GPT-4 when key present; otherwise **transparent rule-based** schema  
- **Financial Planner** → monthly contributions, projected FV, Monte Carlo success  
All agents read/write a common **Agent State** for modularity and traceability. :contentReference[oaicite:7]{index=7}

**Example strategist schema**: action, confidence, position size, risk level, horizon, reasoning. :contentReference[oaicite:8]{index=8}


## Data & preprocessing
- **Markets**: Yahoo Finance OHLCV aligned to daily; missing values handled  
- **Macro**: FRED series timestamp-aligned for comparability  
- **Text**: News headlines cleaned → polarity + aggregate sentiment index  
- **Shared schema** enforces consistent inputs across agents (less glue code later)  
Key insight: early standardization simplified downstream reliability substantially. :contentReference[oaicite:9]{index=9}


## Evaluation & example output
**Backtest (AAPL, 30-day horizon):** Prophet MAE 2.1 / RMSE 2.8; Fallback MAE 3.4 / RMSE 4.7; **Ensemble** MAE 2.0 / RMSE 2.6. :contentReference[oaicite:10]{index=10}

**One sample run (AAPL):** ensemble forecast ~$238.78 (+3.6%), confidence ~81.9%, with strategist = **HOLD (low risk, long horizon)** after weighing bullish technicals vs neutral/bearish macro-sentiment context. :contentReference[oaicite:11]{index=11} :contentReference[oaicite:12]{index=12}

**Planner example:** 10-yr target projection with contribution plan + Monte Carlo success probability; also reports plan Sharpe and expected volatility. :contentReference[oaicite:13]{index=13}

**Lessons learned:** ensemble > single model; LLM reasoning improves interpretability; fallbacks + prompt design were key for resilience. :contentReference[oaicite:14]{index=14}


## What I built personally (skills a team could reuse)
- **Agent orchestration & shared state** (async, modular boundaries) :contentReference[oaicite:15]{index=15}  
- **Forecasting ensemble** (Prophet + ARIMA-style fallback w/ volatility-aware confidence) :contentReference[oaicite:16]{index=16}  
- **LLM strategist w/ rule fallback** (structured outputs; explainability) :contentReference[oaicite:17]{index=17}  
- **Goal-based planner** (FV math + Monte Carlo) :contentReference[oaicite:18]{index=18}  
- **Data plumbing** across Yahoo/FRED/News + early standardization for reliability :contentReference[oaicite:19]{index=19}

