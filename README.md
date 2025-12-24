# GenAI Multi-Agent Hedge Fund

_Agents for market forecasting, strategy, and personal financial planning._

This project explores how **agentic workflows**, time-series models and LLM reasoning can work together for more transparent investing and goal-based planning. It was built as part of the University of Chicago MS-ADS program and implements a **7-agent async pipeline** that integrates Yahoo Finance, FRED, and news APIs, then explains its decisions in plain language.

---

## Key Features

- **7 coordinated agents with shared state**
  - Market
  - Sentiment
  - Macro
  - Forecasting (Prophet + ARIMA fallback)
  - Risk
  - Strategist (LLM or rules)
  - Financial Planner
- **Ensemble forecasting** with volatility-aware confidence and graceful fallbacks when optional dependencies are missing.
- **Personal financial planning**:
  - Monthly contributions.
  - Projected portfolio growth.
  - Monte Carlo–based success probability for reaching targets.
- **Explainability**:
  - Human-readable rationales behind buy/sell/hold decisions.
  - Transparent description of planning assumptions.
- **Live demo app** (Streamlit):  
  `https://a2amultiagenthedgefund-7jxenesgsrcamwahzhhbuf.streamlit.app`

---

## System Architecture

The system is a multi-agent pipeline built around a **shared Agent State**. Each agent reads what it needs, writes its outputs, and passes control to the next stage.

### High-Level Data & Agent Flow

flowchart TD
    YF[Yahoo Finance]
    NEWS[News APIs]
    FRED[FRED]

    MKT[Market Agent]
    SENT[Sentiment Agent]
    MACRO[Macro Agent]
    FCST[Forecasting Agent]
    RISK[Risk Agent]
    STRAT[Strategist Agent]
    PLAN[Financial Planner]

    STATE[(Shared Agent State)]

    YF --> MKT
    NEWS --> SENT
    FRED --> MACRO

    MKT --> STATE
    SENT --> STATE
    MACRO --> STATE

    STATE --> FCST --> STATE
    STATE --> RISK --> STATE
    STATE --> STRAT --> STATE
    STATE --> PLAN --> STATE

## Agents

### 1. **Market Agent**
Pulls OHLCV data from Yahoo Finance and computes:

- Daily returns.
- Volatility measures.
- Basic technical indicators (RSI, simple support/resistance levels).

---

### 2. **Sentiment Agent**
Processes recent news headlines and produces:

- Headline-level polarity scores.
- Aggregated sentiment index + headline volume.
- Heuristic fallback when sentiment APIs are unavailable.

---




## Architecture
- **Market Data**: OHLCV, returns, RSI, volatility, support/resistance.  
- **Sentiment**: News headline polarity + volume (heuristic fallback).
- **Macro Econ**: FRED indicators (GDP, CPI, unemployment, yield curve, VIX).
- **Forecasting**: Prophet + ARIMA fallback -> ensemble w/ volatility-weighted confidence.
- **Risk**: down-weights forecast confidence under high volatility.  
- **Strategist**: GPT-4 when key present; otherwise transparent rule-based schema.  
- **Financial Planner** -> monthly contributions, projected FV, Monte Carlo success.  
All agents read/write a common **Agent State** for modularity and traceability.

## Data & preprocessing
- **Markets**: Yahoo Finance OHLCV aligned to daily, handles missing values.
- **Macro**: FRED series aligned for comparability.
- **Text**: News headlines cleaned and polarity plus aggregate sentiment index.
- **Shared schema** enforces consistent inputs across agents.


## Evaluation & example output
**Backtest (AAPL, 30-day horizon):** 
| Model     | MAE | RMSE |
|-----------|-----|------|
| Prophet   | 2.1 | 2.8  |
| Fallback  | 3.4 | 4.7  |
| **Ensemble** | **2.0** | **2.6** |


