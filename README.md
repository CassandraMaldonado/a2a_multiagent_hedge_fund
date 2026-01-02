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

```mermaid
flowchart TD
    %% Data Sources
    YF[Yahoo Finance]
    NEWS[News APIs]
    FRED[FRED]

    %% Agents
    MKT[Market Agent]
    SENT[Sentiment Agent]
    MACRO[Macro Agent]
    FCST[Forecasting Agent]
    RISK[Risk Agent]
    STRAT[Strategist Agent]
    PLAN[Financial Planner]

    %% Shared State
    STATE[(Shared Agent State)]

    %% Source ingestion
    YF --> MKT
    NEWS --> SENT
    FRED --> MACRO

    %% Write to shared state
    MKT --> STATE
    SENT --> STATE
    MACRO --> STATE

    %% Downstream reasoning
    STATE --> FCST --> STATE
    STATE --> RISK --> STATE
    STATE --> STRAT --> STATE
    STATE --> PLAN --> STATE
