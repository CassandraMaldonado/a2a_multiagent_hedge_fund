# A2A Multi-Agent Hedge Fund System

This project implements an intelligent multi-agent hedge fund simulation using [LangGraph](https://github.com/langchain-ai/langgraph) and Google’s A2A Protocol. It coordinates market analysis, sentiment aggregation, macroeconomic interpretation, forecasting, and risk evaluation through an agentic workflow architecture.

## Project Overview

This system is designed as a modular, real-time investment strategy engine. It leverages specialized AI agents — each focused on one domain (e.g., sentiment, macroeconomics, forecasting) — and orchestrates them via LangGraph to simulate institutional-level financial decision-making.

### Key Components

| Node                     | Description |
|--------------------------|-------------|
| `MarketDataNode`         | Fetches OHLCV data for stocks and crypto (Yahoo Finance, Binance) |
| `SentimentNode`          | Aggregates sentiment from Reddit, Twitter, and financial news using FinBERT |
| `MacroEconNode`          | Analyzes macro indicators (GDP, CPI, Fed Funds) from FRED/IMF |
| `ForecastingNode`        | Produces price forecasts using ARIMA, Prophet, and LSTM models |
| `AgentState`             | Tracks all intermediate data, config, errors, and messages |
| `LangGraph DAG`          | Orchestrates dynamic message passing, consensus-building, and agent collaboration |

---

## Features

- ✅ Structured A2A-style message passing between agents
- ✅ Dynamic task delegation and agent coordination via LangGraph
- ✅ Forecasting via ensemble of ARIMA, Prophet, LSTM
- ✅ Sentiment analysis from Reddit, Twitter/X, News (via Hugging Face models)
- ✅ Macroeconomic regime detection (expansion, contraction, inflation shock, etc.)
- ✅ Full agent state and message logging for inspection

---

## Quickstart (Google Colab)

1. Clone or upload your agent files into Colab
2. Run environment setup:
    ```bash
    !pip install langgraph pandas yfinance pmdarima prophet tensorflow transformers praw tweepy newsapi-python python-binance
    ```
3. Simulate folder structure:
    ```python
    import os; os.makedirs("src/nodes", exist_ok=True)
    ```
4. Move your uploaded files to:
    ```
    src/
      nodes/
        market_data_node.py
        sentiment_node.py
        macro_econ_node.py
        forecasting_risk_strategist.py
      utils/
        utils_files.py
      state/
        agent_state.py
    ```
5. Add path and run the DAG:
    ```python
    import sys
    sys.path.append("/content/src")
    ```

---

## Example Run

```python
from langgraph.graph import StateGraph
from state.agent_state import create_empty_state
from nodes.market_data_node import MarketDataNode
from nodes.sentiment_node import SentimentNode
from nodes.macro_econ_node import MacroEconNode
from nodes.forecasting_risk_strategist import ForecastingNode

graph = StateGraph()
graph.add_node("market", MarketDataNode(config))
graph.add_node("sentiment", SentimentNode(config))
graph.add_node("macro", MacroEconNode(config))
graph.add_node("forecast", ForecastingNode(config))

graph.set_entry_point("market")
graph.add_edge("market", "sentiment")
graph.add_edge("sentiment", "macro")
graph.add_edge("macro", "forecast")
graph.set_exit_point("forecast")
app = graph.compile()

state = create_empty_state(["AAPL", "BTC-USD"], config)
state["start_date"] = "2024-01-01"
state["end_date"] = "2024-02-01"

results = app.invoke(state)
