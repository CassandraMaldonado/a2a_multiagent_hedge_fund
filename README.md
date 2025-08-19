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

- Structured A2A-style message passing between agents
- Dynamic task delegation and agent coordination via LangGraph
- Forecasting via ensemble of ARIMA, Prophet, LSTM
- Sentiment analysis from Reddit, Twitter/X, News (via Hugging Face models)
- Macroeconomic regime detection (expansion, contraction, inflation shock, etc.)
- Full agent state and message logging for inspection
