# AI Stock Dashboard

This is a lightweight AI-powered dashboard built to track key stock metrics, overlay technical indicators, and run basic sentiment analysis on recent news.

Live app:  
https://aidashboardfreesia.streamlit.app/

---

## Overview

The dashboard currently supports up to four stocks at a time (e.g. GLEN.L, AAPL, TSLA, GOOGL). You can toggle between them using the top navigation.

### Features

- **Live Metrics**  
  Displays current price, RSI, P/E ratio, and Debt/Equity for each stock.

- **Technical Charts**  
  Interactive candlestick chart with:
  - Fast/Slow moving averages  
  - Bollinger Bands  
  - Earnings annotations  
  - Volume and On-Balance Volume (OBV)  
  - RSI panel (drawdown to be added)

- **News Sentiment Analysis**  
  - Overall sentiment score (ranging from -1 to +1)  
  - Daily sentiment trendline  
  - Sentiment breakdown (Positive / Neutral / Negative)  
  - Filterable headlines by sentiment

- **Report Export (WIP)**  
  A button has been added as a placeholder for one-click summary exports, pending final requirements.

---

## Tech Stack

- Streamlit
- yfinance
- NewsAPI
- VADER
- Plotly
- Pandas
