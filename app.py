import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import os
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objs as go
import plotly.express as px
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter

# Load environment variables

if "GITHUB_ACTIONS" not in os.environ:  # or check os.getenv('STREAMLIT_CLOUD')
    from dotenv import load_dotenv
    load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY   = os.getenv("NEWS_API_KEY")
if not OPENAI_API_KEY or not NEWS_API_KEY:
    st.error("Missing API keys. Please set OPENAI_API_KEY and NEWS_API_KEY in your .env file.")
    st.stop()
client = OpenAI(api_key=OPENAI_API_KEY)
newsapi = NewsApiClient(api_key=NEWS_API_KEY)
analyzer = SentimentIntensityAnalyzer()

# Page configuration
st.set_page_config(
    page_title="Multi-Stock AI Dashboard",
    layout="wide"
)

# Sidebar: Ticker inputs
st.sidebar.header("Tickers")
tickers = []
for i in range(1, 5):
    t = st.sidebar.text_input(f"Ticker {i}", key=f"ticker_{i}", placeholder="e.g. AAPL")
    if t and t.strip():
        tickers.append(t.strip().upper())
if not tickers:
    st.sidebar.warning("Enter at least one ticker to analyze.")
tickers = tickers[:4]

# Sidebar: Parameters
date_range = st.sidebar.date_input(
    "Date range",
    [
        datetime.date.today() - datetime.timedelta(days=180),
        datetime.date.today()
    ]
)
short_ma = st.sidebar.slider("Fast MA window", 5, 50, 14)
long_ma = st.sidebar.slider("Slow MA window", 20, 200, 50)
rsi_window = st.sidebar.slider("RSI window", 5, 50, 14)
rsi_alert = st.sidebar.number_input(
    "Alert when RSI below", min_value=0, max_value=100, value=30
)
show_news = st.sidebar.checkbox("Show news panel", True)
show_overview = st.sidebar.checkbox("Show overview tab", True)
ai_focus = st.sidebar.multiselect(
    "AI summary focus",
    ["News", "Technicals", "Fundamentals"],
    default=["News", "Technicals", "Fundamentals"]
)

# Data retrieval functions
def fetch_stock(ticker, start, end):
    tk = yf.Ticker(ticker)
    df = tk.history(start=start, end=end)
    info = tk.info
    return df, info

@st.cache_data
def compute_indicators(df):
    df = df.copy()
    df['Fast_MA'] = df['Close'].rolling(short_ma).mean()
    df['Slow_MA'] = df['Close'].rolling(long_ma).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(rsi_window).mean()
    loss = -delta.clip(upper=0).rolling(rsi_window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Vol_Spike'] = df['Volume'] > 2 * df['Volume'].rolling(20).mean()
    return df

@st.cache_data
def fetch_news(query):
    res = newsapi.get_everything(
        q=query,
        language='en',
        sort_by='publishedAt',
        page_size=5
    )
    return res.get('articles', [])

# AI-based summary
def ai_summary(ticker, headlines, curr_rsi, prev_rsi, pe, netm, de):
    pe_str = f"{pe:.2f}" if isinstance(pe, (int, float)) else "N/A"
    netm_str = f"{netm*100:.1f}%" if isinstance(netm, (int, float)) else "N/A"
    de_str = f"{de:.2f}" if isinstance(de, (int, float)) else "N/A"
    parts = []
    if 'News' in ai_focus:
        parts.append("Key news: " + "; ".join(headlines))
    if 'Technicals' in ai_focus:
        parts.append(f"RSI: {curr_rsi:.1f} (prev {prev_rsi:.1f})")
    if 'Fundamentals' in ai_focus:
        parts.append(f"P/E: {pe_str}, Net Margin: {netm_str}, Debt/Equity: {de_str}")
    prompt = (
        f"Summarise {ticker} in 4 bullet points:\n"
        + "\n".join(f"- {p}" for p in parts)
    )
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return resp.choices[0].message.content

# Build tabs
tab_titles = []
if show_overview:
    tab_titles.append("Overview")
tab_titles += tickers
tabs = st.tabs(tab_titles)
offset = 1 if show_overview else 0

# Overview Tab
if show_overview:
    with tabs[0]:
        st.header("Portfolio Overview")
        with st.spinner("🔄 Building overview table..."):
            rows = []
            for t in tickers:
                df_raw, info = fetch_stock(t, *date_range)
                if df_raw.empty:
                    continue
                ind = compute_indicators(df_raw)
                price = info.get('regularMarketPrice')
                price = float(price) if isinstance(price, (int, float)) else np.nan
                rsi_val = ind['RSI'].dropna()
                rsi_val = rsi_val.iloc[-1] if not rsi_val.empty else np.nan
                pe_val = info.get('trailingPE')
                pe_val = float(pe_val) if isinstance(pe_val, (int, float)) else np.nan
                rows.append({
                    'Ticker': t,
                    'Price': price,
                    'RSI': rsi_val,
                    'P/E': pe_val
                })
            df_ov = pd.DataFrame(rows)
            st.dataframe(
                df_ov.style.format({
                    'Price': '{:.2f}',
                    'RSI': '{:.1f}',
                    'P/E': '{:.2f}'
                })
            )

# Per-stock Tabs
for i, ticker in enumerate(tickers):
    with tabs[i + offset]:
        st.header(ticker)
        # Fetch data
        with st.spinner(f"🔄 Loading {ticker} data..."):
            df_raw, info = fetch_stock(ticker, *date_range)
            df = compute_indicators(df_raw) if not df_raw.empty else pd.DataFrame()
        if df.empty:
            st.warning(f"No data available for {ticker} in the selected date range.")
            continue
        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        price = info.get('regularMarketPrice')
        change = info.get('regularMarketChangePercent')
        c1.metric(
            "Price",
            f"{price:.2f}" if isinstance(price, (int, float)) else "N/A",
            delta=f"{change*100:.1f}%" if isinstance(change, (int, float)) else "N/A"
        )
        rsi_series = df['RSI'].dropna()
        rsi_val = rsi_series.iloc[-1]
        prev_rsi = rsi_series.iloc[-2] if len(rsi_series) > 1 else rsi_val
        c2.metric(
            "RSI",
            f"{rsi_val:.1f}",
            delta=f"{rsi_val - prev_rsi:.1f}",
            delta_color="inverse" if rsi_val < rsi_alert else "normal"
        )
        pe = info.get('trailingPE')
        de = info.get('debtToEquity')
        c3.metric(
            "P/E",
            f"{pe:.2f}" if isinstance(pe, (int, float)) else "N/A"
        )
        c4.metric(
            "Debt/Equity",
            f"{de:.2f}" if isinstance(de, (int, float)) else "N/A"
        )
        st.markdown(f"[Yahoo Finance →](https://finance.yahoo.com/quote/{ticker})")
        # Charts: Price, Volume, RSI
        fig_p = go.Figure()
        fig_p.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            )
        )
        fig_p.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Fast_MA'],
                name=f"MA{short_ma}"
            )
        )
        fig_p.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Slow_MA'],
                name=f"MA{long_ma}"
            )
        )
        fig_p.update_layout(height=400, template='plotly_dark')
        st.plotly_chart(fig_p, use_container_width=True)
        # Volume
        fig_v = go.Figure()
        fig_v.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Volume')
        )
        fig_v.update_layout(height=200, template='plotly_dark', yaxis_title='Volume')
        st.plotly_chart(fig_v, use_container_width=True)
        # RSI
        fig_r = go.Figure()
        fig_r.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI')
        )
        fig_r.update_layout(
            height=200,
            template='plotly_dark',
            yaxis=dict(range=[0, 100]),
            yaxis_title='RSI'
        )
        st.plotly_chart(fig_r, use_container_width=True)
        # News & Sentiment
        if show_news:
            name = info.get('longName') or info.get('shortName') or ticker.split('.')[0]
            with st.spinner(f"🗞️ Fetching news for {name}..."):
                articles = fetch_news(name)
            if articles:
                st.subheader("Latest News")
                # Classify sentiment
                sentiments = []
                scores = []
                dates = []
                for art in articles:
                    title = art.get('title', '')
                    pub = art.get('publishedAt', '')
                    try:
                        dt = pd.to_datetime(pub).normalize()
                    except:
                        dt = pd.Timestamp.now().normalize()
                    dates.append(dt)
                    vs = analyzer.polarity_scores(title)
                    comp = vs['compound']
                    scores.append(comp)
                    if comp >= 0.05:
                        sentiments.append('Positive')
                    elif comp <= -0.05:
                        sentiments.append('Negative')
                    else:
                        sentiments.append('Neutral')
                cnt = Counter(sentiments)
                overall = sum(scores) / len(scores) if scores else 0
                st.markdown(f"**Overall Sentiment Score:** {overall:.2f} (–1 to +1)")
                # Trend sparkline
                df_sent = pd.DataFrame({'date': dates, 'compound': scores})
                daily = df_sent.groupby('date', as_index=False).compound.mean()
                fig_sent_line = px.line(
                    daily,
                    x='date', y='compound',
                    title='', height=200
                )
                fig_sent_line.update_layout(
                    margin=dict(t=0, b=0),
                    xaxis_title='',
                    yaxis_title='Sentiment'
                )
                st.plotly_chart(fig_sent_line, use_container_width=True)
                # Distribution
                df_dist = pd.DataFrame({
                    'Sentiment': list(cnt.keys()),
                    'Count': list(cnt.values())
                })
                fig_sent_pie = px.pie(
                    df_dist,
                    names='Sentiment', values='Count',
                    title='', height=200
                )
                fig_sent_pie.update_layout(margin=dict(t=0, b=0))
                st.plotly_chart(fig_sent_pie, use_container_width=True)
                # Optional filter
                sentiment_choice = st.radio(
                    "Filter Sentiment",
                    ['Positive','Neutral','Negative'],
                    key=f"sent_{ticker}"
                )
                for art, s in zip(articles, sentiments):
                    if s == sentiment_choice:
                        st.markdown(
                            f"**{art['source']['name']}** ({art['publishedAt'][:10]}): "
                            f"[{art['title']}]({art['url']}) - *{s}*"
                        )
                with st.expander("AI Summary & Drivers"):
                    heads = [a['title'] for a in articles]
                    summ = ai_summary(
                        ticker, heads, rsi_val, prev_rsi,
                        info.get('trailingPE'), info.get('netMargins'), info.get('debtToEquity')
                    )
                    st.markdown(summ)
        # Alerts & Export
        if rsi_val < rsi_alert:
            st.warning(f"RSI for {ticker} below {rsi_alert}: {rsi_val:.1f}")
        if st.button(f"Download {ticker} Report", key=f"dl_{ticker}"):
            st.success("Report feature coming soon!")
        st.caption("Powered by Streamlit, yFinance, NewsAPI & OpenAI")