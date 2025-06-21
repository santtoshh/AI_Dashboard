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

# Load local environment variables
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# API keys and clients
API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
if not API_KEY or not NEWS_API_KEY:
    st.error("Missing API keys. Set OPENAI_API_KEY & NEWS_API_KEY in .env or Streamlit Secrets.")
    st.stop()
client = OpenAI(api_key=API_KEY)
newsapi = NewsApiClient(api_key=NEWS_API_KEY)
analyzer = SentimentIntensityAnalyzer()

# Page configuration
st.set_page_config(
    page_title="Multi-Stock AI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar: inputs
st.sidebar.header("Controls")
tickers = []
for i in range(1, 5):
    t = st.sidebar.text_input(f"Ticker {i}", key=f"ticker_{i}", placeholder="e.g. AAPL")
    if t and t.strip():
        tickers.append(t.strip().upper())
if not tickers:
    st.sidebar.warning("Enter at least one ticker.")
tickers = tickers[:4]

# Date range input with single-date guard
dr = st.sidebar.date_input(
    "Date range",
    [
        datetime.date.today() - datetime.timedelta(days=180),
        datetime.date.today()
    ]
)
# Handle single-date selection gracefully
if isinstance(dr, (list, tuple)):
    if len(dr) == 2:
        start_date, end_date = dr
    elif len(dr) == 1:
        # User only picked one date
        start_date = end_date = dr[0]
    else:
        # Fallback if weird tuple
        start_date = end_date = dr[-1]
else:
    # A single date object
    start_date = end_date = dr

short_ma = st.sidebar.slider("Fast MA window (days)", 5, 50, 14)
long_ma = st.sidebar.slider("Slow MA window (days)", 20, 200, 50)
rsi_window = st.sidebar.slider("RSI window (days)", 5, 50, 14)
rsi_alert = st.sidebar.number_input("Alert when RSI below", 0, 100, 30)
show_news = st.sidebar.checkbox("Show news panel", True)
show_overview = st.sidebar.checkbox("Show overview tab", True)
ai_focus = st.sidebar.multiselect(
    "AI summary focus", ["News", "Technicals", "Fundamentals"],
    default=["News", "Technicals", "Fundamentals"]
)

# Data functions
@st.cache_data
def fetch_stock(tkr, start, end):
    tk = yf.Ticker(tkr)
    df = tk.history(start=start, end=end)
    return df, tk.info

@st.cache_data
def compute_indicators(df, short_ma, long_ma, rsi_window):
    df = df.copy()
    # Moving Averages
    df['Fast_MA'] = df['Close'].rolling(short_ma).mean()
    df['Slow_MA'] = df['Close'].rolling(long_ma).mean()
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(rsi_window).mean()
    loss = -delta.clip(upper=0).rolling(rsi_window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # Bollinger Bands
    mid = df['Close'].rolling(short_ma).mean()
    std = df['Close'].rolling(short_ma).std()
    df['BB_UP'] = mid + 2 * std
    df['BB_LOW'] = mid - 2 * std
    # On-Balance Volume
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iat[i] > df['Close'].iat[i-1]: obv.append(obv[-1] + df['Volume'].iat[i])
        elif df['Close'].iat[i] < df['Close'].iat[i-1]: obv.append(obv[-1] - df['Volume'].iat[i])
        else: obv.append(obv[-1])
    df['OBV'] = obv
    # Drawdown
    df['Roll_Max'] = df['Close'].cummax()
    df['Drawdown'] = df['Close'] / df['Roll_Max'] - 1
    return df

@st.cache_data
def fetch_news(query):
    res = newsapi.get_everything(q=query, language='en', sort_by='publishedAt', page_size=5)
    return res.get('articles', [])

# AI summary
def ai_summary(tkr, heads, curr, prev, pe, netm, de):
    pe_str = f"{pe:.2f}" if isinstance(pe, (int, float)) else "N/A"
    netm_str = f"{netm*100:.1f}%" if isinstance(netm, (int, float)) else "N/A"
    de_str = f"{de:.2f}" if isinstance(de, (int, float)) else "N/A"
    parts = []
    if 'News' in ai_focus:
        parts.append("Key news: " + "; ".join(heads))
    if 'Technicals' in ai_focus:
        parts.append(f"RSI: {curr:.1f} (prev {prev:.1f})")
    if 'Fundamentals' in ai_focus:
        parts.append(f"P/E: {pe_str}, Net Margin: {netm_str}, Debt/Equity: {de_str}")
    prompt = f"Summarise {tkr} in 4 bullet points:\n" + "\n".join(f"- {p}" for p in parts)
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return resp.choices[0].message.content

# Tabs setup
tab_titles = []
if show_overview: tab_titles.append("Overview")
for t in tickers: tab_titles.append(t)
tabs = st.tabs(tab_titles)
offset = 1 if show_overview else 0

# Overview
if show_overview:
    with tabs[0]:
        st.header("Portfolio Overview")
        with st.spinner("Building overview table..."):
            rows = []
            for t in tickers:
                df_raw, info = fetch_stock(t, start_date, end_date)
                if df_raw.empty: continue
                ind = compute_indicators(df_raw, short_ma, long_ma, rsi_window)
                price = info.get('regularMarketPrice', np.nan)
                rsi = ind['RSI'].dropna().iloc[-1] if not ind['RSI'].dropna().empty else np.nan
                pe_val = info.get('trailingPE', np.nan)
                rows.append({'Ticker': t, 'Price': price, 'RSI': rsi, 'P/E': pe_val})
            df_ov = pd.DataFrame(rows)
            st.dataframe(df_ov.style.format({'Price':'{:.2f}','RSI':'{:.1f}','P/E':'{:.2f}'}))
        # Normalized performance
        st.subheader("Normalized Performance (Base 100)")
        df_norm = pd.DataFrame({
            t: fetch_stock(t, start_date, end_date)[0]['Close']
            for t in tickers
        }).pct_change().add(1).cumprod() * 100
        fig_norm = px.line(df_norm, title="Normalized Performance", height=300)
        fig_norm.update_layout(template='plotly_dark')
        st.plotly_chart(fig_norm, use_container_width=True, key="overview_norm")
        # Correlation
        st.subheader("Return Correlation")
        corr = df_norm.pct_change().corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Correlation Matrix", height=300)
        fig_corr.update_layout(template='plotly_dark')
        st.plotly_chart(fig_corr, use_container_width=True, key="overview_corr")

# Per-stock tabs
for idx, tkr in enumerate(tickers):
    with tabs[idx + offset]:
        st.header(tkr)
        df_raw, info = fetch_stock(tkr, start_date, end_date)
        df = compute_indicators(df_raw, short_ma, long_ma, rsi_window) if not df_raw.empty else pd.DataFrame()
        if df.empty:
            st.warning(f"No data available for {tkr}.")
            continue
        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        price = info.get('regularMarketPrice')
        change = info.get('regularMarketChangePercent')
        c1.metric("Price", f"{price:.2f}" if isinstance(price, (int,float)) else "N/A",
                  delta=f"{change*100:.1f}%" if isinstance(change, (int,float)) else "N/A",
                  help="Close price & % change")
        # Safely compute RSI values
        rsi_s = df['RSI'].dropna()
        if rsi_s.empty:
            curr = prev = np.nan
        else:
            curr = rsi_s.iloc[-1]
            prev = rsi_s.iloc[-2] if len(rsi_s) > 1 else curr
        c2.metric("RSI", f"{curr:.1f}", delta=f"{curr-prev:.1f}",
                  delta_color="inverse" if curr < rsi_alert else "normal",
                  help="Relative Strength Index")
        pe = info.get('trailingPE')
        dep = info.get('debtToEquity')
        c3.metric("P/E", f"{pe:.2f}" if isinstance(pe, (int,float)) else "N/A",
                  help="Trailing P/E ratio")
        c4.metric("Debt/Equity", f"{dep:.2f}" if isinstance(dep, (int,float)) else "N/A",
                  help="Debt-to-Equity ratio")
        st.markdown(f"[Yahoo Finance →](https://finance.yahoo.com/quote/{tkr})")
        # Price + overlays filter
        overlays = st.multiselect(
            "Select indicators to display on Price Chart:",
            ["Fast MA", "Slow MA", "BB Upper", "BB Lower"],
            default=["Fast MA", "Slow MA", "BB Upper", "BB Lower"],
            key=f"overlay_{tkr}"
        )
        # Price + MAs + BBs + earnings annotations
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'
        ))
        if "Fast MA" in overlays:
            fig.add_trace(go.Scatter(x=df.index, y=df['Fast_MA'], name=f"MA{short_ma}"))
        if "Slow MA" in overlays:
            fig.add_trace(go.Scatter(x=df.index, y=df['Slow_MA'], name=f"MA{long_ma}"))
        if "BB Upper" in overlays:
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_UP'], name='BB Upper', line=dict(dash='dash')))
        if "BB Lower" in overlays:
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_LOW'], name='BB Lower', line=dict(dash='dash')))
        ed = info.get('earningsDate')
        if isinstance(ed, (list, tuple)):
            for e in ed:
                try:
                    dt = pd.to_datetime(e).normalize()
                    fig.add_vline(x=dt, line_dash='dot', annotation_text='Earnings', annotation_position='top')
                except:
                    pass
        fig.update_layout(title="Price & Indicators", height=400, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True, key=f"price_{tkr}")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'
        ))
        fig.add_trace(go.Scatter(x=df.index, y=df['Fast_MA'], name=f"MA{short_ma}"))
        fig.add_trace(go.Scatter(x=df.index, y=df['Slow_MA'], name=f"MA{long_ma}"))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_UP'], name='BB Upper', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_LOW'], name='BB Lower', line=dict(dash='dash')))
        ed = info.get('earningsDate')
        if isinstance(ed, (list, tuple)):
            for e in ed:
                try:
                    dt = pd.to_datetime(e).normalize()
                    fig.add_vline(x=dt, line_dash='dot', annotation_text='Earnings', annotation_position='top')
                except: pass
        fig.update_layout(title="Price & Indicators", height=400, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
        # Volume & OBV
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'))
        fig2.add_trace(go.Scatter(x=df.index, y=df['OBV'], name='OBV', yaxis='y2'))
        fig2.update_layout(title="Volume & OBV", height=250, template='plotly_dark',
                           yaxis_title='Volume', yaxis2=dict(overlaying='y', side='right', title='OBV'))
        st.plotly_chart(fig2, use_container_width=True, key=f"vol_{tkr}")
        # RSI & Drawdown
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
        fig3.add_trace(go.Scatter(x=df.index, y=df['Drawdown'], name='Drawdown', yaxis='y2'))
        fig3.update_layout(title="RSI & Drawdown", height=250, template='plotly_dark',
                           yaxis=dict(range=[0,100], title='RSI'),
                           yaxis2=dict(overlaying='y', side='right', title='Drawdown'))
        st.plotly_chart(fig3, use_container_width=True, key=f"rsi_{tkr}")
        # News & Sentiment
        if show_news:
            name = info.get('longName') or info.get('shortName') or tkr
            arts = fetch_news(name)
            if arts:
                st.subheader("Latest News & Sentiment")
                dates, scores, sents = [], [], []
                for art in arts:
                    dates.append(pd.to_datetime(art.get('publishedAt','')).normalize())
                    comp = analyzer.polarity_scores(art.get('title',''))['compound']
                    scores.append(comp)
                    sents.append('Positive' if comp>=0.05 else 'Negative' if comp<=-0.05 else 'Neutral')
                cnt = Counter(sents)
                overall = np.mean(scores) if scores else 0
                # Expanded by default
                with st.expander("AI Summary & Drivers", expanded=True):
                    heads = [a['title'] for a in arts]
                    summ = ai_summary(tkr, heads, curr, prev, info.get('trailingPE'), info.get('netMargins'), info.get('debtToEquity'))
                    st.markdown(summ)
                st.metric("Sentiment Score", f"{overall:.2f}")
                df_sent = pd.DataFrame({'date':dates,'compound':scores})
                daily = df_sent.groupby('date', as_index=False).mean()
                fig_line = px.line(daily, x='date', y='compound', title='Sentiment Trend', height=200)
                fig_line.update_layout(template='plotly_dark', margin=dict(t=0,b=0), yaxis_title='Sentiment')
                st.plotly_chart(fig_line, use_container_width=True, key=f"sent_line_{tkr}")
                fig_pie = px.pie(names=list(cnt.keys()), values=list(cnt.values()), title='Sentiment Distribution', height=200)
                fig_pie.update_layout(template='plotly_dark', margin=dict(t=0,b=0))
                st.plotly_chart(fig_pie, use_container_width=True, key=f"sent_pie_{tkr}")
                choice = st.radio("Filter Sentiment", ['Positive','Neutral','Negative'], key=f"sent_{tkr}")
                for art, s in zip(arts, sents):
                    if s == choice:
                        st.markdown(f"**{art['source']['name']}** ({art['publishedAt'][:10]}): [{art['title']}]({art['url']}) - *{s}*")
        # Alerts & Export
        if curr < rsi_alert:
            st.warning(f"RSI below {rsi_alert}: {curr:.1f}")
        if st.button(f"Download {tkr} Report", key=f"dl_{tkr}"):
            st.success("Report feature coming soon!")
        st.caption("Powered by Streamlit, yFinance, NewsAPI & OpenAI")
