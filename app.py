import streamlit as st
import os, math, random, warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --- Setup ---
warnings.filterwarnings("ignore")
tqdm.pandas()
sns.set(style="whitegrid")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

st.set_page_config(page_title="📊 NIFTY-500 AI Portfolio", layout="wide", page_icon="💹")
st.title("📊 NIFTY-500 AI Portfolio Builder")
st.markdown("Build a top-stock portfolio using ANOVA + RandomForest model with visual insights!")

# --- Sidebar ---
st.sidebar.header("⚙️ Settings")
investment_amount = st.sidebar.number_input("💰 Total Investment Amount (₹)", min_value=1000.0, value=100000.0, step=1000.0)
top_k = st.sidebar.slider("📈 Number of Top Stocks to Select", min_value=5, max_value=50, value=15)

# --- Functions ---
@st.cache_data
def get_nifty500_tickers():
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        df = pd.read_csv(url)
        tickers = [t.strip() + ".NS" for t in df["Symbol"].dropna().unique().tolist()]
        return tickers
    except:
        return []

@st.cache_data
def validate_tickers(tickers):
    valid = []
    for t in tqdm(tickers, desc="Validating tickers"):
        try:
            data = yf.download(t, period="5d", progress=False)
            if not data.empty:
                valid.append(t)
        except:
            pass
    return valid

@st.cache_data
def fetch_price_data(tickers, period="1y", interval="1d"):
    all_data = {}
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.download(batch, period=period, interval=interval, group_by='ticker', progress=False)
            for t in batch:
                df = data[t] if t in data else None
                if df is not None and not df.empty:
                    df = df.reset_index()
                    df.columns = [c.lower() for c in df.columns]
                    df["ticker"] = t
                    all_data[t] = df
        except:
            pass
    return all_data

def build_features(df):
    df = df.copy()
    df["ret_1d"] = df["close"].pct_change()
    df["ret_5d"] = df["close"].pct_change(5)
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["vol_5d"] = df["close"].rolling(5).std()
    df["rsi"] = 100 - (100 / (1 + (df["ret_1d"].rolling(14)
                   .apply(lambda x: (x[x>0].sum() / -x[x<0].sum()) if -x[x<0].sum()!=0 else 0))))
    df["target"] = (df["close"].shift(-5) > df["close"]).astype(int)
    return df.dropna().reset_index(drop=True)

@st.cache_data
def build_feature_dataset(data_dict):
    dfs = []
    for df in data_dict.values():
        f = build_features(df)
        if not f.empty:
            dfs.append(f)
    return pd.concat(dfs, ignore_index=True)

@st.cache_data
def train_anova_rf(df):
    feats = ['ret_1d','ret_5d','ma_5','ma_20','vol_5d','rsi']
    X = df[feats].fillna(0).values
    y = df["target"].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    selector = SelectKBest(f_classif, k=min(5, X_train_s.shape[1]))
    X_train_sel = selector.fit_transform(X_train_s, y_train)
    X_val_sel = selector.transform(X_val_s)
    rf = RandomForestClassifier(n_estimators=150, random_state=SEED)
    rf.fit(X_train_sel, y_train)
    preds = rf.predict(X_val_sel)
    prob = rf.predict_proba(X_val_sel)[:,1]
    metrics = {
        "Accuracy": accuracy_score(y_val, preds),
        "Precision": precision_score(y_val, preds),
        "Recall": recall_score(y_val, preds),
        "F1-Score": f1_score(y_val, preds),
        "ROC-AUC": roc_auc_score(y_val, prob)
    }
    return rf, feats, scaler, selector, metrics

@st.cache_data
def score_all(df, model, feats, scaler, selector):
    X = df[feats].fillna(0).values
    X_scaled = scaler.transform(X)
    X_sel = selector.transform(X_scaled)
    df["score"] = model.predict_proba(X_sel)[:,1]
    latest = df.sort_values('date').groupby('ticker').tail(1)
    latest = latest[['ticker','score','close']].rename(columns={'close':'last_close'}).sort_values('score', ascending=False)
    return latest

def allocate_portfolio(total, df):
    n = len(df)
    equal_amt = total / n
    alloc = []
    for _, r in df.iterrows():
        qty = math.floor(equal_amt / r['last_close'])
        used = qty * r['last_close']
        alloc.append([r['ticker'], r['score'], r['last_close'], qty, used])
    port = pd.DataFrame(alloc, columns=['Ticker','Score','Last Close','Quantity','Used Amount'])
    port['Weight'] = port['Used Amount'] / port['Used Amount'].sum()
    return port

# --- Main ---
if st.button("🚀 Run Portfolio Builder"):
    with st.spinner("Fetching and validating tickers..."):
        TICKERS = get_nifty500_tickers()
        if not TICKERS:
            VALID_TICKERS = [
                "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
                "SBIN.NS","ITC.NS","BHARTIARTL.NS","KOTAKBANK.NS","LT.NS"
            ]
            st.warning("Using fallback 10 reliable tickers.")
        else:
            VALID_TICKERS = validate_tickers(TICKERS)
            st.success(f"Validated {len(VALID_TICKERS)} tickers.")

    with st.spinner("Fetching price data..."):
        price_data = fetch_price_data(VALID_TICKERS, period="1y")
        st.success(f"Fetched data for {len(price_data)} tickers.")

    with st.spinner("Building features..."):
        full_df = build_feature_dataset(price_data)
        st.success(f"Feature dataset shape: {full_df.shape}")

    with st.spinner("Training model..."):
        model, feats, scaler, selector, metrics = train_anova_rf(full_df)
        st.success("Model trained successfully!")

    st.subheader("📊 Model Performance Metrics")
    cols = st.columns(len(metrics))
    for i, (k,v) in enumerate(metrics.items()):
        cols[i].metric(label=k, value=f"{v:.4f}")

    with st.spinner("Scoring stocks and allocating portfolio..."):
        scores = score_all(full_df, model, feats, scaler, selector)
        top_stocks = scores.head(top_k)
        portfolio = allocate_portfolio(investment_amount, top_stocks)
        used = portfolio['Used Amount'].sum()
        leftover = investment_amount - used

    # --- Tabs for dashboard ---
    tabs = st.tabs(["Top Stocks", "Portfolio Allocation", "Visualizations"])
    with tabs[0]:
        st.subheader("🏆 Top Stocks")
        st.dataframe(top_stocks)
        st.download_button("Download Top Stocks CSV", top_stocks.to_csv(index=False), "top_stocks.csv", "text/csv")

    with tabs[1]:
        st.subheader("💼 Portfolio Allocation")
        st.dataframe(portfolio)
        st.write(f"💰 Total: ₹{investment_amount:.2f}, Used: ₹{used:.2f}, Leftover: ₹{leftover:.2f}")
        st.download_button("Download Portfolio CSV", portfolio.to_csv(index=False), "portfolio.csv", "text/csv")

    with tabs[2]:
        st.subheader("📈 Visualizations")
        fig1 = px.bar(top_stocks, x='ticker', y='score', color='score', title="Top Stock Scores", text='score')
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.pie(portfolio, names='Ticker', values='Used Amount', title="Portfolio Distribution")
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.histogram(scores, x='score', nbins=30, title="Score Distribution of All Tickers")
        st.plotly_chart(fig3, use_container_width=True)

st.success("✅ Portfolio Builder Finished!")
