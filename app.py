import streamlit as st
import os, math, random, warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# --- Setup ---
warnings.filterwarnings("ignore")
tqdm.pandas()
sns.set(style="whitegrid")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ================================================================
# Streamlit App
# ================================================================
st.set_page_config(page_title="NIFTY-500 AI Portfolio Builder", layout="wide")
st.title("📊 NIFTY-500 AI Portfolio Builder (ANOVA + RandomForest version)")
st.markdown("""
This app fetches NIFTY-500 tickers, validates them, builds features, trains an ANOVA+RandomForest model, scores stocks, and creates a top-15 portfolio with visualizations.
""")

# Sidebar for inputs
st.sidebar.header("Settings")
investment_amount = st.sidebar.number_input("Total Investment Amount (₹)", min_value=1000.0, value=100000.0, step=1000.0)
top_k = st.sidebar.slider("Top Stocks to Select", min_value=5, max_value=50, value=15)

# Cache data fetching and processing
@st.cache_data
def get_nifty500_tickers():
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        df = pd.read_csv(url)
        tickers = [t.strip() + ".NS" for t in df["Symbol"].dropna().unique().tolist()]
        return tickers
    except Exception as e:
        st.error(f"Could not fetch NSE list: {e}")
        return []

@st.cache_data
def validate_tickers(tickers):
    valid = []
    progress_bar = st.progress(0)
    for i, t in enumerate(tickers):
        try:
            data = yf.download(t, period="5d", progress=False)
            if not data.empty:
                valid.append(t)
        except:
            pass
        progress_bar.progress((i + 1) / len(tickers))
    progress_bar.empty()
    return valid

@st.cache_data
def fetch_price_data(tickers, period="1y", interval="1d"):
    all_data, failed = {}, []
    batch_size = 50
    progress_bar = st.progress(0)
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.download(batch, period=period, interval=interval, group_by='ticker', progress=False)
            for t in batch:
                try:
                    df = data[t] if t in data else None
                    if df is not None and not df.empty:
                        df = df.reset_index()
                        df.columns = [c.lower() for c in df.columns]
                        df["ticker"] = t
                        all_data[t] = df
                    else:
                        failed.append(t)
                except Exception:
                    failed.append(t)
        except Exception:
            failed.extend(batch)
        progress_bar.progress((i + batch_size) / len(tickers))
    progress_bar.empty()
    return all_data

@st.cache_data
def build_feature_dataset(data_dict):
    dfs = []
    for t, df in data_dict.items():
        f = build_features(df)
        if not f.empty:
            dfs.append(f)
    return pd.concat(dfs, ignore_index=True)

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
        "accuracy": accuracy_score(y_val, preds),
        "precision": precision_score(y_val, preds),
        "recall": recall_score(y_val, preds),
        "f1": f1_score(y_val, preds),
        "roc_auc": roc_auc_score(y_val, prob)
    }
    return rf, feats, scaler, selector, metrics

@st.cache_data
def score_all(df, _model, feats, _scaler, _selector):
    X = df[feats].fillna(0).values
    X_scaled = _scaler.transform(X)
    X_sel = _selector.transform(X_scaled)
    df["score"] = _model.predict_proba(X_sel)[:,1]
    latest = df.sort_values('date').groupby('ticker').tail(1)
    latest = latest[['ticker','score','close']].rename(columns={'close':'last_close'})
    latest = latest.sort_values('score', ascending=False).reset_index(drop=True)
    return latest

def allocate_portfolio(total, df):
    n = len(df)
    equal_amt = total / n
    alloc = []
    for _, r in df.iterrows():
        qty = math.floor(equal_amt / r['last_close'])
        used = qty * r['last_close']
        alloc.append([r['ticker'], r['score'], r['last_close'], qty, used])
    port = pd.DataFrame(alloc, columns=['ticker','score','last_close','quantity','used_amount'])
    port['weight'] = port['used_amount'] / port['used_amount'].sum()
    return port

# Main logic
if st.button("Run Portfolio Builder"):
    with st.spinner("Fetching and validating tickers..."):
        TICKERS = get_nifty500_tickers()
        if not TICKERS:
            VALID_TICKERS = [
                "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
                "SBIN.NS","ITC.NS","BHARTIARTL.NS","KOTAKBANK.NS","LT.NS",
                "HINDUNILVR.NS","AXISBANK.NS","BAJFINANCE.NS","ASIANPAINT.NS",
                "MARUTI.NS","SUNPHARMA.NS","ULTRACEMCO.NS","WIPRO.NS","TITAN.NS","HCLTECH.NS"
            ]
            st.warning("Using fallback 20 reliable tickers.")
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
        st.success("Model trained.")
        st.subheader("Model Performance")
        st.json(metrics)

    with st.spinner("Scoring stocks and allocating portfolio..."):
        scores = score_all(full_df, model, feats, scaler, selector)
        top_stocks = scores.head(top_k)
        portfolio = allocate_portfolio(investment_amount, top_stocks)
        used = portfolio['used_amount'].sum()
        left = investment_amount - used

        st.subheader("Top Stocks")
        st.dataframe(top_stocks)

        st.subheader("Portfolio Allocation")
        st.dataframe(portfolio)
        st.write(f"Total: ₹{investment_amount:.2f}, Used: ₹{used:.2f}, Leftover: ₹{left:.2f}")

        # Downloads
        st.download_button("Download Top Stocks CSV", top_stocks.to_csv(index=False), "top_stocks.csv", "text/csv")
        st.download_button("Download Portfolio CSV", portfolio.to_csv(index=False), "portfolio_allocation.csv", "text/csv")
        st.download_button("Download All Scores CSV", scores.to_csv(index=False), "all_scores.csv", "text/csv")

    # Visualizations
    st.subheader("Visualizations")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x='ticker', y='score', data=top_stocks, palette="viridis", ax=ax)
        ax.set_title("Top Stock Scores")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(portfolio['used_amount'], labels=portfolio['ticker'], autopct='%1.1f%%', startangle=140, colors=sns.color_palette("tab20", top_k))
        ax.set_title("Portfolio Allocation")
        st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(scores['score'], bins=30, kde=True, color='skyblue', ax=ax)
    ax.set_title("Score Distribution of All Tickers")
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='ticker', y='weight', data=portfolio, palette="coolwarm", ax=ax)
    ax.set_title("Portfolio Weight Distribution")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='score', y='used_amount', data=portfolio, hue='ticker', s=100, palette="tab20", ax=ax)
    ax.set_title("Score vs Investment Used")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

    # Dashboard
    st.subheader("Portfolio Dashboard")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    sns.barplot(x='ticker', y='weight', data=portfolio, palette="coolwarm", ax=axes[0,0])
    axes[0,0].set_title("Portfolio Weights")
    axes[0,0].tick_params(axis='x', rotation=45)

    sns.barplot(x='ticker', y='used_amount', data=portfolio, palette="magma", ax=axes[0,1])
    axes[0,1].set_title("Investment per Stock")
    axes[0,1].tick_params(axis='x', rotation=45)

    sns.scatterplot(x='score', y='used_amount', data=portfolio, hue='ticker', s=100, palette="tab20", ax=axes[1,0])
    axes[1,0].set_title("Score vs Used Amount")
    axes[1,0].set_xlabel("Score")
    axes[1,0].set_ylabel("Used ₹")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.suptitle("Portfolio Dashboard — Top NIFTY-500 Stocks", fontsize=16, y=1.02)
    st.pyplot(fig)

    st.success("All done!")
