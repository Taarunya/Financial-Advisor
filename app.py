# app.py - HOME PAGE
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="FinSight | AI Financial Advisor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: #0a0e1a;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Navigation */
    .nav-container {
        background: linear-gradient(135deg, rgba(26, 31, 53, 0.9) 0%, rgba(15, 18, 25, 0.9) 100%);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(255,255,255,0.08);
        padding: 20px 40px;
        margin: 0px -60px 30px -60px;
        position: sticky;
        top: 3.8rem;
        z-index: 100;
    }
    
    .nav-logo {
        font-size: 28px;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
        margin-right: 40px;
    }
    
    /* Hero Section */
    .hero {
        background: linear-gradient(135deg, #1a1f35 0%, #0f1219 100%);
        padding: 60px 40px;
        border-radius: 24px;
        margin-bottom: 40px;
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
    }
    
    .hero-title {
        font-size: 56px;
        font-weight: 900;
        background: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
        line-height: 1.2;
    }
    
    .hero-subtitle {
        font-size: 20px;
        color: rgba(255,255,255,0.7);
        margin-bottom: 30px;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .feature-pills {
        margin-top: 25px;
    }
    
    .feature-pill {
        display: inline-block;
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        color: #00d4ff;
        padding: 10px 20px;
        border-radius: 25px;
        font-size: 14px;
        font-weight: 600;
        margin: 5px;
    }
    
    /* Market Ticker */
    .market-ticker {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .market-ticker:hover {
        background: rgba(255,255,255,0.05);
        border-color: rgba(0, 212, 255, 0.2);
        transform: translateY(-3px);
    }
    
    .ticker-name {
        font-size: 13px;
        color: rgba(255,255,255,0.6);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    
    .ticker-price {
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
        margin: 8px 0;
    }
    
    .ticker-change {
        font-size: 15px;
        font-weight: 600;
    }
    
    .positive { color: #00ff88; }
    .negative { color: #ff4757; }
    
    /* Section Cards */
    .section-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 35px;
        transition: all 0.3s ease;
        height: 100%;
        cursor: pointer;
        min-height: 260px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .section-card:hover {
        transform: translateY(-8px);
        border-color: rgba(0, 212, 255, 0.4);
        box-shadow: 0 15px 50px rgba(0, 212, 255, 0.2);
    }
    
    .card-icon {
        font-size: 48px;
        margin-bottom: 20px;
    }
    
    .card-title {
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 12px;
    }
    
    .card-description {
        font-size: 15px;
        color: rgba(255,255,255,0.6);
        line-height: 1.7;
        margin-bottom: 20px;
    }
    
    .section-header {
        font-size: 32px;
        font-weight: 800;
        color: #ffffff;
        margin: 50px 0 30px 0;
    }
    
    /* Buttons */
    div.stButton > button {
        border-radius: 12px !important;
        height: 50px !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%) !important;
        color: #0a0e1a !important;
        border: none !important;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 40px rgba(0, 212, 255, 0.3);
    }
    
    /* Trending Section */
    .trending-stock {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 18px;
        margin: 10px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: all 0.2s ease;
    }
    
    .trending-stock:hover {
        background: rgba(255,255,255,0.06);
        border-color: rgba(0, 212, 255, 0.3);
    }
    
    .stock-name {
        font-size: 16px;
        font-weight: 600;
        color: #ffffff;
    }
    
    .stock-price {
        font-size: 18px;
        font-weight: 700;
        color: #00d4ff;
    }
    
    /* Fix for button containers */
    .button-container {
        margin-top: 20px;
    }
    
    /* Chat button styling */
    .chat-button {
        background: linear-gradient(135deg, #ff6b6b 0%, #ffa500 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: 700 !important;
        cursor: pointer !important;
    }
</style>
""", unsafe_allow_html=True)

# Navigation Bar
st.markdown("""
<div class="nav-container">
    <span class="nav-logo">📊 FinSight</span>
</div>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero">
    <div class="hero-title">Welcome to FinSight AI</div>
    <div class="hero-subtitle">
        Your intelligent financial companion for stocks, mutual funds, and commodities analysis.
        Powered by advanced AI and real-time market data.
    </div>
    <div class="feature-pills">
        <span class="feature-pill">AI-Powered Insights</span>
        <span class="feature-pill">Real-time Data</span>
        <span class="feature-pill">Portfolio Analytics</span>
        <span class="feature-pill">Risk Assessment</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Market Overview
st.markdown('<div class="section-header">Market Overview</div>', unsafe_allow_html=True)

@st.cache_data(ttl=300)
def get_index_data(ticker):
    try:
        t = yf.Ticker(ticker)

        # -------- Method 1: fast_info --------
        try:
            p = t.fast_info.get("last_price")
            prev = t.fast_info.get("previous_close")
            if p and prev:
                change = p - prev
                change_pct = (change / prev) * 100
                return float(p), float(change), float(change_pct)
        except:
            pass

        # -------- Method 2: info --------
        try:
            info = t.info
            p = info.get("regularMarketPrice")
            prev = info.get("regularMarketPreviousClose")
            if p and prev:
                change = p - prev
                change_pct = (change / prev) * 100
                return float(p), float(change), float(change_pct)
        except:
            pass

        # -------- Method 3: history fallback --------
        hist = t.history(period="5d", interval="1d")
        if hist is not None and len(hist) >= 2:
            current = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2])
            change = current - prev
            change_pct = (change / prev) * 100
            return current, change, change_pct

        return None, None, None

    except Exception:
        return None, None, None


indices = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "BANK NIFTY": "^NSEBANK",
    "NASDAQ": "^IXIC"
}

market_cols = st.columns(4)
for idx, (name, ticker) in enumerate(indices.items()):
    with market_cols[idx]:
        price, change, change_pct = get_index_data(ticker)
        
        if price is not None and change is not None and change_pct is not None:
            change_class = "positive" if change >= 0 else "negative"
            sign = "+" if change >= 0 else ""
            
            st.markdown(f"""
            <div class="market-ticker">
                <div class="ticker-name">{name}</div>
                <div class="ticker-price">{price:,.2f}</div>
                <div class="ticker-change {change_class}">{sign}{change:,.2f} ({sign}{change_pct:.2f}%)</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="market-ticker">
                <div class="ticker-name">{name}</div>
                <div class="ticker-price">N/A</div>
                <div class="ticker-change">Loading...</div>
            </div>
            """, unsafe_allow_html=True)

# Quick Access Sections
st.markdown('<div class="section-header">Explore Features</div>', unsafe_allow_html=True)

# -------- ROW 1 --------
row1 = st.columns(2)

# Stock Analysis Card
with row1[0]:
    st.markdown("""
    <div class="section-card">
        <div class="card-title">Stock Analysis</div>
        <div class="card-description">
            Deep dive into stocks with AI-powered insights, fundamentals, risk metrics, 
            and comparison tools. Generate comprehensive reports instantly.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Analyze Stocks →", key="goto_stocks", use_container_width=True):
        try:
            st.switch_page("pages/1_stock.py")
        except Exception as e:
            st.error(f"Navigation error: {str(e)}")


# Mutual Funds Card
with row1[1]:
    st.markdown("""
    <div class="section-card">
        <div class="card-title">Mutual Funds</div>
        <div class="card-description">
            Explore top-performing mutual funds, compare returns, and build your 
            investment portfolio with expert recommendations.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("View Funds →", key="goto_mf", use_container_width=True):
        try:
            st.switch_page("pages/3_Mutual_funds.py")
        except Exception as e:
            st.error(f"Navigation error: {str(e)}")


st.write("")  # spacing between rows


# -------- ROW 2 --------
row2 = st.columns(2)

# Commodities Card
with row2[0]:
    st.markdown("""
    <div class="section-card">
        <div class="card-title">Commodities</div>
        <div class="card-description">
            Track real-time prices of gold, silver, crude oil, and other commodities. 
            Stay updated with market trends.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("View Commodities →", key="goto_comm", use_container_width=True):
        try:
            st.switch_page("pages/2_commodities.py")
        except Exception as e:
            st.error(f"Navigation error: {str(e)}")


# Paper Trading Card (NEW)
with row2[1]:
    st.markdown("""
    <div class="section-card">
        <div class="card-title">Paper Trading</div>
        <div class="card-description">
            Practice stock trading using ₹100,000 virtual money with real market prices.
            Track your portfolio value, profits, losses, and orders safely without any risk.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Start Paper Trading →", key="goto_paper", use_container_width=True):
        try:
            st.switch_page("pages/4_Paper_Trading.py")
        except Exception as e:
            st.error(f"Navigation error: {str(e)}")


# Trending Stocks - Fixed layout with 3 columns
st.markdown('<div class="section-header">Trending Stocks</div>', unsafe_allow_html=True)

# Create 3 columns for trending stocks
trending_cols = st.columns([1, 1, 1])  # Equal width columns

trending_stocks = [
    {"name": "Reliance Industries", "ticker": "RELIANCE.NS"},
    {"name": "TCS", "ticker": "TCS.NS"},
    {"name": "Infosys", "ticker": "INFY.NS"},
    {"name": "HDFC Bank", "ticker": "HDFCBANK.NS"},
    {"name": "ICICI Bank", "ticker": "ICICIBANK.NS"},
    {"name": "Wipro", "ticker": "WIPRO.NS"}
]

# Distribute stocks across 3 columns
for idx, stock in enumerate(trending_stocks):
    col_idx = idx % 3
    with trending_cols[col_idx]:
        price, change, change_pct = get_index_data(stock["ticker"])
        if price is not None and change is not None and change_pct is not None:
            change_class = "positive" if change >= 0 else "negative"
            sign = "+" if change >= 0 else ""
            
            st.markdown(f"""
            <div class="trending-stock">
                <div>
                    <div class="stock-name">{stock['name']}</div>
                    <div class="ticker-change {change_class}">{sign}{change_pct:.2f}%</div>
                </div>
                <div class="stock-price">₹{price:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="trending-stock">
                <div>
                    <div class="stock-name">{stock['name']}</div>
                    <div class="ticker-change">Loading...</div>
                </div>
                <div class="stock-price">-</div>
            </div>
            """, unsafe_allow_html=True)

# Add Chat section
st.markdown("""
<div style="margin-top: 50px; text-align: center;">
    <div style="font-size: 24px; font-weight: 700; color: #ffffff; margin-bottom: 20px;">
        💬 Need Help? Chat with Our AI Assistant
    </div>
    <div style="font-size: 16px; color: rgba(255,255,255,0.6); margin-bottom: 30px; max-width: 600px; margin-left: auto; margin-right: auto;">
        Get instant answers to your financial questions, investment advice, and market insights.
    </div>
</div>
""", unsafe_allow_html=True)

# Chat button
chat_col1, chat_col2, chat_col3 = st.columns([1, 1, 1])
with chat_col2:
    if st.button("💬 Start Chat with AI Assistant", use_container_width=True, 
                 key="chat_button", type="primary"):
        try:
            # Try to open chat if available
            st.info("Chat feature coming soon!")
        except:
            st.info("Chat feature will be available in the next update!")

st.caption("FinSight • AI Financial Advisor • Streamlit + Yahoo Finance")

# Floating chatbot (if exists)
try:
    from utils.floating_chatbot import render_floating_chatbot
    render_floating_chatbot()
except ImportError:
    pass