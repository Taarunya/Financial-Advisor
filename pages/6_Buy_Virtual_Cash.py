import streamlit as st
from utils.auth import require_login
from utils.trading_db import get_balance, add_virtual_cash

st.set_page_config(
    page_title="FinSight | Buy Virtual Cash",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# ---------------------------
# Mandating Login function 
# ---------------------------
require_login()

user = st.session_state["user"]
user_id = user["id"]

# -----------------------------
# HEADER
# -----------------------------
top1, top2 = st.columns([3, 1])

with top1:
    st.markdown("## Buy Virtual Trading Balance")

with top2:
    if st.button("Back to Paper Trading", use_container_width=True):
        st.switch_page("pages/4_Paper_Trading.py")

st.write("")

# -----------------------------
# WALLET INFO
# -----------------------------
wallet = get_balance(user_id)

c1, c2, c3 = st.columns(3)
c1.metric("Current Virtual Balance", f"₹{wallet:,.2f}")
c2.metric("Account Type", "Trading Wallet")
c3.metric("User", user["email"])

st.write("")
st.markdown("### Add Balance Packages")

# -----------------------------
# PACKAGES
# -----------------------------
packages = [
    {"real": 50, "virtual": 50_000},
    {"real": 100, "virtual": 120_000},
    {"real": 250, "virtual": 350_000},
    {"real": 500, "virtual": 800_000},
]

cols = st.columns(4, gap="large")

for i, pkg in enumerate(packages):
    with cols[i]:
        st.markdown(
            f"""
            <div style="
                background: rgba(255,255,255,0.04);
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 20px;
                padding: 20px;
                text-align: center;
                box-shadow: 0 10px 40px rgba(0,0,0,0.35);
            ">
                <h3>₹{pkg['virtual']:,}</h3>
                <p style="opacity:0.8;">Trading Balance</p>
                <p style="font-weight:800;">Pay ₹{pkg['real']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(f"Add ₹{pkg['virtual']:,}", key=f"buy_{pkg['real']}", use_container_width=True):
            add_virtual_cash(user_id, pkg["real"], pkg["virtual"])
            st.success("Balance added successfully")
            st.rerun()

# -----------------------------
# HISTORY SHORTCUT
# -----------------------------
st.write("")
st.divider()

colx, coly = st.columns([3, 1])
with colx:
    st.markdown("#### Why add virtual balance?")
    st.markdown(
        """
        • Trade with higher capital  
        • Test advanced strategies  
        • Build realistic portfolios  
        • Improve risk management  
        """
    )

with coly:
    if st.button("Go to Trading Dashboard", use_container_width=True):
        st.switch_page("pages/4_Paper_Trading.py")

st.caption("FinSight • Virtual Trading Wallet")
