📈 FinSight — AI-Powered Paper Trading & Financial Learning Platform

FinSight · Smart Investing Simulator & Learning Platform

FinSight is a full-stack, AI-powered paper trading and financial exploration platform that allows users to simulate stock market trading using real-time market data, manage a virtual wallet, and learn investing concepts without any financial risk. The platform integrates live market feeds, secure authentication, virtual payments, and an optional AI assistant — all built with scalability in mind.


![Python](https://img.shields.io/badge/Python-3.10+-blue.svg) 
![Streamlit](https://img.shields.io/badge/Streamlit-1.36+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)


✨ Key Features
🔐 User Authentication

Secure signup, login, and logout

Session-based authentication

User-specific portfolios and trade history

📊 Market Exploration

Live stock prices using Yahoo Finance (yfinance)

Indian and US equities support (NSE & NASDAQ)

Market indices overview (NIFTY, SENSEX, BANKNIFTY, NASDAQ)

Search-based stock discovery

🧾 Paper Trading Engine

Buy & sell stocks using virtual INR

Real-time price conversion (USD → INR)

Automatic wallet balance updates

Position tracking with average buy price

Profit & Loss (PnL) calculation

💼 Portfolio Management

Live valuation of holdings

Wallet cash + holdings = total portfolio value

Trade execution history

Order and transaction logs

💰 Virtual Wallet System

Each user gets a default virtual wallet

Wallet balance persists across sessions

Integrated with paper trading engine

💳 Payment & Top-Up Simulation

Razorpay payment gateway integration (Test Mode)

Secure server-side webhook verification

Wallet crediting via webhook events

Payment history logging

Cloud-ready payment architecture

🤖 AI Assistant (Optional)

Floating AI chatbot for finance queries

Market explanations & learning support

Built for future GenAI extensions

🏗️ System Architecture (High Level)

Frontend: Streamlit (UI, dashboards, trading views)

Backend: Python services + Flask webhook server

Database: SQLite (wallets, users, trades, holdings)

Market Data: Yahoo Finance API

Payments: Razorpay (Test Mode + Webhooks)

AI Layer: Optional LLM-based assistant

Tunneling: ngrok (for webhook testing)

🚀 Quick Start
Prerequisites

Python 3.10+

pip

Internet connection (for live market data)

Installation

Clone the repository

git clone https://github.com/your-username/finsight.git
cd finsight


Install dependencies

pip install -r requirements.txt


Run the Streamlit app

streamlit run app.py


(Optional) Run webhook server

cd backend
python server.py


Open in browser

http://localhost:8501

📖 Usage Flow

User logs in or signs up

User lands on the Main Dashboard

Explore stocks and market indices

Perform paper trades (buy/sell)

Portfolio and wallet update automatically

Optional wallet top-up via payment simulation

AI assistant available for guidance

📁 Project Structure
FinSight/
├── app.py                     # Streamlit entry point
├── pages/
│   ├── 1_stock.py
│   ├── 2_commodities.py
│   ├── 3_Mutual_funds.py
│   ├── 4_Paper_Trading.py
│   ├── 5_learners_guide.py
│   └── 6_Buy_Virtual_Cash.py
├── utils/
│   ├── auth.py                # Authentication logic
│   ├── trading_db.py          # Wallet, trades, holdings
│   ├── payments.py            # Razorpay helpers
│   └── floating_chatbot.py    # AI assistant
├── backend/
│   └── server.py              # Razorpay webhook server
├── data/
│   └── finsight.db            # SQLite database
├── requirements.txt
└── README.md

🛠️ Technology Stack

Frontend: Streamlit

Backend: Python, Flask

Database: SQLite

Market Data: Yahoo Finance (yfinance)

Payments: Razorpay (Test Mode)

AI: Optional LLM integration

Deployment Ready: Cloud & serverless compatible

🔒 Security & Design Notes

Payments verified server-side via webhooks

No direct wallet updates from frontend

Session-isolated user data

Safe for cloud deployment (AWS / GCP / Azure)

No real money involved (educational platform)

🎯 Use Cases

Financial education

Stock market learning

Algorithmic trading practice

Academic projects

Patent & research demonstrations

Portfolio simulation

📝 License

This project is licensed under the MIT License.

👤 Author

Taarunya Aggarwal

Project: FinSight

Built for academic, research, and learning purposes
