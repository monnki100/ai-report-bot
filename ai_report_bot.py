import yfinance as yf
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

tickers = ["NVDA","MU","AMD","AVGO","MSFT","AMZN","GOOGL","^GSPC","^VIX"]

data = {}

for t in tickers:
    stock = yf.Ticker(t)
    hist = stock.history(period="5d")
    if len(hist) >= 2:
        change = (hist["Close"].iloc[-1] / hist["Close"].iloc[-2] - 1) * 100
        data[t] = round(change,2)
    else:
        data[t] = 0

score = 50
if data["NVDA"] > 2:
    score += 10
if data["^VIX"] > 5:
    score -= 15

prompt = f"""
AI市場データ:
{data}

市場温度スコア: {score}

市場レポートを作成してください。
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print("===== AI市場レポート =====")
print(response.choices[0].message.content)
