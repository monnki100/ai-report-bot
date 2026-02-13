import yfinance as yf
import datetime

tickers = {
    "NVDA": "NVIDIA",
    "MU": "Micron",
    "AMD": "AMD",
    "AVGO": "Broadcom",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet",
    "^GSPC": "S&P500",
    "^VIX": "VIX"
}

data = {}

for ticker in tickers:
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5d")
    if len(hist) >= 2:
        change = (hist["Close"].iloc[-1] / hist["Close"].iloc[-2] - 1) * 100
        data[ticker] = round(change, 2)
    else:
        data[ticker] = 0

# スコア計算
score = 50

if data["NVDA"] > 2:
    score += 10
if data["AMD"] > 2:
    score += 5
if data["^VIX"] > 5:
    score -= 15
if data["^GSPC"] < -1:
    score -= 10

# 温度判定
if score >= 80:
    temp = "🔥 加速局面"
elif score >= 60:
    temp = "🟢 強気"
elif score >= 40:
    temp = "⚖ 中立"
elif score >= 20:
    temp = "🟡 減速"
else:
    temp = "❄ 崩れ"

# レポート出力
print("===== AI市場レポート =====")
print("日付:", datetime.date.today())
print("市場温度:", score, temp)
print("")

for ticker, name in tickers.items():
    print(f"{name} ({ticker}): {data[ticker]}%")

print("")
print("■ 戦略コメント")

if score >= 60:
    print("・強気維持")
    print("・押し目戦略有効")
elif score >= 40:
    print("・ポジション維持")
    print("・新規は慎重")
else:
    print("・信用縮小検討")
    print("・防御優先")
