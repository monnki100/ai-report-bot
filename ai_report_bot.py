import yfinance as yf
import datetime
import pandas as pd

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

def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

score = 50
report_data = {}

for ticker in tickers:
    stock = yf.Ticker(ticker)
    hist = stock.history(period="6mo")

    if len(hist) < 50:
        continue

    current_price = hist["Close"].iloc[-1]
    prev_price = hist["Close"].iloc[-2]
    change = (current_price / prev_price - 1) * 100

    ma50 = hist["Close"].rolling(50).mean().iloc[-1]
    rsi = calculate_rsi(hist["Close"]).iloc[-1]

    volume_today = hist["Volume"].iloc[-1]
    volume_avg = hist["Volume"].rolling(20).mean().iloc[-1]

    report_data[ticker] = {
        "change": round(change,2),
        "ma50": round(ma50,2),
        "rsi": round(rsi,2),
        "volume_ratio": round(volume_today/volume_avg,2)
    }

    # スコアロジック
    if ticker in ["NVDA","AMD"]:
        if change > 2:
            score += 5
        if current_price > ma50:
            score += 5
        if rsi < 30:
            score += 3
        if rsi > 70:
            score -= 3

    if ticker == "^VIX":
        if change > 5:
            score -= 10

    if ticker == "^GSPC":
        if change > 1:
            score += 5
        if change < -1:
            score -= 5

# 温度判定
if score >= 80:
    temp = "🔥 加速局面"
elif score >= 65:
    temp = "🟢 強気"
elif score >= 45:
    temp = "⚖ 中立"
elif score >= 30:
    temp = "🟡 減速"
else:
    temp = "❄ 崩れ"

# 出力
print("===== AI市場プロレポート =====")
print("日付:", datetime.date.today())
print("市場温度:", score, temp)
print("")

for ticker, name in tickers.items():
    if ticker in report_data:
        d = report_data[ticker]
        print(f"{name} ({ticker})")
        print(f"  前日比: {d['change']}%")
        print(f"  MA50: {d['ma50']}")
        print(f"  RSI: {d['rsi']}")
        print(f"  出来高倍率: {d['volume_ratio']}倍")
        print("")

print("■ 戦略指針")

if score >= 65:
    print("・押し目積極")
    print("・トレンドフォロー有効")
elif score >= 45:
    print("・ポジション維持")
    print("・新規は選別")
else:
    print("・信用縮小")
    print("・ディフェンシブ優先")
