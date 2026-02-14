import yfinance as yf
import datetime
import pandas as pd
import requests
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from googletrans import Translator

# ===== 設定 =====

tickers = {
    "NVDA": "NVIDIA",
    "MU": "Micron",
    "AMD": "AMD",
    "AVGO": "Broadcom",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet",
    "^GSPC": "S&P500",
    "^VIX": "VIX",
    "SOXX": "SOXX",
    "^IXIC": "NASDAQ"
}

negative_keywords = [
    "lawsuit","crash","downgrade","fraud","investigation",
    "recall","bankruptcy","layoff","decline",
    "missed earnings","regulation","antitrust"
]

translator = Translator()

def translate_to_japanese(text):
    try:
        return translator.translate(text, dest="ja").text
    except:
        return text

def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ===== 初期値 =====

score = 50
risk_flag = False
report_data = {}

# ===== テクニカル分析 =====

for ticker in tickers:
    stock = yf.Ticker(ticker)
    hist = stock.history(period="6mo")

    if len(hist) < 50:
        continue

    current = hist["Close"].iloc[-1]
    prev = hist["Close"].iloc[-2]
    change = (current / prev - 1) * 100
    ma50 = hist["Close"].rolling(50).mean().iloc[-1]
    ma200 = hist["Close"].rolling(200).mean().iloc[-1]
    rsi = calculate_rsi(hist["Close"]).iloc[-1]
    volume_ratio = hist["Volume"].iloc[-1] / hist["Volume"].rolling(20).mean().iloc[-1]

    report_data[ticker] = {
        "change": round(change,2),
        "ma50": round(ma50,2),
        "ma200": round(ma200,2),
        "rsi": round(rsi,2),
        "volume_ratio": round(volume_ratio,2)
    }

    if ticker in ["NVDA","AMD"]:
        if change > 2: score += 5
        if current > ma50: score += 5
        if rsi < 30: score += 3
        if rsi > 70: score -= 3

    if ticker == "^VIX" and change > 5:
        score -= 10
        risk_flag = True

    if ticker == "^GSPC":
        if change > 1: score += 5
        if change < -1: score -= 5

# ===== 長期守り強化 =====

soxx = yf.Ticker("SOXX").history(period="1y")
if len(soxx) >= 200:
    if soxx["Close"].iloc[-1] < soxx["Close"].rolling(200).mean().iloc[-1]:
        score -= 15
        risk_flag = True

nasdaq = yf.Ticker("^IXIC").history(period="1y")
if len(nasdaq) >= 200:
    if nasdaq["Close"].iloc[-1] < nasdaq["Close"].rolling(200).mean().iloc[-1]:
        score -= 10
        risk_flag = True

vix = yf.Ticker("^VIX").history(period="5d")
if len(vix) >= 2:
    vix_change = (vix["Close"].iloc[-1] / vix["Close"].iloc[-2] - 1) * 100
    if vix_change > 10:
        score -= 10
        risk_flag = True

# ===== ニュース取得 =====

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def get_ai_news():
    url = (
        "https://newsapi.org/v2/everything?"
        "q=AI+semiconductor+NVIDIA+AMD&"
        "language=en&sortBy=publishedAt&pageSize=5&"
        f"apiKey={NEWS_API_KEY}"
    )
    r = requests.get(url)
    if r.status_code != 200:
        return []
    data = r.json()
    return [a["title"] for a in data.get("articles", []) if a.get("title")]

news = get_ai_news()

negative_count = 0
translated_news = []

for n in news:
    lower = n.lower()
    for word in negative_keywords:
        if word in lower:
            negative_count += 1
    translated_news.append(translate_to_japanese(n))

if negative_count >= 2:
    score -= 10
    risk_flag = True
elif negative_count == 1:
    score -= 5

# ===== 温度判定（最後に） =====

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

# ===== レポート生成 =====

report = ""
def add_line(text=""):
    global report
    report += text + "\n"

add_line("===== AI市場プロレポート =====")
add_line(f"日付: {datetime.date.today()}")
add_line(f"市場温度: {score} {temp}")
add_line("")

for ticker, name in tickers.items():
    if ticker in report_data:
        d = report_data[ticker]
        add_line(f"{name} ({ticker})")
        add_line(f"  前日比: {d['change']}%")
        add_line(f"  MA50: {d['ma50']}")
        add_line(f"  MA200: {d['ma200']}")
        add_line(f"  RSI: {d['rsi']}")
        add_line(f"  出来高倍率: {d['volume_ratio']}倍")
        add_line("")

add_line("■ AI関連最新ニュース")
for n in translated_news:
    add_line(f"- {n}")

if negative_count >= 2:
    add_line("")
    add_line("⚠ ネガティブニュース増加（市場警戒）")

if risk_flag:
    add_line("")
    add_line("⚠ 崩れモード発動")

# ===== メール送信 =====

gmail_user = os.getenv("GMAIL_ADDRESS")
gmail_password = os.getenv("GMAIL_APP_PASSWORD")

subject = "⚠ AI市場警戒アラート" if risk_flag else "Daily AI Stock Report"

html = f"""
<html>
<body style="font-family:Arial;">
<h2>📊 AI市場プロレポート</h2>
<p><b>日付:</b> {datetime.date.today()}</p>
<p><b>市場温度:</b> {score} {temp}</p>
<hr>
<pre>{report}</pre>
</body>
</html>
"""

msg = MIMEMultipart("alternative")
msg["Subject"] = subject
msg["From"] = gmail_user
msg["To"] = gmail_user
msg.attach(MIMEText(report, "plain"))
msg.attach(MIMEText(html, "html"))

with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
    server.login(gmail_user, gmail_password)
    server.send_message(msg)

print("Email sent successfully!")
