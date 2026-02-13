import yfinance as yf
import datetime
import pandas as pd
import requests
import os

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
    ma200 = hist["Close"].rolling(200).mean().iloc[-1]
    rsi = calculate_rsi(hist["Close"]).iloc[-1]

    volume_today = hist["Volume"].iloc[-1]
    volume_avg = hist["Volume"].rolling(20).mean().iloc[-1]

    report_data[ticker] = {
        "change": round(change,2),
        "ma50": round(ma50,2),
        "ma200": round(ma200,2),
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

risk_flag = False

if "SOXX" in report_data and report_data["SOXX"]["change"] < -3:
    risk_flag = True

if "^VIX" in report_data and report_data["^VIX"]["change"] > 7:
    risk_flag = True

# ===== 出力 =====

report = ""

def add_line(text=""):
    print(text)
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

add_line("■ 戦略指針")

if score >= 65:
    add_line("・押し目積極")
    add_line("・トレンドフォロー有効")
elif score >= 45:
    add_line("・ポジション維持")
    add_line("・新規は選別")
else:
    add_line("・信用縮小")
    add_line("・ディフェンシブ優先")

if risk_flag:
    add_line("")
    add_line("⚠ 崩れモード警戒（半導体指数 or VIX急変）")

# ===== SOXX 長期トレンド監視 =====
soxx_hist = yf.Ticker("SOXX").history(period="1y")

if len(soxx_hist) >= 200:
    soxx_ma200 = soxx_hist["Close"].rolling(200).mean().iloc[-1]
    soxx_now = soxx_hist["Close"].iloc[-1]

    if soxx_now < soxx_ma200:
        score -= 15
        risk_flag = True
        add_line("⚠ SOXXが200日線割れ（長期トレンド崩れ）")

# ===== NASDAQ 長期監視 =====
nasdaq_hist = yf.Ticker("^IXIC").history(period="1y")

if len(nasdaq_hist) >= 200:
    nasdaq_ma200 = nasdaq_hist["Close"].rolling(200).mean().iloc[-1]
    nasdaq_now = nasdaq_hist["Close"].iloc[-1]

    if nasdaq_now < nasdaq_ma200:
        score -= 10
        risk_flag = True
        add_line("⚠ NASDAQが200日線割れ（市場全体弱気）")

# ===== VIX急騰強化 =====
vix_hist = yf.Ticker("^VIX").history(period="5d")

if len(vix_hist) >= 2:
    vix_now = vix_hist["Close"].iloc[-1]
    vix_prev = vix_hist["Close"].iloc[-2]

    vix_change = ((vix_now - vix_prev) / vix_prev) * 100

    if vix_change > 10:
        score -= 10
        risk_flag = True
        add_line("⚠ VIX急騰（恐怖拡大）")


# ===== ニュース取得 =====

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def get_ai_news():
    url = f"https://newsapi.org/v2/everything?q=AI+semiconductor&language=en&sortBy=publishedAt&pageSize=5&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    headlines = [a["title"] for a in articles]
    return headlines

news = get_ai_news()

add_line("")
add_line("■ AI関連最新ニュース")

for n in news:
    add_line(f"- {n}")

# ===== メール送信 =====

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

gmail_user = os.getenv("GMAIL_ADDRESS")
gmail_password = os.getenv("GMAIL_APP_PASSWORD")

if risk_flag:
    subject = "⚠ AI市場警戒アラート"
else:
    subject = "Daily AI Stock Report"

html = f"""
<html>
<body style="font-family:Arial;">
<h2>📊 AI市場プロレポート</h2>
<p><b>日付:</b> {datetime.date.today()}</p>
<p><b>市場温度:</b> {score} {temp}</p>

<hr>
<pre>
{report}
</pre>

{"<h3 style='color:red;'>⚠ 崩れモード発動</h3>" if risk_flag else ""}

</body>
</html>
"""

msg = MIMEMultipart("alternative")
msg["Subject"] = subject
msg["From"] = gmail_user
msg["To"] = gmail_user

msg.attach(MIMEText(report, "plain"))
msg.attach(MIMEText(html, "html"))

try:
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(gmail_user, gmail_password)
        server.send_message(msg)
    print("Email sent successfully!")
except Exception as e:
    print("Email failed:", e)
