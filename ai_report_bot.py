import yfinance as yf
import datetime
import pandas as pd
import requests
import os
import smtplib
import json
import logging
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from deep_translator import GoogleTranslator

# ===== ログ設定 =====

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ===== パス設定（スクリプト基準の絶対パス） =====

BASE_DIR = Path(__file__).resolve().parent
ALLOCATION_FILE = BASE_DIR / "last_allocation.json"

# ===== 銘柄設定 =====

tickers = {
    "NVDA": "NVIDIA",
    "MU": "Micron",
    "AMD": "AMD",
    "AVGO": "Broadcom",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet",
    "^GSPC": "S&P500",
    "^IXIC": "NASDAQ",
    "SOXX": "SOXX",
}

# VIXは別管理（二重取得・二重判定を防止）
VIX_TICKER = "^VIX"

negative_keywords = [
    "lawsuit", "crash", "downgrade", "fraud", "investigation",
    "recall", "bankruptcy", "layoff", "decline",
    "missed earnings", "regulation", "antitrust",
]

semiconductor_stocks = ["NVDA", "AMD", "AVGO", "MU"]
ai_large_stocks = ["MSFT", "AMZN", "GOOGL"]

# ===== ユーティリティ =====


def translate_to_japanese(text: str) -> str:
    """Google翻訳で日本語に変換（deep-translator使用）"""
    try:
        return GoogleTranslator(source="en", target="ja").translate(text)
    except Exception as e:
        logger.warning(f"翻訳失敗: {e}")
        return text


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI（相対力指数）を計算"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def clamp(value: float, low: float = 0, high: float = 100) -> float:
    """値を範囲内にクランプ"""
    return max(low, min(high, value))


# ===== テクニカルデータ取得 =====


def fetch_technical_data(ticker_symbol: str) -> dict | None:
    """1銘柄のテクニカル指標を取得。データ不足時はNone。"""
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period="1y")
    except Exception as e:
        logger.error(f"{ticker_symbol} データ取得失敗: {e}")
        return None

    if len(hist) < 200:
        logger.warning(f"{ticker_symbol}: データ不足 ({len(hist)}行 < 200)")
        return None

    close = hist["Close"]
    current = close.iloc[-1]
    prev = close.iloc[-2]

    return {
        "current": round(current, 2),
        "change": round((current / prev - 1) * 100, 2),
        "ma50": round(close.rolling(50).mean().iloc[-1], 2),
        "ma200": round(close.rolling(200).mean().iloc[-1], 2),
        "rsi": round(calculate_rsi(close).iloc[-1], 2),
        "volume_ratio": round(
            hist["Volume"].iloc[-1] / hist["Volume"].rolling(20).mean().iloc[-1], 2
        ),
    }


# ===== スコアリング =====


def calculate_score(report_data: dict, vix_data: dict | None) -> tuple[int, bool]:
    """
    市場温度スコアとリスクフラグを算出。
    VIXは独立して1回だけ評価する。
    """
    score = 50
    risk_flag = False

    # --- 個別銘柄スコアリング ---
    for ticker, d in report_data.items():
        change = d["change"]

        # 半導体主力
        if ticker in ("NVDA", "AMD"):
            if change > 2:
                score += 5
            if d["current"] > d["ma50"]:
                score += 5
            if d["rsi"] < 30:
                score += 3
            if d["rsi"] > 70:
                score -= 3

        # S&P500
        if ticker == "^GSPC":
            if change > 1:
                score += 5
            if change < -1:
                score -= 5

    # --- 長期トレンド判定 ---
    soxx = report_data.get("SOXX")
    if soxx and soxx["current"] < soxx["ma200"]:
        score -= 15
        risk_flag = True

    nasdaq = report_data.get("^IXIC")
    if nasdaq and nasdaq["current"] < nasdaq["ma200"]:
        score -= 10
        risk_flag = True

    # --- VIX判定（1回のみ） ---
    if vix_data:
        vix_change = vix_data["change"]
        if vix_change > 10:
            score -= 15
            risk_flag = True
        elif vix_change > 5:
            score -= 10
            risk_flag = True

    return int(clamp(score)), risk_flag


# ===== 温度ラベル =====


def get_temperature_label(score: int) -> str:
    if score >= 80:
        return "🔥 加速局面"
    elif score >= 65:
        return "🟢 強気"
    elif score >= 45:
        return "⚖ 中立"
    elif score >= 30:
        return "🟡 減速"
    else:
        return "❄ 崩れ"


# ===== ポジション配分 =====


def get_allocation(score: int, risk_flag: bool) -> dict:
    if risk_flag:
        return {"cash": 70, "semiconductor": 5, "ai_large": 5, "defensive": 20}

    if score >= 80:
        return {"cash": 10, "semiconductor": 40, "ai_large": 40, "defensive": 10}
    elif score >= 65:
        return {"cash": 20, "semiconductor": 35, "ai_large": 35, "defensive": 10}
    elif score >= 45:
        return {"cash": 35, "semiconductor": 25, "ai_large": 25, "defensive": 15}
    elif score >= 30:
        return {"cash": 50, "semiconductor": 15, "ai_large": 15, "defensive": 20}
    else:
        return {"cash": 70, "semiconductor": 5, "ai_large": 5, "defensive": 20}


# ===== 銘柄別配分 =====


def distribute(group: list[str], total_weight: float, report_data: dict) -> dict:
    """グループ内銘柄をトレンド・RSIに基づいて配分。"""

    strong, normal, reduced = [], [], []

    for ticker in group:
        d = report_data.get(ticker)
        if d is None:
            continue

        trend_ok = d["ma50"] > d["ma200"]
        rsi = d["rsi"]

        if d["change"] > 2 and trend_ok and 40 <= rsi <= 65:
            strong.append(ticker)
        elif rsi > 70:
            reduced.append(ticker)
        else:
            normal.append(ticker)

    # 全て空なら均等配分にフォールバック
    active = [t for t in group if t in report_data]
    if not active:
        return {}
    if not strong and not normal and not reduced:
        per = round(total_weight / len(active), 1)
        return {t: per for t in active}

    # 重み配分
    if strong:
        weights = {"strong": 0.6, "normal": 0.3, "reduced": 0.1}
    else:
        weights = {"strong": 0.0, "normal": 0.8, "reduced": 0.2}

    result = {}
    for bucket, tickers_list in [
        ("strong", strong),
        ("normal", normal),
        ("reduced", reduced),
    ]:
        if tickers_list:
            per = round(total_weight * weights[bucket] / len(tickers_list), 1)
            for t in tickers_list:
                result[t] = per

    return result


def build_detailed_allocation(
    allocation: dict, report_data: dict, rebalance: bool
) -> dict:
    """銘柄別の詳細配分を構築。リバランス日以外は前回の配分を維持。"""

    if not rebalance:
        try:
            with open(ALLOCATION_FILE, "r") as f:
                saved = json.load(f)
                if saved:
                    logger.info("前回の配分を読み込みました")
                    return saved
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"保存済み配分の読み込み失敗（新規計算します）: {e}")

    # 新規計算
    detailed = {}
    detailed.update(
        distribute(semiconductor_stocks, allocation["semiconductor"], report_data)
    )
    detailed.update(
        distribute(ai_large_stocks, allocation["ai_large"], report_data)
    )

    # 保存
    try:
        with open(ALLOCATION_FILE, "w") as f:
            json.dump(detailed, f, indent=2)
        logger.info("配分を保存しました")
    except Exception as e:
        logger.error(f"配分の保存失敗: {e}")

    return detailed


# ===== NVDAブースト =====


def apply_nvda_boost(
    detailed: dict, score: int, risk_flag: bool, report_data: dict
) -> dict:
    if "NVDA" not in detailed:
        return detailed

    boost = 0
    if score >= 65:
        boost += 5
    if risk_flag:
        boost -= 5
    if "NVDA" in report_data and report_data["NVDA"]["rsi"] < 35:
        boost += 3

    detailed["NVDA"] = max(0, detailed["NVDA"] + boost)
    return detailed


# ===== VIXボラティリティ調整 + 正規化 =====


def apply_vix_adjustment(
    detailed: dict, vix_data: dict | None, allocation: dict
) -> dict:
    vix_change = vix_data["change"] if vix_data else 0

    vol_factor = 1.0
    if vix_change > 5:
        vol_factor = 0.8
    elif vix_change < -3:
        vol_factor = 1.1

    for t in detailed:
        detailed[t] = round(detailed[t] * vol_factor, 1)

    # 正規化
    total_weight = sum(detailed.values())
    target_total = allocation["semiconductor"] + allocation["ai_large"]

    if total_weight > 0:
        scale = target_total / total_weight
        for t in detailed:
            detailed[t] = round(detailed[t] * scale, 1)

    return detailed


# ===== ニュース取得 =====


def get_ai_news() -> list[str]:
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        logger.warning("NEWS_API_KEY が設定されていません")
        return []

    url = (
        "https://newsapi.org/v2/top-headlines?"
        "q=AI+semiconductor+NVIDIA+AMD&"
        "language=en&pageSize=5&"
        f"apiKey={api_key}"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        return [a["title"] for a in data.get("articles", []) if a.get("title")]
    except Exception as e:
        logger.error(f"ニュース取得失敗: {e}")
        return []


def analyze_news(news: list[str]) -> tuple[list[str], int]:
    """ニュースを翻訳し、ネガティブキーワードをカウント。"""
    translated = []
    negative_count = 0

    for title in news:
        lower = title.lower()
        for word in negative_keywords:
            if word in lower:
                negative_count += 1
        translated.append(translate_to_japanese(title))

    return translated, negative_count


# ===== リバランス判定 =====


def is_rebalance_day() -> bool:
    """金曜日（市場終了後）にリバランス判定"""
    return datetime.date.today().weekday() == 4  # 4 = 金曜日


# ===== レポート生成 =====


def generate_report(
    score: int,
    temp: str,
    risk_flag: bool,
    report_data: dict,
    vix_data: dict | None,
    translated_news: list[str],
    negative_count: int,
    allocation: dict,
    detailed_allocation: dict,
    rebalance: bool,
) -> str:
    lines = []

    if rebalance:
        lines.append("🔁 今週はリバランス実行日です")
    else:
        lines.append("📌 今週は配分維持日です")

    lines.append("")
    lines.append("===== AI市場プロレポート =====")
    lines.append(f"日付: {datetime.date.today()}")
    lines.append(f"市場温度: {score} {temp}")
    lines.append("")

    # 銘柄テクニカル
    all_tickers = {**tickers, VIX_TICKER: "VIX"}
    for ticker, name in all_tickers.items():
        d = report_data.get(ticker) if ticker != VIX_TICKER else vix_data
        if d is None:
            continue
        lines.append(f"{name} ({ticker})")
        lines.append(f"  前日比: {d['change']}%")
        lines.append(f"  MA50: {d['ma50']}")
        lines.append(f"  MA200: {d['ma200']}")
        lines.append(f"  RSI: {d['rsi']}")
        lines.append(f"  出来高倍率: {d['volume_ratio']}倍")
        lines.append("")

    # ニュース
    lines.append("■ AI関連最新ニュース")
    if translated_news:
        for n in translated_news:
            lines.append(f"- {n}")
    else:
        lines.append("- ニュースの取得なし")

    if negative_count >= 2:
        lines.append("")
        lines.append("⚠ ネガティブニュース増加（市場警戒）")

    if risk_flag:
        lines.append("")
        lines.append("⚠ 崩れモード発動")

    # 押し目候補
    lines.append("")
    lines.append("■ 押し目候補")
    dip_found = False
    for ticker, d in report_data.items():
        if d["change"] < -4 and d["rsi"] < 35 and d["ma50"] > d["ma200"]:
            lines.append(f"・{ticker} 押し目候補")
            dip_found = True
    if not dip_found:
        lines.append("・該当なし")

    # ポジション配分
    lines.append("")
    lines.append("■ 推奨ポジション配分")
    lines.append(f"現金: {allocation['cash']}%")
    lines.append(f"半導体: {allocation['semiconductor']}%")
    lines.append(f"AI大型株: {allocation['ai_large']}%")
    lines.append(f"ディフェンシブ: {allocation['defensive']}%")

    lines.append("")
    lines.append("■ 銘柄別詳細配分")
    for t, w in detailed_allocation.items():
        lines.append(f"{t}: {w}%")

    return "\n".join(lines)


# ===== メール送信 =====


def send_email(report: str, score: int, temp: str, risk_flag: bool):
    gmail_user = os.getenv("GMAIL_ADDRESS")
    gmail_password = os.getenv("GMAIL_APP_PASSWORD")

    if not gmail_user or not gmail_password:
        logger.error("Gmail認証情報が設定されていません")
        return

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

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_user, gmail_password)
            server.send_message(msg)
        logger.info("メール送信成功")
    except Exception as e:
        logger.error(f"メール送信失敗: {e}")


# ===== メイン処理 =====


def main():
    logger.info("===== AI市場監視Bot 起動 =====")

    # 1. テクニカルデータ取得
    report_data = {}
    for ticker in tickers:
        data = fetch_technical_data(ticker)
        if data:
            report_data[ticker] = data

    # VIXは独立取得
    vix_data = fetch_technical_data(VIX_TICKER)

    # 2. スコアリング
    score, risk_flag = calculate_score(report_data, vix_data)

    # 3. ニュース分析
    news = get_ai_news()
    translated_news, negative_count = analyze_news(news)

    if negative_count >= 2:
        score = int(clamp(score - 10))
        risk_flag = True
    elif negative_count == 1:
        score = int(clamp(score - 5))

    # 4. 温度判定
    temp = get_temperature_label(score)

    # 5. ポジション配分
    allocation = get_allocation(score, risk_flag)

    # 6. リバランス判定
    rebalance = is_rebalance_day()

    # 7. 銘柄別詳細配分
    detailed_allocation = build_detailed_allocation(
        allocation, report_data, rebalance
    )

    # 8. NVDAブースト
    detailed_allocation = apply_nvda_boost(
        detailed_allocation, score, risk_flag, report_data
    )

    # 9. VIX調整 + 正規化
    detailed_allocation = apply_vix_adjustment(
        detailed_allocation, vix_data, allocation
    )

    # 10. レポート生成
    report = generate_report(
        score, temp, risk_flag, report_data, vix_data,
        translated_news, negative_count,
        allocation, detailed_allocation, rebalance,
    )

    logger.info("\n" + report)

    # 11. メール送信
    send_email(report, score, temp, risk_flag)

    logger.info("===== 処理完了 =====")


if __name__ == "__main__":
    main()
