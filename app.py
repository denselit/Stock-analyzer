"""
📈 Stock Master Analyzer - Streamlit Web App
스마트폰/PC 브라우저에서 실행 가능
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

# ✅ Yahoo Finance 봇 차단 우회: 브라우저처럼 보이는 헤더 설정
def get_yf_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    })
    return session

# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title="Stock Master Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS 스타일 (모바일 최적화)
# ============================================================
st.markdown("""
<style>
    /* 전체 폰트 */
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

    /* 메트릭 카드 */
    [data-testid="metric-container"] {
        background: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 10px;
        padding: 12px;
    }

    /* 상단 타이틀 */
    .main-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #4fc3f7;
        margin-bottom: 4px;
    }
    .sub-title {
        font-size: 0.85rem;
        color: #90a4ae;
        margin-bottom: 20px;
    }

    /* 점수 배지 */
    .score-high  { color: #66bb6a; font-weight: 700; }
    .score-mid   { color: #ffa726; font-weight: 700; }
    .score-low   { color: #ef5350; font-weight: 700; }

    /* 모바일: 사이드바 버튼 크게 */
    .stButton > button {
        width: 100%;
        height: 3rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 기본 종목 리스트
# ============================================================
DEFAULT_TICKERS = [
    "NVDA", "AVGO", "ALAB", "CRDO", "CLS", "ARM", "MRVL", "VRT", "ETN",
    "TER", "SERV", "VRTX", "ALNY", "LLY", "RKLB", "AVAV", "AXON", "CRWD",
    "NOW", "PLTR", "GEV", "ORCL",
    "000660.KS", "005930.KS", "012450.KS", "273060.KS", "267260.KS", "278280.KS"
]

# ============================================================
# 분석 로직 (stock_analyzer_v2.py 와 동일)
# ============================================================
def get_macro_weights(market_status):
    if market_status == "bull":   return 0.3, 0.7
    elif market_status == "bear": return 0.7, 0.3
    else:                         return 0.5, 0.5

def get_rsi(prices, window=14):
    delta = prices.diff()
    gain  = delta.where(delta > 0, 0).rolling(window).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

def get_true_mdd(prices):
    rolling_max = prices.cummax()
    return round(((prices - rolling_max) / rolling_max).min() * 100, 1)

def get_momentum_signals(hist):
    close   = hist['Close']
    signals = {}
    if len(close) >= 200:
        ma50  = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1]
        signals['ma_signal'] = "🟢 골든크로스" if ma50 > ma200 else "🔴 데드크로스"
    else:
        signals['ma_signal'] = "N/A"
    if len(hist) >= 60:
        vol_ratio = hist['Volume'].iloc[-5:].mean() / hist['Volume'].iloc[-60:].mean()
        signals['vol_signal'] = "📊 급증" if vol_ratio > 1.5 else "일반"
    else:
        signals['vol_signal'] = "N/A"
    return signals

SECTOR_PSR = {
    "Technology": 15, "Healthcare": 10, "Industrials": 5,
    "Energy": 3, "Consumer Cyclical": 3, "Financial Services": 4,
    "Communication Services": 8, "Basic Materials": 3,
    "Real Estate": 6, "Utilities": 4, "Consumer Defensive": 4,
}

def get_rsi_signal(rsi, expensive):
    if rsi < 30:
        return ("Deeply Oversold BUT Expensive ⚠️", 0.85) if expensive else ("Deeply Oversold 🔥", 1.3)
    elif rsi < 45:
        return ("Mild Oversold ⚠️", 0.9) if expensive else ("Mild Oversold 📉", 1.1)
    elif rsi > 80:   return "Extremely Overbought 🚫", 0.6
    elif rsi > 70:   return "Overbought ⚠️", 0.75
    else:            return "Neutral ➡️", 1.0

def analyze_single(symbol, w_q, w_t):
    try:
        session = get_yf_session()  # ✅ 세션 적용
        stock = yf.Ticker(symbol, session=session)
        info  = stock.info
        hist  = stock.history(period="1y")
        if hist.empty or len(hist) < 20:
            return None

        rev        = info.get("totalRevenue")
        market_cap = info.get("marketCap", 1)
        sector     = info.get("sector", "Unknown")

        fcf_margin   = 0.0 if not rev or rev == 0 else (info.get("freeCashflow", 0) or 0) / rev
        gross_margin = info.get("grossMargins", 0) or 0
        roe          = info.get("returnOnEquity", 0) or 0
        rev_growth   = info.get("revenueGrowth", 0) or 0
        psr          = info.get("priceToSalesTrailing12Months", 0) or 0
        forward_pe   = info.get("forwardPE")
        beta         = info.get("beta", 1.0) or 1.0
        debt_equity  = info.get("debtToEquity", 0) or 0
        ev_ebitda    = info.get("enterpriseToEbitda", 0) or 0
        insider_pct  = info.get("heldPercentInsiders", 0) or 0
        cash_abs     = info.get("totalCash", 0) or 0
        cash_ratio   = cash_abs / (market_cap or 1)

        peg = info.get("pegRatio")
        eg  = info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth") or rev_growth
        if peg is None or peg <= 0:
            peg = (forward_pe / (eg * 100)) if (forward_pe and eg and eg > 0) else 2.5

        rd_intensity = 0.0 if not rev or rev == 0 else (info.get("researchDevelopment", 0) or 0) / rev

        # Q Score
        q = 0
        if peg < 1.0:            q += 3
        elif peg < 1.5:          q += 2
        elif peg < 2.0:          q += 1
        if roe < 0:              q -= 2
        elif roe > 0.25:         q += 3
        elif roe > 0.15:         q += 2
        elif roe > 0.05:         q += 1
        if gross_margin > 0.6:   q += 2
        elif gross_margin > 0.4: q += 1
        if fcf_margin > 0.20:    q += 2
        elif fcf_margin > 0.10:  q += 1
        if debt_equity < 30:     q += 1
        elif debt_equity > 200:  q -= 1

        # T Score
        t = 0
        if rev_growth > 0.40:    t += 4
        elif rev_growth > 0.20:  t += 3
        elif rev_growth > 0.10:  t += 2
        elif rev_growth > 0:     t += 1
        if rd_intensity > 0.10:  t += 3
        elif rd_intensity > 0.05: t += 2
        elif rd_intensity > 0.02: t += 1
        if cash_ratio > 0.10 or cash_abs > 500_000_000:   t += 2
        elif cash_ratio > 0.05 or cash_abs > 100_000_000: t += 1
        if insider_pct > 0.10:   t += 1

        close_prices  = hist['Close']
        current_price = info.get("currentPrice") or float(close_prices.iloc[-1])
        high_52w      = info.get("fiftyTwoWeekHigh", current_price)
        rsi           = get_rsi(close_prices).iloc[-1]
        true_mdd      = get_true_mdd(close_prices)
        momentum      = get_momentum_signals(hist)
        expensive     = (psr > SECTOR_PSR.get(sector, 10)) or (peg > 2.5)
        signal, mult  = get_rsi_signal(rsi, expensive)

        if w_q > w_t and beta > 1.5:
            mult *= 0.9

        final = round(((q * w_q) + (t * w_t)) * mult, 2)

        # 사분면 분류 (나중에 중앙값 기준으로 재분류)
        return {
            "Ticker":          symbol,
            "Name":            info.get("shortName", symbol),
            "Sector":          sector,
            "Final Score":     final,
            "Q Score":         q,
            "T Score":         t,
            "Signal":          signal,
            "MA":              momentum['ma_signal'],
            "Vol":             momentum['vol_signal'],
            "RSI":             round(rsi, 1),
            "PEG":             round(peg, 2),
            "PSR":             round(psr, 1),
            "EV/EBITDA":       round(ev_ebitda, 1),
            "ROE(%)":          round(roe * 100, 1),
            "FCF Margin(%)":   round(fcf_margin * 100, 1),
            "Gross Margin(%)": round(gross_margin * 100, 1),
            "Rev Growth(%)":   round(rev_growth * 100, 1),
            "R&D(%)":          round(rd_intensity * 100, 1),
            "D/E":             round(debt_equity, 1),
            "Insider(%)":      round(insider_pct * 100, 1),
            "Beta":            round(beta, 2),
            "MDD(%)":          true_mdd,
            "52W Drop(%)":     round(((high_52w - current_price) / high_52w) * 100, 1),
            "현재가":           round(current_price, 2),
        }
    except:
        return None

def run_analysis(tickers, macro_env):
    w_q, w_t = get_macro_weights(macro_env)
    results  = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(analyze_single, s, w_q, w_t): s for s in tickers}
        for f in as_completed(futures):
            r = f.result()
            if r:
                results.append(r)
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results).sort_values("Final Score", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", df.index + 1)

    # 사분면 분류
    q_med, t_med = df["Q Score"].median(), df["T Score"].median()
    def classify(row):
        if row["Q Score"] >= q_med and row["T Score"] >= t_med: return "✅ 우량+성장"
        elif row["Q Score"] >= q_med:                            return "🔵 우량주"
        elif row["T Score"] >= t_med:                            return "🚀 텐배거 후보"
        else:                                                    return "⚠️ 요주의"
    df["분류"] = df.apply(classify, axis=1)
    return df

# ============================================================
# Streamlit UI
# ============================================================

# 사이드바
with st.sidebar:
    st.markdown("### ⚙️ 분석 설정")

    macro_env = st.selectbox(
        "매크로 환경",
        options=["stable", "bull", "bear"],
        format_func=lambda x: {"stable": "🟡 Stable (균형)", "bull": "🟢 Bull (성장 중심)", "bear": "🔴 Bear (방어 중심)"}[x],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 📋 종목 관리")
    ticker_input = st.text_area(
        "분석 종목 (줄바꿈 또는 쉼표로 구분)",
        value="\n".join(DEFAULT_TICKERS),
        height=250,
    )

    st.markdown("---")
    run_btn = st.button("🚀 분석 실행", type="primary")

    st.markdown("---")
    st.markdown("#### 💡 점수 가이드")
    st.markdown("""
- **Q Score** : 재무 건전성 (우량주)
- **T Score** : 성장 잠재력 (텐배거)
- **Final**   : Q×가중치 + T×가중치 × RSI배율
    """)

# 메인 화면
st.markdown('<div class="main-title">📈 Stock Master Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Q/T Score 기반 종목 스크리닝 시스템</div>', unsafe_allow_html=True)

if run_btn:
    # 종목 파싱
    raw     = ticker_input.replace(",", "\n").replace(" ", "\n")
    tickers = [t.strip().upper() for t in raw.splitlines() if t.strip()]

    if not tickers:
        st.error("종목을 입력해 주세요.")
        st.stop()

    # 분석 실행
    w_q, w_t = get_macro_weights(macro_env)
    env_label = {"stable": "Stable", "bull": "Bull", "bear": "Bear"}[macro_env]

    with st.spinner(f"⏳ {len(tickers)}개 종목 분석 중... (30~60초 소요)"):
        df = run_analysis(tickers, macro_env)

    if df.empty:
        st.error("분석 결과가 없습니다. 종목 코드를 확인해 주세요.")
        st.stop()

    # ── 상단 요약 지표 ────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("분석 종목",   f"{len(df)}개")
    col2.metric("매크로 환경", env_label)
    col3.metric("Q 가중치",   f"{w_q*100:.0f}%")
    col4.metric("T 가중치",   f"{w_t*100:.0f}%")

    st.markdown("---")

    # ── 탭 구성 ──────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 TOP 10", "📊 전체 결과", "📐 Q vs T 분류", "📁 업종별"])

    with tab1:
        st.markdown("#### 🏆 TOP 10 추천 종목")
        top10 = df.head(10)[["Rank","Ticker","Name","Sector","Q Score","T Score","Final Score","분류","Signal","RSI","현재가"]]
        st.dataframe(top10, use_container_width=True, hide_index=True)

        # 텐배거 후보 별도 강조
        cands = df[df["분류"] == "🚀 텐배거 후보"].head(5)
        if not cands.empty:
            st.markdown("#### 🚀 텐배거 후보 TOP 5")
            st.dataframe(
                cands[["Ticker","Name","Q Score","T Score","Final Score","Rev Growth(%)","R&D(%)","현재가"]],
                use_container_width=True, hide_index=True
            )

    with tab2:
        st.markdown("#### 📊 전체 분석 결과")

        # 필터
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            min_q = st.slider("Q Score 최소값", -5, 10, 0)
        with col_f2:
            min_t = st.slider("T Score 최소값", 0, 11, 0)

        filtered = df[(df["Q Score"] >= min_q) & (df["T Score"] >= min_t)]
        st.dataframe(
            filtered[["Rank","Ticker","Name","Sector","Q Score","T Score","Final Score","분류","RSI","MDD(%)","52W Drop(%)","현재가"]],
            use_container_width=True, hide_index=True
        )
        st.caption(f"필터 적용 후 {len(filtered)}개 종목")

        # Excel 다운로드
        excel_buf = pd.ExcelWriter("/tmp/result.xlsx", engine="openpyxl")
        df.to_excel(excel_buf, index=False, sheet_name="전체분석")
        df.head(10).to_excel(excel_buf, index=False, sheet_name="TOP10")
        excel_buf.close()
        with open("/tmp/result.xlsx", "rb") as f:
            today = datetime.now().strftime("%Y%m%d")
            st.download_button(
                label="📥 Excel 다운로드",
                data=f.read(),
                file_name=f"Stock_Master_{today}_{macro_env}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    with tab3:
        st.markdown("#### 📐 Q vs T 사분면 분류")
        q_med, t_med = df["Q Score"].median(), df["T Score"].median()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("✅ 우량+성장", len(df[df["분류"]=="✅ 우량+성장"]))
        c2.metric("🔵 우량주",   len(df[df["분류"]=="🔵 우량주"]))
        c3.metric("🚀 텐배거",   len(df[df["분류"]=="🚀 텐배거 후보"]))
        c4.metric("⚠️ 요주의",   len(df[df["분류"]=="⚠️ 요주의"]))

        for label in ["✅ 우량+성장", "🚀 텐배거 후보", "🔵 우량주", "⚠️ 요주의"]:
            sub = df[df["분류"] == label]
            if not sub.empty:
                with st.expander(f"{label} ({len(sub)}개)"):
                    st.dataframe(
                        sub[["Ticker","Name","Q Score","T Score","Final Score","Signal","현재가"]],
                        use_container_width=True, hide_index=True
                    )

    with tab4:
        st.markdown("#### 📁 업종별 평균 점수")
        sector_summary = (
            df.groupby("Sector")[["Final Score","Q Score","T Score"]]
            .agg(["mean","count"])
            .round(2)
        )
        st.dataframe(sector_summary, use_container_width=True)

else:
    # 실행 전 안내
    st.info("👈 왼쪽 사이드바에서 설정 후 **🚀 분석 실행** 버튼을 누르세요.")
    st.markdown("""
    #### 사용 방법
    1. **매크로 환경** 선택 (Bull/Bear/Stable)
    2. **종목 목록** 확인 및 수정
    3. **분석 실행** 버튼 클릭
    4. 결과 확인 및 Excel 다운로드

    #### 점수 기준
    | 분류 | Q Score | T Score |
    |---|---|---|
    | ✅ 우량+성장 | 중앙값 이상 | 중앙값 이상 |
    | 🔵 우량주 | 중앙값 이상 | 중앙값 미만 |
    | 🚀 텐배거 후보 | 중앙값 미만 | 중앙값 이상 |
    | ⚠️ 요주의 | 중앙값 미만 | 중앙값 미만 |
    """)
