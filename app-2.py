"""
📈 Stock Master Analyzer - Streamlit Web App v1.2
================================================
v1.1 → v1.2:
- stock.info 대신 fast_info + income_stmt + balance_sheet + cashflow 사용
  (Yahoo Finance 클라우드 차단 우회)
- 재무 데이터 직접 계산으로 정확도 향상
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

# ✅ curl_cffi 세션 (가격 데이터 차단 우회)
def get_yf_session():
    from curl_cffi import requests as curl_requests
    session = curl_requests.Session(impersonate="chrome120")
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

st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
    [data-testid="metric-container"] {
        background: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 10px;
        padding: 12px;
    }
    .main-title { font-size: 1.6rem; font-weight: 700; color: #4fc3f7; margin-bottom: 4px; }
    .sub-title  { font-size: 0.85rem; color: #90a4ae; margin-bottom: 20px; }
    .stButton > button { width: 100%; height: 3rem; font-size: 1rem; font-weight: 600; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 기본 종목
# ============================================================
DEFAULT_TICKERS = [
    "NVDA", "AVGO", "ALAB", "CRDO", "CLS", "ARM", "MRVL", "SNDK", "LITE",
    "TER", "SERV", "VRTX", "ALNY", "LLY", "RKLB", "AVAV", "HWM",
    "AXON", "CRWD", "NOW", "PLTR", "ORCL", "VRT", "ETN", "GEV", "CCJ",
    "000660.KS", "005930.KS", "007660.KS", "012450.KS", "034020.KS", "042700.KS", "079550.KS", "267260.KS", "307950.KS",
]

# ============================================================
# 재무 데이터 추출 (income_stmt / balance_sheet / cashflow)
# ============================================================
def safe_val(df, row_keywords, col_idx=0, default=0):
    """재무제표에서 키워드로 행을 찾아 값 반환"""
    if df is None or df.empty:
        return default
    for kw in row_keywords:
        matches = [i for i in df.index if kw.lower() in str(i).lower()]
        if matches:
            try:
                val = df.loc[matches[0]].iloc[col_idx]
                if pd.notna(val):
                    return float(val)
            except Exception:
                continue
    return default

def get_fundamentals(stock):
    """
    stock.info 대신 재무제표에서 직접 지표 계산
    → Yahoo Finance 클라우드 차단 우회
    """
    data = {}
    try:
        inc = stock.income_stmt       # 손익계산서
        bal = stock.balance_sheet     # 재무상태표
        cf  = stock.cashflow          # 현금흐름표
        fi  = stock.fast_info         # 기본 가격 정보

        # ── 기본 가격 ─────────────────────────────
        data['current_price'] = getattr(fi, 'last_price', 0) or 0
        data['market_cap']    = getattr(fi, 'market_cap', 0) or 0
        data['high_52w']      = getattr(fi, 'fifty_two_week_high', data['current_price']) or data['current_price']
        data['name']          = getattr(fi, 'quote_type', stock.ticker) or stock.ticker

        # ── 매출 & 성장률 ──────────────────────────
        rev_curr = safe_val(inc, ["Total Revenue", "Revenue"], col_idx=0)
        rev_prev = safe_val(inc, ["Total Revenue", "Revenue"], col_idx=1)
        data['revenue']     = rev_curr
        data['rev_growth']  = ((rev_curr - rev_prev) / abs(rev_prev)) if rev_prev and rev_prev != 0 else 0

        # ── 매출총이익률 ───────────────────────────
        gross = safe_val(inc, ["Gross Profit"])
        data['gross_margin'] = (gross / rev_curr) if rev_curr else 0

        # ── R&D 집중도 ─────────────────────────────
        rd = safe_val(inc, ["Research Development", "Research And Development", "R&D"])
        data['rd_intensity'] = abs(rd / rev_curr) if rev_curr else 0

        # ── ROE ────────────────────────────────────
        net_income = safe_val(inc, ["Net Income"])
        equity     = safe_val(bal, ["Stockholders Equity", "Total Equity", "Common Stock Equity"])
        data['roe'] = (net_income / abs(equity)) if equity and equity != 0 else 0

        # ── FCF 마진 ───────────────────────────────
        op_cf  = safe_val(cf, ["Operating Cash Flow", "Cash From Operations"])
        capex  = safe_val(cf, ["Capital Expenditure", "Purchase Of Property"])
        fcf    = op_cf + capex  # capex는 보통 음수로 기록됨
        data['fcf_margin'] = (fcf / rev_curr) if rev_curr else 0

        # ── 부채비율 ───────────────────────────────
        total_debt = safe_val(bal, ["Total Debt", "Long Term Debt"])
        data['debt_equity'] = (total_debt / abs(equity) * 100) if equity and equity != 0 else 0

        # ── 현금 ───────────────────────────────────
        cash_abs = safe_val(bal, ["Cash And Cash Equivalents", "Cash Cash Equivalents"])
        data['cash_abs']   = cash_abs
        data['cash_ratio'] = (cash_abs / data['market_cap']) if data['market_cap'] else 0

        # ── PEG (fast_info에서 PE 가져오기) ────────
        pe = getattr(fi, 'forward_pe', None) or getattr(fi, 'trailing_pe', None)
        if pe and data['rev_growth'] and data['rev_growth'] > 0:
            data['peg'] = pe / (data['rev_growth'] * 100)
        else:
            data['peg'] = 2.5

        # ── PSR ────────────────────────────────────
        data['psr'] = (data['market_cap'] / rev_curr) if rev_curr else 0

        # ── Insider / Beta (fast_info에 없으면 기본값) ──
        data['insider_pct'] = 0.05   # fast_info에 없어서 보수적 기본값
        data['beta']        = getattr(fi, 'beta', 1.0) or 1.0
        data['sector']      = "Unknown"
        data['short_name']  = stock.ticker

    except Exception as e:
        # 완전 실패 시 기본값
        data.setdefault('current_price', 0)
        data.setdefault('market_cap', 0)
        data.setdefault('high_52w', 0)
        data.setdefault('revenue', 0)
        data.setdefault('rev_growth', 0)
        data.setdefault('gross_margin', 0)
        data.setdefault('rd_intensity', 0)
        data.setdefault('roe', 0)
        data.setdefault('fcf_margin', 0)
        data.setdefault('debt_equity', 0)
        data.setdefault('cash_abs', 0)
        data.setdefault('cash_ratio', 0)
        data.setdefault('peg', 2.5)
        data.setdefault('psr', 0)
        data.setdefault('insider_pct', 0.05)
        data.setdefault('beta', 1.0)
        data.setdefault('sector', 'Unknown')
        data.setdefault('short_name', stock.ticker)

    return data

# ============================================================
# 분석 로직
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
    close = hist['Close']
    signals = {}
    if len(close) >= 200:
        ma50, ma200 = close.rolling(50).mean().iloc[-1], close.rolling(200).mean().iloc[-1]
        signals['ma_signal'] = "🟢 골든크로스" if ma50 > ma200 else "🔴 데드크로스"
    else:
        signals['ma_signal'] = "N/A"
    if len(hist) >= 60:
        vr = hist['Volume'].iloc[-5:].mean() / hist['Volume'].iloc[-60:].mean()
        signals['vol_signal'] = "📊 급증" if vr > 1.5 else "일반"
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
        session = get_yf_session()
        stock   = yf.Ticker(symbol, session=session)
        hist    = stock.history(period="1y")

        if hist.empty or len(hist) < 20:
            return None

        # ✅ 재무 데이터: info 대신 재무제표에서 직접 추출
        fd = get_fundamentals(stock)

        rev_growth   = fd['rev_growth']
        rd_intensity = fd['rd_intensity']
        gross_margin = fd['gross_margin']
        roe          = fd['roe']
        fcf_margin   = fd['fcf_margin']
        debt_equity  = fd['debt_equity']
        cash_abs     = fd['cash_abs']
        cash_ratio   = fd['cash_ratio']
        peg          = fd['peg']
        psr          = fd['psr']
        beta         = fd['beta']
        insider_pct  = fd['insider_pct']
        sector       = fd['sector']
        current_price = fd['current_price'] or float(hist['Close'].iloc[-1])
        high_52w      = fd['high_52w'] or float(hist['Close'].max())

        # Q Score
        q = 0
        if peg < 1.0:             q += 3
        elif peg < 1.5:           q += 2
        elif peg < 2.0:           q += 1
        if roe < 0:               q -= 2
        elif roe > 0.25:          q += 3
        elif roe > 0.15:          q += 2
        elif roe > 0.05:          q += 1
        if gross_margin > 0.6:    q += 2
        elif gross_margin > 0.4:  q += 1
        if fcf_margin > 0.20:     q += 2
        elif fcf_margin > 0.10:   q += 1
        if debt_equity < 30:      q += 1
        elif debt_equity > 200:   q -= 1

        # T Score
        t = 0
        if rev_growth > 0.40:     t += 4
        elif rev_growth > 0.20:   t += 3
        elif rev_growth > 0.10:   t += 2
        elif rev_growth > 0:      t += 1
        if rd_intensity > 0.10:   t += 3
        elif rd_intensity > 0.05: t += 2
        elif rd_intensity > 0.02: t += 1
        if cash_ratio > 0.10 or cash_abs > 500_000_000:   t += 2
        elif cash_ratio > 0.05 or cash_abs > 100_000_000: t += 1
        if insider_pct > 0.10:    t += 1

        close_prices = hist['Close']
        rsi          = get_rsi(close_prices).iloc[-1]
        true_mdd     = get_true_mdd(close_prices)
        momentum     = get_momentum_signals(hist)
        expensive    = (psr > SECTOR_PSR.get(sector, 10)) or (peg > 2.5)
        signal, mult = get_rsi_signal(rsi, expensive)

        if w_q > w_t and beta > 1.5:
            mult *= 0.9

        final = round(((q * w_q) + (t * w_t)) * mult, 2)

        return {
            "Ticker":          symbol,
            "Name":            fd['short_name'],
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
            "ROE(%)":          round(roe * 100, 1),
            "FCF Margin(%)":   round(fcf_margin * 100, 1),
            "Gross Margin(%)": round(gross_margin * 100, 1),
            "Rev Growth(%)":   round(rev_growth * 100, 1),
            "R&D(%)":          round(rd_intensity * 100, 1),
            "D/E":             round(debt_equity, 1),
            "Beta":            round(beta, 2),
            "MDD(%)":          true_mdd,
            "52W Drop(%)":     round(((high_52w - current_price) / high_52w) * 100, 1) if high_52w else 0,
            "현재가":           round(current_price, 2),
        }
    except Exception as e:
        return None

def run_analysis(tickers, macro_env):
    w_q, w_t = get_macro_weights(macro_env)
    results  = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(analyze_single, s, w_q, w_t): s for s in tickers}
        for f in as_completed(futures):
            r = f.result()
            if r:
                results.append(r)
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results).sort_values("Final Score", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", df.index + 1)
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
with st.sidebar:
    st.markdown("### ⚙️ 분석 설정")
    macro_env = st.selectbox(
        "매크로 환경",
        options=["stable", "bull", "bear"],
        format_func=lambda x: {"stable": "🟡 Stable", "bull": "🟢 Bull", "bear": "🔴 Bear"}[x],
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
    st.markdown("""
#### 💡 점수 가이드
- **Q Score** : 재무 건전성
- **T Score** : 성장 잠재력
- **Final**   : 가중합 × RSI배율
    """)

st.markdown('<div class="main-title">📈 Stock Master Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Q/T Score 기반 종목 스크리닝</div>', unsafe_allow_html=True)

if run_btn:
    raw     = ticker_input.replace(",", "\n").replace(" ", "\n")
    tickers = [t.strip().upper() for t in raw.splitlines() if t.strip()]

    if not tickers:
        st.error("종목을 입력해 주세요.")
        st.stop()

    w_q, w_t   = get_macro_weights(macro_env)
    env_label  = {"stable": "Stable", "bull": "Bull", "bear": "Bear"}[macro_env]

    with st.spinner(f"⏳ {len(tickers)}개 종목 분석 중... (1~2분 소요)"):
        df = run_analysis(tickers, macro_env)

    if df.empty:
        st.error("분석 결과가 없습니다. 종목 코드를 확인해 주세요.")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("분석 종목",   f"{len(df)}개")
    col2.metric("매크로 환경", env_label)
    col3.metric("Q 가중치",   f"{w_q*100:.0f}%")
    col4.metric("T 가중치",   f"{w_t*100:.0f}%")

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 TOP 10", "📊 전체 결과", "📐 Q vs T 분류", "📁 업종별"])

    with tab1:
        st.markdown("#### 🏆 TOP 10")
        st.dataframe(
            df.head(10)[["Rank","Ticker","Name","Q Score","T Score","Final Score","분류","Signal","RSI","현재가"]],
            use_container_width=True, hide_index=True
        )
        cands = df[df["분류"] == "🚀 텐배거 후보"].head(5)
        if not cands.empty:
            st.markdown("#### 🚀 텐배거 후보 TOP 5")
            st.dataframe(
                cands[["Ticker","Name","Q Score","T Score","Final Score","Rev Growth(%)","R&D(%)","현재가"]],
                use_container_width=True, hide_index=True
            )

    with tab2:
        st.markdown("#### 📊 전체 결과")
        col_f1, col_f2 = st.columns(2)
        with col_f1: min_q = st.slider("Q Score 최소값", -5, 10, 0)
        with col_f2: min_t = st.slider("T Score 최소값",  0, 11, 0)
        filtered = df[(df["Q Score"] >= min_q) & (df["T Score"] >= min_t)]
        st.dataframe(
            filtered[["Rank","Ticker","Name","Q Score","T Score","Final Score","분류","RSI","MDD(%)","현재가"]],
            use_container_width=True, hide_index=True
        )
        # Excel 다운로드
        excel_buf = pd.ExcelWriter("/tmp/result.xlsx", engine="openpyxl")
        df.to_excel(excel_buf, index=False, sheet_name="전체분석")
        df.head(10).to_excel(excel_buf, index=False, sheet_name="TOP10")
        excel_buf.close()
        with open("/tmp/result.xlsx", "rb") as f:
            st.download_button(
                "📥 Excel 다운로드", f.read(),
                file_name=f"Stock_Master_{datetime.now().strftime('%Y%m%d')}_{macro_env}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    with tab3:
        st.markdown("#### 📐 Q vs T 사분면 분류")
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
        st.dataframe(
            df.groupby("Sector")[["Final Score","Q Score","T Score"]].agg(["mean","count"]).round(2),
            use_container_width=True
        )

else:
    st.info("👈 왼쪽 사이드바에서 설정 후 **🚀 분석 실행** 버튼을 누르세요.")
    st.markdown("""
#### 사용 방법
1. **매크로 환경** 선택
2. **종목 목록** 확인 및 수정  
3. **분석 실행** 클릭
4. 결과 확인 및 Excel 다운로드

| 분류 | 의미 |
|---|---|
| ✅ 우량+성장 | Q·T 모두 상위 |
| 🔵 우량주 | Q 상위, T 하위 |
| 🚀 텐배거 후보 | T 상위, Q 하위 |
| ⚠️ 요주의 | Q·T 모두 하위 |
    """)
