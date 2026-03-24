"""
국민연금 보유 종목 모니터 v1.0
- DART Open API 기반 대량보유 공시 조회
- 종목별 지분율 추이 시각화
"""

import streamlit as st
import pandas as pd
import requests
import zipfile
import io
import xml.etree.ElementTree as ET
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

# ─── 페이지 설정 ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="국민연금 지분 모니터",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
    
    .main-header {
        background: linear-gradient(135deg, #1a3a5c 0%, #0d6efd 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; font-weight: 700; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.9rem; }
    
    .metric-card {
        background: white;
        border: 1px solid #e8ecf0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .metric-label { font-size: 0.8rem; color: #6c757d; margin-bottom: 0.4rem; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #1a3a5c; }
    .metric-sub   { font-size: 0.75rem; color: #868e96; margin-top: 0.2rem; }
    
    .info-box {
        background: #f0f7ff;
        border-left: 4px solid #0d6efd;
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        font-size: 0.88rem;
        color: #1a3a5c;
        margin-bottom: 1rem;
    }
    
    .dart-link {
        display: inline-block;
        background: #0d6efd;
        color: white !important;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        font-size: 0.8rem;
        text-decoration: none !important;
        font-weight: 600;
    }
    .dart-link:hover { background: #0b5ed7; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
    }
    
    div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── DART API 함수들 ─────────────────────────────────────────────────────────
DART_BASE = "https://opendart.fss.or.kr/api"


def get_dart_api_key():
    """API 키를 세션 또는 secrets에서 가져옴"""
    return st.session_state.get("dart_api_key", "")


@st.cache_data(ttl=86400, show_spinner=False)
def load_corp_codes(api_key: str) -> pd.DataFrame:
    """DART 전체 기업 코드 목록 (1일 캐시)"""
    resp = requests.get(
        f"{DART_BASE}/corpCode.xml",
        params={"crtfc_key": api_key},
        timeout=30,
    )
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        with z.open("CORPCODE.xml") as f:
            root = ET.parse(f).getroot()
            rows = [
                {
                    "corp_code": item.findtext("corp_code"),
                    "corp_name": item.findtext("corp_name"),
                    "stock_code": item.findtext("stock_code"),
                }
                for item in root.findall("list")
            ]
    df = pd.DataFrame(rows)
    # 상장 기업만 (stock_code 있는 것)
    df = df[df["stock_code"].notna() & (df["stock_code"].str.strip() != "")]
    return df.reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nps_filings(api_key: str, bgn_de: str, end_de: str) -> pd.DataFrame:
    """국민연금 대량보유 공시 목록 조회 (D001: 주식등의대량보유상황보고서)"""
    all_items = []
    page = 1

    while True:
        resp = requests.get(
            f"{DART_BASE}/list.json",
            params={
                "crtfc_key": api_key,
                "pblntf_detail_ty": "D001",
                "bgn_de": bgn_de,
                "end_de": end_de,
                "page_no": page,
                "page_count": 100,
            },
            timeout=15,
        )
        data = resp.json()

        if data.get("status") not in ("000", "013"):  # 013 = 데이터 없음
            st.error(f"DART API 오류: {data.get('message', 'Unknown error')} (status: {data.get('status')})")
            break

        items = data.get("list", [])
        nps_items = [i for i in items if "국민연금" in i.get("flr_nm", "")]
        all_items.extend(nps_items)

        total = int(data.get("total_count", 0))
        if total == 0 or page * 100 >= total:
            break
        page += 1
        time.sleep(0.15)

    if not all_items:
        return pd.DataFrame()

    df = pd.DataFrame(all_items)
    df["rcept_dt"] = pd.to_datetime(df["rcept_dt"], errors="coerce")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_major_stock_history(api_key: str, corp_code: str) -> pd.DataFrame:
    """특정 종목의 주요주주(대량보유) 이력 조회"""
    resp = requests.get(
        f"{DART_BASE}/majorstock.json",
        params={"crtfc_key": api_key, "corp_code": corp_code},
        timeout=15,
    )
    data = resp.json()

    if data.get("status") not in ("000",):
        return pd.DataFrame()

    items = data.get("list", [])
    df = pd.DataFrame(items)

    if df.empty:
        return df

    # 국민연금만 필터
    if "stkholdr_nm" in df.columns:
        df = df[df["stkholdr_nm"].str.contains("국민연금", na=False)].copy()

    return df


def parse_rate(val) -> float:
    """'10.23%' 또는 '10.23' → float"""
    try:
        return float(str(val).replace("%", "").replace(",", "").strip())
    except Exception:
        return None


def build_dart_link(rcept_no: str) -> str:
    return f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}"


# ─── 사이드바 ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 설정")

    api_key_input = st.text_input(
        "DART API Key",
        type="password",
        placeholder="발급받은 키를 입력하세요",
        help="https://opendart.fss.or.kr 에서 무료 발급 (승인 후 사용 가능)",
    )
    if api_key_input:
        st.session_state["dart_api_key"] = api_key_input

    st.markdown(
        '<a class="dart-link" href="https://opendart.fss.or.kr/uat/uia/easyLogin.do" target="_blank">🔑 DART API Key 발급</a>',
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### 📅 조회 기간")
    today = datetime.today().date()
    col_s, col_e = st.columns(2)
    with col_s:
        bgn = st.date_input("시작", value=today - timedelta(days=365), label_visibility="collapsed")
    with col_e:
        end = st.date_input("종료", value=today, label_visibility="collapsed")

    st.caption(f"📌 {bgn.strftime('%Y.%m.%d')} ~ {end.strftime('%Y.%m.%d')}")

    st.divider()
    st.markdown("""
    <div style='font-size:0.78rem; color:#6c757d; line-height:1.6;'>
    <b>데이터 주기</b><br>
    • <b>공시 현황</b>: DART 실시간<br>
    • <b>지분율 추이</b>: 공시 기준 (5% 이상만)<br>
    • API 응답은 1시간 캐시됨
    </div>
    """, unsafe_allow_html=True)

# ─── 헤더 ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🏦 국민연금 보유 종목 모니터</h1>
  <p>DART 전자공시 기반 · 대량보유(5% 이상) 공시 자동 추적</p>
</div>
""", unsafe_allow_html=True)

api_key = get_dart_api_key()

if not api_key:
    st.markdown("""
    <div class="info-box">
    👈 <b>시작하려면</b> 사이드바에서 DART API Key를 입력하세요.<br>
    API Key는 <a href="https://opendart.fss.or.kr" target="_blank">opendart.fss.or.kr</a>에서 <b>무료</b>로 발급받을 수 있습니다.
    (신청 후 1~2일 소요)
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── 탭 ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📋 최근 공시 현황", "📈 종목별 지분율 추이"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 : 최근 공시 현황
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    with st.spinner("DART에서 공시 데이터 불러오는 중..."):
        df_filings = fetch_nps_filings(
            api_key,
            bgn.strftime("%Y%m%d"),
            end.strftime("%Y%m%d"),
        )

    if df_filings.empty:
        st.info("⚠️ 조회 기간 내 국민연금 대량보유 공시가 없습니다.")
    else:
        # 요약 지표
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">총 공시 건수</div>
              <div class="metric-value">{len(df_filings):,}</div>
              <div class="metric-sub">건</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">관련 종목 수</div>
              <div class="metric-value">{df_filings['corp_name'].nunique():,}</div>
              <div class="metric-sub">개 기업</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            latest = df_filings["rcept_dt"].max()
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">최근 공시일</div>
              <div class="metric-value" style="font-size:1.3rem;">{latest.strftime('%m/%d')}</div>
              <div class="metric-sub">{latest.strftime('%Y년')}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            days = (end - bgn).days
            per_month = len(df_filings) / max(days / 30, 1)
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">월평균 공시</div>
              <div class="metric-value">{per_month:.1f}</div>
              <div class="metric-sub">건/월</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 차트 + 테이블 레이아웃
        col_chart, col_table = st.columns([1, 1])

        with col_chart:
            # 월별 공시 건수 추이
            monthly = (
                df_filings.set_index("rcept_dt")
                .resample("ME")
                .size()
                .reset_index(name="count")
            )
            monthly["월"] = monthly["rcept_dt"].dt.strftime("%Y-%m")

            fig_bar = go.Figure(
                go.Bar(
                    x=monthly["월"],
                    y=monthly["count"],
                    marker_color="#0d6efd",
                    marker_line_width=0,
                )
            )
            fig_bar.update_layout(
                title="월별 공시 건수",
                xaxis_title="",
                yaxis_title="건수",
                plot_bgcolor="white",
                paper_bgcolor="white",
                margin=dict(t=40, b=20, l=20, r=10),
                height=260,
            )
            fig_bar.update_xaxes(tickangle=-45, tickfont_size=10)
            st.plotly_chart(fig_bar, use_container_width=True)

            # 공시 다발 종목 Top 15
            top_corps = df_filings["corp_name"].value_counts().head(15)
            fig_h = go.Figure(
                go.Bar(
                    x=top_corps.values,
                    y=top_corps.index,
                    orientation="h",
                    marker_color=px.colors.sequential.Blues_r[: len(top_corps)],
                )
            )
            fig_h.update_layout(
                title="공시 다발 종목 Top 15",
                xaxis_title="공시 건수",
                yaxis=dict(autorange="reversed"),
                plot_bgcolor="white",
                paper_bgcolor="white",
                margin=dict(t=40, b=20, l=10, r=10),
                height=360,
            )
            st.plotly_chart(fig_h, use_container_width=True)

        with col_table:
            st.markdown("##### 공시 목록")
            # 필터
            all_corps = ["전체"] + sorted(df_filings["corp_name"].unique().tolist())
            selected_corp = st.selectbox("종목 필터", all_corps, key="filing_corp_filter")

            filtered = df_filings.copy()
            if selected_corp != "전체":
                filtered = filtered[filtered["corp_name"] == selected_corp]

            display = filtered[["rcept_dt", "corp_name", "report_nm", "rcept_no"]].copy()
            display["rcept_dt"] = display["rcept_dt"].dt.strftime("%Y-%m-%d")
            display["DART"] = display["rcept_no"].apply(
                lambda x: f'<a href="{build_dart_link(x)}" target="_blank">🔗 보기</a>'
            )
            display = display.rename(columns={
                "rcept_dt": "공시일",
                "corp_name": "회사명",
                "report_nm": "보고서명",
            })
            display = display.drop(columns=["rcept_no"])
            display = display.sort_values("공시일", ascending=False).reset_index(drop=True)

            # HTML 테이블 렌더링 (링크 포함)
            html_table = display.to_html(escape=False, index=False)
            styled = f"""
            <div style="overflow-y:auto; max-height:560px; border-radius:8px; border:1px solid #e8ecf0;">
            <style>
                table {{ width:100%; border-collapse:collapse; font-size:0.82rem; }}
                th {{ background:#1a3a5c; color:white; padding:8px 10px; text-align:left; position:sticky; top:0; }}
                td {{ padding:7px 10px; border-bottom:1px solid #f0f0f0; }}
                tr:hover td {{ background:#f5f9ff; }}
            </style>
            {html_table}
            </div>
            """
            st.markdown(styled, unsafe_allow_html=True)

            # CSV 다운로드
            csv = display.drop(columns=["DART"]).to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                "⬇️ CSV 다운로드",
                data=csv,
                file_name=f"nps_filings_{bgn}_{end}.csv",
                mime="text/csv",
            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 : 종목별 지분율 추이
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="info-box">
    💡 DART 대량보유 공시 이력을 기반으로 <b>국민연금의 종목별 지분율 변화</b>를 시각화합니다.<br>
    &nbsp;&nbsp;&nbsp;&nbsp;지분율 5% 미만이거나 보유 이력이 없는 종목은 데이터가 나타나지 않습니다.
    </div>
    """, unsafe_allow_html=True)

    search_term = st.text_input(
        "🔍 종목명 검색",
        placeholder="삼성전자, LG에너지솔루션, SK하이닉스 ...",
        key="stock_search",
    )

    if search_term:
        with st.spinner("기업 목록 불러오는 중..."):
            corp_df = load_corp_codes(api_key)
            matches = corp_df[corp_df["corp_name"].str.contains(search_term, na=False)]

        if matches.empty:
            st.warning(f"'{search_term}'에 해당하는 상장 기업이 없습니다.")
        else:
            selected_name = st.selectbox(
                "종목 선택",
                options=matches["corp_name"].tolist(),
                key="stock_select",
            )
            corp_code = matches.loc[matches["corp_name"] == selected_name, "corp_code"].iloc[0]
            stock_code = matches.loc[matches["corp_name"] == selected_name, "stock_code"].iloc[0]

            with st.spinner(f"'{selected_name}' 지분율 이력 조회 중..."):
                hist_df = fetch_major_stock_history(api_key, corp_code)

            if hist_df.empty:
                st.info(
                    f"**{selected_name}**에 대한 국민연금 대량보유 이력이 없습니다.\n\n"
                    "→ 현재 지분율이 5% 미만이거나 과거에도 5% 이상 보유한 적이 없는 종목입니다."
                )
            else:
                # 지분율 파싱
                rate_col = None
                for candidate in ["sp_stock_lmp_rate", "stock_rate"]:
                    if candidate in hist_df.columns:
                        rate_col = candidate
                        break

                date_col = None
                for candidate in ["report_de", "rcept_dt"]:
                    if candidate in hist_df.columns:
                        date_col = candidate
                        break

                if rate_col and date_col:
                    hist_df["지분율"] = hist_df[rate_col].apply(parse_rate)
                    hist_df["날짜"] = pd.to_datetime(hist_df[date_col], errors="coerce")
                    hist_df = hist_df.dropna(subset=["지분율", "날짜"]).sort_values("날짜")

                    # 주요 지표
                    current_rate = hist_df["지분율"].iloc[-1]
                    max_rate = hist_df["지분율"].max()
                    min_rate = hist_df["지분율"].min()
                    chg = hist_df["지분율"].iloc[-1] - hist_df["지분율"].iloc[-2] if len(hist_df) > 1 else 0

                    c1, c2, c3, c4 = st.columns(4)
                    arrow = "▲" if chg > 0 else ("▼" if chg < 0 else "─")
                    color = "#198754" if chg > 0 else ("#dc3545" if chg < 0 else "#6c757d")
                    with c1:
                        st.markdown(f"""
                        <div class="metric-card">
                          <div class="metric-label">최근 지분율</div>
                          <div class="metric-value">{current_rate:.2f}%</div>
                          <div class="metric-sub" style="color:{color}">{arrow} {abs(chg):.2f}%p 전회比</div>
                        </div>""", unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"""
                        <div class="metric-card">
                          <div class="metric-label">최고 지분율</div>
                          <div class="metric-value">{max_rate:.2f}%</div>
                          <div class="metric-sub">{hist_df.loc[hist_df['지분율'].idxmax(), '날짜'].strftime('%Y.%m')}</div>
                        </div>""", unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"""
                        <div class="metric-card">
                          <div class="metric-label">최저 지분율</div>
                          <div class="metric-value">{min_rate:.2f}%</div>
                          <div class="metric-sub">{hist_df.loc[hist_df['지분율'].idxmin(), '날짜'].strftime('%Y.%m')}</div>
                        </div>""", unsafe_allow_html=True)
                    with c4:
                        st.markdown(f"""
                        <div class="metric-card">
                          <div class="metric-label">공시 횟수</div>
                          <div class="metric-value">{len(hist_df)}</div>
                          <div class="metric-sub">회 (누적)</div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # 지분율 추이 차트
                    fig = go.Figure()

                    # 영역 채우기
                    fig.add_trace(go.Scatter(
                        x=hist_df["날짜"],
                        y=hist_df["지분율"],
                        mode="lines+markers",
                        name="국민연금 지분율",
                        line=dict(color="#0d6efd", width=2.5),
                        marker=dict(size=7, color="#0d6efd", line=dict(width=1.5, color="white")),
                        fill="tozeroy",
                        fillcolor="rgba(13, 110, 253, 0.08)",
                        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>지분율: %{y:.2f}%<extra></extra>",
                    ))

                    # 5% 기준선
                    fig.add_hline(
                        y=5,
                        line_dash="dot",
                        line_color="rgba(220, 53, 69, 0.7)",
                        line_width=1.5,
                        annotation_text="의무공시 기준 5%",
                        annotation_position="top right",
                        annotation_font_size=11,
                        annotation_font_color="#dc3545",
                    )

                    fig.update_layout(
                        title=f"<b>{selected_name}</b> — 국민연금 지분율 추이",
                        xaxis_title="",
                        yaxis_title="지분율 (%)",
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        hovermode="x unified",
                        height=400,
                        margin=dict(t=50, b=20, l=20, r=20),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        xaxis=dict(gridcolor="#f0f0f0"),
                        yaxis=dict(gridcolor="#f0f0f0", zeroline=False),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # 상세 테이블
                    with st.expander("📄 공시 이력 상세 보기"):
                        table_cols = {"날짜": "보고일", "지분율": "지분율(%)"}
                        extra_map = {
                            "sp_stock_lmp_cnt": "보통주 보유수량",
                            "dp_stock_lmp_cnt": "의결권 있는 주식수",
                            "stkholdr_relate": "보유 목적",
                            "rcept_no": "DART 링크",
                        }
                        show_df = hist_df.copy()
                        rename = {"날짜": "보고일", "지분율": "지분율(%)"}
                        for col, label in extra_map.items():
                            if col in show_df.columns:
                                rename[col] = label

                        show_df = show_df.rename(columns=rename)
                        show_df["보고일"] = show_df["보고일"].dt.strftime("%Y-%m-%d")

                        if "DART 링크" in show_df.columns:
                            show_df["DART 링크"] = show_df["DART 링크"].apply(
                                lambda x: f'<a href="{build_dart_link(str(x))}" target="_blank">🔗 보기</a>'
                            )

                        cols_to_show = [c for c in ["보고일", "지분율(%)", "보통주 보유수량", "보유 목적", "DART 링크"] if c in show_df.columns]
                        show_df = show_df[cols_to_show].sort_values("보고일", ascending=False)

                        html_t = show_df.to_html(escape=False, index=False)
                        st.markdown(f"""
                        <div style="overflow-x:auto; border-radius:8px; border:1px solid #e8ecf0;">
                        <style>
                            table {{ width:100%; border-collapse:collapse; font-size:0.83rem; }}
                            th {{ background:#1a3a5c; color:white; padding:8px 12px; text-align:left; }}
                            td {{ padding:7px 12px; border-bottom:1px solid #f0f0f0; }}
                            tr:hover td {{ background:#f5f9ff; }}
                        </style>
                        {html_t}
                        </div>
                        """, unsafe_allow_html=True)

                else:
                    st.warning("지분율 데이터 형식을 파싱할 수 없습니다. DART API 응답을 확인해주세요.")
                    st.dataframe(hist_df, use_container_width=True)

# ─── 푸터 ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; font-size:0.78rem; color:#adb5bd; padding:0.5rem 0 1rem;">
    데이터 출처: DART 전자공시시스템 (금융감독원) &nbsp;|&nbsp; 
    5% 이상 보유 종목만 공시 의무 &nbsp;|&nbsp; 
    매매 내역은 지연 공시됨
</div>
""", unsafe_allow_html=True)
