#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
암호화폐 자금 유입/유출 핵심 4지표를 최근 1년 기준 일별로 수집하여
CSV/차트/HTML을 생성하는 스크립트 (완전 무료 & API 키 불필요 버전).

지표 및 데이터 소스:
1) 스테이블코인 시가총액 & 전일 변화 (DeFiLlama)
2) 현물 ETF(BTC, ETH) 순유입/유출 (Farside)
3) BTC/ETH 실현 시가총액 (Realized Cap, CoinMetrics Community API)
4) BTC/ETH 선물 미결제약정 (Open Interest, Binance Futures openInterestHist)

추가 기능:
- 위 4개 지표로 “자금 유입/유출 점수(0~100)”를 계산
- docs/index.html 상단에 오늘 점수 및 지표별 요약 텍스트 삽입

※ 거래소 보유량(Exchange Reserves)은 유료/키 필요로 인해 제외.
"""

import sys
import time
import datetime as dt
from pathlib import Path
from typing import Optional, Dict, Any, List

import requests
import pandas as pd
import matplotlib.pyplot as plt


# ===========================================
# 공통 설정
# ===========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DOCS_DIR = BASE_DIR / "docs"          # docs 폴더
CHART_DIR = DOCS_DIR / "charts"       # docs/charts 폴더

DATA_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)
CHART_DIR.mkdir(exist_ok=True)

TODAY = dt.date.today()
ONE_YEAR_AGO = TODAY - dt.timedelta(days=365)

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "crypto-liquidity-monitor/1.0 (+github; no-browser)",
    }
)


def safe_get(url: str, params: Optional[Dict[str, Any]] = None,
             max_retries: int = 3, sleep_sec: int = 3) -> requests.Response:
    """단순 재시도 로직이 있는 GET 요청 헬퍼."""
    for attempt in range(1, max_retries + 1):
        try:
            resp = SESSION.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp
        except Exception as e:
            print(f"[WARN] GET 실패 {attempt}/{max_retries}: {url} - {e}", file=sys.stderr)
            if attempt == max_retries:
                raise
            time.sleep(sleep_sec)


# ===========================================
# 1. 스테이블코인 시가총액 & 유입 (DeFiLlama)
# ===========================================

def fetch_stablecoin_mcap_1y() -> pd.DataFrame:
    """
    DeFiLlama stablecoincharts API에서 전체 스테이블코인 시가총액 일별 데이터(1년)를 조회.

    API:
      GET https://stablecoins.llama.fi/stablecoincharts/all?stablecoin=1
    """
    url = "https://stablecoins.llama.fi/stablecoincharts/all"
    params = {"stablecoin": "1"}
    resp = safe_get(url, params=params)
    data = resp.json()

    rows: List[Dict[str, Any]] = []
    for row in data:
        ts = row.get("date")
        total = (row.get("totalCirculatingUSD") or {}).get("peggedUSD")
        if ts is None or total is None:
            continue

        # --- date 필드 타입에 따라 유연하게 처리 ---
        date: Optional[dt.date] = None

        # 1) 숫자형 (유닉스 타임스탬프)
        if isinstance(ts, (int, float)):
            date = dt.datetime.utcfromtimestamp(int(ts)).date()

        # 2) 문자열인 경우
        elif isinstance(ts, str):
            ts_str = ts.strip()
            # 2-1) 숫자 문자열이면 유닉스 타임으로 간주
            if ts_str.isdigit():
                date = dt.datetime.utcfromtimestamp(int(ts_str)).date()
            else:
                # 2-2) YYYY-MM-DD, YYYY-MM-DDTHH:MM:SS 형태 등 처리
                try:
                    # '2024-01-01T00:00:00Z' -> '2024-01-01'
                    ts_str_clean = ts_str.split("T")[0]
                    date = dt.datetime.fromisoformat(ts_str_clean).date()
                except Exception:
                    # 알 수 없는 형식이면 스킵
                    continue
        else:
            # 지원하지 않는 타입이면 스킵
            continue

        if date is None:
            continue

        if not (ONE_YEAR_AGO <= date <= TODAY):
            continue

        rows.append(
            {
                "date": date,
                "stablecoin_mcap_usd": float(total),
            }
        )

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    if df.empty:
        return df

    df["stablecoin_mcap_change_usd"] = df["stablecoin_mcap_usd"].diff()
    return df


# ===========================================
# 2. 현물 ETF 순유입/유출 (Farside)
# ===========================================

def _fetch_farside_etf_all(url: str, total_col_name: str) -> pd.DataFrame:
    """
    Farside 'All Data' 페이지에서 Date/Total 테이블을 읽어오는 공통 함수.
    예:
      - BTC: https://farside.co.uk/bitcoin-etf-flow-all-data/
      - ETH: https://farside.co.uk/ethereum-etf-flow-all-data/
    """
    resp = safe_get(url)
    html = resp.text
    tables = pd.read_html(html)

    target = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("date" in c for c in cols) and any("total" in c for c in cols):
            target = t
            break

    if target is None:
        raise RuntimeError(f"Farside 테이블에서 Date/Total을 찾을 수 없습니다: {url}")

    # 컬럼 매핑
    col_map = {}
    for c in target.columns:
        lc = str(c).lower()
        if "date" in lc:
            col_map[c] = "date"
        elif "total" in lc:
            col_map[c] = total_col_name
        else:
            col_map[c] = None

    target = target.rename(columns=col_map)
    target = target[["date", total_col_name]]

    target["date"] = pd.to_datetime(target["date"], errors="coerce").dt.date
    target[total_col_name] = (
        target[total_col_name]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("(", "-", regex=False)
        .str.replace(")", "", regex=False)
    )
    target[total_col_name] = pd.to_numeric(target[total_col_name], errors="coerce")

    target = target.dropna(subset=["date"])
    target = target[
        (target["date"] >= ONE_YEAR_AGO) & (target["date"] <= TODAY)
    ]

    return target.sort_values("date").reset_index(drop=True)


def fetch_btc_etf_flows_1y() -> pd.DataFrame:
    url = "https://farside.co.uk/bitcoin-etf-flow-all-data/"
    return _fetch_farside_etf_all(url, "btc_etf_net_flow_usd_mn")


def fetch_eth_etf_flows_1y() -> pd.DataFrame:
    url = "https://farside.co.uk/ethereum-etf-flow-all-data/"
    return _fetch_farside_etf_all(url, "eth_etf_net_flow_usd_mn")


# ===========================================
# 3. 실현 시가총액 (CoinMetrics Community API)
# ===========================================

def fetch_realized_cap_1y(assets=("btc", "eth")) -> pd.DataFrame:
    """
    CoinMetrics Community API로 BTC/ETH CapRealUSD(Realized Cap) 1년치 일별 조회.
    API root: https://community-api.coinmetrics.io/v4
    """
    url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    params = {
        "assets": ",".join(assets),
        "metrics": "CapRealUSD",
        "frequency": "1d",
        "start_time": ONE_YEAR_AGO.isoformat(),
        "end_time": TODAY.isoformat(),
        "page_size": 10000,
    }
    resp = safe_get(url, params=params)
    j = resp.json()
    data = j.get("data", [])

    rows = []
    for row in data:
        t = row.get("time")
        asset = row.get("asset")
        val = row.get("CapRealUSD")
        if t is None or asset is None or val is None:
            continue
        date = dt.datetime.fromisoformat(t.replace("Z", "+00:00")).date()
        if not (ONE_YEAR_AGO <= date <= TODAY):
            continue
        rows.append(
            {
                "date": date,
                "asset": asset.lower(),
                "realized_cap_usd": float(val),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df_pivot = df.pivot(index="date", columns="asset", values="realized_cap_usd").reset_index()

    # 컬럼명 정리
    rename = {a: f"{a}_realized_cap_usd" for a in assets if a in df_pivot.columns}
    df_pivot = df_pivot.rename(columns=rename)
    df_pivot = df_pivot[
        (df_pivot["date"] >= ONE_YEAR_AGO) & (df_pivot["date"] <= TODAY)
    ]
    return df_pivot.sort_values("date").reset_index(drop=True)


# ===========================================
# 4. 선물 미결제약정 (Binance Futures openInterestHist)
# ===========================================

def fetch_binance_oi(symbol: str, asset_label: str) -> pd.DataFrame:
    """
    Binance USDⓈ-M Futures Open Interest History (1d).
    주의: 공식 문서 기준 최근 30일 데이터만 제공됨.
    → 기존 CSV에 누적 저장하면서 1년치 빌드업하는 구조로 사용.
    """
    url = "https://fapi.binance.com/futures/data/openInterestHist"
    params = {
        "symbol": symbol,
        "period": "1d",
        "limit": 365,  # 실제로는 최근 30일 정도만 리턴될 수 있음
    }
    resp = safe_get(url, params=params)
    data = resp.json()

    rows = []
    for row in data:
        ts = row.get("timestamp")
        val = row.get("sumOpenInterestValue")
        if ts is None or val is None:
            continue
        date = dt.datetime.utcfromtimestamp(int(ts) / 1000).date()
        if not (ONE_YEAR_AGO <= date <= TODAY):
            continue
        rows.append(
            {
                "date": date,
                f"{asset_label}_oi_value_usd": float(val),
            }
        )

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def fetch_oi_with_history(csv_path: Path) -> pd.DataFrame:
    """
    open_interest.csv 기존 파일이 있다면 과거 데이터 유지,
    Binance에서 가져온 최근 데이터와 병합하여 1년 범위 내에서 누적 관리.
    """
    # 기존 데이터 로드
    if csv_path.exists():
        df_old = pd.read_csv(csv_path, parse_dates=["date"])
        df_old["date"] = df_old["date"].dt.date
    else:
        df_old = pd.DataFrame()

    # 새로운 데이터 가져오기
    df_btc = fetch_binance_oi("BTCUSDT", "btc")
    df_eth = fetch_binance_oi("ETHUSDT", "eth")

    df_new = pd.merge(df_btc, df_eth, on="date", how="outer")

    # 과거 + 새로운 데이터 merge
    if df_old.empty:
        df_all = df_new
    else:
        df_all = pd.merge(
            df_old, df_new, on="date", how="outer", suffixes=("_old", "")
        )

        # 중복 열 정리 (새 데이터가 있으면 새 데이터 우선)
        for col in ["btc_oi_value_usd", "eth_oi_value_usd"]:
            col_old = f"{col}_old"
            if col_old in df_all.columns:
                df_all[col] = df_all[col].combine_first(df_all[col_old])
                df_all = df_all.drop(columns=[col_old])

    # 1년 범위만 유지
    df_all = df_all[
        (df_all["date"] >= ONE_YEAR_AGO) & (df_all["date"] <= TODAY)
    ].sort_values("date").reset_index(drop=True)

    return df_all


# ===========================================
# 차트 생성
# ===========================================

def plot_stablecoin(df: pd.DataFrame, out_path: Path):
    if df.empty:
        return
    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    ax1.plot(df["date"], df["stablecoin_mcap_usd"])
    ax1.set_title("Stablecoin Total Market Cap (USD)")
    ax1.set_ylabel("Market Cap (USD)")
    ax1.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_etf_flows(df: pd.DataFrame, out_path: Path):
    if df.empty:
        return
    plt.figure(figsize=(10, 6))
    if "btc_etf_net_flow_usd_mn" in df.columns:
        plt.plot(df["date"], df["btc_etf_net_flow_usd_mn"], label="BTC ETF Net Flow (US$m)")
    if "eth_etf_net_flow_usd_mn" in df.columns:
        plt.plot(df["date"], df["eth_etf_net_flow_usd_mn"], label="ETH ETF Net Flow (US$m)")
    plt.title("Spot ETF Net Flows (Daily, US$m)")
    plt.ylabel("Net Flow (US$m)")
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_realized_cap(df: pd.DataFrame, out_path: Path):
    if df.empty:
        return
    plt.figure(figsize=(10, 6))
    if "btc_realized_cap_usd" in df.columns:
        plt.plot(df["date"], df["btc_realized_cap_usd"], label="BTC Realized Cap (USD)")
    if "eth_realized_cap_usd" in df.columns:
        plt.plot(df["date"], df["eth_realized_cap_usd"], label="ETH Realized Cap (USD)")
    plt.title("Realized Market Cap (BTC & ETH)")
    plt.ylabel("Realized Cap (USD)")
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_open_interest(df: pd.DataFrame, out_path: Path):
    if df.empty:
        return
    plt.figure(figsize=(10, 6))
    if "btc_oi_value_usd" in df.columns:
        plt.plot(df["date"], df["btc_oi_value_usd"], label="BTC Futures OI Value (USD)")
    if "eth_oi_value_usd" in df.columns:
        plt.plot(df["date"], df["eth_oi_value_usd"], label="ETH Futures OI Value (USD)")
    plt.title("Futures Open Interest (USD Value)")
    plt.ylabel("OI Value (USD)")
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ===========================================
# 0~100 스코어 계산 로직
# ===========================================

def _percentile_score(series: pd.Series, value: float) -> Optional[float]:
    """과거 1년 분포에서 value의 분위 점수(0~100)를 계산."""
    s = series.dropna()
    if s.empty or value is None or pd.isna(value):
        return None
    rank = (s <= value).sum() / len(s)
    score = float(rank * 100.0)
    if score < 0:
        score = 0.0
    if score > 100:
        score = 100.0
    return score


def compute_liquidity_score_and_summary(df_all: pd.DataFrame):
    """
    병합된 df_all을 기반으로
    - latest_date
    - total_score (0~100)
    - components: [{name, score, detail}, ...]
    - summary_main: 한 줄 요약
    을 반환.
    """
    if df_all.empty:
        return None, None, [], "데이터가 없어 점수를 계산할 수 없습니다."

    df = df_all.sort_values("date").copy()
    latest_row = df.iloc[-1]
    latest_date = latest_row["date"]
    if isinstance(latest_date, pd.Timestamp):
        latest_date_str = latest_date.date().isoformat()
    else:
        latest_date_str = str(latest_date)

    components: List[Dict[str, Any]] = []

    def add_component(name: str, series: pd.Series, value: float, unit_desc: str):
        score = _percentile_score(series, value)
        if score is None:
            return
        if score >= 80:
            quality = "매우 높은 수준 (상위 20%)"
        elif score >= 60:
            quality = "다소 높은 수준 (상위 40%)"
        elif score >= 40:
            quality = "중간 수준"
        elif score >= 20:
            quality = "다소 낮은 수준 (하위 40%)"
        else:
            quality = "매우 낮은 수준 (하위 20%)"
        if unit_desc:
            detail = f"{name}: {unit_desc}. 과거 1년 분포 기준 {score:.0f}점, {quality}입니다."
        else:
            detail = f"{name}: 과거 1년 분포 기준 {score:.0f}점, {quality}입니다."
        components.append(
            {
                "name": name,
                "score": round(score),
                "detail": detail,
            }
        )

    # 1) 스테이블코인 시총 변화
    if "stablecoin_mcap_change_usd" in df.columns:
        s = df["stablecoin_mcap_change_usd"]
        val = latest_row.get("stablecoin_mcap_change_usd")
        unit_desc = ""
        if val is not None and not pd.isna(val):
            unit_desc = f"일일 변화 {val / 1e9:.2f}B USD"
        add_component("스테이블코인 시가총액 일일 변화", s, val, unit_desc)

    # 2) ETF 순유입 (BTC+ETH 합)
    has_btc_etf = "btc_etf_net_flow_usd_mn" in df.columns
    has_eth_etf = "eth_etf_net_flow_usd_mn" in df.columns
    if has_btc_etf or has_eth_etf:
        etf_total = pd.Series(index=df.index, dtype=float)
        etf_total[:] = 0.0
        if has_btc_etf:
            etf_total = etf_total.add(df["btc_etf_net_flow_usd_mn"].fillna(0), fill_value=0)
        if has_eth_etf:
            etf_total = etf_total.add(df["eth_etf_net_flow_usd_mn"].fillna(0), fill_value=0)
        val_btc = latest_row.get("btc_etf_net_flow_usd_mn") if has_btc_etf else 0.0
        val_eth = latest_row.get("eth_etf_net_flow_usd_mn") if has_eth_etf else 0.0
        val_btc = 0.0 if val_btc is None or pd.isna(val_btc) else float(val_btc)
        val_eth = 0.0 if val_eth is None or pd.isna(val_eth) else float(val_eth)
        val_total = val_btc + val_eth
        unit_desc = f"BTC+ETH 합산 {val_total:.1f}M USD"
        add_component("현물 ETF 순유입(합산)", etf_total, val_total, unit_desc)

    # 3) 실현 시총 변화 (BTC+ETH 합)
    has_btc_real = "btc_realized_cap_usd" in df.columns
    has_eth_real = "eth_realized_cap_usd" in df.columns
    if has_btc_real or has_eth_real:
        real_total = pd.Series(index=df.index, dtype=float)
        real_total[:] = 0.0
        if has_btc_real:
            real_total = real_total.add(df["btc_realized_cap_usd"].fillna(0), fill_value=0)
        if has_eth_real:
            real_total = real_total.add(df["eth_realized_cap_usd"].fillna(0), fill_value=0)
        real_change = real_total.diff()
        val_real = real_change.iloc[-1]
        unit_desc = ""
        if val_real is not None and not pd.isna(val_real):
            unit_desc = f"일일 변화 {val_real / 1e9:.2f}B USD"
        add_component("BTC+ETH 실현 시가총액 일일 변화", real_change, val_real, unit_desc)

    # 4) 선물 OI (BTC+ETH 합)
    has_btc_oi = "btc_oi_value_usd" in df.columns
    has_eth_oi = "eth_oi_value_usd" in df.columns
    if has_btc_oi or has_eth_oi:
        oi_total = pd.Series(index=df.index, dtype=float)
        oi_total[:] = 0.0
        if has_btc_oi:
            oi_total = oi_total.add(df["btc_oi_value_usd"].fillna(0), fill_value=0)
        if has_eth_oi:
            oi_total = oi_total.add(df["eth_oi_value_usd"].fillna(0), fill_value=0)
        val_oi = oi_total.iloc[-1]
        unit_desc = ""
        if val_oi is not None and not pd.isna(val_oi):
            unit_desc = f"총 OI {val_oi / 1e9:.2f}B USD"
        add_component("BTC+ETH 선물 미결제약정(USD 기준)", oi_total, val_oi, unit_desc)

    if not components:
        return latest_date_str, None, [], "지표 데이터가 부족하여 점수를 계산할 수 없습니다."

    scores = [c["score"] for c in components if c.get("score") is not None]
    if not scores:
        return latest_date_str, None, components, "지표 데이터가 부족하여 점수를 계산할 수 없습니다."

    total_score = sum(scores) / len(scores)

    # 메인 요약 문구
    if total_score >= 80:
        summary_main = (
            "과거 1년 기준으로 자금 유입이 매우 강한 구간입니다. "
            "공격적인 리스크 온 환경으로 볼 수 있으나, 과열 신호와 레버리지 과도 여부도 함께 점검하는 것이 좋습니다."
        )
    elif total_score >= 60:
        summary_main = (
            "자금 유입이 우세한 강세 환경입니다. "
            "스윙/추세 매매에 유리한 국면으로, 리스크를 관리하면서 롱 포지션 비중 확대를 검토할 수 있는 구간입니다."
        )
    elif total_score >= 40:
        summary_main = (
            "전반적으로 중립에 가까운 구간입니다. "
            "지표가 뚜렷한 방향을 제시하지 않으므로, 개별 코인·섹터 이슈와 미시적인 수급 요인에 더 민감한 장일 수 있습니다."
        )
    elif total_score >= 20:
        summary_main = (
            "자금 유출이 다소 우세한 구간입니다. "
            "레버리지 축소와 부분 익절, 손절 라인 재점검 등 리스크 관리 비중을 높이는 것이 유리할 수 있습니다."
        )
    else:
        summary_main = (
            "과거 1년 기준으로 강한 자금 유출 구간에 해당합니다. "
            "방어적인 포지셔닝과 현금 비중 확대, 레버리지 축소가 필요한 구간으로 해석할 수 있습니다."
        )

    return latest_date_str, round(total_score), components, summary_main


# ===========================================
# HTML 생성 (docs/index.html)
# ===========================================

def generate_html(output_path: Path,
                  latest_date_str: Optional[str],
                  total_score: Optional[float],
                  components: List[Dict[str, Any]],
                  summary_main: str):
    """
    docs/index.html 생성: 상단에 점수/요약, 아래에 4개 지표 차트 embed.
    (이미지는 docs/charts/*.png 기준 상대 경로)
    """
    today_str = TODAY.isoformat()

    # 점수 카드 HTML
    if latest_date_str is None or total_score is None:
        score_block = f"""
        <section class="score-card">
          <h2>오늘의 자금 유입/유출 점수</h2>
          <p>최근 데이터가 부족하여 점수를 계산할 수 없습니다.</p>
          <p>{summary_main}</p>
        </section>
        """
    else:
        items_html = ""
        for c in components:
            items_html += (
                f"<li><strong>{c['name']}</strong>: {c['score']}점 – {c['detail']}</li>"
            )
        score_block = f"""
        <section class="score-card">
          <h2>오늘의 자금 유입/유출 점수: {total_score:.0f} / 100</h2>
          <p>기준일: {latest_date_str}</p>
          <p>{summary_main}</p>
          <ul>
            {items_html}
          </ul>
        </section>
        """

    sections = []

    def add_section(title: str, img_rel_path: str, description: str):
        sections.append(
            f"""
            <section>
              <h2>{title}</h2>
              <p>{description}</p>
              <img src="{img_rel_path}" alt="{title}" style="max-width:100%;height:auto;">
            </section>
            """
        )

    add_section(
        "1. Stablecoin Market Cap & Daily Change",
        "charts/stablecoin.png",
        "DeFiLlama 기준 전체 USD 스테이블코인 시가총액 일별 추이입니다.",
    )
    add_section(
        "2. Spot ETF Net Flows (BTC & ETH)",
        "charts/etf_flows.png",
        "Farside 기반 미국 BTC/ETH 현물 ETF 일별 순유입/유출 (US$m)입니다.",
    )
    add_section(
        "3. BTC & ETH Realized Market Cap",
        "charts/realized_cap.png",
        "CoinMetrics Community API 기준 BTC/ETH 실현 시가총액(Realized Cap)입니다.",
    )
    add_section(
        "4. BTC & ETH Futures Open Interest",
        "charts/open_interest.png",
        "Binance USDⓈ-M 선물 기준 BTC/ETH 미결제약정(OI, USD 가치)입니다.",
    )

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>Crypto Liquidity Dashboard (Safe Version A)</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 1100px; margin: 20px auto; padding: 0 16px; }}
    h1 {{ text-align: center; }}
    section {{ margin-bottom: 40px; }}
    img {{ border: 1px solid #ddd; }}
    .updated {{ text-align: right; color: #666; font-size: 0.9em; }}
    .score-card {{
      border: 1px solid #ddd;
      padding: 16px;
      border-radius: 8px;
      background-color: #f9f9f9;
    }}
    .score-card h2 {{ margin-top: 0; }}
  </style>
</head>
<body>
  <h1>암호화폐 자금 유입/유출 4대 지표 (1년 일봉)</h1>
  <div class="updated">데이터 기준일: {today_str}</div>
  {score_block}
  {''.join(sections)}
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    print(f"[INFO] HTML 생성: {output_path}")


# ===========================================
# 메인
# ===========================================

def main():
    print(f"[INFO] 기준일: {TODAY}, 최근 1년 시작일: {ONE_YEAR_AGO}")

    # 1. Stablecoin
    print("[STEP] 1/4 스테이블코인 시가총액 수집...")
    df_stable = fetch_stablecoin_mcap_1y()
    stable_csv = DATA_DIR / "stablecoin.csv"
    df_stable.to_csv(stable_csv, index=False)
    print(f"[OK] {stable_csv}")

    # 2. ETF Flows
    print("[STEP] 2/4 BTC/ETH 현물 ETF 순유입 수집...")
    df_btc_etf = fetch_btc_etf_flows_1y()
    df_eth_etf = fetch_eth_etf_flows_1y()
    df_etf = pd.merge(df_btc_etf, df_eth_etf, on="date", how="outer").sort_values("date")
    etf_csv = DATA_DIR / "etf_flows.csv"
    df_etf.to_csv(etf_csv, index=False)
    print(f"[OK] {etf_csv}")

    # 3. Realized Cap
    print("[STEP] 3/4 BTC/ETH Realized Cap 수집...")
    df_realized = fetch_realized_cap_1y(("btc", "eth"))
    realized_csv = DATA_DIR / "realized_cap.csv"
    df_realized.to_csv(realized_csv, index=False)
    print(f"[OK] {realized_csv}")

    # 4. Futures OI
    print("[STEP] 4/4 BTC/ETH 선물 OI 수집 & 누적...")
    oi_csv = DATA_DIR / "open_interest.csv"
    df_oi = fetch_oi_with_history(oi_csv)
    df_oi.to_csv(oi_csv, index=False)
    print(f"[OK] {oi_csv}")

    # 병합 CSV
    print("[STEP] 병합 CSV 생성...")
    df_all = df_stable.copy()
    for part in [df_etf, df_realized, df_oi]:
        if not part.empty:
            df_all = pd.merge(df_all, part, on="date", how="outer")
    df_all = df_all.sort_values("date").reset_index(drop=True)
    merged_csv = DATA_DIR / "metrics_1y_merged.csv"
    df_all.to_csv(merged_csv, index=False)
    print(f"[OK] {merged_csv}")

    # 차트 생성 (docs/charts)
    print("[STEP] 차트 생성...")
    plot_stablecoin(df_stable, CHART_DIR / "stablecoin.png")
    plot_etf_flows(df_etf, CHART_DIR / "etf_flows.png")
    plot_realized_cap(df_realized, CHART_DIR / "realized_cap.png")
    plot_open_interest(df_oi, CHART_DIR / "open_interest.png")

    # 스코어 & 요약 계산
    print("[STEP] 점수 및 요약 계산...")
    latest_date_str, total_score, components, summary_main = compute_liquidity_score_and_summary(df_all)

    # HTML 생성 (docs/index.html)
    print("[STEP] HTML 생성...")
    generate_html(DOCS_DIR / "index.html", latest_date_str, total_score, components, summary_main)

    print("[DONE] 모든 작업 완료.")


if __name__ == "__main__":
    main()
