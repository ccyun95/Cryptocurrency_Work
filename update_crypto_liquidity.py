#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Crypto Liquidity Monitor - Safe & Stable Local Version

목표:
- 4개 핵심 지표를 1년 일봉으로 수집/갱신하여 CSV/차트/HTML 생성
- 0~100 '자금 유입/유출 점수' + 요약 텍스트를 docs/index.html 상단에 자동 삽입
- 외부 소스 403/451 등으로 막혀도 파이프라인이 절대 중단되지 않게 설계
- 새 수집이 실패하면 기존 CSV를 유지하여 대시보드가 끊기지 않게 함

지표:
1) 스테이블코인 시가총액 & 일일 변화 (DeFiLlama)
2) 현물 ETF 순유입/유출 (BTC/ETH, Farside)
3) 실현 시가총액 Realized Cap (BTC/ETH, CoinMetrics Community)
4) 선물 미결제약정 OI (BTC/ETH, Binance Futures)

주의:
- 일부 사이트는 GitHub Actions IP를 차단하는 경우가 많음
- 본 스크립트는 '내 PC' 운영을 최우선으로 안정화

필수 폴더:
- data/
- docs/
- docs/charts/
"""

import sys
import time
import datetime as dt
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import requests
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 기본 경로/환경
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = BASE_DIR / "docs"
CHART_DIR = DOCS_DIR / "charts"

DATA_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)
CHART_DIR.mkdir(exist_ok=True)

TODAY: dt.date = dt.date.today()
ONE_YEAR_AGO: dt.date = TODAY - dt.timedelta(days=365)

# 운영 로그 레벨 유사 동작: 너무 시끄러운 로그 줄이기
VERBOSE = False  # True면 디버그 수준의 더 많은 메시지 출력

# 실패/스킵 상황을 마지막에 요약 출력하기 위한 기록
WARNINGS: List[str] = []
INFOS: List[str] = []


def log_info(msg: str):
    INFOS.append(msg)
    if VERBOSE:
        print(f"[INFO] {msg}")


def log_warn(msg: str):
    WARNINGS.append(msg)
    print(f"[WARN] {msg}", file=sys.stderr)


# =========================================================
# HTTP 세션
# =========================================================

SESSION = requests.Session()
SESSION.headers.update(
    {
        # 로컬 PC에서도 과도한 봇 느낌 줄이기
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.8,ko;q=0.6",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
)


def safe_get(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    max_retries: int = 2,
    sleep_sec: int = 2,
) -> requests.Response:
    """
    재시도는 하되, 경고 스팸을 줄이기 위해
    '최종 실패 시 한 줄만 경고' 찍는 방식.
    """
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = SESSION.get(url, params=params, timeout=20)
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_exc = e
            if attempt < max_retries:
                time.sleep(sleep_sec)

    # 여기까지 왔으면 최종 실패
    raise last_exc  # 호출자에서 상태 코드별 처리


# =========================================================
# 공통 유틸: 실패 시 '이전 CSV 유지'
# =========================================================

def load_existing_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        return df
    except Exception as e:
        log_warn(f"기존 CSV 로드 실패: {csv_path.name} - {e}")
        return pd.DataFrame()


def save_or_keep(old_df: pd.DataFrame, new_df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    """
    new_df가 비었거나 너무 짧으면(old보다 악화) 기존 CSV를 유지.
    """
    if new_df is None or new_df.empty:
        if not old_df.empty:
            log_warn(f"{csv_path.name}: 새 데이터 비어 있음 → 기존 CSV 유지")
            return old_df
        else:
            log_warn(f"{csv_path.name}: 새 데이터 비어 있음 & 기존도 없음")
            return pd.DataFrame()

    # 너무 짧은 데이터가 들어와 기존보다 퇴보하는 경우 방지
    if not old_df.empty and len(new_df) < max(10, int(len(old_df) * 0.3)):
        log_warn(f"{csv_path.name}: 새 데이터가 비정상적으로 짧음 → 기존 CSV 유지")
        return old_df

    try:
        new_df.to_csv(csv_path, index=False)
        log_info(f"CSV 저장 완료: {csv_path.name} ({len(new_df)} rows)")
    except Exception as e:
        log_warn(f"{csv_path.name}: CSV 저장 실패 → 기존 CSV 유지 - {e}")
        return old_df

    return new_df


# =========================================================
# 1) Stablecoin - DeFiLlama
# =========================================================

def fetch_stablecoin_mcap_1y() -> pd.DataFrame:
    url = "https://stablecoins.llama.fi/stablecoincharts/all"
    params = {"stablecoin": "1"}

    try:
        resp = safe_get(url, params=params)
        data = resp.json()
    except Exception as e:
        log_warn(f"DeFiLlama Stablecoin 수집 실패 → 스킵: {e}")
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    for row in data:
        ts = row.get("date")
        total = (row.get("totalCirculatingUSD") or {}).get("peggedUSD")
        if ts is None or total is None:
            continue

        # --- date 타입 유연 처리 ---
        date: Optional[dt.date] = None
        try:
            if isinstance(ts, (int, float)):
                date = dt.datetime.utcfromtimestamp(int(ts)).date()
            elif isinstance(ts, str):
                ts_str = ts.strip()
                if ts_str.isdigit():
                    date = dt.datetime.utcfromtimestamp(int(ts_str)).date()
                else:
                    ts_str_clean = ts_str.split("T")[0]
                    date = dt.datetime.fromisoformat(ts_str_clean).date()
        except Exception:
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


# =========================================================
# 2) ETF Flows - Farside
# =========================================================

def _fetch_farside_etf_all(url: str, total_col_name: str) -> pd.DataFrame:
    try:
        resp = safe_get(url)
    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status == 403:
            log_warn(f"Farside 403 → ETF 데이터 스킵: {url}")
            return pd.DataFrame()
        log_warn(f"Farside HTTP 에러({status}) → ETF 데이터 스킵: {url}")
        return pd.DataFrame()
    except Exception as e:
        log_warn(f"Farside 요청 실패 → ETF 데이터 스킵: {url} - {e}")
        return pd.DataFrame()

    try:
        tables = pd.read_html(resp.text)
    except Exception as e:
        log_warn(f"Farside HTML 파싱 실패 → ETF 데이터 스킵: {e}")
        return pd.DataFrame()

    target = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("date" in c for c in cols) and any("total" in c for c in cols):
            target = t
            break

    if target is None:
        log_warn(f"Farside 테이블(Date/Total) 미탐지 → ETF 스킵: {url}")
        return pd.DataFrame()

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
    return _fetch_farside_etf_all(
        "https://farside.co.uk/bitcoin-etf-flow-all-data/",
        "btc_etf_net_flow_usd_mn"
    )


def fetch_eth_etf_flows_1y() -> pd.DataFrame:
    return _fetch_farside_etf_all(
        "https://farside.co.uk/ethereum-etf-flow-all-data/",
        "eth_etf_net_flow_usd_mn"
    )


# =========================================================
# 3) Realized Cap - CoinMetrics Community
# =========================================================

def fetch_realized_cap_1y(assets=("btc", "eth")) -> pd.DataFrame:
    url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    params = {
        "assets": ",".join(assets),
        "metrics": "CapRealUSD",
        "frequency": "1d",
        "start_time": ONE_YEAR_AGO.isoformat(),
        "end_time": TODAY.isoformat(),
        "page_size": 10000,
    }

    try:
        resp = safe_get(url, params=params)
        j = resp.json()
    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status == 403:
            log_warn("CoinMetrics 403 → Realized Cap 스킵")
            return pd.DataFrame()
        log_warn(f"CoinMetrics HTTP 에러({status}) → Realized Cap 스킵")
        return pd.DataFrame()
    except Exception as e:
        log_warn(f"CoinMetrics 요청 실패 → Realized Cap 스킵: {e}")
        return pd.DataFrame()

    data = j.get("data", [])
    rows = []
    for row in data:
        t = row.get("time")
        asset = row.get("asset")
        val = row.get("CapRealUSD")
        if t is None or asset is None or val is None:
            continue
        try:
            date = dt.datetime.fromisoformat(t.replace("Z", "+00:00")).date()
        except Exception:
            continue
        if not (ONE_YEAR_AGO <= date <= TODAY):
            continue
        rows.append({"date": date, "asset": asset.lower(), "realized_cap_usd": float(val)})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df_pivot = df.pivot(index="date", columns="asset", values="realized_cap_usd").reset_index()

    rename = {a: f"{a}_realized_cap_usd" for a in assets if a in df_pivot.columns}
    df_pivot = df_pivot.rename(columns=rename)

    df_pivot = df_pivot[
        (df_pivot["date"] >= ONE_YEAR_AGO) & (df_pivot["date"] <= TODAY)
    ].sort_values("date").reset_index(drop=True)

    return df_pivot


# =========================================================
# 4) Open Interest - Binance Futures
# =========================================================

def fetch_binance_oi(symbol: str, asset_label: str) -> pd.DataFrame:
    url = "https://fapi.binance.com/futures/data/openInterestHist"
    params = {"symbol": symbol, "period": "1d", "limit": 365}

    try:
        resp = safe_get(url, params=params)
        data = resp.json()
    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status in (403, 451):
            log_warn(f"Binance OI {status} → OI 스킵 ({symbol})")
            return pd.DataFrame()
        log_warn(f"Binance OI HTTP 에러({status}) → OI 스킵 ({symbol})")
        return pd.DataFrame()
    except Exception as e:
        log_warn(f"Binance OI 요청 실패 → OI 스킵 ({symbol}): {e}")
        return pd.DataFrame()

    rows = []
    for row in data:
        ts = row.get("timestamp")
        val = row.get("sumOpenInterestValue")
        if ts is None or val is None:
            continue
        try:
            date = dt.datetime.utcfromtimestamp(int(ts) / 1000).date()
        except Exception:
            continue
        if not (ONE_YEAR_AGO <= date <= TODAY):
            continue
        rows.append({"date": date, f"{asset_label}_oi_value_usd": float(val)})

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def fetch_oi_with_history(csv_path: Path) -> pd.DataFrame:
    old_df = load_existing_csv(csv_path)

    df_btc = fetch_binance_oi("BTCUSDT", "btc")
    df_eth = fetch_binance_oi("ETHUSDT", "eth")

    if df_btc.empty and df_eth.empty:
        # 새 수집이 둘 다 막힌 경우: 기존 유지
        if not old_df.empty:
            log_warn("BTC/ETH OI 모두 수집 실패 → 기존 OI CSV 유지")
            return old_df
        else:
            log_warn("BTC/ETH OI 모두 수집 실패 & 기존 없음")
            return pd.DataFrame()

    df_new = pd.merge(df_btc, df_eth, on="date", how="outer")

    if old_df.empty:
        df_all = df_new
    else:
        df_all = pd.merge(old_df, df_new, on="date", how="outer", suffixes=("_old", ""))

        for col in ["btc_oi_value_usd", "eth_oi_value_usd"]:
            col_old = f"{col}_old"
            if col_old in df_all.columns:
                df_all[col] = df_all[col].combine_first(df_all[col_old])
                df_all = df_all.drop(columns=[col_old])

    df_all = df_all[
        (df_all["date"] >= ONE_YEAR_AGO) & (df_all["date"] <= TODAY)
    ].sort_values("date").reset_index(drop=True)

    return df_all


# =========================================================
# 차트
# =========================================================

def _basic_plot(df: pd.DataFrame, series_cols: List[Tuple[str, str]], title: str, ylabel: str, out_path: Path):
    if df is None or df.empty:
        return
    plt.figure(figsize=(10, 6))
    for col, label in series_cols:
        if col in df.columns:
            plt.plot(df["date"], df[col], label=label)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True)
    if len(series_cols) > 1:
        plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_stablecoin(df: pd.DataFrame):
    _basic_plot(
        df,
        [("stablecoin_mcap_usd", "Stablecoin Market Cap")],
        "Stablecoin Total Market Cap (USD)",
        "Market Cap (USD)",
        CHART_DIR / "stablecoin.png",
    )


def plot_etf_flows(df: pd.DataFrame):
    _basic_plot(
        df,
        [
            ("btc_etf_net_flow_usd_mn", "BTC ETF Net Flow (US$m)"),
            ("eth_etf_net_flow_usd_mn", "ETH ETF Net Flow (US$m)"),
        ],
        "Spot ETF Net Flows (Daily)",
        "Net Flow (US$m)",
        CHART_DIR / "etf_flows.png",
    )


def plot_realized_cap(df: pd.DataFrame):
    _basic_plot(
        df,
        [
            ("btc_realized_cap_usd", "BTC Realized Cap"),
            ("eth_realized_cap_usd", "ETH Realized Cap"),
        ],
        "Realized Market Cap (BTC & ETH)",
        "Realized Cap (USD)",
        CHART_DIR / "realized_cap.png",
    )


def plot_open_interest(df: pd.DataFrame):
    _basic_plot(
        df,
        [
            ("btc_oi_value_usd", "BTC OI Value"),
            ("eth_oi_value_usd", "ETH OI Value"),
        ],
        "Futures Open Interest (USD Value)",
        "OI Value (USD)",
        CHART_DIR / "open_interest.png",
    )


# =========================================================
# 스코어 계산
# =========================================================

def _percentile_score(series: pd.Series, value: float) -> Optional[float]:
    s = series.dropna()
    if s.empty or value is None or pd.isna(value):
        return None
    rank = (s <= value).sum() / len(s)
    score = float(rank * 100.0)
    return max(0.0, min(100.0, score))


def compute_liquidity_score_and_summary(df_all: pd.DataFrame):
    if df_all is None or df_all.empty:
        return None, None, [], "데이터가 없어 점수를 계산할 수 없습니다."

    df = df_all.sort_values("date").copy()
    latest_row = df.iloc[-1]
    latest_date = latest_row["date"]
    latest_date_str = (
        latest_date.isoformat() if isinstance(latest_date, dt.date)
        else str(latest_date)
    )

    components: List[Dict[str, Any]] = []

    def add_component(name: str, series: pd.Series, value: float, unit_desc: str = ""):
        score = _percentile_score(series, value)
        if score is None:
            return
        components.append(
            {
                "name": name,
                "score": round(score),
                "detail": unit_desc,
            }
        )

    # 1) Stablecoin change
    if "stablecoin_mcap_change_usd" in df.columns:
        val = latest_row.get("stablecoin_mcap_change_usd")
        unit = ""
        if val is not None and not pd.isna(val):
            unit = f"일일 변화 {val/1e9:.2f}B USD"
        add_component("스테이블코인 시총 일일 변화", df["stablecoin_mcap_change_usd"], val, unit)

    # 2) ETF total
    has_btc = "btc_etf_net_flow_usd_mn" in df.columns
    has_eth = "eth_etf_net_flow_usd_mn" in df.columns
    if has_btc or has_eth:
        etf_total = pd.Series(index=df.index, dtype=float)
        etf_total[:] = 0.0
        if has_btc:
            etf_total = etf_total.add(df["btc_etf_net_flow_usd_mn"].fillna(0), fill_value=0)
        if has_eth:
            etf_total = etf_total.add(df["eth_etf_net_flow_usd_mn"].fillna(0), fill_value=0)

        val_b = latest_row.get("btc_etf_net_flow_usd_mn") if has_btc else 0.0
        val_e = latest_row.get("eth_etf_net_flow_usd_mn") if has_eth else 0.0
        val_b = 0.0 if val_b is None or pd.isna(val_b) else float(val_b)
        val_e = 0.0 if val_e is None or pd.isna(val_e) else float(val_e)
        val_total = val_b + val_e
        unit = f"BTC+ETH 합 {val_total:.1f}M USD"
        add_component("현물 ETF 순유입 합산", etf_total, val_total, unit)

    # 3) Realized cap change
    has_br = "btc_realized_cap_usd" in df.columns
    has_er = "eth_realized_cap_usd" in df.columns
    if has_br or has_er:
        total = pd.Series(index=df.index, dtype=float)
        total[:] = 0.0
        if has_br:
            total = total.add(df["btc_realized_cap_usd"].fillna(0), fill_value=0)
        if has_er:
            total = total.add(df["eth_realized_cap_usd"].fillna(0), fill_value=0)
        change = total.diff()
        val = change.iloc[-1]
        unit = ""
        if val is not None and not pd.isna(val):
            unit = f"일일 변화 {val/1e9:.2f}B USD"
        add_component("BTC+ETH 실현 시총 일일 변화", change, val, unit)

    # 4) OI total level
    has_bo = "btc_oi_value_usd" in df.columns
    has_eo = "eth_oi_value_usd" in df.columns
    if has_bo or has_eo:
        total = pd.Series(index=df.index, dtype=float)
        total[:] = 0.0
        if has_bo:
            total = total.add(df["btc_oi_value_usd"].fillna(0), fill_value=0)
        if has_eo:
            total = total.add(df["eth_oi_value_usd"].fillna(0), fill_value=0)
        val = total.iloc[-1]
        unit = ""
        if val is not None and not pd.isna(val):
            unit = f"총 OI {val/1e9:.2f}B USD"
        add_component("BTC+ETH 선물 OI(수준)", total, val, unit)

    if not components:
        return latest_date_str, None, [], "가용 지표가 부족하여 점수를 계산할 수 없습니다."

    scores = [c["score"] for c in components if c.get("score") is not None]
    if not scores:
        return latest_date_str, None, components, "가용 지표가 부족하여 점수를 계산할 수 없습니다."

    total_score = round(sum(scores) / len(scores))

    # 점수 해석 문구
    if total_score >= 80:
        summary_main = "유입 강도가 매우 높은 구간입니다. 공격적 리스크 온 환경으로 해석 가능합니다."
    elif total_score >= 60:
        summary_main = "유입 우위 구간입니다. 추세 추종에 우호적일 수 있습니다."
    elif total_score >= 40:
        summary_main = "중립 구간입니다. 방향성보다 종목/섹터 이슈 민감도가 높을 수 있습니다."
    elif total_score >= 20:
        summary_main = "유출 우위 구간입니다. 레버리지 축소 및 리스크 관리 강화가 유리합니다."
    else:
        summary_main = "유출 강도가 매우 높은 구간입니다. 방어적 포지셔닝이 필요합니다."

    return latest_date_str, total_score, components, summary_main


# =========================================================
# HTML 생성
# =========================================================

def generate_html(output_path: Path,
                  latest_date_str: Optional[str],
                  total_score: Optional[int],
                  components: List[Dict[str, Any]],
                  summary_main: str):
    today_str = TODAY.isoformat()

    # 컴포넌트 상세 문구 구성
    li_lines = ""
    for c in components:
        detail = c.get("detail", "")
        li_lines += f"<li><strong>{c['name']}</strong>: {c['score']}점"
        if detail:
            li_lines += f" <span style='color:#666;'>({detail})</span>"
        li_lines += "</li>"

    if total_score is None:
        score_block = f"""
        <section class="score-card">
          <h2>오늘의 자금 유입/유출 점수</h2>
          <p>가용 지표가 부족하여 점수를 계산할 수 없습니다.</p>
          <p>{summary_main}</p>
        </section>
        """
    else:
        score_block = f"""
        <section class="score-card">
          <h2>오늘의 자금 유입/유출 점수: {total_score} / 100</h2>
          <p>기준일: {latest_date_str}</p>
          <p>{summary_main}</p>
          <ul>{li_lines}</ul>
        </section>
        """

    sections = []

    def add_section(title: str, img_rel: str, desc: str):
        sections.append(
            f"""
            <section>
              <h2>{title}</h2>
              <p>{desc}</p>
              <img src="{img_rel}" alt="{title}" style="max-width:100%;height:auto;">
            </section>
            """
        )

    add_section("1. Stablecoin Market Cap",
                "charts/stablecoin.png",
                "DeFiLlama 기준 전체 USD 스테이블코인 시가총액 추이.")
    add_section("2. Spot ETF Net Flows",
                "charts/etf_flows.png",
                "Farside 기반 BTC/ETH 현물 ETF 일별 순유입/유출.")
    add_section("3. Realized Market Cap",
                "charts/realized_cap.png",
                "CoinMetrics 기반 BTC/ETH 실현 시가총액.")
    add_section("4. Futures Open Interest",
                "charts/open_interest.png",
                "Binance 기반 BTC/ETH 선물 미결제약정(USD).")

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>Crypto Liquidity Dashboard (Stable Local)</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 1100px; margin: 20px auto; padding: 0 16px; }}
    h1 {{ text-align: center; }}
    section {{ margin-bottom: 40px; }}
    img {{ border: 1px solid #ddd; }}
    .updated {{ text-align: right; color: #666; font-size: 0.9em; }}
    .score-card {{
      border: 1px solid #ddd;
      padding: 16px;
      border-radius: 10px;
      background: #fafafa;
    }}
    .score-card ul {{ margin-top: 8px; }}
  </style>
</head>
<body>
  <h1>암호화폐 자금 유입/유출 모니터 (1년 일봉)</h1>
  <div class="updated">Last updated: {today_str}</div>
  {score_block}
  {''.join(sections)}
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    log_info("HTML 생성 완료")


# =========================================================
# 메인
# =========================================================

def main():
    print(f"[INFO] 기준일: {TODAY}, 최근 1년 시작일: {ONE_YEAR_AGO}")

    # --- 기존 CSV 로드 ---
    stable_csv = DATA_DIR / "stablecoin.csv"
    etf_csv = DATA_DIR / "etf_flows.csv"
    realized_csv = DATA_DIR / "realized_cap.csv"
    oi_csv = DATA_DIR / "open_interest.csv"
    merged_csv = DATA_DIR / "metrics_1y_merged.csv"

    old_stable = load_existing_csv(stable_csv)
    old_etf = load_existing_csv(etf_csv)
    old_real = load_existing_csv(realized_csv)
    old_oi = load_existing_csv(oi_csv)

    # 1) Stablecoin
    print("[STEP] 1/4 스테이블코인 시가총액 수집...")
    new_stable = fetch_stablecoin_mcap_1y()
    df_stable = save_or_keep(old_stable, new_stable, stable_csv)
    print(f"[OK] {stable_csv}")

    # 2) ETF
    print("[STEP] 2/4 BTC/ETH 현물 ETF 순유입 수집...")
    df_btc_etf = fetch_btc_etf_flows_1y()
    df_eth_etf = fetch_eth_etf_flows_1y()

    if df_btc_etf.empty and df_eth_etf.empty:
        log_warn("BTC/ETH ETF 새 수집 실패 → 기존 ETF CSV 유지")
        df_etf = old_etf
    elif df_btc_etf.empty:
        df_etf = df_eth_etf
    elif df_eth_etf.empty:
        df_etf = df_btc_etf
    else:
        df_etf = pd.merge(df_btc_etf, df_eth_etf, on="date", how="outer")

    if df_etf is None:
        df_etf = pd.DataFrame()

    if not df_etf.empty:
        df_etf = df_etf.sort_values("date").reset_index(drop=True)

    # 새 데이터가 있으면 저장, 아니면 기존 유지 로직
    df_etf = save_or_keep(old_etf, df_etf, etf_csv)
    print(f"[OK] {etf_csv}")

    # 3) Realized Cap
    print("[STEP] 3/4 BTC/ETH Realized Cap 수집...")
    new_real = fetch_realized_cap_1y(("btc", "eth"))
    df_real = save_or_keep(old_real, new_real, realized_csv)
    print(f"[OK] {realized_csv}")

    # 4) OI
    print("[STEP] 4/4 BTC/ETH 선물 OI 수집 & 누적...")
    # fetch_oi_with_history 내부가 이미 '기존 유지' 로직 일부 포함
    new_oi = fetch_oi_with_history(oi_csv)
    df_oi = save_or_keep(old_oi, new_oi, oi_csv)
    print(f"[OK] {oi_csv}")

    # --- 병합 CSV ---
    print("[STEP] 병합 CSV 생성...")
    df_all = df_stable.copy() if not df_stable.empty else pd.DataFrame(columns=["date"])

    for part in [df_etf, df_real, df_oi]:
        if part is not None and not part.empty:
            df_all = pd.merge(df_all, part, on="date", how="outer")

    if not df_all.empty:
        df_all = df_all.sort_values("date").reset_index(drop=True)

    try:
        df_all.to_csv(merged_csv, index=False)
        log_info(f"CSV 저장 완료: {merged_csv.name} ({len(df_all)} rows)")
    except Exception as e:
        log_warn(f"{merged_csv.name}: 저장 실패 - {e}")

    print(f"[OK] {merged_csv}")

    # --- 차트 ---
    print("[STEP] 차트 생성...")
    try:
        plot_stablecoin(df_stable)
        plot_etf_flows(df_etf)
        plot_realized_cap(df_real)
        plot_open_interest(df_oi)
    except Exception as e:
        log_warn(f"차트 생성 중 일부 실패: {e}")

    # --- 스코어/요약 ---
    print("[STEP] 점수 및 요약 계산...")
    latest_date_str, total_score, components, summary_main = compute_liquidity_score_and_summary(df_all)

    # --- HTML ---
    print("[STEP] HTML 생성...")
    try:
        generate_html(DOCS_DIR / "index.html", latest_date_str, total_score, components, summary_main)
    except Exception as e:
        log_warn(f"HTML 생성 실패: {e}")

    # --- 최종 운영 요약 ---
    print("\n[RUN SUMMARY]")
    print(f"- Stablecoin rows: {0 if df_stable is None else len(df_stable)}")
    print(f"- ETF rows:        {0 if df_etf is None else len(df_etf)}")
    print(f"- Realized rows:   {0 if df_real is None else len(df_real)}")
    print(f"- OI rows:         {0 if df_oi is None else len(df_oi)}")
    print(f"- Total score:     {total_score if total_score is not None else 'N/A'}")
    print(f"- HTML:            {DOCS_DIR / 'index.html'}")

    if WARNINGS:
        print("\n[WARNINGS]")
        # 너무 길어지지 않게 마지막 10개만
        for w in WARNINGS[-10:]:
            print(f"* {w}")

    print("\n[DONE] 모든 작업 완료.")


if __name__ == "__main__":
    main()
