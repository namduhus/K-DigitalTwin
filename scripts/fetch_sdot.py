from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

BASE = "http://openapi.seoul.go.kr:8088"
SERVICE = "IotVdata017"
PAGE = 1000  # 서울 OpenAPI 1회 최대 행수
RAW = Path("data/raw")

# 결측률을 반드시 확인해야 하는 컬럼.
# 정지우·남진(2022)은 2021년 데이터에서 풍향·흑구온도 등 9개 항목의
# 결측률이 95% 이상이라 제거했다. 최신 데이터가 개선됐는지가 Phase 0의 결정 게이트다.
WATCH = {
    "AVG_TP": "기온(℃)",
    "AVG_HUM": "습도(%)",
    "AVG_WSPD": "풍속(m/s)",
    "AVG_WD": "풍향",
    "AVG_GT": "흑구온도(℃)",
    "AVG_INILLU": "조도(lux)",
    "AVG_UV": "자외선",
    "AVG_NIS": "소음(dB)",
}


def key() -> str:
    k = os.getenv("SMART_SEOUL_ENV") or os.getenv("SMART_SEOUL_ENV_LIVE")
    if not k:
        sys.exit("[error] .env에 SMART_SEOUL_ENV 없음")
    return k.strip()


def call(start: int, end: int, gu: str | None = None, date: str | None = None) -> dict:
    url = f"{BASE}/{key()}/json/{SERVICE}/{start}/{end}/"
    if date and not gu:
        url += "%20/"
    elif gu:
        url += f"{gu}/"
    if date:
        url += f"{date}/"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    body = r.json()

    if SERVICE in body:
        payload = body[SERVICE]
        code = payload.get("RESULT", {}).get("CODE", "")
        if code and not code.startswith("INFO-000"):
            sys.exit(f"[error] {code}: {payload['RESULT'].get('MESSAGE')}")
        return payload

    # 서비스 키가 없으면 최상위에 RESULT만 온다 (인증 실패 등)
    res = body.get("RESULT", body)
    sys.exit(f"[error] {res.get('CODE')}: {res.get('MESSAGE')}")


def probe() -> None:
    p = call(1, 5)
    rows = p.get("row", [])
    print(f"list_total_count = {p.get('list_total_count'):,}")
    print(f"반환 행수 = {len(rows)}")
    if rows:
        print(f"컬럼 {len(rows[0])}개: {', '.join(rows[0].keys())}")
        print("\n--- 첫 행 ---")
        print(json.dumps(rows[0], ensure_ascii=False, indent=2)[:1400])


def pull(gu: str | None, date: str | None, limit: int | None = None) -> pd.DataFrame:
    head = call(1, 1, gu, date)
    total = int(head.get("list_total_count", 0))
    if total == 0:
        sys.exit(
            f"[error] 데이터 0건 (gu={gu}, date={date})\n"
            "  이 API는 약 1개월 롤링 윈도우다. 과거 날짜는 조회되지 않는다.\n"
            "  과거 데이터는 열린데이터광장의 연도별 파일 데이터셋을 받아야 한다."
        )
    cap = min(total, limit) if limit else total
    print(f"총 {total:,}건 중 {cap:,}건 수집", file=sys.stderr)

    frames, start = [], 1
    while start <= cap:
        end = min(start + PAGE - 1, cap)
        rows = call(start, end, gu, date).get("row", [])
        if not rows:
            break
        frames.append(pd.DataFrame(rows))
        print(f"  {end:,}/{cap:,}", end="\r", file=sys.stderr)
        start = end + 1
        time.sleep(0.2)
    print(file=sys.stderr)
    return pd.concat(frames, ignore_index=True)


def report_missing(df: pd.DataFrame) -> None:
    print(f"\n행수 {len(df):,} · 센서 {df['SN'].nunique()}개")
    if "MSRMT_HR" in df:
        print(f"측정시간 범위 {df['MSRMT_HR'].min()} ~ {df['MSRMT_HR'].max()}")

    print(f"\n{'컬럼':<12} {'항목':<14} {'결측률':>8}  {'유효':>7}  판정")
    print("-" * 62)
    for col, label in WATCH.items():
        if col not in df.columns:
            print(f"{col:<12} {label:<14} {'컬럼없음':>8}")
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        # 센서 미측정은 결측 대신 0 또는 음수 센티널로 들어오는 경우가 있다
        valid = s.notna() & (s > -900)
        rate = 1 - valid.mean()
        verdict = "사용가능" if rate < 0.3 else ("주의" if rate < 0.95 else "사실상 불가")
        print(f"{col:<12} {label:<14} {rate:>7.1%}  {valid.sum():>7,}  {verdict}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "missing", "fetch"])
    ap.add_argument("--gu", help="자치구 영문명 (예: Gangnam-gu)")
    ap.add_argument("--date", help="등록일시 yyyy-mm-dd")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    if a.cmd == "probe":
        probe()
        return 0

    df = pull(a.gu, a.date, a.limit)
    report_missing(df)

    if a.cmd == "fetch":
        RAW.mkdir(parents=True, exist_ok=True)

        # 파일명에 **실제 수집된 데이터 범위**를 넣는다.
        # 고정 이름(sdot_all.parquet)을 쓰면 재실행 때 덮어쓰는데, 이 API는
        # 롤링 윈도우라 앞쪽이 떨어져 나간 뒤 덮어쓰면 그 구간이 영구 손실된다.
        # 주간 파일은 약 1주 지연 발행이므로 최근 구간은 이 parquet이 유일한 사본이다.
        ts = pd.to_datetime(df["MSRMT_HR"], format="%Y-%m-%d_%H:%M:%S", errors="coerce")
        span = f"{ts.min():%Y%m%d}-{ts.max():%Y%m%d}" if ts.notna().any() else "unknown"
        parts = [x for x in [a.gu, span] if x]
        out = RAW / f"sdot_api_{'_'.join(parts)}.parquet"

        if out.exists():
            print(f"\n[skip] 같은 범위의 파일이 이미 있다: {out}")
        else:
            df.to_parquet(out, index=False)
            print(f"\n→ {out} ({out.stat().st_size / 1e6:.1f} MB)")
        print("  ※ raw/ 의 API parquet은 복구 불가이므로 삭제하지 않는다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
