from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
import urllib.parse as up
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import dotenv_values

URL = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
BOUNDARY = Path("data/processing/seoul_boundary.parquet")
OUT = Path("data/processing/kma_treg.parquet")
RAW = Path("data/raw/kma")           # 원본 응답 보관 — 예보는 지나가면 못 받는다

BASE_TIMES = ["2300", "2000", "1700", "1400", "1100", "0800", "0500", "0200"]

# 기상청 격자 변환 상수 (공개 표준)
RE, GRID = 6371.00877, 5.0
SLAT1, SLAT2, OLON, OLAT, XO, YO = 30.0, 60.0, 126.0, 38.0, 43, 136


def latlon_to_grid(lat, lon):
    DEG = np.pi / 180.0
    re_, sl1, sl2, ol, oa = RE / GRID, SLAT1 * DEG, SLAT2 * DEG, OLON * DEG, OLAT * DEG
    sn = np.log(np.cos(sl1) / np.cos(sl2)) / np.log(
        np.tan(np.pi * 0.25 + sl2 * 0.5) / np.tan(np.pi * 0.25 + sl1 * 0.5))
    sf = (np.tan(np.pi * 0.25 + sl1 * 0.5) ** sn) * np.cos(sl1) / sn
    ro = re_ * sf / (np.tan(np.pi * 0.25 + oa * 0.5) ** sn)
    ra = re_ * sf / (np.tan(np.pi * 0.25 + np.asarray(lat) * DEG * 0.5) ** sn)
    theta = np.asarray(lon) * DEG - ol
    theta = np.where(theta > np.pi, theta - 2 * np.pi, theta)
    theta = np.where(theta < -np.pi, theta + 2 * np.pi, theta) * sn
    return (np.floor(ra * np.sin(theta) + XO + 0.5).astype(int),
            np.floor(ro - ra * np.cos(theta) + YO + 0.5).astype(int))


def key() -> str:
    for k, v in dotenv_values(".env").items():
        if "KMA" in k.upper() and v and v.strip():
            # Encoding 키(`%3D`로 끝남)를 params에 넣으면 이중 인코딩돼 인증이 실패한다
            return up.unquote(v.strip())
    sys.exit("[error] .env에서 기상청 인증키(KMA_*)를 찾을 수 없다")


def seoul_grids() -> pd.DataFrame:
    from pyproj import Transformer
    from shapely import from_wkt
    from shapely.ops import unary_union
    b = pd.read_parquet(BOUNDARY)
    seoul = unary_union([from_wkt(w) for w in b[b.sig_cd != "11000"].wkt])
    x0, y0, x1, y1 = seoul.bounds
    # 2km 간격으로 훑어 격자 번호를 모은다 (기상청 격자가 5km이므로 충분히 촘촘하다)
    gx, gy = np.meshgrid(np.arange(x0, x1, 2000.0), np.arange(y0, y1, 2000.0))
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    tr = Transformer.from_crs(5186, 4326, always_xy=True)
    lon, lat = tr.transform(pts[:, 0], pts[:, 1])
    from shapely.geometry import Point
    inside = np.array([seoul.contains(Point(x, y)) for x, y in pts])
    nx, ny = latlon_to_grid(lat[inside], lon[inside])
    g = pd.DataFrame({"nx": nx, "ny": ny}).drop_duplicates().sort_values(["nx", "ny"])
    return g.reset_index(drop=True)


def call(k, d, t, nx, ny, rows=1000, tries=3):
    for i in range(tries):
        try:
            r = requests.get(URL, timeout=30, params={
                "serviceKey": k, "pageNo": 1, "numOfRows": rows, "dataType": "JSON",
                "base_date": d, "base_time": t, "nx": int(nx), "ny": int(ny)})
            j = r.json()["response"]
            code = j["header"]["resultCode"]
            if code == "00":
                return j["body"]["items"]["item"]
            return code
        except Exception as e:
            if i == tries - 1:
                return f"ERR:{e}"
            time.sleep(2 ** i)


def latest_base(k) -> tuple[str, str]:
    now = dt.datetime.now()
    for dd in range(0, 3):
        d = (now - dt.timedelta(days=dd)).strftime("%Y%m%d")
        for t in BASE_TIMES:
            if isinstance(call(k, d, t, 60, 127, rows=1), list):
                return d, t
    sys.exit("[error] 최근 3일 안에 자료가 있는 발표 시각이 없다")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="발표시각 가용성만 확인")
    ap.add_argument("--base", nargs=2, metavar=("YYYYMMDD", "HHMM"))
    a = ap.parse_args()
    k = key()

    if a.probe:
        now = dt.datetime.now()
        print(f"시스템 시각 {now:%Y-%m-%d %H:%M}\n")
        print(f"  {'날짜':<10}" + "".join(f"{t:>6}" for t in reversed(BASE_TIMES)))
        for dd in range(0, 4):
            d = (now - dt.timedelta(days=dd)).strftime("%Y%m%d")
            marks = []
            for t in reversed(BASE_TIMES):
                r = call(k, d, t, 60, 127, rows=1)
                marks.append("  O  " if isinstance(r, list) else "  .  ")
            print(f"  {d:<10}" + "".join(f"{m:>6}" for m in marks))
        print("\n  O = 자료 있음 · . = NO_DATA")
        print("  과거 예보는 약 2일치만 남는다 — 소급 검증은 불가, 전향 검증으로 간다")
        return 0

    d, t = a.base if a.base else latest_base(k)
    print(f"[발표] {d} {t}시")

    g = seoul_grids()
    print(f"[격자] 서울을 덮는 기상청 격자 **{len(g)}개** "
          f"(nx {g.nx.min()}~{g.nx.max()} · ny {g.ny.min()}~{g.ny.max()})")

    RAW.mkdir(parents=True, exist_ok=True)
    recs, fail = [], 0
    t0 = time.time()
    for i, r in g.iterrows():
        items = call(k, d, t, r.nx, r.ny)
        if not isinstance(items, list):
            fail += 1
            continue
        for it in items:
            if it["category"] == "TMP":            # 1시간 기온
                recs.append({"nx": r.nx, "ny": r.ny,
                             "fcst": it["fcstDate"] + it["fcstTime"],
                             "tmp": float(it["fcstValue"])})
        print(f"  {i+1}/{len(g)} · 누적 {len(recs):,}건 · {time.time()-t0:.0f}초", end="\r")
    print()
    if fail:
        print(f"  실패 격자 {fail}개")

    df = pd.DataFrame(recs)
    # 원본 보관 — 예보는 지나가면 다시 못 받는다 (`fetch_boundary.py`와 같은 규약)
    raw_p = RAW / f"vilage_{d}_{t}.parquet"
    df.to_parquet(raw_p, index=False)
    print(f"  원본 저장: {raw_p} ({len(df):,}행)")

    df["ts"] = pd.to_datetime(df.fcst, format="%Y%m%d%H%M")
    treg = df.groupby("ts").agg(t_reg=("tmp", "median"),
                                n_grid=("tmp", "size"),
                                spread=("tmp", lambda s: s.max() - s.min())).reset_index()
    treg["base"] = f"{d}{t}"
    treg.to_parquet(OUT, index=False)

    print(f"\n[t_reg] 예보 시각 {len(treg)}개 · "
          f"{treg.ts.min():%m-%d %H시} ~ {treg.ts.max():%m-%d %H시}")
    print(f"        기온 {treg.t_reg.min():.1f} ~ {treg.t_reg.max():.1f}℃")
    print(f"        격자 간 폭 중위 **{treg.spread.median():.1f}℃** — "
          f"기상청 5km 해상도가 서울 안에서 내는 차이다")
    print(f"\n  {'시각':<14}{'t_reg':>8}{'격자':>7}{'격자간 폭':>10}")
    print("  " + "-" * 39)
    for _, r in treg.head(12).iterrows():
        print(f"  {r.ts:%m-%d %H시}   {r.t_reg:>8.1f}{int(r.n_grid):>7}{r.spread:>10.1f}")
    print(f"\n→ {OUT}")
    print(f"   출처: 기상청 단기예보 조회서비스 (공공누리 제1유형 · 무료)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
