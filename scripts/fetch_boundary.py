from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import dotenv_values
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

BASE = "https://api.vworld.kr/req/wfs"
TYPENAME = "lt_c_adsigg_info"
SEOUL_BBOX = (176_000.0, 534_500.0, 217_700.0, 569_000.0)   # EPSG:5186
SEOUL_SIG_PREFIX = "11"          # 서울특별시 시군구 코드 앞 2자리
OUT = Path("data/processing/seoul_boundary.parquet")
RAW = Path("data/raw/vworld_adsigg.gml")   # 원본 응답 보관 — 아래 주석 참조

FEAT_RE = re.compile(r"<gml:featureMember>(.*?)</gml:featureMember>", re.S)
POLY_RE = re.compile(r"<gml:Polygon[^>]*>(.*?)</gml:Polygon>", re.S)
OUTER_RE = re.compile(r"<gml:outerBoundaryIs>.*?<gml:coordinates[^>]*>([^<]+)"
                      r"</gml:coordinates>.*?</gml:outerBoundaryIs>", re.S)
INNER_RE = re.compile(r"<gml:innerBoundaryIs>.*?<gml:coordinates[^>]*>([^<]+)"
                      r"</gml:coordinates>.*?</gml:innerBoundaryIs>", re.S)


def key() -> str:
    for k, v in dotenv_values(".env").items():
        if "WORLD" in k.upper() and v and v.strip():
            return v.strip()
    sys.exit("[error] .env에서 브이월드 인증키를 찾을 수 없다")


def coords(s: str) -> np.ndarray:
    return np.array([[float(a) for a in p.split(",")[:2]] for p in s.split()])


def fetch(k: str, tries: int = 5) -> str:
    if RAW.exists():
        print(f"  원본 캐시 사용: {RAW} ({RAW.stat().st_size/1e6:.1f} MB)")
        return RAW.read_text(encoding="utf-8")
    p = {"SERVICE": "WFS", "REQUEST": "GetFeature", "VERSION": "1.1.0",
         "TYPENAME": TYPENAME, "BBOX": ",".join(f"{v:.1f}" for v in SEOUL_BBOX),
         "SRSNAME": "EPSG:5186", "MAXFEATURES": "1000", "OUTPUT": "GML2",
         "KEY": k, "DOMAIN": "localhost"}
    for i in range(tries):
        try:
            r = requests.get(BASE, params=p, timeout=180)
            r.encoding = "utf-8"          # 서버가 charset을 안 주면 한글이 깨진다
            if r.status_code == 200 and "FeatureCollection" in r.text:
                RAW.parent.mkdir(parents=True, exist_ok=True)
                RAW.write_text(r.text, encoding="utf-8")
                print(f"  원본 저장: {RAW} ({len(r.text)/1e6:.1f} MB)")
                return r.text
            print(f"  [retry {i+1}] HTTP {r.status_code}", file=sys.stderr)
        except requests.RequestException as e:
            print(f"  [retry {i+1}] {e}", file=sys.stderr)
        if i < tries - 1:
            time.sleep(min(2 ** i * 2, 60))
    sys.exit("[error] WFS 요청 실패")


def parse(xml: str) -> pd.DataFrame:
    rows = []
    for fm in FEAT_RE.findall(xml):
        def tag(name: str) -> str | None:
            m = re.search(rf"<sop:{name}>([^<]*)</sop:{name}>", fm)
            return m.group(1).strip() if m else None

        polys = []
        for pb in POLY_RE.findall(fm):
            outer = OUTER_RE.findall(pb)
            if not outer:
                continue
            inners = [coords(s) for s in INNER_RE.findall(pb)]
            polys.append(Polygon(coords(outer[0]), inners))
        if not polys:
            continue
        geom = polys[0] if len(polys) == 1 else MultiPolygon(polys)
        rows.append({"sig_cd": tag("sig_cd"), "sig_kor_nm": tag("sig_kor_nm"),
                     "full_nm": tag("full_nm"), "geom": geom})
    return pd.DataFrame(rows)


def main() -> int:
    print(f"브이월드 WFS {TYPENAME} 요청 — bbox {SEOUL_BBOX}")
    xml = fetch(key())
    print(f"  {len(xml)/1e6:.1f} MB 수신")

    df = parse(xml)
    print(f"  파싱 {len(df):,}개 시군구 (bbox 안이므로 경기·인천 포함)")

    seoul = df[df["sig_cd"].str.startswith(SEOUL_SIG_PREFIX, na=False)].copy()
    print(f"  서울(코드 {SEOUL_SIG_PREFIX}*) **{len(seoul)}개 자치구**")
    if len(seoul) != 25:
        print(f"  서울은 25개 자치구여야 한다 — {len(seoul)}개다. 코드 필터를 확인한다")

    seoul["area_km2"] = seoul["geom"].apply(lambda g: g.area / 1e6)
    seoul = seoul.sort_values("area_km2", ascending=False).reset_index(drop=True)
    valid = seoul["geom"].apply(lambda g: g.is_valid)
    print(f"  지오메트리 유효 {int(valid.sum())}/{len(seoul)}")
    if not valid.all():
        seoul.loc[~valid, "geom"] = seoul.loc[~valid, "geom"].apply(lambda g: g.buffer(0))
        print(f"  무효 {int((~valid).sum())}개를 buffer(0)으로 보정했다")

    union = unary_union(seoul["geom"].tolist())
    tot = union.area / 1e6
    print(f"\n서울 전체 면적 **{tot:.1f} km²** (공식 605.2 km², 차 {tot-605.2:+.1f})")
    b = union.bounds
    print(f"  bbox x {b[0]:.0f}~{b[2]:.0f} ({(b[2]-b[0])/1000:.1f} km) · "
          f"y {b[1]:.0f}~{b[3]:.0f} ({(b[3]-b[1])/1000:.1f} km)")
    for cell in (25, 50):
        n_bbox = int((b[2]-b[0])//cell + 1) * int((b[3]-b[1])//cell + 1)
        print(f"  {cell}m 격자: bbox {n_bbox:,}셀 · **경계 내 약 {int(tot*1e6/cell**2):,}셀** "
              f"({tot*1e6/cell**2/n_bbox*100:.0f}%)")

    print(f"\n자치구 면적 (km²) — 상위·하위 3개")
    for _, r in pd.concat([seoul.head(3), seoul.tail(3)]).iterrows():
        print(f"  {r['sig_kor_nm']:<8}{r['area_km2']:>8.2f}")

    out = pd.concat([
        seoul[["sig_cd", "sig_kor_nm", "full_nm", "area_km2"]].assign(
            wkt=seoul["geom"].apply(lambda g: g.wkt)),
        pd.DataFrame([{"sig_cd": "11000", "sig_kor_nm": "서울특별시",
                       "full_nm": "서울특별시", "area_km2": tot, "wkt": union.wkt}]),
    ], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"\n→ {OUT} ({OUT.stat().st_size/1e6:.1f} MB, 자치구 {len(seoul)} + 전체 1)")
    print("  출처: 국토교통부 행정구역도(공공데이터포털 15059008) · 공공누리 제1유형 · 무료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
