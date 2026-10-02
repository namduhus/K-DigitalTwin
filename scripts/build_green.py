from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.strtree import STRtree

UQ153 = Path("data/raw/park/UPIS_C_UQ153.shp")
ZON216 = Path("data/raw/park/UPIS_SHP_ZON216.shp")
SENSORS = Path("data/processing/sdot_locations.csv")
OUT = Path("data/processing/sensor_green.parquet")

TARGET_CRS = 5186
RADIUS = 600.0          # build_horizon.py와 동일 반경 — 피처 간 비교가 가능해진다

GREEN_LCLAS = {"UQT200", "UQT300"}          # 공원, 녹지
PAVED_LCLAS = {"UQT100", "UQT500"}          # 광장, 공공공지 — 포장면


def load(path: Path, label: str) -> gpd.GeoDataFrame:
    g = gpd.read_file(path, encoding="cp949")
    src = g.crs.to_epsg()
    g = g.to_crs(TARGET_CRS)
    print(f"{label}: {len(g):,}개 · EPSG:{src} → {TARGET_CRS}")
    return g


def agg(tree: STRtree, geoms: np.ndarray, pt: Point) -> tuple[int, float, float, float]:
    cand = tree.query(pt.buffer(RADIUS))
    if len(cand) == 0:
        return 0, 0.0, 0.0, np.nan
    buf = pt.buffer(RADIUS)
    inter = [geoms[j].intersection(buf).area for j in cand]
    areas = [geoms[j].area for j in cand]
    dist = min(geoms[j].distance(pt) for j in cand)
    total = sum(inter)
    return int(sum(1 for a in inter if a > 0)), total / (np.pi * RADIUS**2), max(areas), dist


def main() -> int:
    for p in (UQ153, ZON216, SENSORS):
        if not p.exists():
            sys.exit(f"[error] {p} 없음")

    uq = load(UQ153, "도시계획시설(공간시설)")
    zn = load(ZON216, "생활권계획 공원")
    sen = pd.read_csv(SENSORS)

    # 대분류로 녹지/포장면 분리
    grn = uq[uq["LCLAS_CL"].isin(GREEN_LCLAS)]
    pav = uq[uq["LCLAS_CL"].isin(PAVED_LCLAS)]
    print(f"\n녹지(공원+녹지) {len(grn):,}개 · {grn.geometry.area.sum()/1e6:.1f} km²")
    print(f"포장면(광장+공공공지) {len(pav):,}개 · {pav.geometry.area.sum()/1e6:.1f} km²")
    print(f"생활권 공원 {len(zn):,}개 · {zn.geometry.area.sum()/1e6:.1f} km²")

    # 두 출처의 녹지를 합집합으로 쓴다 (중복은 교차면적 계산에서 이중계상되나,면적비를 1로 클립해 처리한다. 서로 보완 관계라 합치는 편이 누락이 적다)
    green_all = pd.concat([grn.geometry, zn.geometry], ignore_index=True)
    gg = np.array(green_all.values, dtype=object)
    pp = np.array(pav.geometry.values, dtype=object)
    t_g, t_p = STRtree(gg), STRtree(pp)

    rows = []
    for i, s in sen.iterrows():
        pt = Point(float(s["x"]), float(s["y"]))
        gn, gf, gmax, gd = agg(t_g, gg, pt)
        pn, pf, _, pd_ = agg(t_p, pp, pt)
        rows.append({
            "sn": s["sn"],
            "grn_n": gn,
            "grn_frac": min(gf, 1.0),          # 두 출처 중복으로 1을 넘을 수 있다
            "grn_ar_max": gmax,
            "grn_d_min": gd,
            "pave_n": pn,
            "pave_frac": min(pf, 1.0),
        })
        if (i + 1) % 100 == 0 or i + 1 == len(sen):
            print(f"  {i+1:>5}/{len(sen)}", end="\r", file=sys.stderr)
    print(file=sys.stderr)

    df = pd.DataFrame(rows)
    print(f"\n{'항목':<14} {'중위':>10} {'최소':>10} {'최대':>10} {'결측':>7}")
    print("-" * 56)
    for c, lab in [("grn_frac", "녹지 면적비"), ("grn_d_min", "최근접 녹지(m)"),
                   ("grn_ar_max", "최대 녹지(㎡)"), ("grn_n", "반경내 녹지수"),
                   ("pave_frac", "포장면 면적비")]:
        s = df[c]
        print(f"{lab:<14} {s.median():>10,.3f} {s.min():>10,.3f} {s.max():>10,.3f} {s.isna().mean():>6.1%}")

    df.to_parquet(OUT, index=False)
    print(f"\n→ {OUT} ({OUT.stat().st_size/1e6:.2f} MB, {len(df):,}센서)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
