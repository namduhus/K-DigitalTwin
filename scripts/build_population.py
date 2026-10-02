from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

RAW = Path("data/raw")
GRID_CSV = RAW / "_census_reqdoc_1787791099191" / "2024년_인구_다사_100M.csv"
GRID_SHP = RAW / "_grid_border_grid_2025_grid_다사_grid_다사" / "grid_다사_100M.shp"
OA_AGE = RAW / "인구_census_reqdoc_1787792255236" / "11_2024년_성연령별인구.csv"
OA_TOT = RAW / "인구_census_reqdoc_1787792255236" / "11_2024년_인구총괄(총인구).csv"
OA_SHP = RAW / "bnd_oa_11_2025_2Q" / "bnd_oa_11_2025_2Q.shp"

CELLS = Path("data/processing/grid_features.parquet")
BOUNDARY = Path("data/processing/seoul_boundary.parquet")
OUT = Path("data/processing/grid_population.parquet")

ENC = "cp949"                 # 셸 도구(`cut`·`head`)는 여기서 죽는다
ELDER_FROM = 14               # in_age_014 = 65~69세
N_BAND = 21
CRS_SRC, CRS_DST = 5179, 5186


def long_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, header=None, names=["yr", "cd", "item", "val"],
                       encoding=ENC, dtype={"cd": str, "item": str})


def main() -> int:
    for p in (GRID_CSV, GRID_SHP, OA_AGE, OA_TOT, OA_SHP, CELLS, BOUNDARY):
        if not p.exists():
            sys.exit(f"[error] 없음: {p}")

    # ── ① 집계구 고령비율 ──
    age = long_csv(OA_AGE)
    age["n"] = age.item.str[-3:].astype(int)
    whole = age[age.n <= N_BAND]                       # 전체 블록만 (남·여 블록은 중복)
    tot_oa = whole.groupby("cd").val.sum()
    eld_oa = whole[whole.n >= ELDER_FROM].groupby("cd").val.sum()
    oa = pd.DataFrame({"pop_oa": tot_oa, "eld_oa": eld_oa.reindex(tot_oa.index).fillna(0)})
    oa["elder_frac"] = np.where(oa.pop_oa > 0, oa.eld_oa / oa.pop_oa, np.nan)
    print(f"[집계구] {len(oa):,}개 · 전체 {oa.pop_oa.sum():,.0f}명 · "
          f"65세+ {oa.eld_oa.sum():,.0f}명 (**{oa.eld_oa.sum()/oa.pop_oa.sum()*100:.1f}%**)")

    chk = long_csv(OA_TOT)
    t1 = chk[chk.item == "to_in_001"].val.sum()
    print(f"          검산 — 인구총괄 총인구 {t1:,} vs 성연령별 합 {oa.pop_oa.sum():,.0f} "
          f"({oa.pop_oa.sum()/t1-1:+.2%} · BSCA 노이즈)")

    # ── ② 집계구 경계 ──
    g_oa = gpd.read_file(OA_SHP)[["TOT_OA_CD", "ADM_CD", "geometry"]].to_crs(CRS_DST)
    hit = g_oa.TOT_OA_CD.isin(oa.index)
    print(f"[경계]   집계구 {len(g_oa):,} · 통계와 조인 {int(hit.sum()):,} "
          f"(**{oa.index.isin(g_oa.TOT_OA_CD).mean()*100:.2f}%** of 통계)")
    if oa.index.isin(g_oa.TOT_OA_CD).mean() < 0.99:
        print("  조인율 99% 미만 — 통계와 경계의 연도가 어긋났을 수 있다", file=sys.stderr)
    g_oa = g_oa.merge(oa[["elder_frac"]], left_on="TOT_OA_CD", right_index=True, how="left")

    # ── ③ 100m 격자 인구 ──
    gp = long_csv(GRID_CSV)
    gp = gp[gp.item == "to_in_001"][["cd", "val"]].rename(columns={"val": "pop"})
    g100 = gpd.read_file(GRID_SHP)[["GRID_CD", "geometry"]]
    g100 = g100.merge(gp, left_on="GRID_CD", right_on="cd", how="inner").to_crs(CRS_DST)
    n_all = len(g100)

    # 서울로 먼저 자른다. 다사 타일은 **수도권 전체**를 덮어서(합계 2,528만명)
    # 자르지 않고 집계구 매칭을 하면 실패율이 84.5%로 나온다 — 서울 밖이라 당연한
    # 실패인데 **진짜 실패율이 그 안에 가려진다.** 자르고 재야 매칭 품질이 보인다.
    from shapely import from_wkt
    from shapely.ops import unary_union
    b = pd.read_parquet(BOUNDARY)
    seoul = unary_union([from_wkt(w) for w in b[b.sig_cd != "11000"].wkt])
    g100 = g100[g100.geometry.centroid.within(seoul)].copy()
    print(f"[격자]   100m {n_all:,}개(다사 타일 전체 {gp['pop'].sum():,}명) "
          f"→ **서울 {len(g100):,}개 · {g100['pop'].sum():,}명**")

    # 격자 중심점으로 집계구를 찾는다. 면적 안분은 노이즈 대비 이득이 없다 —
    # 100m 격자가 집계구(중위 약 0.03 km²)보다 작아 대부분 하나에 온전히 들어간다.
    cen = g100.copy()
    cen["geometry"] = g100.geometry.centroid
    j = gpd.sjoin(cen, g_oa[["elder_frac", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated()]                       # 경계 겹침 시 첫 건만
    miss = j.elder_frac.isna()
    print(f"          집계구 매칭 실패 {int(miss.sum()):,} "
          f"(**{miss.mean()*100:.2f}%**) → 서울 평균 비율로 채운다")
    if miss.mean() > 0.05:
        print("  매칭 실패가 5%를 넘는다 — 경계·좌표계를 확인할 것", file=sys.stderr)
    j.loc[miss, "elder_frac"] = float(oa.eld_oa.sum() / oa.pop_oa.sum())
    j["pop_elder"] = j["pop"] * j.elder_frac

    # ── ④ 25m 격자에 붙인다 ──
    cells = pd.read_parquet(CELLS, columns=["gx", "gy", "x", "y", "in_building"])
    # 100m 격자 하나에 25m 셀 16개가 들어간다. **인구를 나눠 주지 않고 밀도로 준다** —
    # 25m 셀의 "그 자리 인구밀도"가 필요한 것이지 인구를 쪼개는 것이 목적이 아니다.
    px = np.floor(j.geometry.x.to_numpy() / 100).astype(np.int64)
    py = np.floor(j.geometry.y.to_numpy() / 100).astype(np.int64)
    key = pd.DataFrame({"px": px, "py": py, "pop": j["pop"].to_numpy(),
                        "pop_elder": j.pop_elder.to_numpy(),
                        "elder_frac": j.elder_frac.to_numpy()}).groupby(["px", "py"]).sum(
                        ).assign(elder_frac=lambda d: np.where(d["pop"] > 0,
                                                              d.pop_elder / d["pop"], np.nan))

    cells["px"] = np.floor(cells.x / 100).astype(np.int64)
    cells["py"] = np.floor(cells.y / 100).astype(np.int64)
    out = cells.merge(key, on=["px", "py"], how="left")
    for c in ("pop", "pop_elder"):
        out[c] = out[c].fillna(0.0) / 16.0             # 100m 인구를 25m 셀 16개로 균등 분배
    out["elder_frac"] = out.elder_frac.fillna(float(oa.eld_oa.sum() / oa.pop_oa.sum()))

    have = out["pop"] > 0
    print(f"\n[25m]    셀 {len(out):,} · 인구>0 {int(have.sum()):,} ({have.mean()*100:.1f}%)")
    print(f"          총인구 {out['pop'].sum():,.0f}명 · 고령 {out.pop_elder.sum():,.0f}명 "
          f"(**{out.pop_elder.sum()/out['pop'].sum()*100:.1f}%**)")
    bld = out.in_building.astype(bool)
    print(f"          건물 내부 셀의 인구 비중 {out.loc[bld,'pop'].sum()/out['pop'].sum()*100:.1f}% "
          f"(사람은 건물에 산다 — 노출 가중에서는 건물 내부를 빼지 않는다)")

    out = out[["gx", "gy", "pop", "pop_elder", "elder_frac"]]
    out.to_parquet(OUT, index=False)
    print(f"\n→ {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")
    print(f"   출처: 국가데이터처 SGIS 격자통계·집계구통계 (이용허락범위 제한 없음 · 무료)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
