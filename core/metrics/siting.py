from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from core.baseline.b2_gbm import BLD_F, TERR_F
from core.eval.dataset import load_base, load_sensor_meta

GRID = Path("data/processing/grid_features.parquet")
POP = Path("data/processing/grid_population.parquet")
TRAIN = Path("data/processing/train.parquet")
OUT = Path("data/metric/siting.parquet")

# 정적 피처만. `hz_sun`·`shadow_margin`·`is_shadow`는 시각마다 변해 셀 속성이 아니다.
STATIC_SHADE = ["svf", "svf_bld", "svf_terr", "hz_mean", "hz_max"]
FEATS = BLD_F + TERR_F + STATIC_SHADE
BLK = 20                      # 500m — 처방 지도와 같은 단위


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topn", type=int, default=20)
    a = ap.parse_args()

    # ── 센서 쪽 피처 ──
    base = load_base(FEATS, verbose=False)
    sen = base.groupby("sn")[FEATS].first().dropna()
    meta = load_sensor_meta()
    sxy = meta.loc[sen.index, ["x", "y"]].to_numpy(float)
    print(f"[센서] {len(sen):,}개 · 정적 피처 {len(FEATS)}개")

    mu = sen.mean().to_numpy(dtype=float)
    sd = sen.std().to_numpy(dtype=float).copy()   # pandas 반환이 읽기전용일 수 있다
    sd[sd == 0] = 1.0
    S = ((sen.to_numpy() - mu) / sd).astype(np.float32)

    # ── 격자 쪽 ──
    g = pd.read_parquet(GRID, columns=["gx", "gy", "x", "y", "gu"] + FEATS)
    ok = g[FEATS].notna().all(axis=1)
    print(f"[격자] {len(g):,}셀 · 피처 완전 {int(ok.sum()):,} ({ok.mean()*100:.1f}%)")
    g = g[ok].reset_index(drop=True)
    G = ((g[FEATS].to_numpy() - mu) / sd).astype(np.float32)

    from scipy.spatial import cKDTree
    tree = cKDTree(S)

    # 기준선 — 센서끼리의 최근접 거리. 이보다 멀면 확실히 외삽이다.
    d_sen, _ = tree.query(S, k=2)          # 자기 자신 제외
    ref = np.percentile(d_sen[:, 1], [50, 90, 95])
    print(f"       센서 간 최근접 거리(피처 공간)  중위 {ref[0]:.2f} · "
          f"P90 {ref[1]:.2f} · P95 {ref[2]:.2f} (표준편차 단위)")

    g["fdist"], _ = tree.query(G, k=1)
    print(f"       격자 → 최근접 센서 거리        중위 {g.fdist.median():.2f} · "
          f"P90 {g.fdist.quantile(.90):.2f} · 최대 {g.fdist.max():.2f}")
    for lab, thr in (("센서 중위", ref[0]), ("센서 P90", ref[1]), ("센서 P95", ref[2])):
        print(f"       {lab}({thr:.2f})보다 먼 셀: **{(g.fdist > thr).mean()*100:.1f}%**")

    # 공간 거리도 함께 — 피처가 비슷해도 물리적으로 멀면 관측이 없는 것이다
    stree = cKDTree(sxy)
    g["sdist"], _ = stree.query(g[["x", "y"]].to_numpy(float), k=1)
    print(f"       격자 → 최근접 센서 **물리 거리** 중위 {g.sdist.median():.0f}m · "
          f"P95 {g.sdist.quantile(.95):.0f}m · 최대 {g.sdist.max():.0f}m")

    # ── 인구 ──
    pop = pd.read_parquet(POP, columns=["gx", "gy", "pop", "pop_elder"])
    g = g.merge(pop, on=["gx", "gy"], how="left")
    g[["pop", "pop_elder"]] = g[["pop", "pop_elder"]].fillna(0.0)

    # ── 500m 블록 우선순위 ──
    g["bx"], g["by"] = g.gx // BLK, g.gy // BLK
    blk = g.groupby(["bx", "by"]).agg(
        fdist=("fdist", "median"), sdist=("sdist", "median"),
        pop=("pop", "sum"), pop_elder=("pop_elder", "sum"),
        x=("x", "mean"), y=("y", "mean"), n=("gx", "size"))
    blk = blk[blk.n >= BLK * BLK * 0.5].reset_index()
    blk["prio"] = blk.fdist.rank(pct=True) * blk["pop"].rank(pct=True)
    print(f"\n[블록] 500m {len(blk):,}개 · 인구 합계 {blk['pop'].sum():,.0f}명")

    top = blk.nlargest(a.topn, "prio").copy()
    # 자치구 이름 — 블록 중심에서 가장 가까운 센서의 자치구로 표시한다(근사)
    _, ix = stree.query(top[["x", "y"]].to_numpy(float), k=1)
    top["gu"] = meta.loc[sen.index[ix], "gu"].to_numpy()

    print(f"\n{'='*84}\n다음 S-DoT 후보 — 상위 {a.topn} 블록 (외삽 정도 × 인구)\n{'='*84}")
    print(f"  {'#':>3}{'자치구':>9}{'인구':>9}{'고령':>8}"
          f"{'피처거리':>10}{'최근접센서':>11}{'중심 x,y':>20}")
    print("  " + "-" * 70)
    for i, (_, r) in enumerate(top.iterrows(), 1):
        print(f"  {i:>3}{str(r.gu):>9}{int(r['pop']):>9,}{int(r.pop_elder):>8,}"
              f"{r.fdist:>10.2f}{r.sdist:>10.0f}m{r.x:>11.0f}{r.y:>9.0f}")

    print(f"\n  상위 {a.topn} 블록 합계 인구 **{top['pop'].sum():,.0f}명** "
          f"(고령 {top.pop_elder.sum():,.0f}명) · "
          f"피처거리 중위 {top.fdist.median():.2f} (전체 {blk.fdist.median():.2f})")

    # ── 자치구별 요약 — 센서 편차와 대조 ──
    # 셀의 자치구는 `grid_features`의 `gu`(경계로 부여된 값)를 쓴다.
    # 최근접 센서의 자치구로 근사하면 **센서가 많은 구가 셀을 더 가져가** 순환이 생긴다.
    gu = g.groupby("gu").agg(cells=("gx", "size"), pop=("pop", "sum"),
                             fdist=("fdist", "median"))
    gu["sensors"] = meta.loc[sen.index].groupby("gu").size().reindex(gu.index).fillna(0)
    gu["pop_per_sensor"] = gu["pop"] / gu.sensors.replace(0, np.nan)
    gu = gu.sort_values("pop_per_sensor", ascending=False)
    print(f"\n{'='*84}\n자치구별 — 센서 1대가 담당하는 인구\n{'='*84}")
    print(f"  {'자치구':<10}{'센서':>6}{'인구':>12}{'센서당 인구':>13}{'피처거리 중위':>14}")
    print("  " + "-" * 55)
    for k, v in gu.head(8).iterrows():
        print(f"  {k:<10}{int(v.sensors):>6}{int(v['pop']):>12,}"
              f"{v.pop_per_sensor:>13,.0f}{v.fdist:>14.2f}")
    print("  ...")
    for k, v in gu.tail(3).iterrows():
        print(f"  {k:<10}{int(v.sensors):>6}{int(v['pop']):>12,}"
              f"{v.pop_per_sensor:>13,.0f}{v.fdist:>14.2f}")
    print(f"\n  센서당 인구 최대/최소 **{gu.pop_per_sensor.max()/gu.pop_per_sensor.min():.1f}배**")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    g[["gx", "gy", "fdist", "sdist", "pop", "pop_elder"]].to_parquet(OUT, index=False)
    print(f"\n→ {OUT}")
    print(f"\n  이것은 **최적 설계가 아니다.** 새 센서가 주변 불확실성을 얼마나 줄이는지")
    print(f"    풀려면 공분산 구조와 순차 선택이 필요하다. 여기서는 **어디가 비어 있는지**만 보인다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
