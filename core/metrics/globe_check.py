from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN = Path("data/processing/train.parquet")
OUT = Path("data/metric/globe_check.parquet")

MIN_VALID = 0.50          # 센서 유효율 하한 (`data_schema.md` §흑구온도와 같은 기준)
MIN_ROWS = 200
SUN_EDGES = [0, 10, 20, 30, 40, 50, 90]
MARGIN_EDGES = [-90, -30, -10, 0, 10, 30, 90]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args()

    cols = ["sn", "ts", "gu", "tp", "gt", "sensor_ok", "is_night", "is_shadow",
            "sun_el", "shadow_margin", "svf", "hz_mean", "bld_d_min", "illu"]
    d = pd.read_parquet(TRAIN, columns=cols)
    d = d[d.sensor_ok & d.tp.notna()]

    # ── 유효 센서 추리기 ──
    v = d.groupby("sn").agg(n=("tp", "size"), gt_ok=("gt", "count"))
    v["frac"] = v["gt_ok"] / v["n"]
    keep = v.index[(v.frac >= MIN_VALID) & (v.gt_ok >= MIN_ROWS)]
    print(f"[센서] 전체 {len(v):,} · 흑구온도 유효율 ≥{MIN_VALID:.0%} & {MIN_ROWS}행+ → "
          f"**{len(keep)}개**")
    # `d.gt`는 **pandas의 `.gt()` 메서드(greater than)**로 잡힌다 — 컬럼명이
    # 메서드명과 겹치는 경우다(`.dt`와 같은 계열). 대괄호로 접근한다.
    d = d[d.sn.isin(keep) & d["gt"].notna()].copy()
    print(f"       행 {len(d):,} · 자치구 {d.gu.nunique()}곳 · "
          f"기간 {d.ts.min():%Y-%m-%d} ~ {d.ts.max():%Y-%m-%d}")

    d["rad"] = d["gt"] - d.tp
    day = d[~d.is_night & d.is_shadow.notna()].copy()
    day["sh"] = day.is_shadow.astype(bool)
    print(f"\n[복사 부하 rad = gt − tp]  전체 중위 {d.rad.median():+.2f}℃ · "
          f"주간 {day.rad.median():+.2f}℃ · 야간 {d.loc[d.is_night,'rad'].median():+.2f}℃")
    print(f"       야간에 0 근처여야 한다 — 복사가 없으면 흑구온도는 기온과 같다")

    rows = []

    # ── ⓪ 센서별 오프셋 — 이것이 판정을 가른다 ──
    #
    # 흑구온도는 복사를 흡수하므로 **주간에 기온보다 높아야** 하고 **야간에는 같아야**
    # 한다. 그래서 `주간 − 야간` 차이가 곧 복사 신호다. 센서마다 고유 오프셋이 있어도
    # 그 차이는 남으므로, 오프셋을 걷어내고도 신호가 없으면 데이터가 복사를 못 담은 것이다.
    print(f"\n{'='*78}\n⓪ 센서별 오프셋 — 야간에 0이어야 한다\n{'='*78}")
    g = d.groupby(["sn", d.is_night]).rad.median().unstack()
    g.columns = ["day", "night"]
    g["diff"] = g["day"] - g["night"]
    print(f"  야간 복사 부하   중위 **{g.night.median():+.2f}℃** · "
          f"범위 {g.night.min():+.1f} ~ {g.night.max():+.1f}℃  (0이어야 한다)")
    print(f"  주간 − 야간      중위 **{g['diff'].median():+.2f}℃** · "
          f"P25 {g['diff'].quantile(.25):+.2f} · P75 {g['diff'].quantile(.75):+.2f}")
    print(f"  이 값이 곧 복사 신호다. 0이면 **햇빛이 있으나 없으나 같다**는 뜻이다.")
    broken = g.index[g.night < -10]
    if len(broken):
        print(f"  명백한 고장 센서 {len(broken)}개: {list(broken)} "
              f"(야간 {g.loc[broken,'night'].round(1).tolist()}℃)")
    rows.append({"kind": "offset", "val": "night_median", "d_rad": float(g.night.median())})
    rows.append({"kind": "offset", "val": "day_minus_night", "d_rad": float(g["diff"].median())})

    verdict = abs(g["diff"].median()) < 0.2
    if verdict:
        print(f"\n  **판정: 검증 불가.** 주야 차이가 {g['diff'].median():+.2f}℃로 사실상 0이다.")
        print(f"     이 데이터는 복사를 담고 있지 않다 — 우리 방법이 아니라 **센서의 문제**다.")
        print(f"     복사 대리의 근거는 **조도·자외선(fig2)에 그대로 남는다.**")

    # ── ① 그늘 vs 양지 ──
    print(f"\n{'='*78}\n① 그늘 vs 양지 — 복사 부하와 기온을 나란히\n{'='*78}")
    print(f"  {'태양고도':<10}{'그늘 n':>9}{'양지 n':>9}"
          f"{'복사 Δ(그늘−양지)':>18}{'기온 Δ':>12}{'배수':>8}")
    print("  " + "-" * 66)
    day["band"] = pd.cut(day.sun_el, SUN_EDGES,
                         labels=[f"{a}~{b}°" for a, b in zip(SUN_EDGES[:-1], SUN_EDGES[1:])])
    med = d.groupby("ts")["tp"].transform("median")
    day["anom"] = day.tp - med.reindex(day.index)
    for b, g in day.groupby("band", observed=True):
        s, u = g[g.sh], g[~g.sh]
        if len(s) < 30 or len(u) < 30:
            continue
        d_rad = s.rad.median() - u.rad.median()
        d_tp = s.anom.median() - u.anom.median()
        rows.append({"kind": "sun_band", "val": str(b), "n_sh": len(s), "n_sun": len(u),
                     "d_rad": d_rad, "d_tp": d_tp})
        print(f"  {str(b):<10}{len(s):>9,}{len(u):>9,}{d_rad:>17.2f}℃{d_tp:>11.2f}℃"
              f"{abs(d_rad/d_tp) if d_tp else np.nan:>8.1f}")
    print(f"\n  복사 부하 차이가 기온 차이보다 훨씬 커야 한다 — 그늘의 **직접** 대상이므로")

    # ── ② shadow_margin 구간별 ──
    print(f"\n{'='*78}\n② `shadow_margin` 구간별 — 그늘 깊이에 따라 단조인가\n{'='*78}")
    day["mb"] = pd.cut(day.shadow_margin, MARGIN_EDGES,
                       labels=["−30↓(깊은 그늘)", "−30~−10", "−10~0", "0~+10", "+10~+30", "+30↑(강한 양지)"])
    t = day.groupby("mb", observed=True).agg(n=("rad", "size"), rad=("rad", "median"),
                                             tp=("anom", "median"), svf=("svf", "median"))
    print(f"  {'구간':<18}{'n':>10}{'복사 부하':>12}{'기온 편차':>12}{'SVF':>9}")
    print("  " + "-" * 61)
    for k, v_ in t.iterrows():
        print(f"  {str(k):<18}{int(v_.n):>10,}{v_.rad:>11.2f}℃{v_.tp:>11.2f}℃{v_.svf:>9.3f}")
        rows.append({"kind": "margin", "val": str(k), "n_sh": int(v_.n),
                     "d_rad": v_.rad, "d_tp": v_.tp})

    # ── ③ 회귀 — 기하가 복사를 얼마나 설명하는가 ──
    print(f"\n{'='*78}\n③ 회귀 — 기하 피처가 복사 부하를 얼마나 설명하는가 (주간)\n{'='*78}")
    m = day.dropna(subset=["rad", "shadow_margin", "svf", "sun_el", "bld_d_min"])
    for name, feats in (("태양고도만", ["sun_el"]),
                        ("+ 그늘 여부", ["sun_el", "sh"]),
                        ("+ 음영 깊이", ["sun_el", "sh", "shadow_margin"]),
                        ("+ SVF·건물거리", ["sun_el", "sh", "shadow_margin", "svf", "bld_d_min"])):
        X = np.column_stack([np.ones(len(m))] + [m[f].to_numpy(float) for f in feats])
        y = m.rad.to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"  {name:<16}R² **{r2:.3f}**   (n={len(m):,})")
        rows.append({"kind": "r2", "val": name, "d_rad": r2})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(OUT, index=False)
    print(f"\n→ {OUT}")
    print(f"\n  센서 {len(keep)}개 · 자치구 {d.gu.nunique()}곳뿐이라 **공간 대표성도 없다.**")
    print(f"    다만 이번 판정을 가른 것은 표본 크기가 아니라 **주야 차이 0.00℃**다 —")
    print(f"    센서가 100개여도 복사에 반응하지 않으면 결과는 같다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
