from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from core.eval.dataset import PREDS, load_base

OUT = Path("data/metric/trajectory.parquet")

TROPICAL_C = 25.0       # 열대야 기준
HOT_C = 33.0            # 폭염 기준 (근거 수치와 같은 임계값)
NIGHT_START_H = 18      # 18:01~다음날 09:00
NIGHT_LEN_H = 15
MIN_NIGHT_OBS = 12
MIN_DAY_OBS = 20


def keys(ts: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    shifted = ts - pd.Timedelta(hours=NIGHT_START_H)
    night_key = shifted.dt.floor("D")
    in_night = shifted.dt.hour < NIGHT_LEN_H
    return night_key, ts.dt.floor("D"), in_night


def derive(df: pd.DataFrame, col: str) -> pd.DataFrame:
    n = df[df["in_night"]].groupby(["sn", "night_key"], observed=True).agg(
        obs=(col, "size"), tropical_h=(col, lambda s: int((s >= TROPICAL_C).sum())))
    n = n[n["obs"] >= MIN_NIGHT_OBS].drop(columns="obs")

    d = df.groupby(["sn", "day_key"], observed=True).agg(
        obs=(col, "size"),
        hot_h=(col, lambda s: int((s >= HOT_C).sum())),
        peak_h=(col, "idxmax"))
    d = d[d["obs"] >= MIN_DAY_OBS].drop(columns="obs")
    d["peak_h"] = df.loc[d["peak_h"], "hour"].to_numpy()
    return n, d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    a = ap.parse_args()

    files = ([PREDS / f"{m}.parquet" for m in a.models] if a.models
             else sorted(PREDS.glob("*.parquet")))
    files = [f for f in files if f.exists()]
    if not files:
        sys.exit(f"[error] {PREDS} 에 예측이 없다")

    base = load_base(["hour"])
    base["night_key"], base["day_key"], base["in_night"] = keys(base["ts"])
    base = base[base["fold"] >= 0].reset_index(drop=True)      # always-train은 예측이 없다

    truth_n, truth_d = derive(base, "tp")
    print(f"실측 궤적 — 열대야 구간 {len(truth_n):,}개 (≥{MIN_NIGHT_OBS}h 관측) · "
          f"달력일 {len(truth_d):,}개 (≥{MIN_DAY_OBS}h)")
    print(f"  열대야 지속시간: 중위 {truth_n['tropical_h'].median():.0f}h · "
          f"0h {(truth_n['tropical_h']==0).mean()*100:.0f}% · "
          f"15h(밤새) {(truth_n['tropical_h']==NIGHT_LEN_H).mean()*100:.0f}%")
    print(f"  33℃ 초과 시간:  중위 {truth_d['hot_h'].median():.0f}h · "
          f"0h {(truth_d['hot_h']==0).mean()*100:.0f}% · 최대 {truth_d['hot_h'].max()}h")
    print(f"  일최고 발생시각: 중위 {truth_d['peak_h'].median():.0f}시 · "
          f"IQR {truth_d['peak_h'].quantile(.25):.0f}~{truth_d['peak_h'].quantile(.75):.0f}시")

    key = pd.MultiIndex.from_arrays([base["sn"], base["ts"]])
    rows = []
    for f in files:
        p = pd.read_parquet(f)
        s = pd.Series(p["y_pred"].to_numpy(np.float32),
                      index=pd.MultiIndex.from_arrays([p["sn"].astype(str), p["ts"]]))
        base["yp"] = s.reindex(key).to_numpy(np.float32)
        if base["yp"].isna().all():
            continue
        pn, pd_ = derive(base.dropna(subset=["yp"]), "yp")

        jn = truth_n.join(pn, how="inner", lsuffix="_t", rsuffix="_p")
        jd = truth_d.join(pd_, how="inner", lsuffix="_t", rsuffix="_p")
        e_trop = (jn["tropical_h_p"] - jn["tropical_h_t"]).to_numpy(float)
        e_hot = (jd["hot_h_p"] - jd["hot_h_t"]).to_numpy(float)
        e_peak = (jd["peak_h_p"] - jd["peak_h_t"]).to_numpy(float)
        # 열대야 발생 여부(이진) 정확도 — 지속시간보다 정책이 먼저 보는 값
        yes_t, yes_p = jn["tropical_h_t"] > 0, jn["tropical_h_p"] > 0
        rows.append({
            "model": f.stem,
            "n_night": len(jn), "n_day": len(jd),
            "trop_mae": np.abs(e_trop).mean(), "trop_bias": e_trop.mean(),
            "trop_acc": float((yes_t == yes_p).mean()),
            "hot_mae": np.abs(e_hot).mean(), "hot_bias": e_hot.mean(),
            "peak_mae": np.abs(e_peak).mean(),
        })

    r = pd.DataFrame(rows).set_index("model")
    print(f"\n{'='*74}\n궤적 파생량 오차 (MAE, 단위 = 시간)\n{'='*74}")
    print(f"  {'model':<14}{'열대야 지속':>11}{'편향':>8}{'열대야 판정':>11}"
          f"{'33℃ 초과':>10}{'편향':>8}{'일최고 시각':>11}")
    print("  " + "-" * 71)
    for m, v in r.iterrows():
        print(f"  {m:<14}{v['trop_mae']:>11.3f}{v['trop_bias']:>+8.2f}"
              f"{v['trop_acc']*100:>10.1f}%{v['hot_mae']:>10.3f}{v['hot_bias']:>+8.2f}"
              f"{v['peak_mae']:>11.2f}")

    if {"b3_noshade", "b3_hybrid"} <= set(r.index):
        A, B = r.loc["b3_noshade"], r.loc["b3_hybrid"]
        print(f"\n음영·SVF 기여 (b3_noshade → b3_hybrid)")
        for k, lab in [("trop_mae", "열대야 지속시간"), ("hot_mae", "33℃ 초과 시간"),
                       ("peak_mae", "일최고 발생시각")]:
            d = A[k] - B[k]
            print(f"  {lab:<16}{A[k]:.3f} → {B[k]:.3f}  = {d:+.4f}h ({d/A[k]*100:+.2f}%)")
        d = B["trop_acc"] - A["trop_acc"]
        print(f"  {'열대야 판정 정확도':<16}{A['trop_acc']*100:.2f}% → {B['trop_acc']*100:.2f}%"
              f"  = {d*100:+.3f}pp")
        print(f"\n  비교 — 시각별 MAE 기여는 +1.11% (전체) / +2.20% (깊은 그늘)이었다")

    if "b1_idw" in r.index and "b3_hybrid" in r.index:
        A, B = r.loc["b1_idw"], r.loc["b3_hybrid"]
        print(f"\nB3의 B1 대비 개선 (공간 피처 전체)")
        for k, lab in [("trop_mae", "열대야 지속시간"), ("hot_mae", "33℃ 초과 시간"),
                       ("peak_mae", "일최고 발생시각")]:
            print(f"  {lab:<16}{A[k]:.3f} → {B[k]:.3f}  = {(A[k]-B[k])/A[k]*100:+.2f}%")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    r.to_parquet(OUT)
    print(f"\n→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
