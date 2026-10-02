from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core.baseline.b2_gbm import PARAMS
from core.baseline.b3_hybrid import ALL_F, BASE_F, EARLY, FEATS, N_ROUND, TREG_IX
from core.baseline.grid_infer import AZ_STEP, GRID, N_AZ, QUANTILES, sun_features
from core.eval.dataset import iter_folds, load_base, load_regional
from core.model.traj_dist import NORMAL_IQ, sample

KMA = Path("data/processing/kma_treg.parquet")
DIST = Path("data/metric/traj_dist.npz")
CALIB = Path("data/metric/calib.npz")
OUT_DIR = Path("data/metric")

TIME_F_NAMES = ["hour", "doy", "year", "sun_el", "sun_az", "is_night", "t_reg"]
N_DRAW = 30
CHUNK = 60_000
HOT_C = 33.0
SEED = 20260828


def fit_models(base, regional):
    import lightgbm as lgb
    fd = next(iter_folds(base, regional))
    X = np.zeros((len(base), len(ALL_F)), np.float32)
    X[:, [ALL_F.index(c) for c in BASE_F]] = base[BASE_F].to_numpy(np.float32)
    X[:, TREG_IX] = fd.t_reg
    y = fd.delta
    cols = [ALL_F.index(c) for c in FEATS["b3_hybrid"]]      # ← IDW 없음

    rng = np.random.default_rng(SEED)
    sn = base["sn"].to_numpy()
    val_sn = rng.choice(np.unique(sn), size=max(len(np.unique(sn)) // 5, 1), replace=False)
    m_va = np.isin(sn, val_sn)
    m_tr = ~m_va

    out = {}
    for name, obj in [("point", None)] + [(k, v) for k, v in QUANTILES.items() if k != "q50"]:
        p = dict(PARAMS) if obj is None else dict(PARAMS, objective="quantile", alpha=obj)
        dtr = lgb.Dataset(X[m_tr][:, cols], label=y[m_tr])
        dva = lgb.Dataset(X[m_va][:, cols], label=y[m_va], reference=dtr)
        t0 = time.time()
        out[name] = lgb.train(p, dtr, num_boost_round=N_ROUND, valid_sets=[dva],
                              callbacks=[lgb.early_stopping(EARLY, verbose=False)])
        print(f"  {name:<6} 트리 {out[name].best_iteration:>4} · {time.time()-t0:.0f}초")
    return out, cols


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--hours", type=int, default=0, help="0이면 예보 전체")
    a = ap.parse_args()
    from scipy import stats
    rng = np.random.default_rng(SEED)

    kma = pd.read_parquet(KMA).sort_values("ts")
    print(f"[예보] {KMA} · 발표 {kma.base.iloc[0]} · 시각 {len(kma)}개 "
          f"({kma.ts.min():%m-%d %H시} ~ {kma.ts.max():%m-%d %H시})")
    print(f"       기온 {kma.t_reg.min():.0f}~{kma.t_reg.max():.0f}℃ · "
          f"격자 간 폭 중위 {kma.spread.median():.1f}℃")

    base = load_base(BASE_F)
    regional = load_regional()
    print(f"\n추론 모델 학습 — **IDW 없음** (미래엔 이웃 관측이 없다)")
    models, cols = fit_models(base, regional)

    hzc = [f"hz_{i:03d}" for i in range(N_AZ)]
    static = [c for c in ALL_F if c not in TIME_F_NAMES + ["hz_sun", "shadow_margin",
                                                           "is_shadow"]]
    g = pd.read_parquet(GRID, columns=["gx", "gy", "in_building"] + static + hzc)
    n = len(g)
    HZ = g[hzc].to_numpy(np.float32)
    print(f"\n격자 {n:,}셀 · 정적 피처 {len(static)}개")

    z, cb = np.load(DIST), np.load(CALIB)
    R, nu = z["R"], float(z["nu"])
    edges, Qb, hot_c = cb["sigma_edges"], cb["Q_bin"], float(cb["hot_c"])
    t90 = float(stats.t.ppf(0.95, nu) * np.sqrt((nu - 2) / nu))
    fac = Qb / t90
    print(f"보정 — Mondrian conformal (폭염 × σ 구간 {Qb.shape[1]}개) · t(ν={nu:.0f})")

    want = pd.DatetimeIndex(kma.ts)
    if a.probe:
        want = want[:3]
    elif a.hours:
        want = want[:a.hours]
    sun = sun_features(want).set_index("ts")
    treg_map = pd.Series(kma.t_reg.to_numpy(), index=pd.DatetimeIndex(kma.ts))

    rows, t0 = [], time.time()
    for k, ts in enumerate(want):
        s = sun.loc[ts]
        ai = int(s.sun_az / AZ_STEP) % N_AZ
        hz_sun = HZ[:, ai].astype(np.float32)
        is_night = s.sun_el <= 0
        margin = np.full(n, np.nan, np.float32)
        is_shadow = np.full(n, np.nan, np.float32)
        if not is_night:
            margin = (s.sun_el - hz_sun).astype(np.float32)
            is_shadow = (margin < 0).astype(np.float32)

        t_reg = float(treg_map.loc[ts])
        X = np.empty((n, len(ALL_F)), np.float32)
        for c in static:
            X[:, ALL_F.index(c)] = g[c].to_numpy(np.float32)
        X[:, ALL_F.index("hour")] = ts.hour
        X[:, ALL_F.index("doy")] = ts.dayofyear
        X[:, ALL_F.index("year")] = ts.year
        X[:, ALL_F.index("sun_el")] = s.sun_el
        X[:, ALL_F.index("sun_az")] = s.sun_az
        X[:, ALL_F.index("is_night")] = float(is_night)
        X[:, ALL_F.index("hz_sun")] = hz_sun
        X[:, ALL_F.index("shadow_margin")] = margin
        X[:, ALL_F.index("is_shadow")] = is_shadow
        X[:, TREG_IX] = t_reg

        Xc = X[:, cols]
        r = {"gx": g.gx.to_numpy(), "gy": g.gy.to_numpy(), "ts": np.repeat(ts, n),
             "t_reg": np.full(n, t_reg, np.float32)}
        for name, m in models.items():
            key = "t_hat" if name == "point" else name
            r[key] = (t_reg + m.predict(Xc, num_iteration=m.num_trees())).astype(np.float32)
        rows.append(pd.DataFrame(r))
        el = time.time() - t0
        print(f"  {k+1:>2}/{len(want)} {ts:%m-%d %H시} · t_reg {t_reg:.0f}℃ · "
              f"{el:.0f}초 · 남은 {el/(k+1)*(len(want)-k-1):.0f}초", end="\r")
    print()

    df = pd.concat(rows, ignore_index=True)
    out = OUT_DIR / "forecast_grid.parquet"
    df.to_parquet(out, index=False)
    print(f"\n→ {out} ({out.stat().st_size/1e6:.0f} MB · {len(df):,}행)")

    # ── 요약 ──
    print(f"\n{'='*74}\n예보 다운스케일링 결과\n{'='*74}")
    print(f"  {'시각':<12}{'기상청 t_reg':>13}{'우리 중위':>11}{'셀 P05~P95':>16}"
          f"{'골목 격차':>11}")
    print("  " + "-" * 63)
    for ts, gr in df.groupby("ts"):
        lo, hi = np.percentile(gr.t_hat, [5, 95])
        print(f"  {ts:%m-%d %H시}  {gr.t_reg.iloc[0]:>13.0f}{gr.t_hat.median():>11.2f}"
              f"{f'{lo:.1f}~{hi:.1f}':>16}{hi-lo:>11.2f}")

    spread = df.groupby("ts").t_hat.agg(lambda s: np.percentile(s, 95) - np.percentile(s, 5))
    print(f"\n  골목 격차(P95−P05) 중위 **{spread.median():.2f}℃** · "
          f"최대 {spread.max():.2f}℃")
    print(f"  기상청 5km 격자 간 폭은 {kma.spread.median():.1f}℃였다 — "
          f"**같은 시각 25m 안에서 {spread.median():.2f}℃가 더 나온다**")

    # 보정 척도 — 예보는 폭염 아님(t_reg < 33)이라 평시 축을 탄다
    sg_raw = ((df.q90 - df.q10) / NORMAL_IQ).to_numpy()
    hot = (df.t_reg.to_numpy() >= hot_c).astype(int)
    b = np.clip(np.searchsorted(edges, sg_raw, side="right") - 1, 0, Qb.shape[1] - 1)
    sg_cal = sg_raw * fac[hot, b]
    print(f"\n  보정 척도 중위 {np.nanmedian(sg_cal):.3f}℃ "
          f"(원 {np.nanmedian(sg_raw):.3f}℃ · 배수 {np.median(fac[hot, b]):.2f}) · "
          f"폭염 시각 {int(hot.sum()/n)}/{len(want)}")
    print(f"\n  정확도는 아직 모른다 — 9/5 Phase 2.5에서 S-DoT 관측이 쌓이면 대조한다.")
    print(f"     이 실행이 증명하는 것은 **파이프라인이 예보 입력으로 끝까지 돈다**는 것이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
