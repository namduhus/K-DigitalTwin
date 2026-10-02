from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core.baseline.b1_interp import idw_oof, sensor_grid
from core.baseline.b2_gbm import BLD_F, PARAMS, SHADE_F, TERR_F, TIME_F
from core.baseline.b3_hybrid import ALL_F, BASE_F, FEATS, IDW_IX, IDW_K, IDW_P, TREG_IX
from core.baseline.counterfactual import SCENARIOS, apply_scenario
from core.eval.dataset import (
    iter_folds,
    load_base,
    load_regional,
    load_sensor_meta,
    trusted_sensors,
)

GRID = Path("data/processing/grid_features.parquet")
OUT_DIR = Path("data/metric")

# `build_train.py:43`과 동일 — 서울 중심
LAT, LON, TZ = 37.5665, 126.9780, "Asia/Seoul"
AZ_STEP, N_AZ = 2.0, 180
N_ROUND, EARLY = 800, 100

# 실측으로 고른 대표일 (00:00 ~ 다음날 09:00 = 34시각, 열대야 구간 포함)
DAYS = {"2026-08-06": "폭염일", "2026-07-15": "평온일"}
HOURS_AHEAD = 34

QUANTILES = {"q10": 0.10, "q50": 0.50, "q90": 0.90}
CF_SCENARIO = "소공원 (복합)"      # 처방 지도용


def sun_features(ts: pd.DatetimeIndex) -> pd.DataFrame:
    import pvlib
    idx = ts.tz_localize(TZ, nonexistent="shift_forward")
    sp = pvlib.solarposition.spa_python(idx, LAT, LON)
    return pd.DataFrame({
        "ts": idx.tz_localize(None),
        "sun_el": sp["apparent_elevation"].to_numpy(),   # 대기굴절 보정 겉보기 고도
        "sun_az": sp["azimuth"].to_numpy(),
    })


def fit_inference_models(base, regional, D, sensor_fold, sn_col, ts_row, n_ts, untrusted):
    import lightgbm as lgb
    fd = next(iter_folds(base, regional))          # t_reg는 fold 0 기준 = 학습센서 중위
    X = np.zeros((len(base), len(ALL_F) + 1), np.float32)
    X[:, [ALL_F.index(c) for c in BASE_F]] = base[BASE_F].to_numpy(np.float32)
    X[:, TREG_IX] = fd.t_reg
    X[:, IDW_IX] = idw_oof(fd, base, D, np.full_like(sensor_fold, -1), sn_col, ts_row,
                           n_ts, untrusted, IDW_P, IDW_K)
    y = fd.delta
    cols = [ALL_F.index(c) for c in FEATS["b3_hybrid"]] + [IDW_IX]

    # 조기중단용 내부 검증 — 센서 20%를 떼어 쓴다 (추론 모델이므로 최종 학습에 다시 포함)
    rng = np.random.default_rng(20260821)
    val_sn = rng.choice(np.unique(sn_col), size=max(len(np.unique(sn_col)) // 5, 1),
                        replace=False)
    m_val = np.isin(sn_col, val_sn)

    models, meta = {}, {}
    for name, obj in [("point", None)] + [(k, v) for k, v in QUANTILES.items()]:
        p = dict(PARAMS)
        if obj is not None:
            p.update(objective="quantile", alpha=obj, metric="quantile")
        dtr = lgb.Dataset(X[~m_val][:, cols], label=y[~m_val])
        dva = lgb.Dataset(X[m_val][:, cols], label=y[m_val], reference=dtr)
        bst = lgb.train(p, dtr, num_boost_round=N_ROUND, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(EARLY, verbose=False)])
        n_best = bst.best_iteration
        # 최종 모델은 전체 센서로 다시 학습 (트리 수는 위에서 찾은 값 고정)
        full = lgb.train(p, lgb.Dataset(X[:, cols], label=y), num_boost_round=n_best)
        models[name] = full
        meta[name] = n_best
        print(f"  {name:<6} 트리 {n_best:>4} (내부검증 센서 {len(val_sn)}개로 결정)")
    return models, cols, fd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="폭염일 14시 한 시각만")
    ap.add_argument("--days", nargs="*", default=None, help="기본: 2026-08-06 2026-07-15")
    a = ap.parse_args()

    t00 = time.time()
    base = load_base(BASE_F)
    regional = load_regional()
    meta_sen = load_sensor_meta()
    sns, tss, sn_col, ts_row = sensor_grid(base)
    xy = meta_sen.loc[sns, ["x", "y"]].to_numpy(np.float64)
    D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    sensor_fold = base.groupby("sn")["fold"].first().reindex(sns).to_numpy()
    trust = set(trusted_sensors(base))
    untrusted = np.array([i for i, s in enumerate(sns) if s not in trust])

    print(f"\n추론용 모델 학습 — 전체 {len(sns):,}센서 (평가용 5-fold와 별개)")
    models, cols, fd0 = fit_inference_models(base, regional, D, sensor_fold, sn_col,
                                             ts_row, len(tss), untrusted)

    # ── 격자 로드 ────────────────────────────────────────────────
    hzc = [f"hz_{i:03d}" for i in range(N_AZ)]
    static = [c for c in ALL_F if c not in TIME_F + ["hz_sun", "shadow_margin", "is_shadow"]]
    g = pd.read_parquet(GRID, columns=["gx", "gy", "x", "y", "gu", "in_building"]
                        + static + hzc)
    n = len(g)
    HZ = g[hzc].to_numpy(np.float32)
    print(f"\n격자 {n:,}셀 · 정적 피처 {len(static)}개 · 수평선 {N_AZ}방위")

    # 격자 IDW 가중치 — 기하가 고정이라 한 번만 만든다
    from scipy.spatial import cKDTree
    t0 = time.time()
    tr_ix = np.setdiff1d(np.arange(len(sns)), untrusted)
    kt = cKDTree(xy[tr_ix])
    dd, ii = kt.query(g[["x", "y"]].to_numpy(np.float64), k=IDW_K, workers=-1)
    W = (1.0 / np.maximum(dd, 1.0) ** IDW_P).astype(np.float32)
    print(f"  IDW 가중치 {n:,}셀 × k={IDW_K} ({W.nbytes/1e6:.0f} MB) · {time.time()-t0:.1f}초")

    # 센서 편차 행렬 (시각 × 센서) — 전체 센서 기준
    M = np.full((len(tss), len(sns)), np.nan, np.float32)
    M[ts_row, sn_col] = fd0.delta
    t_reg_all = pd.Series(fd0.t_reg, index=base["ts"]).groupby(level=0).first()

    days = a.days or list(DAYS)
    for day in days:
        # 시각을 만들지 말고 **실제 관측 시각에서 고른다.** S-DoT은 정시가 아니다
        # (예: 최고 시각이 `2026-08-06 16:07`). `d0 + h시간`으로 만들면 교집합이 비어버린다.
        d0 = pd.Timestamp(day)
        ts_all = pd.DatetimeIndex(tss)
        want = ts_all[(ts_all >= d0) & (ts_all < d0 + pd.Timedelta(hours=HOURS_AHEAD))]
        if a.probe:
            want = want[want.hour == 14][:1]
        sun = sun_features(want).set_index("ts")
        print(f"\n[{day} {DAYS.get(day,'')}] {len(want)}시각 · "
              f"태양고도 {sun.sun_el.min():.1f}~{sun.sun_el.max():.1f}°")

        rows, t0 = [], time.time()
        for k, ts in enumerate(want):
            s = sun.loc[ts]
            # ── 시각 피처 (build_train.py와 동일 정의) ──
            ai = int(s.sun_az / AZ_STEP) % N_AZ
            hz_sun = HZ[:, ai].astype(np.float32)
            is_night = s.sun_el <= 0
            margin = np.full(n, np.nan, np.float32)
            is_shadow = np.full(n, np.nan, np.float32)
            if not is_night:
                margin = (s.sun_el - hz_sun).astype(np.float32)
                is_shadow = (margin < 0).astype(np.float32)

            # ── IDW ──
            col = np.searchsorted(tss, ts)
            dv = M[col, tr_ix]
            ok = ~np.isnan(dv)
            num = np.nansum(np.where(ok[ii], dv[ii], 0.0) * W, axis=1)
            den = (ok[ii] * W).sum(axis=1)
            idw = np.where(den > 0, num / den, 0.0).astype(np.float32)

            # ── 피처 행렬 조립 ──
            X = np.empty((n, len(ALL_F) + 1), np.float32)
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
            X[:, TREG_IX] = t_reg = float(t_reg_all.loc[ts])
            X[:, IDW_IX] = idw

            r = {"gx": g.gx.to_numpy(), "gy": g.gy.to_numpy(),
                 "ts": np.repeat(ts, n)}
            for name, m in models.items():
                key = "t_hat" if name == "point" else name
                r[key] = (t_reg + m.predict(X[:, cols], num_iteration=m.num_trees())
                          ).astype(np.float32)
            # 처방 — 소공원 시나리오 치환 재예측
            Xc = apply_scenario(X, SCENARIOS[CF_SCENARIO])
            r["cf_park"] = (models["point"].predict(Xc[:, cols],
                            num_iteration=models["point"].num_trees()).astype(np.float32)
                            - (r["t_hat"] - t_reg))
            rows.append(pd.DataFrame(r))
            el = time.time() - t0
            print(f"  {k+1:>2}/{len(want)} {ts:%m-%d %H시} · t_reg {t_reg:.1f}℃ · "
                  f"{el:.0f}초 · 남은 {el/(k+1)*(len(want)-k-1):.0f}초", end="\r")
        print()

        df = pd.concat(rows, ignore_index=True)
        out = OUT_DIR / f"grid_pred_{day}.parquet"
        df.to_parquet(out, index=False)
        print(f"  → {out} ({out.stat().st_size/1e6:.0f} MB, {len(df):,}행)")
        print(f"    t_hat 중위 {df.t_hat.median():.2f}℃ · 범위 {df.t_hat.min():.1f}~{df.t_hat.max():.1f}")
        # q10~q90은 **80% 구간**이다 (90%가 아니다 — 8/26 정정).
        # 게다가 이 이름값 80%의 hold-out 실제 적중률이 **60.3%**다. 보정 전 값이므로
        # 신청서 인용은 conformal 보정 후 값으로 한다. `record.md`(8/26) 참조.
        print(f"    80% 구간 폭 중위 {(df.q90-df.q10).median():.3f}℃ · "
              f"P95 {(df.q90-df.q10).quantile(.95):.3f}℃  보정 전")
        print(f"    처방({CF_SCENARIO}) 효과 중위 {df.cf_park.median():+.3f}℃ · "
              f"P05 {df.cf_park.quantile(.05):+.3f}℃")

    print(f"\n총 {(time.time()-t00)/60:.1f}분")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
