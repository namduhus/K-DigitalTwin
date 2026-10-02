from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from core.eval.dataset import (
    N_FOLDS,
    iter_folds,
    load_base,
    load_regional,
    save_preds,
)

# `t_reg`(그 시각 광역 기준값)를 시간 피처에 넣는다. 폭염 시각에 편차가 커진다는 것을
# 이미 안다(P95-P05 3.19 → 3.77℃). 그 정보를 주지 않으면 모델이 "더울 때 공간 차이가
# 벌어진다"를 학습할 수 없다. fold마다 값이 달라 X의 마지막 열에 매번 덮어쓴다.
TIME_F = ["hour", "doy", "year", "sun_el", "sun_az", "is_night", "t_reg"]
BLD_F = ["bld_n", "bld_ar_frac", "bld_h_max", "bld_h_mean", "bld_d_min"]
TERR_F = ["elev", "slope", "aspect", "th_mean", "th_max", "terr_dominant",
          "grn_n", "grn_frac", "grn_ar_max", "grn_d_min", "pave_n", "pave_frac"]
SHADE_F = ["svf", "svf_bld", "svf_terr", "hz_mean", "hz_max", "hz_sun",
           "shadow_margin", "is_shadow"]

TIERS = {
    "a1_time":   TIME_F,
    "a2_bld":    TIME_F + BLD_F,
    "a3_terr":   TIME_F + BLD_F + TERR_F,
    "a4_shade":  TIME_F + BLD_F + TERR_F + SHADE_F,   # = B2 본체
}
ALL_F = TIERS["a4_shade"]
BASE_F = [c for c in ALL_F if c != "t_reg"]      # train.parquet에서 읽는 것
T_REG_IX = ALL_F.index("t_reg")

# 네이티브 API를 쓴다 — `lightgbm.sklearn`은 scikit-learn을 요구하는데 의존성을 하나
# 더 지을 이유가 없다. objective="l1"은 평가 지표가 MAE이므로 목적함수도 맞춘 것이다.
# l2로 학습하면 이상값(42℃대 기온·열대야 포화 — 신호라 남겨둔 것들)에 과하게 끌린다.
# 파라미터는 실측으로 골랐다 (fold 0, 296만 행, 50트리 기준)
#   l1 leaves=127 lr=0.05  80ms/트리  MAE 0.6445
#   l1 leaves= 63 lr=0.10  58ms/트리  MAE 0.6405   ← 더 빠르고 더 정확
#   l2 leaves=127 lr=0.05  73ms/트리  MAE 0.6500
#   l2 leaves= 63 lr=0.10  48ms/트리  MAE 0.6457
# l1이 l2보다 낫고 평가 지표(MAE)와도 일치한다. 20회 학습 × 800라운드 ≈ 12분.
PARAMS = dict(objective="l1", metric="l1", num_leaves=63, learning_rate=0.1,
              max_bin=127, min_data_in_leaf=200, bagging_fraction=0.8, bagging_freq=1,
              feature_fraction=0.8, num_threads=0, verbose=-1, seed=20260821)
N_ROUND, EARLY = 800, 30


def fit_predict(tier: str, feats: list[str], fd, base, X, sensor_fold) -> tuple[np.ndarray, dict]:
    import lightgbm as lgb

    inner_val_sn = np.where(sensor_fold == (fd.f + 1) % N_FOLDS)[0]
    sn_ix = base["_sn_ix"].to_numpy()
    m_test = fd.is_test
    m_val = np.isin(sn_ix, inner_val_sn) & ~m_test
    m_tr = ~m_test & ~m_val

    cols = [ALL_F.index(c) for c in feats]
    dtr = lgb.Dataset(X[m_tr][:, cols], label=fd.delta[m_tr], feature_name=feats)
    dva = lgb.Dataset(X[m_val][:, cols], label=fd.delta[m_val], reference=dtr)
    bst = lgb.train(PARAMS, dtr, num_boost_round=N_ROUND, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(EARLY, verbose=False)])
    pred = bst.predict(X[m_test][:, cols], num_iteration=bst.best_iteration)
    # gain 중요도를 쓴다 — split 횟수는 카디널리티 높은 피처를 과대평가한다
    info = {"n_tree": bst.best_iteration, "n_tr": int(m_tr.sum()), "n_val": int(m_val.sum()),
            "imp": pd.Series(bst.feature_importance("gain"), index=feats)}
    return pred.astype(np.float32), info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="fold 0 · A4만")
    a = ap.parse_args()

    base = load_base(BASE_F)
    regional = load_regional()

    sns = np.array(sorted(base["sn"].unique()))
    base["_sn_ix"] = base["sn"].map({s: i for i, s in enumerate(sns)})
    sensor_fold = base.groupby("sn")["fold"].first().reindex(sns).to_numpy()

    # 결측 확인 — LightGBM은 NaN을 처리하지만 어느 피처가 비었는지는 알아야 한다
    # `shadow_margin`·`is_shadow`는 야간에 정의되지 않는다 (태양이 없다). 구조적
    # 결측이므로 채우지 않고 LightGBM의 NaN 처리에 맡긴다 — 0으로 채우면
    # "태양 고도와 장애물 앙각이 같다"는 뜻이 되어 낮의 경계 상황과 섞인다.
    na = base[BASE_F].isna().mean()
    night = float(base["is_night"].mean())
    if (na > 0).any():
        print("\n피처 결측률 (0 초과만):")
        for c, v in na[na > 0].sort_values(ascending=False).items():
            tag = "  ← 야간 비율과 일치 = 구조적 결측" if abs(v - night) < 0.005 else ""
            print(f"  {c:<16}{v*100:>6.2f}%{tag}")
        print(f"  (야간 비율 {night*100:.2f}%)")

    X = np.empty((len(base), len(ALL_F)), np.float32)
    X[:, [ALL_F.index(c) for c in BASE_F]] = base[BASE_F].to_numpy(np.float32)
    print(f"\n피처 {len(ALL_F)}개 · 행 {len(base):,} · {X.nbytes/1e6:.0f} MB")
    for t, f in TIERS.items():
        print(f"  {t:<10}{len(f):>3}개")

    tiers = {"a4_shade": TIERS["a4_shade"]} if a.probe else TIERS
    folds = [0] if a.probe else list(range(N_FOLDS))

    preds = {t: np.full(len(base), np.nan, np.float32) for t in tiers}
    imps: dict[str, list] = {t: [] for t in tiers}

    print(f"\n{'tier':<11}{'fold':<6}{'학습행':>11}{'내부검증':>10}{'트리':>7}{'내부MAE':>9}{'초':>7}")
    print("-" * 62)
    for fd in iter_folds(base, regional):
        if fd.f not in folds:
            continue
        X[:, T_REG_IX] = fd.t_reg          # fold마다 광역 기준값이 다르다
        for t, feats in tiers.items():
            t0 = time.time()
            pr, info = fit_predict(t, feats, fd, base, X, sensor_fold)
            preds[t][fd.is_test] = fd.t_reg[fd.is_test] + pr
            imps[t].append(info["imp"])
            truth = fd.delta[fd.is_test]
            mae = float(np.abs(pr - truth).mean())
            print(f"{t:<11}{fd.f:<6}{info['n_tr']:>11,}{info['n_val']:>10,}"
                  f"{info['n_tree']:>7}{mae:>9.4f}{time.time()-t0:>7.1f}")

    print()
    y_true = base["tp"].to_numpy(np.float32)
    m = base["fold"].to_numpy() >= 0
    for t in tiers:
        name = "b2_gbm" if t == "a4_shade" else f"b2_{t}"
        p = save_preds(name, base["sn"], base["ts"], preds[t])
        ok = m & ~np.isnan(preds[t])
        print(f"→ {p}   전체 MAE(참고) {np.abs(preds[t][ok] - y_true[ok]).mean():.3f}℃")

    if not a.probe:
        imp = pd.concat(imps["a4_shade"], axis=1).mean(axis=1).sort_values(ascending=False)
        print(f"\nA4 피처 중요도 (5-fold 평균, 상위 15)")
        for c, v in imp.head(15).items():
            grp = ("음영·SVF" if c in SHADE_F else "DEM·녹지" if c in TERR_F
                   else "건물밀도" if c in BLD_F else "시간")
            print(f"  {c:<16}{v:>9.0f}   {grp}")
        by = imp.groupby(lambda c: "음영·SVF" if c in SHADE_F else "DEM·녹지" if c in TERR_F
                         else "건물밀도" if c in BLD_F else "시간").sum()
        print(f"\n  그룹별 합계 (비율)")
        for g, v in by.sort_values(ascending=False).items():
            print(f"    {g:<10}{v:>10.0f}   {v/by.sum()*100:>5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
