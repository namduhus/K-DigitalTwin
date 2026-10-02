from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from core.baseline.b1_interp import idw_oof, sensor_grid
from core.baseline.b2_gbm import BLD_F, PARAMS, SHADE_F, TERR_F, TIME_F
from core.eval.dataset import (
    N_FOLDS,
    iter_folds,
    load_base,
    load_regional,
    load_sensor_meta,
    save_preds,
    trusted_sensors,
)

IDW_P, IDW_K = 1.0, 50          # B1 중첩 튜닝 결과 (5 fold 중 4개)
N_ROUND, EARLY = 800, 100       # EARLY를 100으로 — B2에서 트리 수가 fold별 60~798로 흔들렸다

# 주야 분리 학습은 fold 0 실측에서 통합보다 **나빴다** (0.5315 vs 0.5300).
# 표본을 반으로 쪼개는 손실이 부호 반전 이득보다 크다. 그래서 통합 학습을 본선으로
# 하고 `is_night`을 피처로 준다 — GBM이 상호작용을 직접 배운다. 주야별 기여는
# 학습을 나누지 않고 **평가 슬라이스**로 본다 (`core/metrics/evaluate.py`).
# 분리 학습은 그 판단의 근거로 남겨 함께 돌린다.
STATIC = BLD_F + TERR_F
FEATS = {
    "b3_hybrid":  TIME_F + STATIC + SHADE_F,      # 통합 · 전체 피처 ← 본선
    "b3_noshade": TIME_F + STATIC,                # 통합 · 음영·SVF 제외 ← 절제
    "b3_split":   TIME_F + STATIC + SHADE_F,      # 주야 분리 ← 분리가 이득인지 확인
}
ALL_F = TIME_F + STATIC + SHADE_F                # `t_reg`가 TIME_F에 포함돼 있다
BASE_F = [c for c in ALL_F if c != "t_reg"]      # train.parquet에서 읽는 것
TREG_IX = ALL_F.index("t_reg")                   # fold마다 덮어쓴다
IDW_IX = len(ALL_F)                              # X 맨 뒤에 붙이는 한 열


def fit(cols, X, y, m_tr, m_va, m_te):
    import lightgbm as lgb
    dtr = lgb.Dataset(X[m_tr][:, cols], label=y[m_tr], feature_name=[str(c) for c in cols])
    dva = lgb.Dataset(X[m_va][:, cols], label=y[m_va], reference=dtr)
    bst = lgb.train(PARAMS, dtr, num_boost_round=N_ROUND, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(EARLY, verbose=False)])
    return bst.predict(X[m_te][:, cols], num_iteration=bst.best_iteration), bst.best_iteration


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()

    base = load_base(BASE_F)
    regional = load_regional()
    meta = load_sensor_meta()

    sns, tss, sn_col, ts_row = sensor_grid(base)
    xy = meta.loc[sns, ["x", "y"]].to_numpy(np.float64)
    D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    sensor_fold = base.groupby("sn")["fold"].first().reindex(sns).to_numpy()
    trust = set(trusted_sensors(base))
    untrusted = np.array([i for i, s in enumerate(sns) if s not in trust])
    night = base["is_night"].to_numpy().astype(bool)
    sn_ix = sn_col

    # X = [정적·시간 피처] + [idw] + [t_reg].  뒤 두 열은 fold마다 덮어쓴다
    # `t_reg`와 `idw`는 fold마다 값이 달라 매 fold에 덮어쓴다. `np.zeros`를 쓴다 —
    # `np.empty`로 두면 fold 루프 전에 읽히는 열이 초기화되지 않은 채 남을 수 있다.
    X = np.zeros((len(base), len(ALL_F) + 1), np.float32)
    X[:, [ALL_F.index(c) for c in BASE_F]] = base[BASE_F].to_numpy(np.float32)

    print(f"\n행 {len(base):,} · 피처 {len(ALL_F)+1}개 · {X.nbytes/1e6:.0f} MB")
    print(f"  주간 {int((~night).sum()):,} · 야간 {int(night.sum()):,} ({night.mean()*100:.1f}%)")
    print(f"  IDW: p={IDW_P:.0f}, k={IDW_K} (B1 중첩 튜닝 결과) · 미판정 센서 {len(untrusted)}개 이웃 제외")

    models = ["b3_hybrid", "b3_noshade", "b3_split"]
    preds = {m: np.full(len(base), np.nan, np.float32) for m in models}
    folds = [0] if a.probe else list(range(N_FOLDS))
    rows = []

    print(f"\n{'fold':<5}{'모델':<12}{'구간':<6}{'학습':>10}{'트리':>6}{'MAE':>9}{'초':>6}")
    print("-" * 56)
    for fd in iter_folds(base, regional):
        if fd.f not in folds:
            continue
        X[:, IDW_IX] = idw_oof(fd, base, D, sensor_fold, sn_col, ts_row,
                               len(tss), untrusted, IDW_P, IDW_K)
        X[:, TREG_IX] = fd.t_reg
        y = fd.delta

        inner_val_sn = np.where(sensor_fold == (fd.f + 1) % N_FOLDS)[0]
        m_te0 = fd.is_test
        m_va0 = np.isin(sn_ix, inner_val_sn) & ~m_te0
        m_tr0 = ~m_te0 & ~m_va0

        for name in models:
            split = ([("주간", ~night), ("야간", night)] if name == "b3_split"
                     else [("통합", np.ones(len(base), bool))])
            cols = [ALL_F.index(c) for c in FEATS[name]] + [IDW_IX]
            for lab, seg in split:
                t0 = time.time()
                pr, nt = fit(cols, X, y, m_tr0 & seg, m_va0 & seg, m_te0 & seg)
                m = m_te0 & seg
                preds[name][m] = fd.t_reg[m] + pr
                mae = float(np.abs(pr - y[m]).mean())
                rows.append((fd.f, name, lab, mae))
                print(f"{fd.f:<5}{name:<12}{lab:<6}{int((m_tr0&seg).sum()):>10,}"
                      f"{nt:>6}{mae:>9.4f}{time.time()-t0:>6.1f}")

    print()
    y_true = base["tp"].to_numpy(np.float32)
    m_all = base["fold"].to_numpy() >= 0
    for name in models:
        p = save_preds(name, base["sn"], base["ts"], preds[name])
        ok = m_all & ~np.isnan(preds[name])
        print(f"→ {p}   전체 MAE(참고) {np.abs(preds[name][ok]-y_true[ok]).mean():.4f}℃")

    r = pd.DataFrame(rows, columns=["fold", "model", "seg", "mae"])
    print(f"\nfold 평균 ± 표준편차")
    piv = r.groupby(["model", "seg"])["mae"].agg(["mean", "std"])
    print(piv.round(4).to_string())

    if len(folds) == N_FOLDS:
        # 행 수 가중으로 비교한다. 주야 MAE를 단순 평균하면 틀린다 —
        # 주간이 야간보다 행이 33% 많다 (2,082,253 vs 1,564,067).
        ok = {n: m_all & ~np.isnan(preds[n]) for n in ("b3_hybrid", "b3_split")}
        agg = {n: float(np.abs(preds[n][k] - y_true[k]).mean()) for n, k in ok.items()}
        hy, sp = agg["b3_hybrid"], agg["b3_split"]
        print(f"\n주야 분리가 이득인가 (행 수 가중): 통합 {hy:.4f} vs 분리 {sp:.4f} "
              f"→ {'분리가 유리' if sp < hy else '**통합이 유리**'} ({sp-hy:+.4f}℃)")
        un = r[r.model == "b3_split"].groupby("seg")["mae"].mean()
        print(f"  (구간 단순평균 {un.mean():.4f}는 쓰지 않는다 — 주간 행이 33% 많아 왜곡된다)")
        print(f"  조건별 음영·SVF 기여는 `core/metrics/evaluate.py`의 주/야 슬라이스로 본다:")
        print(f"    uv run python -m core.metrics.evaluate --models b1_idw b3_noshade b3_hybrid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
