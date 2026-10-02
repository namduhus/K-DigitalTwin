from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core.baseline.b1_interp import idw_weights, pivot, predict, sensor_grid
from core.baseline.b2_gbm import PARAMS
from core.baseline.b3_hybrid import (
    ALL_F, BASE_F, EARLY, FEATS, IDW_K, IDW_P, N_ROUND, TREG_IX,
)
from core.eval.dataset import (
    N_FOLDS, iter_folds, load_base, load_regional, load_sensor_meta, trusted_sensors,
)

OUT = Path("data/metric/probe_stidw.parquet")

# 새로 붙이는 열 이름 — X의 뒤쪽에 순서대로 쌓는다
EXTRA = ["idw", "idw_lag1", "idw_lag2", "idw_lag24", "idw_d1", "treg_d1", "treg_d3"]

# 변인 하나씩 분리해서 본다. 전부 넣은 것만 보면 어느 것이 벌었는지 모른다.
VARIANTS = {
    "V0 기준 (현재 B3)":        ["idw"],
    "V1 +이웃 과거 1·2h":       ["idw", "idw_lag1", "idw_lag2", "idw_d1"],
    "V2 +앵커 변화율":          ["idw", "treg_d1", "treg_d3"],
    "V3 +이웃 과거 24h":        ["idw", "idw_lag1", "idw_lag2", "idw_lag24", "idw_d1"],
    "V4 전부":                  EXTRA,
}


def lag_index(tss: np.ndarray, hours: int) -> np.ndarray:
    idx = pd.DatetimeIndex(tss)
    pos = pd.Series(np.arange(len(idx)), index=idx)
    src = pos.reindex(idx - pd.Timedelta(hours=hours)).to_numpy(dtype=np.float64)
    return np.where(np.isnan(src), -1, np.nan_to_num(src)).astype(np.int64)


def lag_matrix(PR: np.ndarray, tss: np.ndarray, hours: int) -> np.ndarray:
    src = lag_index(tss, hours)
    out = np.full_like(PR, np.nan)
    ok = src >= 0
    out[ok] = PR[src[ok]]
    return out


def fit_eval(cols, X, y, m_tr, m_va, m_te):
    import lightgbm as lgb
    dtr = lgb.Dataset(X[m_tr][:, cols], label=y[m_tr])
    dva = lgb.Dataset(X[m_va][:, cols], label=y[m_va], reference=dtr)
    bst = lgb.train(PARAMS, dtr, num_boost_round=N_ROUND, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(EARLY, verbose=False)])
    pr = bst.predict(X[m_te][:, cols], num_iteration=bst.best_iteration)
    return float(np.abs(pr - y[m_te]).mean()), bst.best_iteration


def build_fold_features(fd, X, ix, sns, tss, sn_col, ts_row, D, sensor_fold, untrusted):
    X[:, TREG_IX] = fd.t_reg
    M = pivot(fd, sn_col, ts_row, len(tss), len(sns))
    tr = np.setdiff1d(np.where(sensor_fold != fd.f)[0], untrusted)
    W = idw_weights(D[np.ix_(tr, np.arange(len(sns)))], IDW_P, IDW_K)
    for i, s in enumerate(tr):
        W[i, s] = 0.0                       # LOO — 자기 관측 제거
    PR = predict(M[:, tr], W)
    X[:, ix["idw"]] = PR[ts_row, sn_col]
    for h in (1, 2, 24):
        X[:, ix[f"idw_lag{h}"]] = lag_matrix(PR, tss, h)[ts_row, sn_col]
    X[:, ix["idw_d1"]] = X[:, ix["idw"]] - X[:, ix["idw_lag1"]]
    treg_ts = pd.Series(fd.t_reg).groupby(ts_row).first().reindex(range(len(tss))).to_numpy()
    for h in (1, 3):
        src = lag_index(tss, h)
        prev = np.where(src >= 0, treg_ts[np.maximum(src, 0)], np.nan)
        X[:, ix[f"treg_d{h}"]] = (treg_ts - prev)[ts_row]


def paired(a, base, regional, X, ix, n_base, sns, tss, sn_col, ts_row,
           D, sensor_fold, untrusted, sn_ix) -> int:
    from scipy import stats
    keys = [k for k in VARIANTS if k.split()[0] in set(a.paired)] or list(VARIANTS)[:2]
    b3_cols = [ALL_F.index(c) for c in FEATS["b3_hybrid"]]
    print(f"\n짝지은 비교 — {' vs '.join(k.split()[0] for k in keys)} · 5-fold")
    print(f"{'fold':<6}" + "".join(f"{k.split()[0]:>10}" for k in keys) + f"{'차이':>10}")
    print("-" * (6 + 10 * len(keys) + 10))
    rows = []
    for fd in iter_folds(base, regional):
        build_fold_features(fd, X, ix, sns, tss, sn_col, ts_row, D, sensor_fold, untrusted)
        y = fd.delta
        inner_val_sn = np.where(sensor_fold == (fd.f + 1) % N_FOLDS)[0]
        m_te = fd.is_test
        m_va = np.isin(sn_ix, inner_val_sn) & ~m_te
        m_tr = ~m_te & ~m_va
        maes = []
        for k in keys:
            cols = b3_cols + [ix[c] for c in VARIANTS[k]]
            mae, nt = fit_eval(cols, X, y, m_tr, m_va, m_te)
            maes.append(mae)
            rows.append({"fold": fd.f, "variant": k, "mae": mae, "trees": nt})
        print(f"{fd.f:<6}" + "".join(f"{m:>10.4f}" for m in maes)
              + f"{maes[0]-maes[-1]:>+10.4f}")

    r = pd.DataFrame(rows)
    piv = r.pivot(index="fold", columns="variant", values="mae")
    d = piv[keys[0]] - piv[keys[-1]]          # 양수면 뒤쪽(새 변이)이 낫다
    t, p = stats.ttest_rel(piv[keys[0]], piv[keys[-1]])
    print(f"\n평균 " + "".join(f"{piv[k].mean():>10.4f}" for k in keys)
          + f"{d.mean():>+10.4f}")
    gain = d.mean() / piv[keys[0]].mean() * 100
    print(f"\n개선 {gain:+.2f}% · 짝지은 `t({len(d)-1})={t:.2f}` · p={p:.3f} "
          f"· fold 부호 {'전부 같다' if (d > 0).all() or (d < 0).all() else '갈린다'}")
    print("  판정: " + ("**유의** — 5-fold 전부 같은 방향" if p < 0.05 and ((d > 0).all() or (d < 0).all())
                      else "**구별 불가** — 이 차이로 설계 결정을 하지 않는다"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    r.to_parquet(OUT.with_name("probe_stidw_paired.parquet"), index=False)
    print(f"\n→ {OUT.with_name('probe_stidw_paired.parquet')}  (동결본 미변경)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--paired", nargs="*", default=None,
                    help="지정한 변이만 5-fold 전체로 돌려 짝지은 검정 (예: --paired V0 V3)")
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
    sn_ix = sn_col

    n_base = len(ALL_F)
    X = np.zeros((len(base), n_base + len(EXTRA)), np.float32)
    X[:, [ALL_F.index(c) for c in BASE_F]] = base[BASE_F].to_numpy(np.float32)
    ix = {c: n_base + i for i, c in enumerate(EXTRA)}

    print(f"행 {len(base):,} · 센서 {len(sns):,} · 시각 {len(tss):,} · fold {a.fold}")

    if a.paired is not None:
        return paired(a, base, regional, X, ix, n_base, sns, tss, sn_col, ts_row,
                      D, sensor_fold, untrusted, sn_ix)

    fd = next(f for f in iter_folds(base, regional) if f.f == a.fold)
    y = fd.delta
    X[:, TREG_IX] = fd.t_reg

    # ── IDW 행렬을 직접 만든다. `idw_oof`는 행 단위 값만 주는데 시간축으로 밀어야 한다
    t0 = time.time()
    M = pivot(fd, sn_col, ts_row, len(tss), len(sns))
    tr = np.setdiff1d(np.where(sensor_fold != fd.f)[0], untrusted)
    W = idw_weights(D[np.ix_(tr, np.arange(len(sns)))], IDW_P, IDW_K)
    for i, s in enumerate(tr):
        W[i, s] = 0.0                       # LOO — 자기 관측 제거 (b1_interp와 동일)
    PR = predict(M[:, tr], W)               # (시각 × 센서)
    X[:, ix["idw"]] = PR[ts_row, sn_col]
    for h in (1, 2, 24):
        X[:, ix[f"idw_lag{h}"]] = lag_matrix(PR, tss, h)[ts_row, sn_col]
    X[:, ix["idw_d1"]] = X[:, ix["idw"]] - X[:, ix["idw_lag1"]]

    # ── 광역 앵커 변화율. 시각 단위로 만든 뒤 행에 뿌린다
    treg_ts = pd.Series(fd.t_reg).groupby(ts_row).first().reindex(range(len(tss))).to_numpy()
    for h in (1, 3):
        src = lag_index(tss, h)
        prev = np.where(src >= 0, treg_ts[np.maximum(src, 0)], np.nan)
        X[:, ix[f"treg_d{h}"]] = (treg_ts - prev)[ts_row]
    print(f"  피처 생성 {time.time()-t0:.1f}초 · "
          f"lag1 결측 {np.isnan(X[:, ix['idw_lag1']]).mean()*100:.1f}% · "
          f"lag24 결측 {np.isnan(X[:, ix['idw_lag24']]).mean()*100:.1f}%")

    inner_val_sn = np.where(sensor_fold == (fd.f + 1) % N_FOLDS)[0]
    m_te = fd.is_test
    m_va = np.isin(sn_ix, inner_val_sn) & ~m_te
    m_tr = ~m_te & ~m_va
    print(f"  학습 {int(m_tr.sum()):,} · 내부검증 {int(m_va.sum()):,} · test {int(m_te.sum()):,}")

    b3_cols = [ALL_F.index(c) for c in FEATS["b3_hybrid"]]
    print(f"\n{'변이':<22}{'피처':>6}{'트리':>7}{'MAE':>10}{'V0 대비 개선':>14}{'초':>7}")
    print("-" * 66)
    rows, base_mae = [], None
    for name, extra in VARIANTS.items():
        cols = b3_cols + [ix[c] for c in extra]
        t1 = time.time()
        mae, nt = fit_eval(cols, X, y, m_tr, m_va, m_te)
        if base_mae is None:
            base_mae = mae
        # 부호 규약: 개선율 = (기준 − 대상)/기준. 양수면 좋아진 것 (`CLAUDE.md`)
        gain = (base_mae - mae) / base_mae * 100
        print(f"{name:<22}{len(cols):>6}{nt:>7}{mae:>10.4f}{gain:>+13.2f}%{time.time()-t1:>7.1f}")
        rows.append({"fold": a.fold, "variant": name, "n_feat": len(cols),
                     "trees": nt, "mae": mae, "gain_pct": gain})

    r = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    r.to_parquet(OUT, index=False)
    print(f"\n→ {OUT}  (동결본·preds·metrics 미변경)")
    print("\nfold 하나는 판정이 아니다 — 0.001~0.002 차이로 결정하지 않는다"
          " (`CLAUDE.md` 작업원칙 10).")
    print("   유망하면 5-fold 짝지은 검정으로 확인한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
