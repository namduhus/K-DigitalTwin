from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core.baseline.b1_interp import idw_oof, sensor_grid
from core.baseline.b2_gbm import PARAMS
from core.baseline.b3_hybrid import (
    ALL_F,
    BASE_F,
    EARLY,
    FEATS,
    IDW_K,
    IDW_P,
    N_ROUND,
    TREG_IX,
)
from core.eval.dataset import (
    N_FOLDS,
    iter_folds,
    load_base,
    load_regional,
    load_sensor_meta,
    trusted_sensors,
)

OUT = Path("data/metric/buffer_cv.parquet")
RADII = [0, 300, 600, 900, 1200]        # m — 격자 최근접 거리 중위(374)~P95(1,320)를 걸친다
SEED = 20260903
MODEL = "b3_hybrid"


def buffered(D: np.ndarray, test_ix: np.ndarray, r: float) -> np.ndarray:
    if r <= 0:
        return np.array([], dtype=int)
    near = (D[test_ix] <= r).any(0)
    near[test_ix] = False                # test 자신은 어차피 학습에 없다
    return np.where(near)[0]


def realized_dist(D, test_ix, keep_ix):
    if len(keep_ix) == 0:
        return {"med": np.nan, "p95": np.nan, "mean": np.nan}
    d = D[np.ix_(test_ix, keep_ix)].min(1)
    return {"med": float(np.median(d)), "p95": float(np.percentile(d, 95)),
            "mean": float(d.mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="표본 크기만 출력하고 끝낸다")
    ap.add_argument("--probe", action="store_true", help="fold 0만")
    ap.add_argument("--radii", type=float, nargs="+", default=RADII)
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

    nn = np.sort(D + np.eye(len(sns)) * 1e9, axis=1)[:, 0]
    print(f"\n센서 {len(sns):,} · 최근접 거리 중위 {np.median(nn):.0f}m "
          f"· P95 {np.percentile(nn, 95):.0f}m")
    print(f"참고 — 격자 셀의 최근접 센서: 중위 374m · P95 1,320m (plan.md §3)")

    # ── 표본 크기 먼저 (학습 없이) ──
    print(f"\n{'r(m)':>6}{'fold':>6}{'test':>7}{'버퍼제외':>10}{'학습센서':>10}"
          f"{'학습비율':>10}")
    print("-" * 49)
    plan = {}
    for r in a.radii:
        for f in range(N_FOLDS):
            te = np.where(sensor_fold == f)[0]
            bf = buffered(D, te, r)
            n_tr = len(sns) - len(te) - len(bf)
            plan[(r, f)] = (te, bf, n_tr)
            print(f"{r:>6.0f}{f:>6}{len(te):>7}{len(bf):>10}{n_tr:>10}"
                  f"{n_tr/(len(sns)-len(te))*100:>9.1f}%")
    if a.dry:
        print("\n--dry — 여기까지. 학습 비율이 너무 낮으면 반경을 줄인다.")
        return 0

    # ── 학습 ──
    import lightgbm as lgb

    X = np.zeros((len(base), len(ALL_F) + 1), np.float32)
    X[:, [ALL_F.index(c) for c in BASE_F]] = base[BASE_F].to_numpy(np.float32)
    IDW_IX = len(ALL_F)
    cols = [ALL_F.index(c) for c in FEATS[MODEL]] + [IDW_IX]
    sn_ix = sn_col
    y_true = base["tp"].to_numpy(np.float32)
    folds = [0] if a.probe else list(range(N_FOLDS))
    rng = np.random.default_rng(SEED)

    rows = []
    print(f"\n{'r(m)':>6}{'fold':>5}{'방식':<8}{'학습센서':>9}{'실거리':>9}"
          f"{'학습행':>11}{'트리':>6}{'MAE':>9}{'초':>7}")
    print("-" * 71)
    for fd in iter_folds(base, regional):
        if fd.f not in folds:
            continue
        X[:, TREG_IX] = fd.t_reg
        y = fd.delta
        m_te = fd.is_test
        te, _, _ = plan[(a.radii[0], fd.f)]

        for r in a.radii:
            te, bf, n_tr = plan[(r, fd.f)]
            # 무작위 대조군 — 같은 수를 공간과 무관하게 뺀다
            pool = np.setdiff1d(np.where(sensor_fold != fd.f)[0], te)
            variants = [("버퍼", bf)]
            if len(bf):
                variants.append(("무작위", rng.choice(pool, len(bf), replace=False)))

            for kind, drop in variants:
                t0 = time.time()
                # IDW 이웃 풀에서도 뺀다 — 한쪽만 빼면 다른 쪽으로 샌다
                excl = np.union1d(untrusted, drop).astype(int)
                X[:, IDW_IX] = idw_oof(fd, base, D, sensor_fold, sn_col, ts_row,
                                       len(tss), excl, IDW_P, IDW_K)
                keep = np.setdiff1d(np.setdiff1d(np.arange(len(sns)), te), drop)
                rd = realized_dist(D, te, keep)
                m_drop = np.isin(sn_ix, drop)
                inner = np.where(sensor_fold == (fd.f + 1) % N_FOLDS)[0]
                m_va = np.isin(sn_ix, inner) & ~m_te & ~m_drop
                m_tr = ~m_te & ~m_va & ~m_drop

                dtr = lgb.Dataset(X[m_tr][:, cols], label=y[m_tr],
                                  feature_name=[str(c) for c in cols])
                dva = lgb.Dataset(X[m_va][:, cols], label=y[m_va], reference=dtr)
                bst = lgb.train(PARAMS, dtr, num_boost_round=N_ROUND, valid_sets=[dva],
                                callbacks=[lgb.early_stopping(EARLY, verbose=False)])
                pr = bst.predict(X[m_te][:, cols], num_iteration=bst.best_iteration)
                mae = float(np.abs(pr - y[m_te]).mean())
                rows.append({"r": r, "fold": fd.f, "kind": kind,
                             "n_drop": len(drop),
                             "n_train_sn": len(sns) - len(te) - len(drop),
                             "n_train_row": int(m_tr.sum()), "mae": mae,
                             "d_med": rd["med"], "d_p95": rd["p95"],
                             "trees": bst.best_iteration})
                print(f"{r:>6.0f}{fd.f:>5}{kind:<8}{len(sns)-len(te)-len(drop):>9}"
                      f"{rd['med']:>9.0f}{int(m_tr.sum()):>11,}"
                      f"{bst.best_iteration:>6}{mae:>9.4f}{time.time()-t0:>7.1f}")

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    if len(folds) == N_FOLDS:
        print(f"\n{'='*62}\n거리별 성능 감쇠 — fold 평균 ± 표준편차\n{'='*62}")
        piv = df.groupby(["kind", "r"])["mae"].agg(["mean", "std", "count"])
        print(piv.round(4).to_string())

        b0 = df[(df.kind == "버퍼") & (df.r == 0)].set_index("fold")["mae"]
        print(f"\n{'r(m)':>6}{'버퍼거리':>10}{'무작위거리':>11}{'버퍼':>9}{'무작위':>9}"
              f"{'국소공백':>10}{'표본밀도':>10}{'짝지은 t':>10}")
        print("-" * 78)
        for r in sorted(df.r.unique()):
            if r == 0:
                continue
            bu = df[(df.kind == "버퍼") & (df.r == r)].set_index("fold")["mae"]
            rd = df[(df.kind == "무작위") & (df.r == r)].set_index("fold")["mae"]
            if rd.empty:
                continue
            d_dist = (bu - rd).dropna()          # 거리 효과
            d_samp = (rd - b0).dropna()          # 표본 효과
            from scipy import stats
            t = stats.ttest_rel(bu.reindex(d_dist.index), rd.reindex(d_dist.index))
            dbu = df[(df.kind == "버퍼") & (df.r == r)]["d_med"].mean()
            drd = df[(df.kind == "무작위") & (df.r == r)]["d_med"].mean()
            print(f"{r:>6.0f}{dbu:>10.0f}{drd:>11.0f}{bu.mean():>9.4f}{rd.mean():>9.4f}"
                  f"{d_dist.mean():>+10.4f}{d_samp.mean():>+10.4f}"
                  f"{t.statistic:>10.2f}")
        print("\n실거리 = test 센서 → 남은 학습 센서 최근접 거리 중위 (fold 평균)")
        print("  국소공백 = 버퍼 − 무작위 (같은 표본 수에서 「구멍이 뚫려서」 나빠진 몫)")
        print("  표본밀도 = 무작위 − r=0 (「센서가 줄어서」 나빠진 몫)")
        print("  무작위 제거도 밀도를 낮춰 거리를 늘린다 — 둘의 차를 「순수 거리 효과」로 부르지 않는다.")
        print("  신청서에 쓸 축은 명목 r이 아니라 **실거리**다 (격자 중위 374m · P95 1,320m와 대응).")
    print(f"\n→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
