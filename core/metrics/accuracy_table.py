from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from core.eval.dataset import N_FOLDS, PREDS, hot_timestamps, load_base
from core.metrics.evaluate import BIN_FEATURES, build_slices

DIST = Path("data/metric/traj_dist.npz")
CALIB = Path("data/metric/calib.npz")
OUT = Path("data/metric/accuracy_table.parquet")

POINT_MODELS = ["b0a", "b0b", "b1_idw", "b2_gbm", "b3_hybrid"]
LABEL = {"b0a": "B0a 서울단일", "b0b": "B0b 5km", "b1_idw": "B1 IDW",
         "b2_gbm": "B2 GBM", "b3_hybrid": "B3 하이브리드", "dist": "분포 t+conformal"}
MAIN_SLICES = [("all", "all", "전체"), ("hot", "hot", "폭염 시각"),
               ("daynight", "day", "주간"), ("daynight", "night", "야간"),
               ("sparse_gu", "sparse", "희소 자치구")]
N_SAMP = 100
CHUNK = 500_000
SEED = 20260826


def crps_t(mu, sg, y, nu, n_samp, rng):
    out = np.empty(len(mu))
    tq = np.sqrt((nu - 2) / nu)
    for i in range(0, len(mu), CHUNK):
        j = min(i + CHUNK, len(mu))
        X = mu[i:j, None] + rng.standard_t(nu, (j - i, n_samp)) * tq * sg[i:j, None]
        Xs = np.sort(X, axis=1)
        k = np.arange(n_samp, dtype=np.float64)
        out[i:j] = (np.abs(Xs - y[i:j, None]).mean(1)
                    - 0.5 * (Xs * (2 * k - n_samp + 1)).sum(1) * (2.0 / n_samp ** 2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=N_SAMP)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)

    cols = ["is_night", "is_shadow", "sun_el", "shadow_margin"] + BIN_FEATURES
    base = load_base(cols)
    base = base[base.fold >= 0].reset_index(drop=True)

    preds = {}
    for m in POINT_MODELS:
        f = PREDS / f"{m}.parquet"
        if not f.exists():
            print(f"  {m} 예측 없음 — 건너뜀")
            continue
        p = pd.read_parquet(f); p["sn"] = p["sn"].astype(str)
        s = pd.Series(p.y_pred.to_numpy(np.float32),
                      index=pd.MultiIndex.from_arrays([p.sn, p.ts]))
        preds[m] = s.reindex(pd.MultiIndex.from_arrays([base.sn, base.ts])).to_numpy(np.float32)

    # ── 분포 모델 조립 ──
    z, cb = np.load(DIST), np.load(CALIB)
    nu = float(z["nu"])
    edges, Qb = cb["sigma_edges"], cb["Q_bin"]
    from scipy import stats
    t90 = float(stats.t.ppf(0.95, nu) * np.sqrt((nu - 2) / nu))
    fac = Qb / t90

    sc = pd.read_parquet(PREDS / "b3_scale.parquet"); sc["sn"] = sc["sn"].astype(str)
    ss = pd.Series(sc.sigma.to_numpy(np.float64),
                   index=pd.MultiIndex.from_arrays([sc.sn, sc.ts]))
    sig = ss.reindex(pd.MultiIndex.from_arrays([base.sn, base.ts])).to_numpy(np.float64)
    # 보정 격자는 (폭염 여부 × σ 구간) 2차원이다 — σ만으로는 폭염 적중률이
    # 87.4%였다. 폭염이 핵심 용도이므로 그 축을 따로 잡는다 (`calibrate.py`).
    hot_row = base.ts.isin(set(hot_timestamps(base))).to_numpy().astype(int)
    b = np.clip(np.searchsorted(edges, sig, side="right") - 1, 0, Qb.shape[1] - 1)
    Qrow = Qb[hot_row, b]
    sig_cal = sig * (Qrow / t90)

    y = base.tp.to_numpy(np.float64)
    mu = preds["b3_hybrid"].astype(np.float64)
    ok = np.isfinite(mu) & np.isfinite(sig_cal)
    print(f"[행] {len(base):,} · 분포 모델 유효 {int(ok.sum()):,} "
          f"· 보정 σ 중위 {np.nanmedian(sig_cal):.3f}℃")

    cover = np.full(len(base), np.nan)
    cover[ok] = (np.abs(y[ok] - mu[ok]) <= Qrow[ok] * sig[ok]).astype(float)
    width = np.full(len(base), np.nan)
    width[ok] = 2 * Qrow[ok] * sig[ok]
    crps = np.full(len(base), np.nan)
    crps[ok] = crps_t(mu[ok], sig_cal[ok], y[ok], nu, a.samples, rng)

    # ── 슬라이스별 집계 ──
    sl = build_slices(base)
    fold = base.fold.to_numpy()
    rows = []
    for kind, lab in sl.items():
        labv = lab.to_numpy()
        for val in pd.unique(labv):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            m0 = labv == val
            for f in range(N_FOLDS):
                m = m0 & (fold == f)
                if not m.any():
                    continue
                r = {"slice_kind": kind, "slice_val": str(val), "fold": f, "n": int(m.sum())}
                for name, yp in preds.items():
                    r[f"mae_{name}"] = float(np.abs(yp[m] - y[m]).mean())
                mm = m & ok
                r["mae_dist"] = float(np.abs(mu[mm] - y[mm]).mean())
                r["crps_dist"] = float(crps[mm].mean())
                r["cover_dist"] = float(cover[mm].mean())
                r["width_dist"] = float(width[mm].mean())
                rows.append(r)
    M = pd.DataFrame(rows)
    S = (M.groupby(["slice_kind", "slice_val"], observed=True)
           .agg(["mean", "std"]).drop(columns=[("fold", "mean"), ("fold", "std")]))
    M.to_parquet(OUT, index=False)

    def get(kind, val, col, stat="mean"):
        try:
            return S.loc[(kind, val), (col, stat)]
        except KeyError:
            return np.nan

    # ── 표 1: MAE 4분할 ──
    have = [m for m in POINT_MODELS if m in preds] + ["dist"]
    print(f"\n{'='*100}\n표 1 — MAE (℃) · 5-fold 평균 ± 표준편차\n{'='*100}")
    head = f"  {'구간':<12}{'행':>11}" + "".join(f"{LABEL[m]:>18}" for m in have)
    print(head); print("  " + "-" * (len(head) - 2))
    for kind, val, name in MAIN_SLICES:
        n = get(kind, val, "n", "mean")
        if not np.isfinite(n):
            continue
        line = f"  {name:<12}{int(n * N_FOLDS):>11,}"
        for m in have:
            line += f"{get(kind, val, f'mae_{m}'):>12.3f} ±{get(kind, val, f'mae_{m}', 'std'):.3f}"
        print(line)
    print(f"\n  ※ 분포 모델의 MAE는 B3와 같다 — **평균은 B3를 그대로 쓰기 때문이다.**")
    print(f"    분포 모델이 더하는 것은 MAE가 아니라 아래 두 표다.")

    # ── 표 2: 분포 지표 ──
    print(f"\n{'='*100}\n표 2 — 분포 지표 · **CRPS는 점추정에서 MAE와 같다**\n{'='*100}")
    head = (f"  {'구간':<12}{'행':>11}{'CRPS B3(=MAE)':>16}{'CRPS 분포':>13}"
            f"{'개선':>9}{'90% 적중률':>13}{'90% 구간폭':>13}")
    print(head); print("  " + "-" * (len(head) - 2))
    for kind, val, name in MAIN_SLICES:
        n = get(kind, val, "n", "mean")
        if not np.isfinite(n):
            continue
        b3 = get(kind, val, "mae_b3_hybrid")
        cd = get(kind, val, "crps_dist")
        print(f"  {name:<12}{int(n * N_FOLDS):>11,}{b3:>16.4f}{cd:>13.4f}"
              f"{(b3-cd)/b3*100:>8.1f}%{get(kind, val, 'cover_dist')*100:>12.1f}%"
              f"{get(kind, val, 'width_dist'):>12.3f}℃")
    print(f"\n  ※ 보정 격자는 **(σ 구간 10개 × 폭염 여부)**다. 폭염 축을 넣기 전에는")
    print(f"    폭염 적중률이 **87.4%**였다 — 핵심 용도에서 구간이 좁았다는 뜻이라 고쳤다.")
    print(f"    남은 편차(희소 자치구 89.6%)는 **지역 축을 넣지 않았기 때문**이고,")
    print(f"    그것이 지금 조건부 보정의 한계다.")

    # ── 표 3: 피처 분위별 ──
    print(f"\n{'='*100}\n표 3 — 피처 분위별 MAE · 범주(`rgn`) 대신 연속 피처로 본다\n{'='*100}")
    for c in BIN_FEATURES:
        sub = S.loc[c] if c in S.index.get_level_values(0) else None
        if sub is None:
            continue
        print(f"\n  [{c}]")
        head = f"  {'분위':<8}{'행':>11}" + "".join(f"{LABEL[m]:>18}" for m in have)
        print(head); print("  " + "-" * (len(head) - 2))
        for val in sorted(sub.index):
            line = f"  {val:<8}{int(get(c, val, 'n', 'mean') * N_FOLDS):>11,}"
            for m in have:
                line += (f"{get(c, val, f'mae_{m}'):>12.3f} "
                         f"±{get(c, val, f'mae_{m}', 'std'):.3f}")
            print(line)

    print(f"\n→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
