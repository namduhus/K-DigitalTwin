from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from core.eval.dataset import load_regional
from core.model.traj_dist import NORMAL_IQ, sample

GRID_PRED = {d: Path(f"data/metric/grid_pred_{d}.parquet")
             for d in ("2026-08-06", "2026-07-15")}
DIST = Path("data/metric/traj_dist.npz")
CALIB = Path("data/metric/calib.npz")
OUT_DIR = Path("data/metric")

N_DRAW = 30
CHUNK = 60_000
HOT_C = 33.0
SEED = 20260826
QS = [5, 25, 50, 75, 95]


def load_day(day: str):
    t = pq.read_table(GRID_PRED[day], columns=["gx", "gy", "ts", "t_hat", "q10", "q90"])
    d = t.to_pandas()
    d = d[d.ts.dt.normalize() == pd.Timestamp(day)]
    hours = np.sort(d.ts.unique())
    if len(hours) != 24:
        print(f"  {day} 달력일 시각이 {len(hours)}개다 (24 아님) — 그대로 진행")
    d["h"] = d.ts.map({t: i for i, t in enumerate(hours)})

    cells = d[["gx", "gy"]].drop_duplicates().sort_values(["gx", "gy"]).reset_index(drop=True)
    key = pd.MultiIndex.from_frame(cells)
    pos = pd.Series(np.arange(len(cells)), index=key)
    r = pos.reindex(pd.MultiIndex.from_arrays([d.gx, d.gy])).to_numpy()

    n, H = len(cells), len(hours)
    M = np.full((n, H), np.nan, np.float32)
    S = np.full((n, H), np.nan, np.float32)
    M[r, d.h.to_numpy()] = d.t_hat.to_numpy(np.float32)
    S[r, d.h.to_numpy()] = ((d.q90 - d.q10) / NORMAL_IQ).to_numpy(np.float32)
    return cells, hours, M, S


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--days", nargs="*", default=list(GRID_PRED))
    ap.add_argument("--no-hourly", action="store_true", help="시각별 분위 파일을 만들지 않는다")
    a = ap.parse_args()

    from scipy import stats
    z = np.load(DIST)
    R, nu, z_sd = z["R"], float(z["nu"]), float(z["z_sd"])

    # 확대 배수를 **Mondrian conformal**에서 가져온다 (8/26 변경).
    #
    # 전역 배수 `z_sd`(2.086)를 쓰면 σ가 작은 곳에서 좁고(적중 80.1%) 큰 곳에서
    # 넓다(94.5%) — **분위 GBM의 σ가 차이를 과장하기 때문**이다. 격자 셀은 σ가
    # 큰 쪽에 몰려 있어(폭염일 중위 0.872℃) 전역 배수를 쓰면 **과대**가 된다.
    # σ 구간별로 보정하면 구간별 적중률 편차가 **14.4%p → 0.1%p**로 떨어지고
    # 평균 구간폭도 **7.7% 좁아진다**. `record.md`(8/26) · `core/model/calibrate.py`.
    cb = np.load(CALIB)
    edges, Qb, HOT_TREG = cb["sigma_edges"], cb["Q_bin"], float(cb["hot_c"])
    t90 = float(stats.t.ppf(0.95, nu) * np.sqrt((nu - 2) / nu))   # 단위 t의 90% 반폭
    fac = Qb / t90                                                # (폭염, σ구간) 배수
    print(f"[분포] t(ν={nu:.1f}) · 상관 24×24 (인접 {np.mean([R[i,i+1] for i in range(23)]):.3f})")
    print(f"[보정] Mondrian conformal · **폭염 여부 × σ 구간 {Qb.shape[1]}개** · "
          f"척도 배수 {fac.min():.2f}~{fac.max():.2f} (전역 z_sd={z_sd:.2f} 대신)")

    # 폭염 판정은 그 시각 광역값 ≥ 33℃다. 센서 중위인 `t_reg`가 그 값이고
    # 격자 추론도 같은 값을 쓴다 — 그래서 격자에 그대로 적용된다.
    treg = load_regional()[0]

    for day in a.days:
        t0 = time.time()
        cells, hours, M, S = load_day(day)
        H = M.shape[1]
        if H != R.shape[0]:
            print(f"  {day} 시각 {H}개 ≠ 상관행렬 {R.shape[0]}차원 — 건너뛴다")
            continue
        if a.probe:
            cells, M, S = cells.iloc[:50_000].copy(), M[:50_000], S[:50_000]
        n = len(cells)
        bad = int(np.isnan(M).any(1).sum())
        hot_h = (treg.reindex(pd.DatetimeIndex(hours)).to_numpy() >= HOT_TREG).astype(int)
        bb = np.clip(np.searchsorted(edges, S, side="right") - 1, 0, Qb.shape[1] - 1)
        print(f"\n[{day}] 셀 {n:,} · 시각 {H} · 결측 셀 {bad:,} · "
              f"평균 {np.nanmin(M):.1f}~{np.nanmax(M):.1f}℃ · 원 척도 중위 {np.nanmedian(S):.3f}℃")
        print(f"       폭염 시각 {int(hot_h.sum())}/{H} · "
              f"보정 후 척도 중위 {np.nanmedian(S * fac[hot_h[None, :], bb]):.3f}℃ · "
              f"배수 중위 {np.median(fac[hot_h[None, :], bb]):.2f} "
              f"(셀의 {np.mean(bb == Qb.shape[1]-1)*100:.0f}%가 최상위 σ 구간)")

        rng = np.random.default_rng(SEED)
        hq = np.full((n, H, len(QS)), np.nan, np.float32) if not a.no_hourly else None
        out = {k: np.full(n, np.nan, np.float32) for k in
               ("hot_mean", "hot_q10", "hot_q90", "p_hot1", "p_hot6",
                "peak_mean", "peak_sd", "tmin_mean", "trange_mean")}

        for i0 in range(0, n, CHUNK):
            i1 = min(i0 + CHUNK, n)
            # 셀·시각마다 자기 σ가 속한 구간의 배수를 붙인다
            s_raw = S[i0:i1]
            b = np.clip(np.searchsorted(edges, s_raw, side="right") - 1, 0, Qb.shape[1] - 1)
            mu, sg = M[i0:i1], s_raw * fac[hot_h[None, :], b]
            ok = np.isfinite(mu).all(1) & np.isfinite(sg).all(1)
            if not ok.any():
                continue
            dr = sample(mu[ok], sg[ok], R, nu, N_DRAW, rng)      # (30, m, 24)
            idx = np.arange(i0, i1)[ok]

            if hq is not None:
                hq[idx] = np.percentile(dr, QS, axis=0).transpose(1, 2, 0)

            hot = (dr >= HOT_C).sum(2).astype(np.float32)        # (30, m)
            out["hot_mean"][idx] = hot.mean(0)
            out["hot_q10"][idx] = np.percentile(hot, 10, axis=0)
            out["hot_q90"][idx] = np.percentile(hot, 90, axis=0)
            out["p_hot1"][idx] = (hot >= 1).mean(0)
            out["p_hot6"][idx] = (hot >= 6).mean(0)
            pk = dr.argmax(2).astype(np.float32)
            out["peak_mean"][idx] = pk.mean(0)
            out["peak_sd"][idx] = pk.std(0)
            out["tmin_mean"][idx] = dr.min(2).mean(0)
            out["trange_mean"][idx] = (dr.max(2) - dr.min(2)).mean(0)

        der = pd.concat([cells.reset_index(drop=True), pd.DataFrame(out)], axis=1)
        p = OUT_DIR / f"grid_traj_{day}.parquet"
        der.to_parquet(p, index=False)
        v = der.dropna(subset=["hot_mean"])
        print(f"  → {p}  ({p.stat().st_size/1e6:.0f} MB · {time.time()-t0:.0f}초)")
        print(f"     33℃ 초과 시간  평균 {v.hot_mean.mean():.2f}h · "
              f"셀 중위 {v.hot_mean.median():.2f}h · P95 {v.hot_mean.quantile(.95):.2f}h")
        print(f"     셀 안 불확실성 — q10~q90 폭 중위 "
              f"**{(v.hot_q90 - v.hot_q10).median():.1f}시간**")
        print(f"     P(33℃ 초과 ≥1h) 중위 {v.p_hot1.median()*100:.1f}% · "
              f"≥6h 중위 {v.p_hot6.median()*100:.1f}%")
        print(f"     일최고 시각 평균 {v.peak_mean.mean():.1f}시 · "
              f"셀 내 표준편차 중위 {v.peak_sd.median():.2f}시간")

        if hq is not None:
            hp = OUT_DIR / f"grid_traj_q_{day}.parquet"
            m = np.isfinite(hq[:, 0, 0])
            g = cells[m].reset_index(drop=True)
            q = pd.DataFrame({
                "gx": np.repeat(g.gx.to_numpy(), H),
                "gy": np.repeat(g.gy.to_numpy(), H),
                "ts": np.tile(hours, m.sum()),
                **{f"p{q_}": hq[m][:, :, j].ravel() for j, q_ in enumerate(QS)},
            })
            q.to_parquet(hp, index=False)
            print(f"  → {hp}  ({hp.stat().st_size/1e6:.0f} MB · {len(q):,}행)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
