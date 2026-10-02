from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from core.eval.dataset import load_regional
from core.model.traj_dist import NORMAL_IQ

GRID = Path("data/processing/grid_features.parquet")
OUT = Path("data/metric/fan_cells.parquet")
DIST = Path("data/metric/traj_dist.npz")
CALIB = Path("data/metric/calib.npz")
DAY = "2026-08-06"
STREET_XY = (193402.0, 543399.0)      # fig1 관악구 창 중심


def main() -> int:
    g = pd.read_parquet(GRID, columns=["gx", "gy", "x", "y", "gu", "svf", "grn_frac",
                                       "bld_ar_frac", "bld_d_min", "in_building"])
    g = g[~g.in_building.astype(bool)]

    # 대비가 읽히는 셀을 고른다 — 하나는 밀집, 하나는 개방 녹지, 하나는 골목 그림의 창
    # `grn_frac > quantile(0.98)`로 잡았더니 **공집합**이었다 — 98분위가 이미
    # 1.0(포화)이라 초과하는 값이 없다. 상한이 있는 비율 피처에서 흔한 함정이다.
    # 상위 N개를 뽑는 방식으로 바꾼다.
    dense = g.nlargest(len(g) // 20, "bld_ar_frac").nsmallest(1, "svf")
    openg = g.nlargest(20_000, "grn_frac").nlargest(1, "svf")
    dxy = (g.x - STREET_XY[0]) ** 2 + (g.y - STREET_XY[1]) ** 2
    street = g.loc[[dxy.idxmin()]]
    pick = pd.concat([dense.assign(tag="밀집 시가지"),
                      street.assign(tag="관악 골목 (그림1)"),
                      openg.assign(tag="개방 녹지")]).reset_index(drop=True)
    print(pick[["tag", "gu", "gx", "gy", "svf", "grn_frac", "bld_d_min"]].to_string(index=False))

    key = set(zip(pick.gx, pick.gy))
    t = pq.read_table(f"data/metric/grid_pred_{DAY}.parquet",
                      columns=["gx", "gy", "ts", "t_hat", "q10", "q90"]).to_pandas()
    t = t[t.ts.dt.normalize() == pd.Timestamp(DAY)]
    t = t[[(a, b) in key for a, b in zip(t.gx, t.gy)]].copy()

    # 보정 척도 — (폭염 여부 × σ 구간). `grid_traj.py`와 같은 규칙이어야 한다
    from scipy import stats
    nu = float(np.load(DIST)["nu"])
    cb = np.load(CALIB)
    edges, Qb, hot_c = cb["sigma_edges"], cb["Q_bin"], float(cb["hot_c"])
    fac = Qb / float(stats.t.ppf(0.95, nu) * np.sqrt((nu - 2) / nu))
    treg = load_regional()[0]
    t["t_reg"] = treg.reindex(pd.DatetimeIndex(t.ts)).to_numpy()
    t["hot"] = (t.t_reg >= hot_c).astype(int)
    sg = (t.q90 - t.q10) / NORMAL_IQ
    b = np.clip(np.searchsorted(edges, sg, side="right") - 1, 0, Qb.shape[1] - 1)
    t["sigma_cal"] = sg.to_numpy() * fac[t.hot.to_numpy(), b]

    out = t.merge(pick[["gx", "gy", "tag", "gu", "svf", "grn_frac", "bld_d_min"]],
                  on=["gx", "gy"]).sort_values(["tag", "ts"])
    out.to_parquet(OUT, index=False)
    print(f"\n→ {OUT}  ({OUT.stat().st_size/1e3:.0f} KB · {len(out)}행 · "
          f"셀 {out.groupby(['gx','gy']).ngroups}개 × 시각 {out.ts.nunique()})")
    print(f"   폭염 시각 {int(t.groupby('ts').hot.first().sum())}/{t.ts.nunique()} · "
          f"보정 척도 {out.sigma_cal.min():.2f}~{out.sigma_cal.max():.2f}℃")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
