from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from core.baseline.grid_infer import AZ_STEP, GRID, N_AZ, sun_features

PRED = {d: Path(f"data/metric/grid_pred_{d}.parquet") for d in ("2026-08-06", "2026-07-15")}
OUT = Path("data/metric/shadow_probe.parquet")

CELL_M = 25.0
MIN_CELLS = 100          # 창 하나에 유효 셀이 이만큼은 있어야 통계를 낸다
SHADE_LO, SHADE_HI = 0.10, 0.90   # 그늘/양지가 둘 다 있어야 창 안 비교가 성립한다


def load_grid(hz_cols: list[str]) -> pd.DataFrame:
    cols = ["gx", "gy", "x", "y", "gu", "svf", "bld_d_min", "bld_h_max",
            "bld_ar_frac", "in_building"] + hz_cols
    g = pd.read_parquet(GRID, columns=cols)
    print(f"[grid] {len(g):,}셀 · 건물 내부 {int(g.in_building.sum()):,} "
          f"({g.in_building.mean()*100:.1f}%)")
    return g


def window_stats(d: pd.DataFrame, win: int) -> pd.DataFrame:
    d = d.assign(wx=d.gx // win, wy=d.gy // win)
    grp = d.groupby(["wx", "wy"], sort=False)

    def agg(s: pd.DataFrame) -> pd.Series:
        sh = s.is_shadow.to_numpy(bool)
        t = s.t_hat.to_numpy(np.float64)
        f = float(sh.mean())
        # 컬럼명을 `dt`로 두면 안 된다 — `row.dt`가 pandas의 datetime 접근자로 잡힌다
        out = {"n": len(s), "shade_frac": f,
               "t_std": t.std(), "t_range": t.max() - t.min(),
               "bld_h": s.bld_h_max.mean(), "svf": s.svf.mean(),
               "x": s.x.mean(), "y": s.y.mean()}          # 그림 크롭용 중심 (EPSG:5186)
        if not (SHADE_LO <= f <= SHADE_HI):
            for k in ("dtemp", "r_shadow", "r_margin", "r_svf", "r_dmin", "r2_all"):
                out[k] = np.nan
            return pd.Series(out)

        out["dtemp"] = t[sh].mean() - t[~sh].mean()
        for k, v in (("r_shadow", sh.astype(float)),
                     ("r_margin", s.shadow_margin.to_numpy(np.float64)),
                     ("r_svf", s.svf.to_numpy(np.float64)),
                     ("r_dmin", s.bld_d_min.to_numpy(np.float64))):
            out[k] = np.corrcoef(v, t)[0, 1] if np.std(v) > 0 else np.nan

        # 창 안 기온 변동을 4개 피처가 얼마나 설명하는가 (골목 해상도의 정체)
        Z = np.column_stack([s.shadow_margin, s.svf, s.bld_d_min, s.bld_h_max]).astype(np.float64)
        Z = np.column_stack([np.ones(len(Z)), Z])
        keep = np.isfinite(Z).all(1)
        if keep.sum() > 10 and t[keep].std() > 0:
            beta, *_ = np.linalg.lstsq(Z[keep], t[keep], rcond=None)
            res = t[keep] - Z[keep] @ beta
            out["r2_all"] = 1.0 - res.var() / t[keep].var()
        else:
            out["r2_all"] = np.nan
        return pd.Series(out)

    w = grp.apply(agg, include_groups=False)
    w["gu"] = grp["gu"].agg(lambda s: s.iloc[0])
    return w[w["n"] >= MIN_CELLS]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default="2026-08-06")
    ap.add_argument("--win", type=int, default=16, help="창 한 변의 셀 수 (16 = 400m)")
    ap.add_argument("--hour", type=int, default=None,
                    help="후보 창을 뽑을 시각을 강제한다 (기본: 그늘 양 × 대비로 자동)")
    a = ap.parse_args()

    pred = PRED[a.day]
    ts_all = pd.DatetimeIndex(pd.unique(pq.read_table(pred, columns=["ts"])["ts"].to_pandas()))
    ts_day = ts_all[ts_all.normalize() == pd.Timestamp(a.day)].sort_values()
    sun = sun_features(ts_day).set_index("ts")
    day_ts = sun.index[sun.sun_el > 3.0]          # 주간만. 저고도는 수평선 잡음이 크다
    print(f"[{a.day}] 주간 {len(day_ts)}시각 (태양고도 >3°) · "
          f"창 {a.win}×{a.win}셀 = {a.win*CELL_M:.0f}m")

    ai = {ts: int(sun.loc[ts, "sun_az"] / AZ_STEP) % N_AZ for ts in day_ts}
    hz_cols = sorted({f"hz_{v:03d}" for v in ai.values()})
    g = load_grid(hz_cols)
    g = g[~g.in_building.astype(bool)].reset_index(drop=True)   # 사람이 걷지 않는 곳
    print(f"       건물 내부 제외 후 {len(g):,}셀\n")

    key = pd.MultiIndex.from_arrays([g.gx, g.gy])
    rows, keep = [], {}
    print(f"  {'시각':>6}{'고도':>7}{'방위':>7}{'그늘셀':>9}{'창 수':>7}"
          f"{'중위 Δt':>10}{'중위 r':>9}{'창내 표준편차':>13}")
    print("  " + "-" * 70)
    for ts in day_ts:
        s = sun.loc[ts]
        p = pq.read_table(pred, columns=["gx", "gy", "t_hat"],
                          filters=[("ts", "=", ts)]).to_pandas()
        t = (pd.Series(p.t_hat.to_numpy(np.float32),
                       index=pd.MultiIndex.from_arrays([p.gx, p.gy]))
             .reindex(key).to_numpy(np.float32))

        hz_sun = g[f"hz_{ai[ts]:03d}"].to_numpy(np.float32)
        margin = s.sun_el - hz_sun
        d = pd.DataFrame({"gx": g.gx, "gy": g.gy, "x": g.x, "y": g.y,
                          "t_hat": t, "gu": g.gu,
                          "is_shadow": margin < 0, "shadow_margin": margin,
                          "svf": g.svf, "bld_d_min": g.bld_d_min,
                          "bld_h_max": g.bld_h_max}).dropna(subset=["t_hat"])

        w = window_stats(d, a.win)
        ok = w.dropna(subset=["dtemp"])
        rows.append({"ts": ts, "sun_el": s.sun_el, "sun_az": s.sun_az,
                     "shade_cell_frac": float(d.is_shadow.mean()),
                     "n_win": len(w), "n_win_mixed": len(ok),
                     "dt_med": ok.dtemp.median(), "dt_p10": ok.dtemp.quantile(.10),
                     "r_shadow_med": ok.r_shadow.median(),
                     "r_margin_med": ok.r_margin.median(),
                     "r_svf_med": ok.r_svf.median(),
                     "r_dmin_med": ok.r_dmin.median(),
                     "r2_med": ok.r2_all.median(),
                     "neg_frac": float((ok.dtemp < 0).mean()) if len(ok) else np.nan,
                     "t_std_med": w.t_std.median()})
        keep[ts] = w
        r = rows[-1]
        print(f"  {ts:%H:%M}{s.sun_el:>7.1f}°{s.sun_az:>7.1f}°"
              f"{r['shade_cell_frac']*100:>8.1f}%{r['n_win_mixed']:>7,}"
              f"{r['dt_med']:>+10.3f}{r['r_shadow_med']:>+9.3f}{r['t_std_med']:>13.3f}")

    R = pd.DataFrame(rows).set_index("ts")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    R.to_parquet(OUT)

    print(f"\n{'='*72}\n판정\n{'='*72}")
    # 시각 선택은 Δt 하나로 하면 안 된다. 정오는 그늘 셀이 0.1%뿐이라 혼합 창이
    # 2개인데도 Δt가 가장 깊게 나온다 — **그림에 쓰면 그림자가 화면에 없다.**
    # 그림이 되려면 ① 그늘이 눈에 보이는 양이고 ② 기온 대비가 있어야 한다.
    if a.hour is not None:
        sel = R.index[R.index.hour == a.hour]
        if not len(sel):
            raise SystemExit(f"{a.hour}시가 주간 시각 목록에 없다: {list(R.index.hour)}")
        best = sel[0]
        b = R.loc[best]
        print(f"  지정 시각: {best:%H:%M} "
              f"(고도 {b.sun_el:.1f}° · 방위 {b.sun_az:.1f}° · 그늘 셀 {b.shade_cell_frac*100:.1f}%)")
    else:
        cand = R[(R.shade_cell_frac >= 0.03) & (R.shade_cell_frac <= 0.35) & (R.dt_med < 0)]
        if cand.empty:
            cand = R[R.dt_med < 0]
        best = (-cand.dt_med * np.sqrt(cand.shade_cell_frac)).idxmax()
        b = R.loc[best]
        print(f"  그림에 쓸 시각 (그늘 양 × 기온 대비): {best:%H:%M} "
              f"(고도 {b.sun_el:.1f}° · 그늘 셀 {b.shade_cell_frac*100:.1f}%)")
        print(f"    ※ Δt만 보면 13시가 최대지만 그늘 셀이 0.1%라 그림에 그림자가 없다")
    print(f"    창 안 Δt(그늘−양지) 중위 : {b.dt_med:+.3f}℃  · 하위10% {b.dt_p10:+.3f}℃")
    print(f"    Δt < 0 인 창 비율        : {b.neg_frac*100:.1f}%")
    print(f"    창 안 상관 중위          : is_shadow {b.r_shadow_med:+.3f} · "
          f"margin {b.r_margin_med:+.3f} · svf {b.r_svf_med:+.3f} · "
          f"bld_d_min {b.r_dmin_med:+.3f}")
    print(f"    창 안 기온 표준편차 중위 : {b.t_std_med:.3f}℃")
    print(f"    4피처 설명력 R² 중위     : {b.r2_med:.3f}  "
          f"← 골목 규모 변동을 무엇이 만드는가")

    print(f"\n  주야 반전 — 저고도에서 Δt 부호가 뒤집힌다 (fig2의 0~10° 역전과 정합)")
    for ts, v in R.iterrows():
        if v.sun_el < 20 or ts == best:
            print(f"    {ts:%H:%M} 고도 {v.sun_el:>4.1f}° · 그늘 {v.shade_cell_frac*100:>4.1f}% "
                  f"· Δt {v.dt_med:>+.3f}℃ · Δt<0 창 {v.neg_frac*100:>5.1f}%")

    w = keep[best].dropna(subset=["dtemp"]).copy()
    w["score"] = -w.dtemp * np.sqrt(w.t_std)
    top = w.sort_values("score", ascending=False).head(12)
    print(f"\n  그림 후보 창 12개 ({best:%H:%M}) — Δt가 크고 창 안 변동도 큰 곳")
    print(f"  {'창(wx,wy)':>13}{'자치구':>9}{'셀':>6}{'그늘':>7}{'Δt':>8}"
          f"{'표준편차':>9}{'기온폭':>8}{'건물높이':>9}{'R²':>6}{'중심 x,y (EPSG:5186)':>24}")
    print("  " + "-" * 98)
    for (wx, wy), v in top.iterrows():
        print(f"  {f'({wx},{wy})':>13}{str(v.gu):>9}{int(v.n):>6}{v.shade_frac*100:>6.0f}%"
              f"{v.dtemp:>+8.3f}{v.t_std:>9.3f}{v.t_range:>8.3f}{v.bld_h:>8.1f}m{v.r2_all:>6.2f}"
              f"{v.x:>13.0f}{v.y:>11.0f}")

    print(f"\n→ {OUT}")
    print("\n  해석 기준 — Δt 중위가 −0.1℃보다 얕고 창 안 표준편차가 0.1℃ 미만이면")
    print("  골목 규모에서 그림자가 기온 지도에 **보이지 않는다**는 뜻이고,")
    print("  그때는 3D 확대 패널을 그려도 읽히지 않는다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
