from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("demo/data")
GRID = Path("data/processing/grid_features.parquet")
BLD = Path("data/processing/buildings.parquet")
POP = Path("data/processing/grid_population.parquet")
DIST = Path("data/metric/traj_dist.npz")
CALIB = Path("data/metric/calib.npz")
pred_path = lambda day: Path(f"data/metric/grid_pred_{day}.parquet")

# 두 날을 굽는다 — 폭염일 하나로는 **「폭염일에 격차가 벌어진다」를 화면으로
#   보여줄 수 없다**. fig8이 숫자로 낸 것(센서 1.98배 / 격자 1.30배)을 데모에서는
#   토글로 보여준다.
#
#   절대 기온은 두 날이 8℃ 넘게 달라 **공통 색 범위를 쓰면 평온일이 통째로
#      한 색**이 된다 (fig8에서 이미 겪었다 — `make_figures.py` fig8 주석).
#      편차는 절대 수준이 빠지므로 공통으로 묶을 수 있고, 묶어야 격차가 보인다.
#      → 범위 결정은 뷰어가 두 날을 다 읽고 한다 (`demo/index.html` precompute).
DAYS = ["2026-08-06", "2026-07-15"]
LABELS = {"2026-08-06": "폭염일", "2026-07-15": "평온일"}
CX, CY, HALF = 193402.0, 543399.0, 300.0      # fig1과 같은 창
MARGIN = 300.0                                # 창 밖 건물도 그림자를 던진다
LAT, LON = 37.5665, 126.9780
HOT_C = 33.0
NORMAL_IQ = 2.5631031310892016
NDRAW = 200
SEED = 20260831


def build(day: str) -> dict:
    PRED = pred_path(day)
    box = (CX - HALF, CX + HALF, CY - HALF, CY + HALF)

    import pyarrow.parquet as pq
    import pvlib
    from pyproj import Transformer
    from scipy import stats
    from shapely import from_wkt

    # ── 원점 (METER_OFFSETS용) ──
    tr = Transformer.from_crs(5186, 4326, always_xy=True)
    olon, olat = tr.transform(CX, CY)
    print(f"[원점] EPSG:5186 ({CX:.0f}, {CY:.0f}) → WGS84 ({olon:.6f}, {olat:.6f})")

    # ── 시각 + 태양 ──
    ts_all = pd.DatetimeIndex(pd.unique(
        pq.read_table(PRED, columns=["ts"])["ts"].to_pandas()))
    ts = pd.DatetimeIndex(np.sort(ts_all[ts_all.normalize() == pd.Timestamp(day)]))
    sp = pvlib.solarposition.spa_python(ts.tz_localize("Asia/Seoul"), LAT, LON)
    sun = [{"el": round(float(e), 2), "az": round(float(a), 2),
            "utc": int(pd.Timestamp(t).tz_localize("Asia/Seoul").timestamp() * 1000)}
           for e, a, t in zip(sp["apparent_elevation"], sp["azimuth"], ts)]
    print(f"[시각] {len(ts)}개 · 태양고도 {min(s['el'] for s in sun):.0f}"
          f"~{max(s['el'] for s in sun):.0f}°")

    # ── 격자 ──
    g = pd.read_parquet(GRID, columns=["gx", "gy", "x", "y", "svf", "bld_d_min",
                                       "grn_frac", "in_building"])
    g = g[(g.x.between(box[0], box[1])) & (g.y.between(box[2], box[3]))].copy()
    pop = pd.read_parquet(POP, columns=["gx", "gy", "pop", "pop_elder"])
    g = g.merge(pop, on=["gx", "gy"], how="left").fillna({"pop": 0.0, "pop_elder": 0.0})
    g = g.sort_values(["gy", "gx"]).reset_index(drop=True)
    print(f"[격자] {len(g):,}셀 · 건물 내부 {int(g.in_building.sum())}")

    p = pq.read_table(PRED, columns=["gx", "gy", "ts", "t_hat", "q10", "q90"]).to_pandas()
    p = p[p.ts.isin(ts)]
    key = pd.MultiIndex.from_arrays([g.gx, g.gy])
    pos = pd.Series(np.arange(len(g)), index=key)
    p = p[p.set_index(["gx", "gy"]).index.isin(key)].copy()
    p["ci"] = pos.reindex(pd.MultiIndex.from_arrays([p.gx, p.gy])).to_numpy()
    p["hi"] = p.ts.map({t: i for i, t in enumerate(ts)})

    n, H = len(g), len(ts)
    T = np.full((n, H), np.nan, np.float64)
    SG = np.full((n, H), np.nan, np.float64)
    T[p.ci.to_numpy(), p.hi.to_numpy()] = p.t_hat.to_numpy()
    SG[p.ci.to_numpy(), p.hi.to_numpy()] = ((p.q90 - p.q10) / NORMAL_IQ).to_numpy()

    # ── 궤적 분위 (팬차트) ──
    z, cb = np.load(DIST), np.load(CALIB)
    R, nu = z["R"], float(z["nu"])
    edges, Qb, hot_c = cb["sigma_edges"], cb["Q_bin"], float(cb["hot_c"])
    t90 = float(stats.t.ppf(0.95, nu) * np.sqrt((nu - 2) / nu))
    fac = Qb / t90
    treg = np.nanmedian(T, axis=0)                       # 창 중위로 광역값 근사
    hot = (treg >= hot_c).astype(int)
    print(f"[보정] 폭염 시각 {int(hot.sum())}/{H} · t(ν={nu:.0f})")

    ok = ~np.isnan(T).any(1)
    rng = np.random.default_rng(SEED)
    L = np.linalg.cholesky(R * (nu - 2) / nu)
    mu, sr = T[ok], SG[ok]
    bidx = np.clip(np.searchsorted(edges, sr, side="right") - 1, 0, Qb.shape[1] - 1)
    sg = sr * fac[np.tile(hot, (mu.shape[0], 1)), bidx]
    draws = mu + sg * ((rng.standard_normal((NDRAW, mu.shape[0], H)) @ L.T)
                       / np.sqrt(rng.chisquare(nu, size=(NDRAW, mu.shape[0], 1)) / nu))
    q = np.percentile(draws, [10, 50, 90], axis=0)       # (3, 셀, 시각)
    hoth = (draws >= HOT_C).sum(2)                        # (draw, 셀)
    print(f"[궤적] 셀 {int(ok.sum()):,} × {H}시각 × {NDRAW}샘플")

    def pad(arr2d):
        full = np.full((n, H), np.nan)
        full[ok] = arr2d
        return [[None if np.isnan(v) else round(float(v), 1) for v in r] for r in full]

    hot_stat = np.full((n, 3), np.nan)
    hot_stat[ok] = np.column_stack([hoth.mean(0),
                                    np.percentile(hoth, 10, axis=0),
                                    np.percentile(hoth, 90, axis=0)])

    # ── 건물 ──
    b = pd.read_parquet(BLD, columns=["cx", "cy", "height_m", "wkt", "geom_ok"])
    b = b[(b.cx.between(box[0] - MARGIN, box[1] + MARGIN))
          & (b.cy.between(box[2] - MARGIN, box[3] + MARGIN))
          & b.geom_ok & b.height_m.notna()]
    polys, heights = [], []
    for w, h in zip(b.wkt, b.height_m):
        geo = from_wkt(w)
        for qq in (geo.geoms if geo.geom_type == "MultiPolygon" else [geo]):
            # 0.5m로 단순화 — 25m 격자 데모에 그 이하 굴곡은 안 보이고 용량만 먹는다
            c = np.asarray(qq.simplify(0.5).exterior.coords)[:-1]
            if len(c) < 3:
                continue
            polys.append([[round(float(x - CX), 1), round(float(y - CY), 1)] for x, y in c])
            heights.append(round(float(h), 1))
    print(f"[건물] {len(polys):,}동 · 높이 중위 {np.median(heights):.1f}m · "
          f"최고 {max(heights):.0f}m")

    data = {
        "day": day, "label": LABELS.get(day, day),
        "origin": [round(olon, 7), round(olat, 7)],
        "hours": [int(t.hour) for t in ts],
        "sun": sun,
        "cell": 25.0,
        # 셀 중심을 원점 기준 미터 오프셋으로
        "cx": [round(float(v - CX), 1) for v in g.x],
        "cy": [round(float(v - CY), 1) for v in g.y],
        "in_bld": [bool(v) for v in g.in_building],
        "t": [[None if np.isnan(v) else round(float(v), 1) for v in r] for r in T],
        "q10": pad(q[0]), "q50": pad(q[1]), "q90": pad(q[2]),
        "hot": [None if np.isnan(a) else [round(float(a), 1), int(lo), int(hi)]
                for a, lo, hi in hot_stat],
        "attr": [{"svf": None if pd.isna(s) else round(float(s), 3),
                  "dmin": None if pd.isna(d) else round(float(d), 1),
                  "grn": None if pd.isna(gr) else round(float(gr), 3),
                  "eld": round(float(e), 1)}
                 for s, d, gr, e in zip(g.svf, g.bld_d_min, g.grn_frac, g.pop_elder)],
        "buildings": {"poly": polys, "h": heights},
        "meta": {"gu": "관악구", "center5186": [CX, CY], "half": HALF,
                 "note": "fig1과 같은 창 — 검증 1이 고른 창 (그늘 32% · 4피처 R² 0.82)"},
    }
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="+", default=DAYS)
    a = ap.parse_args()

    for p in (GRID, BLD, POP, DIST, CALIB):
        if not p.exists():
            sys.exit(f"[error] 없음: {p}")
    for d in a.days:
        if not pred_path(d).exists():
            sys.exit(f"[error] 없음: {pred_path(d)}")
    (OUT / "district").mkdir(parents=True, exist_ok=True)

    index = []
    for d in a.days:
        print(f"\n{'='*66}\n[{LABELS.get(d, d)}] {d}\n{'='*66}")
        data = build(d)
        f = OUT / "district" / f"{d}.json"
        f.write_text(json.dumps(data, separators=(",", ":")))
        print(f"\n→ {f} ({f.stat().st_size/1e6:.1f} MB)")
        index.append({"key": d, "label": LABELS.get(d, d),
                      "district": f"data/district/{d}.json"})

    idx = OUT / "days.json"
    idx.write_text(json.dumps({"days": index, "primary": a.days[0]},
                              ensure_ascii=False, separators=(",", ":")))
    print(f"\n→ {idx}  ({len(index)}일: {', '.join(x['label'] for x in index)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
