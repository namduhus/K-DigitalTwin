from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import Point
from shapely.strtree import STRtree

DEM = Path("data/raw/dem90/한반도90m_GRS80.img")
GRID = Path("data/processing/grid_features.parquet")
SENSORS = Path("data/processing/sdot_locations.csv")
SEN_FEAT = Path("data/processing/sensor_features.parquet")
SEN_GREEN = Path("data/processing/sensor_green.parquet")
UQ153 = Path("data/raw/park/UPIS_C_UQ153.shp")
ZON216 = Path("data/raw/park/UPIS_SHP_ZON216.shp")

AZ_STEP, N_AZ = 2.0, 180
SENSOR_H = 3.0
TERRAIN_KM, STEP_M, BUFFER_M = 15.0, 90.0, 20_000.0
GREEN_R = 600.0
GREEN_LCLAS = {"UQT200", "UQT300"}
PAVED_LCLAS = {"UQT100", "UQT500"}
CHUNK = 1000


def svf(hz: np.ndarray, axis=None) -> np.ndarray:
    return 1.0 - np.mean(np.sin(np.radians(hz)) ** 2, axis=axis)


class Dem:
    def __init__(self):
        s = pd.read_csv(SENSORS)
        x0, x1 = s["x"].min() - BUFFER_M, s["x"].max() + BUFFER_M
        y0, y1 = s["y"].min() - BUFFER_M, s["y"].max() + BUFFER_M
        with rasterio.open(DEM) as ds:
            win = from_bounds(x0, y0, x1, y1, ds.transform)
            self.a = ds.read(1, window=win).astype(np.float32)
            self.tf = ds.window_transform(win)
            nd = ds.nodata
        self.a = np.where(self.a == nd, np.nan, self.a)
        self.H, self.W = self.a.shape
        self.px = self.tf.a
        print(f"DEM {self.W:,}×{self.H:,} px ({self.px:.0f}m) · "
              f"표고 {np.nanmin(self.a):.0f}~{np.nanmax(self.a):.0f}m")

    def sample(self, xs, ys):
        cols = ((xs - self.tf.c) / self.px).astype(np.int32)
        rows = ((self.tf.f - ys) / self.px).astype(np.int32)
        ok = (cols >= 0) & (cols < self.W) & (rows >= 0) & (rows < self.H)
        out = np.full(np.shape(xs), np.nan, np.float32)
        out[ok] = self.a[rows[ok], cols[ok]]
        return out


def terrain_batch(dem: Dem, X, Y, BH):
    n = len(X)
    az = np.arange(N_AZ) * AZ_STEP
    dist = np.arange(STEP_M, TERRAIN_KM * 1000 + 1, STEP_M)
    ux, uy = np.sin(np.radians(az))[:, None], np.cos(np.radians(az))[:, None]
    dx, dy = ux * dist[None, :], uy * dist[None, :]          # (180, 166)

    z0 = dem.sample(X, Y)
    p = dem.px
    nb = np.stack([dem.sample(X - p, Y), dem.sample(X + p, Y),
                   dem.sample(X, Y - p), dem.sample(X, Y + p)])
    gx, gy = (nb[1] - nb[0]) / (2 * p), (nb[3] - nb[2]) / (2 * p)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    aspect = np.degrees(np.arctan2(gx, gy)) % 360

    zz = dem.sample(X[:, None, None] + dx[None], Y[:, None, None] + dy[None])  # (n,180,166)
    rise = zz - (z0[:, None, None] + SENSOR_H)
    with np.errstate(invalid="ignore"):
        el = np.degrees(np.arctan2(rise, dist[None, None, :]))
    th = np.nanmax(np.where(np.isfinite(el), el, -90.0), axis=2)
    th = np.clip(th, 0.0, None)                              # (n, 180)
    comb = np.maximum(BH, th)
    return dict(elev=z0, slope=slope, aspect=aspect,
                svf=svf(comb, axis=1), svf_bld=svf(BH, axis=1), svf_terr=svf(th, axis=1),
                th_mean=th.mean(1), th_max=th.max(1),
                hz_mean=comb.mean(1), hz_max=comb.max(1),
                terr_dominant=(th > BH).mean(1))


def load_green():
    def rd(p):
        g = gpd.read_file(p, encoding="cp949")
        src = g.crs.to_epsg()
        g = g.to_crs(5186)
        print(f"  {p.name}: {len(g):,}개 (EPSG:{src} → 5186)")
        return g
    print("녹지·포장면 로드")
    uq, zn = rd(UQ153), rd(ZON216)
    grn = np.concatenate([uq[uq["LCLAS_CL"].isin(GREEN_LCLAS)].geometry.to_numpy(),
                          zn.geometry.to_numpy()])
    pav = uq[uq["LCLAS_CL"].isin(PAVED_LCLAS)].geometry.to_numpy()
    print(f"  → 녹지 {len(grn):,}개 · 포장면 {len(pav):,}개")
    return grn, STRtree(grn), pav, STRtree(pav)


def green_batch(X, Y, grn, gt, pav, pt_):
    out = {k: np.zeros(len(X), np.float32) for k in
           ("grn_n", "grn_frac", "grn_ar_max", "grn_d_min", "pave_n", "pave_frac")}
    out["grn_d_min"][:] = np.nan
    A = np.pi * GREEN_R ** 2
    for i, (x, y) in enumerate(zip(X, Y)):
        pt = Point(x, y); buf = pt.buffer(GREEN_R)
        for geoms, tree, pre in ((grn, gt, "grn"), (pav, pt_, "pave")):
            cand = tree.query(buf)
            if not len(cand):
                continue
            inter = np.array([geoms[j].intersection(buf).area for j in cand])
            out[f"{pre}_n"][i] = int((inter > 0).sum())
            out[f"{pre}_frac"][i] = min(inter.sum() / A, 1.0)   # 두 출처 중복으로 1 초과 가능
            if pre == "grn":
                out["grn_ar_max"][i] = max(geoms[j].area for j in cand)
                out["grn_d_min"][i] = min(geoms[j].distance(pt) for j in cand)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="센서 재현 검증만 하고 종료")
    ap.add_argument("--n-check", type=int, default=60)
    a = ap.parse_args()

    hzc = [f"hz_{i:03d}" for i in range(N_AZ)]

    # ── 검증: 센서 좌표에서 같은 코드를 돌려 기존값과 대조 ──────────────
    sen = pd.read_csv(SENSORS)
    sf = pd.read_parquet(SEN_FEAT).set_index("sn")
    sg = pd.read_parquet(SEN_GREEN).set_index("sn")
    sh = pd.read_parquet("data/processing/sensor_horizon.parquet").set_index("sn")
    sub = sen[sen.sn.isin(sf.index)].head(a.n_check).reset_index(drop=True)
    X, Y = sub.x.to_numpy(np.float64), sub.y.to_numpy(np.float64)
    BH = sh.loc[sub.sn, hzc].to_numpy(np.float64)

    dem = Dem()
    t0 = time.time()
    tr = terrain_batch(dem, X, Y, BH)
    print(f"지형 {len(sub)}점 {time.time()-t0:.1f}초 → {(time.time()-t0)/len(sub)*1000:.1f} ms/점")

    print(f"\n=== 지형 재현 검증 ({len(sub)}센서) ===")
    ok = True
    for k in ("elev", "slope", "aspect", "svf", "svf_bld", "svf_terr",
              "th_mean", "th_max", "terr_dominant"):
        d = np.abs(tr[k] - sf.loc[sub.sn, k].to_numpy())
        mark = "통과" if np.nanmax(d) < 1e-4 else "불일치"
        if np.nanmax(d) >= 1e-4:
            ok = False
        print(f"  {k:<16}최대차 {np.nanmax(d):.2e}  {mark}")

    grn, gt, pav, pt_ = load_green()
    t0 = time.time()
    gb = green_batch(X, Y, grn, gt, pav, pt_)
    print(f"녹지 {len(sub)}점 {time.time()-t0:.1f}초 → {(time.time()-t0)/len(sub)*1000:.1f} ms/점")
    print(f"\n=== 녹지 재현 검증 ({len(sub)}센서) ===")
    for k in ("grn_n", "grn_frac", "grn_ar_max", "grn_d_min", "pave_n", "pave_frac"):
        if k not in sg.columns:
            print(f"  {k:<16}(센서 파일에 없음)"); continue
        ref = sg.loc[sub.sn, k].to_numpy()
        d = np.abs(gb[k] - ref)
        rel = np.nanmax(d / np.maximum(np.abs(ref), 1.0))   # 큰 값은 상대 오차로 본다
        mark = "통과" if rel < 1e-5 else "불일치"
        if rel >= 1e-5:
            ok = False
        print(f"  {k:<16}최대차 {np.nanmax(d):.2e} · 상대 {rel:.2e}  {mark}")

    if not ok:
        print("\n재현 실패 — 격자 계산을 진행하지 않는다. 정의 차이를 먼저 찾는다")
        return 1
    print("\n전 항목 재현 — 격자 계산 가능")
    if a.check:
        return 0

    # ── 격자 계산 ────────────────────────────────────────────────
    if not GRID.exists():
        sys.exit(f"[error] {GRID} 없음 — 먼저 build_grid.py를 돌린다")
    g = pd.read_parquet(GRID)
    print(f"\n격자 {len(g):,}셀 로드")
    GX, GY = g.x.to_numpy(np.float64), g.y.to_numpy(np.float64)

    parts, t0 = [], time.time()
    for s in range(0, len(g), CHUNK):
        e = min(s + CHUNK, len(g))
        bh = g.iloc[s:e][hzc].to_numpy(np.float64)
        r = terrain_batch(dem, GX[s:e], GY[s:e], bh)
        r.update(green_batch(GX[s:e], GY[s:e], grn, gt, pav, pt_))
        parts.append(pd.DataFrame(r))
        el = time.time() - t0
        print(f"  {e:>9,}/{len(g):,} ({e/len(g)*100:>5.1f}%) · {el/60:>5.1f}분 · "
              f"남은 {el/e*(len(g)-e)/60:>5.1f}분", end="\r", file=sys.stderr)
    print(file=sys.stderr)

    add = pd.concat(parts, ignore_index=True)
    # 건물 전용 hz_mean/hz_max/svf는 지형 합산 값으로 덮어쓴다 (센서와 같은 규칙)
    for c in add.columns:
        g[c] = add[c].to_numpy()
    g.to_parquet(GRID, index=False)
    print(f"\n→ {GRID} ({GRID.stat().st_size/1e6:.0f} MB, {len(g):,}셀 × {g.shape[1]}컬럼)")
    print(f"총 {(time.time()-t0)/60:.1f}분")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
