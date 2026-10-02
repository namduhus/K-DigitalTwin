from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

DEM = Path("data/raw/dem90/한반도90m_GRS80.img")
HZ = Path("data/processing/sensor_horizon.parquet")
SENSORS = Path("data/processing/sdot_locations.csv")
OUT = Path("data/processing/sensor_features.parquet")

AZ_STEP = 2.0
N_AZ = 180
SENSOR_H = 3.0          # build_horizon.py와 동일 기준
TERRAIN_KM = 15.0       # 지형 차폐 탐색 거리(km). 북한산·관악산·남산을 포함한다
STEP_M = 90.0           # DEM 해상도와 맞춘 광선 샘플 간격
BUFFER_M = 20_000.0     # DEM 잘라낼 여유 (탐색 거리보다 크게)


def svf(hz: np.ndarray, axis=None) -> np.ndarray:
    return 1.0 - np.mean(np.sin(np.radians(hz)) ** 2, axis=axis)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()

    for p in (DEM, HZ, SENSORS):
        if not p.exists():
            sys.exit(f"[error] {p} 없음")

    sen = pd.read_csv(SENSORS)
    hz = pd.read_parquet(HZ)
    if a.probe:
        sen = sen.sample(40, random_state=0).reset_index(drop=True)
        print(f"[probe] 센서 {len(sen)}개")

    # ── DEM에서 서울 + 여유 구간만 메모리에 올린다 ────────────────────
    x0, x1 = sen["x"].min() - BUFFER_M, sen["x"].max() + BUFFER_M
    y0, y1 = sen["y"].min() - BUFFER_M, sen["y"].max() + BUFFER_M
    with rasterio.open(DEM) as ds:
        win = from_bounds(x0, y0, x1, y1, ds.transform)
        dem = ds.read(1, window=win).astype(np.float32)
        tf = ds.window_transform(win)
        nodata = ds.nodata
    dem = np.where(dem == nodata, np.nan, dem)
    H, W = dem.shape
    px = tf.a                      # 픽셀 크기(x). y는 -px
    print(f"DEM 잘라냄 {W:,}×{H:,} px ({px:.0f}m) · 표고 "
          f"{np.nanmin(dem):.0f}~{np.nanmax(dem):.0f}m · 결측 {np.isnan(dem).mean():.2%}")

    def sample(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        cols = ((xs - tf.c) / px).astype(np.int32)
        rows = ((tf.f - ys) / px).astype(np.int32)
        ok = (cols >= 0) & (cols < W) & (rows >= 0) & (rows < H)
        out = np.full(xs.shape, np.nan, np.float32)
        out[ok] = dem[rows[ok], cols[ok]]
        return out

    # ── 광선 샘플 좌표 미리 계산 ──────────────────────────────────────
    az = np.arange(N_AZ) * AZ_STEP                       # (180,)
    dist = np.arange(STEP_M, TERRAIN_KM * 1000 + 1, STEP_M)  # (166,)
    # 방위: 북=0°, 시계방향
    ux = np.sin(np.radians(az))[:, None]
    uy = np.cos(np.radians(az))[:, None]
    dx = ux * dist[None, :]                              # (180, 166)
    dy = uy * dist[None, :]
    print(f"지형 광선 {N_AZ}방위 × {len(dist)}샘플 (최대 {TERRAIN_KM:.0f}km)")

    hz_cols = [f"hz_{i:03d}" for i in range(N_AZ)]
    bld_hz = hz.set_index("sn")[hz_cols]

    rows = []
    for i, s in sen.iterrows():
        sx, sy = float(s["x"]), float(s["y"])
        z0 = float(sample(np.array([sx]), np.array([sy]))[0])

        # 경사·사면향 — 중심차분 (3×3 이웃)
        nb = sample(np.array([sx - px, sx + px, sx, sx]), np.array([sy, sy, sy - px, sy + px]))
        gx = (nb[1] - nb[0]) / (2 * px)
        gy = (nb[3] - nb[2]) / (2 * px)
        slope = float(np.degrees(np.arctan(np.hypot(gx, gy))))
        aspect = float(np.degrees(np.arctan2(gx, gy)) % 360)

        # 지형 차폐: 각 방위·거리의 표고에서 앙각을 구해 최대값
        zz = sample(sx + dx, sy + dy)                    # (180, 166)
        rise = zz - (z0 + SENSOR_H)
        with np.errstate(invalid="ignore"):
            el = np.degrees(np.arctan2(rise, dist[None, :]))
        th = np.nanmax(np.where(np.isfinite(el), el, -90.0), axis=1)
        th = np.clip(th, 0.0, None)                      # 아래로 내려가는 지형은 0

        bh = bld_hz.loc[s["sn"]].to_numpy(float) if s["sn"] in bld_hz.index else np.zeros(N_AZ)
        comb = np.maximum(bh, th)

        rec = {
            "sn": s["sn"], "elev": z0, "slope": slope, "aspect": aspect,
            "svf": float(svf(comb)), "svf_bld": float(svf(bh)), "svf_terr": float(svf(th)),
            "th_mean": float(th.mean()), "th_max": float(th.max()),
            "hz_mean": float(comb.mean()), "hz_max": float(comb.max()),
            "terr_dominant": float((th > bh).mean()),   # 지형이 건물보다 높은 방위 비율
        }
        rec.update({f"th_{k:03d}": float(v) for k, v in enumerate(th)})
        rec.update({f"hz_{k:03d}": float(v) for k, v in enumerate(comb)})
        rows.append(rec)
        if (i + 1) % 50 == 0 or i + 1 == len(sen):
            print(f"  {i+1:>5}/{len(sen)}", end="\r", file=sys.stderr)
    print(file=sys.stderr)

    df = pd.DataFrame(rows)
    # 건물 전용 피처는 그대로 이어붙인다
    keep = ["sn", "bld_n", "bld_d_min", "bld_h_max", "bld_h_mean", "bld_ar_frac"]
    df = df.merge(hz[[c for c in keep if c in hz.columns]], on="sn", how="left")

    print(f"\n{'항목':<16} {'중위':>9} {'최소':>9} {'최대':>9}")
    print("-" * 48)
    for c, lab in [("elev", "표고(m)"), ("slope", "경사(°)"),
                   ("th_mean", "지형 앙각(°)"), ("th_max", "지형 최대(°)"),
                   ("svf_bld", "SVF 건물만"), ("svf_terr", "SVF 지형만"),
                   ("svf", "SVF 합산"), ("terr_dominant", "지형우세 방위비")]:
        print(f"{lab:<16} {df[c].median():>9.3f} {df[c].min():>9.3f} {df[c].max():>9.3f}")

    print(f"\n지형 차폐로 SVF가 낮아진 정도: 중위 {(df.svf_bld - df.svf).median():.4f} · "
          f"최대 {(df.svf_bld - df.svf).max():.4f}")

    df.to_parquet(OUT, index=False)
    print(f"\n→ {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(df):,}센서)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
