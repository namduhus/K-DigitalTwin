from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import from_wkt
from shapely.geometry import Point
from shapely.strtree import STRtree

SENSORS = Path("data/processing/sdot_locations.csv")
BUILDINGS = Path("data/processing/buildings.parquet")
OUT = Path("data/processing/sensor_horizon.parquet")

AZ_STEP = 2.0                      # 방위 분해능(도) → 180개 구간
N_AZ = int(360 / AZ_STEP)
RADIUS = 600.0                     # 장애물 탐색 반경(m). 근거는 아래 주석
SAMPLE_R = 5.0                     # 센서 주변 샘플 반경(m)
SAMPLE_N = 8                       # 원주 샘플 수 (+ 중심 1개)
BOUNDARY_STEP = 3.0                # 건물 외곽선 샘플 간격(m)

# S-DoT 설치 높이. 정지우·남진(2022)에 "CCTV 폼대나 주민센터 등 2m~4m 이내"로 기록돼 있어 중간값을 쓴다. 센서 개별 높이는 공개 데이터에 없다.

# 이 보정이 필요한 이유: 장애물 앙각은 센서 눈높이에서 본 값이어야 한다.
# 지면 기준으로 계산하면 근거리에서 크게 과대평가된다 — 15m 건물이 10m 거리에 있을 때 지면 기준 56.3° vs 3m 기준 50.2° (차 6.1°).
SENSOR_H = 3.0

# RADIUS 근거: 그림자 길이 = H / tan(고도). 태양 고도 10°에서 100m 건물이 567m. 서울 건물 높이 99%분위가 64m이므로 600m면 실질적으로 충분하다.
# 초고층(최대 404m)의 저각 그림자는 놓치지만, 그 시각은 이미 태양이 매우 낮아 광역 지형·원거리 건물의 영향이 지배해 건물 단위 판정의 의미가 약하다.


def densify(coords: np.ndarray, step: float) -> np.ndarray:
    out = [coords[:-1]]
    seg = np.diff(coords, axis=0)
    length = np.hypot(seg[:, 0], seg[:, 1])
    for i, L in enumerate(length):
        if L > step:
            n = int(L / step)
            t = np.arange(1, n + 1)[:, None] / (n + 1)
            out.append(coords[i] + t * seg[i])
    return np.vstack(out)


def horizon_at(px: float, py: float, near: list[tuple[np.ndarray, float]]) -> np.ndarray:
    hz = np.zeros(N_AZ)
    for pts, h in near:
        dx = pts[:, 0] - px
        dy = pts[:, 1] - py
        d = np.hypot(dx, dy)
        d = np.maximum(d, 0.5)  # 건물 안/경계에 붙은 경우 발산 방지
        # 방위: 북=0°, 시계방향 (pvlib azimuth와 같은 규약)
        az = np.degrees(np.arctan2(dx, dy)) % 360.0
        el = np.degrees(np.arctan2(h, d))
        idx = (az / AZ_STEP).astype(np.int32) % N_AZ
        np.maximum.at(hz, idx, el)
    return hz


def svf_from_horizon(hz: np.ndarray) -> float:
    return float(1.0 - np.mean(np.sin(np.radians(hz)) ** 2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="센서 30개만 시험")
    a = ap.parse_args()

    for p in (SENSORS, BUILDINGS):
        if not p.exists():
            sys.exit(f"[error] {p} 없음")

    sen = pd.read_csv(SENSORS)
    bld = pd.read_parquet(BUILDINGS, columns=["gid", "height_m", "shadow_ok", "wkt"])
    bld = bld[bld["shadow_ok"]].reset_index(drop=True)
    print(f"센서 {len(sen):,}개 · 건물 {len(bld):,}동")

    geom = from_wkt(bld["wkt"].to_numpy())
    heights = bld["height_m"].to_numpy(float)
    tree = STRtree(geom)

    if a.probe:
        sen = sen.sample(30, random_state=0).reset_index(drop=True)
        print(f"[probe] 센서 {len(sen)}개만 계산")

    # 센서 주변 샘플 점 (중심 + 원주 8개)
    ang = np.arange(SAMPLE_N) * (2 * np.pi / SAMPLE_N)
    off = np.vstack([[0.0, 0.0], np.c_[SAMPLE_R * np.cos(ang), SAMPLE_R * np.sin(ang)]])

    rows = []
    for i, s in sen.iterrows():
        sx, sy = float(s["x"]), float(s["y"])
        pt = Point(sx, sy)

        # STRtree.query는 bbox 교차 후보를 준다. 실제 반경으로 다시 걸러야
        # 밀도 피처가 맞는다 (한 센서에서 후보 2,245동 → 실제 600m 이내 1,586동).
        cand = tree.query(pt.buffer(RADIUS))
        dist = np.array([geom[j].distance(pt) for j in cand])
        inr = cand[dist <= RADIUS]

        near = []
        for j in inr:
            pts = densify(np.asarray(geom[j].exterior.coords), BOUNDARY_STEP)
            # 센서 눈높이 기준 유효 높이. 센서보다 낮은 건물은 시야를 가리지 않는다.
            near.append((pts, max(heights[j] - SENSOR_H, 0.0)))

        hz = np.mean([horizon_at(sx + dx, sy + dy, near) for dx, dy in off], axis=0)

        # 건물 밀도 피처 — 축열·통풍의 대리 변수
        h_in = heights[inr]
        ar = sum(geom[j].area for j in inr)
        rec = {
            "sn": s["sn"],
            "svf": svf_from_horizon(hz),
            "hz_mean": float(hz.mean()),
            "hz_max": float(hz.max()),
            "bld_n": int(len(inr)),
            "bld_d_min": float(dist.min()) if len(dist) else np.nan,
            "bld_h_max": float(h_in.max()) if len(inr) else 0.0,
            "bld_h_mean": float(h_in.mean()) if len(inr) else 0.0,
            "bld_ar_frac": float(ar / (np.pi * RADIUS**2)),
        }
        rec.update({f"hz_{k:03d}": float(v) for k, v in enumerate(hz)})
        rows.append(rec)

        if (i + 1) % 50 == 0 or i + 1 == len(sen):
            print(f"  {i+1:>5}/{len(sen)}", end="\r", file=sys.stderr)

    print(file=sys.stderr)
    df = pd.DataFrame(rows)

    print(f"\n{'항목':<14} {'중위':>8} {'최소':>8} {'최대':>8}")
    print("-" * 42)
    for c, lab in [("svf", "천공률"), ("hz_mean", "평균 앙각(°)"), ("hz_max", "최대 앙각(°)"),
                   ("bld_n", "반경내 건물"), ("bld_d_min", "최근접(m)"), ("bld_h_max", "최고 건물(m)"),
                   ("bld_ar_frac", "건물 면적비")]:
        print(f"{lab:<14} {df[c].median():>8.3f} {df[c].min():>8.3f} {df[c].max():>8.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\n→ {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(df):,}센서 × {N_AZ}방위)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
