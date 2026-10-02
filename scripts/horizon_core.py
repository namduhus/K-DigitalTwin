from __future__ import annotations

import numpy as np
import pandas as pd
from shapely import from_wkt, get_coordinates, get_parts
from shapely.geometry import Point
from shapely.strtree import STRtree

AZ_STEP = 2.0
N_AZ = int(360 / AZ_STEP)
RADIUS = 600.0
SAMPLE_R = 5.0
SAMPLE_N = 8
BOUNDARY_STEP = 3.0
SENSOR_H = 3.0
MIN_D = 0.5          # 거리 0 발산 방지 (건물 안/경계에 붙은 경우)


# ────────────────────────────────────────────────── 전역 사전계산

def load_buildings(path: str = "data/processing/buildings.parquet"):
    b = pd.read_parquet(path, columns=["gid", "height_m", "shadow_ok", "wkt"])
    b = b[b["shadow_ok"]].reset_index(drop=True)
    return from_wkt(b["wkt"].to_numpy()), b["height_m"].to_numpy(np.float64)


def densify_all(geoms, heights, step: float = BOUNDARY_STEP):
    rings = np.array([g.exterior for g in geoms], dtype=object)
    co = get_coordinates(rings)
    nvert = np.array([len(g.exterior.coords) for g in geoms])
    off = np.r_[0, np.cumsum(nvert)]

    xs, ys, hs = [], [], []
    for i in range(len(geoms)):
        c = co[off[i]:off[i + 1]]
        v = c[:-1]                                    # 닫는 점 제외
        seg = np.diff(c, axis=0)
        L = np.hypot(seg[:, 0], seg[:, 1])
        parts = [v]
        long = np.where(L > step)[0]
        for j in long:
            n = int(L[j] / step)
            t = (np.arange(1, n + 1) / (n + 1))[:, None]
            parts.append(c[j] + t * seg[j])
        p = np.vstack(parts)
        xs.append(p[:, 0]); ys.append(p[:, 1])
        hs.append(np.full(len(p), heights[i]))
    lens = np.array([len(a) for a in xs])
    return (np.concatenate(xs).astype(np.float64),
            np.concatenate(ys).astype(np.float64),
            np.concatenate(hs).astype(np.float32),
            np.r_[0, np.cumsum(lens)].astype(np.int64))   # 건물 → 점 범위 오프셋


def build_index(px, py):
    from scipy.spatial import cKDTree
    return cKDTree(np.c_[px, py])


# ────────────────────────────────────────────────── 질의

def _ranges_to_idx(starts: np.ndarray, lens: np.ndarray) -> np.ndarray:
    total = int(lens.sum())
    if total == 0:
        return np.empty(0, np.int64)
    out = np.ones(total, np.int64)
    csum = np.cumsum(lens)
    out[0] = starts[0]
    if len(starts) > 1:
        out[csum[:-1]] = starts[1:] - (starts[:-1] + lens[:-1]) + 1
    return np.cumsum(out)


def horizon_one(qx: float, qy: float, tree, px, py, ph, bld_off, eye_h: float = SENSOR_H,
                sample: bool = True) -> np.ndarray:
    if sample:
        ang = np.arange(SAMPLE_N) * (2 * np.pi / SAMPLE_N)
        off = np.vstack([[0.0, 0.0], np.c_[SAMPLE_R * np.cos(ang), SAMPLE_R * np.sin(ang)]])
    else:
        off = np.zeros((1, 2))

    hit = tree.query_ball_point((qx, qy), RADIUS)
    if not len(hit):
        return np.zeros(N_AZ)
    # 점 → 건물 → 그 건물의 전체 점
    bids = np.unique(np.searchsorted(bld_off, np.asarray(hit), side="right") - 1)
    idx = _ranges_to_idx(bld_off[bids], bld_off[bids + 1] - bld_off[bids])
    cx, cy = px[idx], py[idx]
    ch = np.maximum(ph[idx] - eye_h, 0.0)

    acc = np.zeros(N_AZ)
    for dx0, dy0 in off:
        dx, dy = cx - (qx + dx0), cy - (qy + dy0)
        d = np.maximum(np.hypot(dx, dy), MIN_D)
        az = np.degrees(np.arctan2(dx, dy)) % 360.0
        el = np.degrees(np.arctan2(ch, d))
        hz = np.zeros(N_AZ)
        np.maximum.at(hz, (az / AZ_STEP).astype(np.int32) % N_AZ, el)
        acc += hz
    return acc / len(off)


def svf_from_horizon(hz: np.ndarray) -> np.ndarray:
    return 1.0 - np.mean(np.sin(np.radians(hz)) ** 2, axis=-1)


def density_one(qx: float, qy: float, ptree, geoms, heights) -> dict:
    pt = Point(qx, qy)
    cand = ptree.query(pt.buffer(RADIUS))
    if not len(cand):
        return dict(bld_n=0, bld_d_min=np.nan, bld_h_max=0.0, bld_h_mean=0.0, bld_ar_frac=0.0)
    dist = np.array([geoms[j].distance(pt) for j in cand])
    inr = cand[dist <= RADIUS]
    if not len(inr):
        return dict(bld_n=0, bld_d_min=float(dist.min()), bld_h_max=0.0,
                    bld_h_mean=0.0, bld_ar_frac=0.0)
    h = heights[inr]
    return dict(bld_n=int(len(inr)), bld_d_min=float(dist.min()),
                bld_h_max=float(h.max()), bld_h_mean=float(h.mean()),
                bld_ar_frac=float(sum(geoms[j].area for j in inr) / (np.pi * RADIUS ** 2)))
