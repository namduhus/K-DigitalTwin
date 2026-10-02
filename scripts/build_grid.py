from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import from_wkt
from shapely.geometry import Point
from shapely.strtree import STRtree

sys.path.insert(0, "scripts")
import horizon_core as hc                                    # noqa: E402

BOUNDARY = Path("data/processing/seoul_boundary.parquet")
OUT = Path("data/processing/grid_features.parquet")
TILE = 400          # 타일 한 변 (셀 수). 400×400 = 16만 셀 단위로 나눈다

_G: dict = {}       # 워커가 fork로 공유하는 전역 사전계산


def make_grid(cell: float) -> pd.DataFrame:
    b = pd.read_parquet(BOUNDARY)
    gu = b[b["sig_cd"] != "11000"].copy()
    gu["geom"] = from_wkt(gu["wkt"].to_numpy())
    seoul = from_wkt(b.loc[b["sig_cd"] == "11000", "wkt"].iloc[0])

    x0, y0, x1, y1 = seoul.bounds
    gx = np.arange(int(x0 // cell), int(x1 // cell) + 1)
    gy = np.arange(int(y0 // cell), int(y1 // cell) + 1)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    X = (GX + 0.5) * cell
    Y = (GY + 0.5) * cell
    print(f"bbox 격자 {len(gx)} × {len(gy)} = {GX.size:,}셀")

    # 자치구 폴리곤으로 한 번에 판정 — 서울 경계 내 여부와 자치구 라벨을 동시에 얻는다
    tree = STRtree(gu["geom"].to_numpy())
    pts = np.array([Point(a, b_) for a, b_ in zip(X.ravel(), Y.ravel())], dtype=object)
    hit = tree.query(pts, predicate="within")                # (2, n) — [질의idx, 폴리곤idx]
    lab = np.full(GX.size, -1, np.int32)
    lab[hit[0]] = hit[1]
    keep = lab >= 0
    print(f"  서울 경계 내 **{int(keep.sum()):,}셀** ({keep.mean()*100:.0f}%)")

    return pd.DataFrame({
        "gx": GX.ravel()[keep], "gy": GY.ravel()[keep],
        "x": X.ravel()[keep], "y": Y.ravel()[keep],
        "gu": gu["sig_kor_nm"].to_numpy()[lab[keep]],
    })


def precompute() -> None:
    if "tree" in _G:
        return
    t0 = time.time()
    geoms, heights = hc.load_buildings()
    px, py, ph, off = hc.densify_all(geoms, heights)
    _G.update(geoms=geoms, heights=heights, px=px, py=py, ph=ph, off=off,
              tree=hc.build_index(px, py), ptree=STRtree(geoms))
    print(f"사전계산 — 건물 {len(geoms):,}동 · 점 {len(px):,} "
          f"({(px.nbytes+py.nbytes+ph.nbytes)/1e6:.0f} MB) · {time.time()-t0:.1f}초")


def work(chunk: pd.DataFrame) -> pd.DataFrame:
    g = _G
    hzs, dens, inb = [], [], []
    for x, y in zip(chunk["x"].to_numpy(), chunk["y"].to_numpy()):
        hzs.append(hc.horizon_one(x, y, g["tree"], g["px"], g["py"], g["ph"], g["off"]))
        dens.append(hc.density_one(x, y, g["ptree"], g["geoms"], g["heights"]))
        inb.append(bool(len(g["ptree"].query(Point(x, y), predicate="within"))))
    H = np.asarray(hzs, np.float32)
    cols = {"svf": hc.svf_from_horizon(H).astype(np.float32),
            "hz_mean": H.mean(1).astype(np.float32),
            "hz_max": H.max(1).astype(np.float32)}
    for k in ("bld_n", "bld_d_min", "bld_h_max", "bld_h_mean", "bld_ar_frac"):
        cols[k] = np.array([d[k] for d in dens], np.float32)
    cols["in_building"] = np.array(inb)
    # 수평선 프로파일 180열은 float16으로 둔다. 969,212셀 × 180 × 4B = 700 MB가
    # 350 MB로 줄고, 앙각 정밀도는 20° 부근에서 0.01° 수준이라 음영 판정에 충분하다.
    # (프로파일은 버릴 수 없다 — 임의 시각의 `hz_sun`을 뽑는 데 필요하다.)
    for i in range(hc.N_AZ):
        cols[f"hz_{i:03d}"] = H[:, i].astype(np.float16)
    return pd.concat([chunk.reset_index(drop=True), pd.DataFrame(cols)], axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=float, default=25.0)
    ap.add_argument("--probe", action="store_true", help="1,000셀만")
    ap.add_argument("--jobs", type=int, default=0, help="0 = CPU-2")
    a = ap.parse_args()

    t0 = time.time()
    grid = make_grid(a.cell)
    if a.probe:
        grid = grid.sample(1000, random_state=0).reset_index(drop=True)
        print(f"[probe] {len(grid):,}셀만 계산")

    precompute()

    n_job = a.jobs or max(mp.cpu_count() - 2, 1)
    # DataFrame에 np.array_split을 쓰면 ndarray로 변환돼 컬럼 접근이 깨진다.
    # 인덱스를 쪼개고 iloc으로 슬라이스한다.
    n_chunk = max(len(grid) // 8_000, n_job)
    chunks = [grid.iloc[i] for i in np.array_split(np.arange(len(grid)), n_chunk)]
    print(f"\n워커 {n_job}개 · 청크 {len(chunks)}개 (평균 {len(grid)//len(chunks):,}셀)")
    print(f"  실측 10.6 ms/셀(수평선 6.5 + 밀도 4.1) 기준 예상 "
          f"{len(grid)*10.6/1000/60/n_job:.0f}분")

    ctx = mp.get_context("fork")        # spawn이면 274MB 사전계산을 워커마다 다시 만든다
    done, parts = 0, []
    with ctx.Pool(n_job) as pool:
        for r in pool.imap_unordered(work, chunks):
            parts.append(r)
            done += len(r)
            el = time.time() - t0
            print(f"  {done:>9,}/{len(grid):,} ({done/len(grid)*100:>5.1f}%) · "
                  f"{el/60:>5.1f}분 경과 · 남은 {el/max(done,1)*(len(grid)-done)/60:>5.1f}분",
                  end="\r", file=sys.stderr)
    print(file=sys.stderr)

    df = pd.concat(parts, ignore_index=True).sort_values(["gx", "gy"]).reset_index(drop=True)
    # 격자 분포 vs 센서 분포 — 외삽 위험을 정량화한다.
    # 센서는 서울의 대표 표본이 아니다(밀집지에 설치됨). 격자 셀의 상당 비율이
    # 센서 분포의 꼬리에 있으면 모델이 외삽하게 되고, 그건 신청서 한계로 써야 한다.
    sen = pd.read_parquet("data/processing/sensor_horizon.parquet")
    print(f"\n{'피처':<14}{'격자 중위':>10}{'센서 중위':>10}{'격자 P05':>10}{'격자 P95':>10}"
          f"{'센서 P95 초과':>14}")
    print("-" * 70)
    for c in ("svf", "hz_mean", "hz_max", "bld_n", "bld_d_min", "bld_h_max", "bld_ar_frac"):
        if c not in sen.columns:
            print(f"{c:<14}{df[c].median():>10.3f}{'—':>10}")
            continue
        p95 = sen[c].quantile(0.95); p05 = sen[c].quantile(0.05)
        outside = ((df[c] > p95) | (df[c] < p05)).mean()
        print(f"{c:<14}{df[c].median():>10.3f}{sen[c].median():>10.3f}"
              f"{df[c].quantile(.05):>10.3f}{df[c].quantile(.95):>10.3f}{outside*100:>13.1f}%")
    print("  마지막 열 = 격자 셀 중 **센서 P05~P95 범위 밖** 비율 = 외삽 구간")
    print(f"\n건물 내부 셀 {int(df['in_building'].sum()):,} ({df['in_building'].mean()*100:.1f}%)")
    print(f"자치구 {df['gu'].nunique()}개 · 셀 최다 {df['gu'].value_counts().idxmax()} "
          f"{df['gu'].value_counts().max():,} · 최소 {df['gu'].value_counts().idxmin()} "
          f"{df['gu'].value_counts().min():,}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\n→ {OUT} ({OUT.stat().st_size/1e6:.0f} MB, {len(df):,}셀 × {df.shape[1]}컬럼)")
    print(f"총 {(time.time()-t0)/60:.1f}분")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
