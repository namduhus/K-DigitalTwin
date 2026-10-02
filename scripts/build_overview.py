from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("demo/data/overview")
GRID = Path("data/processing/grid_features.parquet")
BOUNDARY = Path("data/processing/seoul_boundary.parquet")
pred_path = lambda day: Path(f"data/metric/grid_pred_{day}.parquet")

# 두 날의 **편차 색 범위를 공통으로** 묶는다 — 이게 이 토글의 전부다
#
#   실측 |편차| p90: 폭염일 1.76℃ · 평온일 1.25℃ (**1.41배**). 날마다 자기
#   범위를 쓰면 두 지도가 똑같이 진해 보여서 **격차 확대가 사라진다.**
#
#   반대로 **절대 기온은 공통으로 묶으면 안 된다.** 두 날의 중위가 38.4 vs
#      27.4℃로 11℃ 차이라 공통 범위(23.0~39.1)에서 평온일이 램프의 34%에
#      눌린다 — fig8에서 이미 겪은 함정이다 (`make_figures.py` fig8 주석).
#      절대 기온은 날마다 자기 범위를 쓰고, 범례가 숫자를 보여준다.
DAYS = ["2026-08-06", "2026-07-15"]
LABELS = {"2026-08-06": "폭염일", "2026-07-15": "평온일"}
CELL = 25.0
SRC_EPSG, DST_EPSG = 5186, 4326

# 램프는 `demo/index.html`과 **같은 값**이어야 한다. 그래서 여기서 굽고
#   manifest에 함께 실어 보낸다 — 뷰어가 전역 범례를 manifest로 그리면
#   파이썬이 칠한 픽셀과 범례가 원리적으로 어긋날 수 없다.
DIVERGING = [[44, 92, 163], [92, 140, 201], [158, 192, 225], [214, 229, 240],
             [246, 246, 246], [253, 219, 199], [244, 165, 130], [214, 96, 77],
             [165, 15, 21]]
SEQUENTIAL = [[255, 255, 204], [254, 217, 118], [254, 178, 76], [253, 141, 60],
              [252, 78, 42], [227, 26, 28], [177, 0, 38]]


# 색이 아니라 **값**을 굽는다 — 모드마다 이미지를 따로 만들지 않는다
#
#   기온을 색으로 칠해 저장하면 모드 수만큼 이미지가 늘어난다(실측: 48장 28.9 MB).
#   대신 기온을 8bit로 양자화해 **회색조 팔레트 PNG 한 벌(24장)**로 굽고,
#   뷰어가 램프를 입힌다. 그러면
#
#     ① 용량이 절반 — 두 모드가 같은 이미지를 공유한다
#     ② 동시각 편차가 **공짜** — `dev = t − median[시각]`이고 중위 24개는 manifest에 있다
#     ③ 범례가 원리적으로 어긋날 수 없다 — 값 하나에서 색과 눈금이 같이 나온다
#     ④ 모드를 하나 더 붙여도 이미지가 안 늘어난다
#
#   양자화 범위는 **분위가 아니라 실제 min/max**로 잡는다. P02–P98로 자르면
#      잘린 4%의 값이 뭉개지고, 그 위에서 계산하는 `dev`까지 같이 틀어진다.
#      표시용 클램프는 뷰어에서 한다 — **저장은 손실 없이, 표시만 잘라낸다.**
LEVELS = 255                                   # 인덱스 1..255 (0은 투명)


def quantize(a: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    idx = np.zeros(a.shape, np.uint8)
    m = np.isfinite(a)
    u = np.clip((a[m] - vmin) / (vmax - vmin), 0, 1)
    idx[m] = np.rint(u * (LEVELS - 1)).astype(np.uint8) + 1
    return idx


# 회색조 팔레트 — `palette[i] = (i, i, i)`라서 뷰어가 캔버스에서 읽은 **R 채널이
# 곧 인덱스**다. 색 관리가 끼어들 여지가 없도록 단조 회색으로 둔다.
GRAY_PALETTE = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, 1)


# |편차| 히스토그램 — 두 날을 합쳐 공통 범위를 잡되 배열을 통째로 들고 있지 않는다.
# (2일 × 24시각 × 1.68M px = 8,600만 개, float32로 344 MB다.)
HIST_HI, HIST_BINS = 8.0, 4000
HIST_EDGES = np.linspace(0.0, HIST_HI, HIST_BINS + 1)


def hist_quantile(hist: np.ndarray, q: float) -> float:
    c = np.cumsum(hist)
    return float(HIST_EDGES[1:][np.searchsorted(c, c[-1] * q)])


def hist_frac_above(hist: np.ndarray, v: float) -> float:
    c = np.cumsum(hist)
    return float(1.0 - c[np.searchsorted(HIST_EDGES[1:], v)] / c[-1]) * 100


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="+", default=DAYS)
    a = ap.parse_args()

    for p in (GRID, BOUNDARY):
        if not p.exists():
            sys.exit(f"[error] 없음: {p}")
    for d in a.days:
        if not pred_path(d).exists():
            sys.exit(f"[error] 없음: {pred_path(d)}")
    OUT.mkdir(parents=True, exist_ok=True)

    import pyarrow.parquet as pq
    import rasterio
    from PIL import Image
    from pyproj import Transformer
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    # ── 격자 틀 ──
    g = pq.read_table(GRID, columns=["gx", "gy", "x", "y", "in_building"]).to_pandas()
    gx0, gx1 = int(g.gx.min()), int(g.gx.max())
    gy0, gy1 = int(g.gy.min()), int(g.gy.max())
    nx, ny = gx1 - gx0 + 1, gy1 - gy0 + 1
    # 행 0이 **북쪽**이어야 이미지 위가 북쪽이 된다 — gy가 클수록 북쪽이므로 뒤집는다
    col = (g.gx.to_numpy() - gx0).astype(np.int32)
    row = (gy1 - g.gy.to_numpy()).astype(np.int32)
    outdoor = ~g.in_building.to_numpy(bool)
    print(f"[격자] {nx} × {ny} · 채워진 셀 {len(g):,} ({len(g)/(nx*ny):.1%}) · "
          f"옥외 {outdoor.sum():,}")

    # 좌상단 픽셀의 **모서리** 좌표 (셀 중심이 x = gx*25 + 12.5 이므로 12.5를 뺀다)
    west = gx0 * CELL
    north = (gy1 + 1) * CELL
    src_tf = rasterio.transform.from_origin(west, north, CELL, CELL)

    # ── 재투영 틀 (날짜와 무관하므로 한 번만) ──
    dst_tf, dw, dh = calculate_default_transform(
        f"EPSG:{SRC_EPSG}", f"EPSG:{DST_EPSG}", nx, ny,
        left=west, bottom=north - ny * CELL, right=west + nx * CELL, top=north)
    W, S = dst_tf * (0, dh)
    E, N = dst_tf * (dw, 0)
    print(f"[재투영] {nx}×{ny} (EPSG:{SRC_EPSG}) → {dw}×{dh} (EPSG:{DST_EPSG})")
    print(f"         bounds W {W:.6f} S {S:.6f} E {E:.6f} N {N:.6f}")
    src_kw = dict(src_transform=src_tf, src_crs=f"EPSG:{SRC_EPSG}",
                  dst_transform=dst_tf, dst_crs=f"EPSG:{DST_EPSG}",
                  resampling=Resampling.nearest)

    # ── 서울 경계 (방위 감각용 · 날짜 무관) ──
    from shapely import from_wkt
    from shapely.ops import transform as shp_transform
    b = pd.read_parquet(BOUNDARY, columns=["sig_kor_nm", "wkt"])
    tr = Transformer.from_crs(SRC_EPSG, DST_EPSG, always_xy=True).transform
    paths = []
    for w in b.wkt:
        geo = shp_transform(tr, from_wkt(w))
        for poly in (geo.geoms if geo.geom_type == "MultiPolygon" else [geo]):
            c = np.asarray(poly.exterior.simplify(0.0002).coords)
            if len(c) >= 4:
                paths.append([[round(float(x), 6), round(float(y), 6)] for x, y in c])
    print(f"[경계] {len(b)}개 구 → 링 {len(paths)}개")

    key = pd.MultiIndex.from_arrays([g.gx, g.gy])
    gidx = pd.Series(np.arange(len(g)), index=key)
    out_r, out_c = row[outdoor], col[outdoor]

    # ── 날짜별로 값 래스터를 굽는다 ──
    per_day, total = [], 0
    for day_key in a.days:
        print(f"\n{'='*66}\n[{LABELS.get(day_key, day_key)}] {day_key}\n{'='*66}")
        dst = OUT / day_key
        dst.mkdir(parents=True, exist_ok=True)

        tt = pq.read_table(pred_path(day_key),
                           columns=["gx", "gy", "ts", "t_hat"]).to_pandas()
        tt["ts"] = pd.to_datetime(tt.ts)
        tt = tt[tt.ts.dt.normalize() == pd.Timestamp(day_key)]
        hours = sorted(tt.ts.dt.hour.unique().tolist())
        print(f"[예측] {len(tt):,}행 · 시각 {len(hours)}개")

        cube = np.full((len(hours), ny, nx), np.nan, np.float32)
        pos = {h: i for i, h in enumerate(hours)}
        ci = gidx.reindex(pd.MultiIndex.from_arrays([tt.gx, tt.gy])).to_numpy()
        if np.isnan(ci).any():
            sys.exit(f"[error] 예측에 격자에 없는 셀 {int(np.isnan(ci).sum()):,}개")
        ci = ci.astype(np.int64)
        cube[tt.ts.dt.hour.map(pos).to_numpy(), row[ci], col[ci]] = tt.t_hat.to_numpy()

        # 동시각 편차 — 옥외 셀 중위 대비 (CLAUDE.md 정의)
        med = np.array([np.nanmedian(cube[k, out_r, out_c]) for k in range(len(hours))])
        dev = cube - med[:, None, None]
        print(f"[중위] 옥외 {outdoor.sum():,}셀 · {med.min():.1f} ~ {med.max():.1f}℃ "
              f"(16시 {med[16]:.1f})")

        # 절대 기온 범위는 **날마다 따로** (공통이면 평온일이 눌린다)
        fin_t = cube[np.isfinite(cube)]
        lo_t, hi_t = float(np.quantile(fin_t, .02)), float(np.quantile(fin_t, .98))
        clamp_t = float(((fin_t < lo_t) | (fin_t > hi_t)).mean() * 100)
        # 편차는 히스토그램만 모아 두고 범위는 두 날을 합쳐 뒤에서 정한다
        hist = np.histogram(np.abs(dev[np.isfinite(dev)]), bins=HIST_EDGES)[0]
        print(f"[범위] t {lo_t:.1f}~{hi_t:.1f}℃ (밖 {clamp_t:.1f}%) · "
              f"|편차| p90 {hist_quantile(hist, .90):.2f}℃")

        vmin, vmax = float(np.nanmin(cube)), float(np.nanmax(cube))
        step = (vmax - vmin) / (LEVELS - 1)
        print(f"[양자화] {vmin:.2f} ~ {vmax:.2f}℃ · {LEVELS}단계 · 눈금 {step:.3f}℃")

        bias = {"n": 0.0, "med": 0.0, "mean": 0.0, "std": 0.0}
        err, size = 0.0, 0
        for k, h in enumerate(hours):
            # 값을 먼저 재투영하고 **그 다음에** 양자화한다.
            warped = np.full((dh, dw), np.nan, np.float32)
            reproject(source=cube[k], destination=warped,
                      src_nodata=np.nan, dst_nodata=np.nan, **src_kw)
            idx = quantize(warped, vmin, vmax)
            m = np.isfinite(warped)
            back = vmin + (idx[m].astype(np.float64) - 1) / (LEVELS - 1) * (vmax - vmin)
            err = max(err, float(np.abs(back - warped[m]).max()))

            s_, w_ = cube[k][np.isfinite(cube[k])], warped[m]
            bias["n"] = w_.size / s_.size
            for kk, f in (("med", np.median), ("mean", np.mean), ("std", np.std)):
                bias[kk] = max(bias[kk], abs(float(f(w_)) - float(f(s_))))

            im = Image.fromarray(idx, "P")
            im.putpalette(GRAY_PALETTE.flatten().tolist())
            fp = dst / f"v_{h:02d}.png"
            im.save(fp, optimize=True, transparency=0)
            size += fp.stat().st_size
        total += size
        print(f"  값 래스터 {len(hours)}장 {size/1e6:.1f} MB · 왕복 최대오차 {err:.4f}℃ "
              f"(눈금 절반 {step/2:.4f} 이하여야 정상)")
        if err > step / 2 + 1e-6:
            sys.exit(f"[error] 양자화 왕복 오차 {err:.4f}℃ > 눈금 절반 {step/2:.4f}")
        print(f"  재투영 편향: 픽셀수 {bias['n']:.3f}배 · 중위차 {bias['med']:.4f}℃ · "
              f"평균차 {bias['mean']:.4f}℃ · 표준편차차 {bias['std']:.4f}℃")
        if max(bias["med"], bias["mean"]) > 0.02:
            sys.exit("[error] 재투영이 값을 편향시켰다 — 목적 해상도를 올려야 한다")

        per_day.append({
            "key": day_key, "label": LABELS.get(day_key, day_key), "hours": hours,
            "median": [round(float(v), 2) for v in med], "hist": hist,
            "t_range": [lo_t, hi_t], "t_clamp": clamp_t,
            "encode": {"vmin": vmin, "vmax": vmax, "levels": LEVELS, "step": step,
                       "file": "v_{HH}.png"},
        })

    # ── 편차 색 범위는 **두 날 공통** ──
    #   날마다 자기 범위를 쓰면 두 지도가 똑같이 진해 보여 격차 확대가 사라진다.
    #   합집합 분포의 p90을 쓰지 않고 **각 날 p90의 최대**를 쓴다 — 시각 수가
    #   같아 합집합 p90은 두 날의 중간으로 끌려가고, 그러면 더운 날의 클램프가
    #   과해진다. 최대를 쓰면 평온일이 반드시 안에 들어온다.
    p90s = {d["key"]: hist_quantile(d["hist"], .90) for d in per_day}
    m_d = max(.3, round(max(p90s.values()), 1))
    print(f"\n[공통 편차 범위] ±{m_d:.1f}℃  "
          f"(날별 p90: {', '.join(f'{k} {v:.2f}' for k, v in p90s.items())})")

    for d in per_day:
        dev_clamp = hist_frac_above(d["hist"], m_d)
        manifest = {
            "day": d["key"], "label": d["label"], "hours": d["hours"],
            "bounds": [W, S, E, N],          # BitmapLayer 규약: [W, S, E, N]
            "size": [int(dw), int(dh)],
            "cell_m": CELL, "cells": int(len(g)), "outdoor": int(outdoor.sum()),
            # 값 복원: v = vmin + (R − 1) / (levels − 1) × (vmax − vmin)
            "encode": d["encode"],
            "median": d["median"],
            "ranges": {"t": d["t_range"], "dev": [-m_d, m_d]},
            "clamp": {"t": round(d["t_clamp"], 2), "dev": round(dev_clamp, 2)},
            "ramps": {"t": SEQUENTIAL, "dev": DIVERGING},
            "layers": {
                "t":   {"label": "기온 (절대)", "unit": "℃", "diverging": False,
                        "note": "하루 전체 P02–P98 · 날마다 범위가 다르다"},
                "dev": {"label": "동시각 편차", "unit": "℃", "diverging": True,
                        "note": "같은 시각 서울 전역 옥외 셀 중위와의 차이 · 두 날 공통 범위"},
            },
            "boundary": paths,
        }
        f = OUT / d["key"] / "manifest.json"
        f.write_text(json.dumps(manifest, separators=(",", ":")))
        print(f"  {d['label']} {d['key']}: 편차 범위 밖 {dev_clamp:.1f}% · "
              f"manifest {f.stat().st_size/1e3:.0f} KB")

    print(f"\n→ {OUT}  PNG 합계 {total/1e6:.1f} MB · {len(per_day)}일")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
