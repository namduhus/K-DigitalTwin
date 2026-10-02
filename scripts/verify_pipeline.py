from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# 리포 루트에서 실행한다. `scripts/`를 직접 실행하면 sys.path[0]이 scripts/라
# `core` 패키지를 못 찾으므로 루트도 넣는다.
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

# ── 기준값 (2026-08-24, Apple M4 Pro) ─────────────────────────────────
REF = {
    "센서(QC 통과)": 1127,
    "평가 대상 센서": 1076,
    "always-train 센서": 51,
    "궤적(20h+)": 126456,
    "평가 정본 행": 3646320,
    "시각": 4773,
    "자치구": 25,
    "서울 면적 km²": 605.7,
    "격자 셀": 969212,
    "격자 컬럼": 208,
    "건물 내부 셀": 122522,
    "B0a MAE": 0.7624,
    "B0b MAE": 0.6622,
    "B1 IDW MAE": 0.6217,
    "B3 MAE": 0.5582,
    # 분포 모델 (8/26 추가). 저장된 산출물을 읽어 대조한다 — 재계산하지 않는다.
    # 이 값들이 신청서에 그대로 들어가므로 플랫폼이 바뀌면 여기서 걸려야 한다.
    "t 자유도 ν": 4.0,
    "인접시각 상관": 0.861,
    "conformal Q(90%)": 2.909,
    "분포 CRPS": 0.4133,
    "90% 적중률": 0.900,
}
TOL = {"exact": 0, "geom": 1e-4, "numpy": 1e-3, "lgbm": 5e-3}

_rows: list[tuple] = []


def chk(name: str, actual, kind: str = "exact", ref=None) -> bool:
    exp = REF.get(name) if ref is None else ref
    if exp is None:
        _rows.append((name, "—", actual, "정보", ""))
        return True
    if kind == "exact":
        ok = actual == exp
        d = f"{actual - exp:+d}" if isinstance(actual, int) else "—"
    else:
        tol = TOL[kind]
        d = abs(actual - exp) / (abs(exp) if kind == "geom" else 1.0)
        ok = d <= tol
        d = f"{d:.2e} (허용 {tol:.0e})"
    _rows.append((name, exp, actual, "통과" if ok else "실패", d))
    return ok


def run(mod: str, *args) -> str:
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", mod, *args], capture_output=True, text=True)
    print(f"  {mod} {' '.join(args)} — {time.time()-t0:.0f}초 (exit {r.returncode})",
          file=sys.stderr)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:], file=sys.stderr)
        sys.exit(f"[error] {mod} 실패")
    return r.stdout


def run_script(path: str, *args) -> str:
    t0 = time.time()
    r = subprocess.run([sys.executable, path, *args], capture_output=True, text=True)
    print(f"  {path} {' '.join(args)} — {time.time()-t0:.0f}초 (exit {r.returncode})",
          file=sys.stderr)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:], file=sys.stderr)
        sys.exit(f"[error] {path} 실패")
    return r.stdout


# ── 개별 검증 ────────────────────────────────────────────────────────

def v_splits() -> None:
    run("core.eval.build_splits")
    sp = json.loads(Path("data/metric/splits.json").read_text())
    chk("평가 대상 센서", sp["n_sensor_eval"])
    chk("always-train 센서", sp["n_sensor_always_train"])
    chk("궤적(20h+)", sp["n_traj_total"])
    reg = pd.read_parquet("data/metric/regional.parquet")
    chk("시각", int(reg["ts"].nunique()))


def v_base_and_b0() -> None:
    from core.eval.dataset import load_base, load_regional, iter_folds
    base = load_base(verbose=False)
    chk("평가 정본 행", len(base))
    chk("센서(QC 통과)", int(base["sn"].nunique()))

    run("core.baseline.b0_regional")
    y = base["tp"].to_numpy(np.float32)
    fold = base["fold"].to_numpy()
    key = pd.MultiIndex.from_arrays([base["sn"], base["ts"]])
    maes = {}
    for name in ("b0a", "b0b"):
        p = pd.read_parquet(f"data/metric/preds/{name}.parquet")
        s = pd.Series(p["y_pred"].to_numpy(np.float32),
                      index=pd.MultiIndex.from_arrays([p["sn"].astype(str), p["ts"]]))
        v = s.reindex(key).to_numpy(np.float32)
        per = [float(np.abs(v[fold == f] - y[fold == f]).mean()) for f in range(5)]
        maes[name] = float(np.mean(per))
    chk("B0a MAE", round(maes["b0a"], 4), "numpy")
    chk("B0b MAE", round(maes["b0b"], 4), "numpy")

    # 하네스 검산 — B0a는 Δ̂=0 이므로 MAE가 mean(|Δ|)와 같아야 한다
    p = pd.read_parquet("data/metric/preds/b0a.parquet")
    s = pd.Series(p["y_pred"].to_numpy(np.float32),
                  index=pd.MultiIndex.from_arrays([p["sn"].astype(str), p["ts"]]))
    v = s.reindex(key).to_numpy(np.float32)
    worst = 0.0
    for fd in iter_folds(base, load_regional()):
        m = fold == fd.f
        worst = max(worst, abs(float(np.abs(fd.delta[m]).mean())
                               - float(np.abs(v[m] - y[m]).mean())))
    chk("하네스 검산 최대차", round(worst, 8), "numpy", ref=0.0)


def v_horizon(n: int = 40) -> None:
    import horizon_core as hc
    geoms, heights = hc.load_buildings()
    px, py, ph, off = hc.densify_all(geoms, heights)
    tree = hc.build_index(px, py)
    sen = pd.read_csv("data/processing/sdot_locations.csv")
    old = pd.read_parquet("data/processing/sensor_horizon.parquet").set_index("sn")
    sub = sen[sen.sn.isin(old.index)].head(n)
    hzc = [f"hz_{i:03d}" for i in range(180)]
    H = np.array([hc.horizon_one(float(s.x), float(s.y), tree, px, py, ph, off)
                  for _, s in sub.iterrows()])
    O = old.loc[sub.sn, hzc].to_numpy()
    chk("수평선 앙각 최대차 °", float(np.abs(H - O).max()), "numpy", ref=0.0)
    chk("SVF 최대차", float(np.abs(hc.svf_from_horizon(H) - old.loc[sub.sn, "svf"]).max()),
        "numpy", ref=0.0)


def v_terrain_green() -> None:
    out = run_script("scripts/build_grid_terrain.py", "--check", "--n-check", "40")
    ok = "전 항목 재현" in out
    _rows.append(("지형·녹지 재현", "전 항목 재현", "통과" if ok else "실패",
                  "통과" if ok else "실패", ""))


def v_boundary() -> None:
    b = pd.read_parquet("data/processing/seoul_boundary.parquet")
    chk("자치구", int((b["sig_cd"] != "11000").sum()))
    chk("서울 면적 km²", round(float(b.loc[b.sig_cd == "11000", "area_km2"].iloc[0]), 1), "geom")
    raw = Path("data/raw/vworld_adsigg.gml")
    _rows.append(("경계 원본 GML", "존재", "존재" if raw.exists() else "없음",
                  "통과" if raw.exists() else "실패",
                  f"{raw.stat().st_size/1e6:.1f} MB" if raw.exists() else "재요청 필요"))


def v_grid() -> None:
    import pyarrow.parquet as pq
    f = pq.ParquetFile("data/processing/grid_features.parquet")
    chk("격자 셀", f.metadata.num_rows)
    chk("격자 컬럼", len(f.schema_arrow.names))
    d = pd.read_parquet("data/processing/grid_features.parquet", columns=["in_building"])
    chk("건물 내부 셀", int(d["in_building"].sum()))


def v_b1() -> None:
    out = run("core.baseline.b1_interp")
    from core.eval.dataset import load_base
    base = load_base(verbose=False)
    y = base["tp"].to_numpy(np.float32); fold = base["fold"].to_numpy()
    key = pd.MultiIndex.from_arrays([base["sn"], base["ts"]])
    p = pd.read_parquet("data/metric/preds/b1_idw.parquet")
    s = pd.Series(p["y_pred"].to_numpy(np.float32),
                  index=pd.MultiIndex.from_arrays([p["sn"].astype(str), p["ts"]]))
    v = s.reindex(key).to_numpy(np.float32)
    per = [float(np.abs(v[fold == f] - y[fold == f]).mean()) for f in range(5)]
    chk("B1 IDW MAE", round(float(np.mean(per)), 4), "numpy")
    # 배리오그램 nugget 비율은 출력에서 뽑는다
    for line in out.splitlines():
        if "nugget 비율" in line:
            _rows.append(("배리오그램 nugget", "57%", line.split("nugget 비율")[1].strip()[:5],
                          "정보", ""))
            break


def v_b3() -> None:
    run("core.baseline.b3_hybrid")
    from core.eval.dataset import load_base
    base = load_base(verbose=False)
    y = base["tp"].to_numpy(np.float32); fold = base["fold"].to_numpy()
    key = pd.MultiIndex.from_arrays([base["sn"], base["ts"]])
    p = pd.read_parquet("data/metric/preds/b3_hybrid.parquet")
    s = pd.Series(p["y_pred"].to_numpy(np.float32),
                  index=pd.MultiIndex.from_arrays([p["sn"].astype(str), p["ts"]]))
    v = s.reindex(key).to_numpy(np.float32)
    per = [float(np.abs(v[fold == f] - y[fold == f]).mean()) for f in range(5)]
    chk("B3 MAE", round(float(np.mean(per)), 4), "lgbm")


def v_dist() -> None:
    d = Path("data/metric")
    if not (d / "traj_dist.npz").exists():
        _rows.append(("분포 모델 산출물", "존재", "없음 — core.model.traj_dist 미실행", "정보", ""))
        return
    z, cb = np.load(d / "traj_dist.npz"), np.load(d / "calib.npz")
    R = z["R"]
    chk("t 자유도 ν", float(z["nu"]), "numpy")
    chk("인접시각 상관", float(np.mean([R[i, i + 1] for i in range(R.shape[0] - 1)])), "lgbm")
    Q = cb["Q"][list(cb["levels"]).index(0.90)]
    chk("conformal Q(90%)", float(Q), "lgbm")

    at = d / "accuracy_table.parquet"
    if at.exists():
        t = pd.read_parquet(at)
        a = t[(t.slice_kind == "all") & (t.slice_val == "all")]
        chk("분포 CRPS", float(a.crps_dist.mean()), "lgbm")
        chk("90% 적중률", float(a.cover_dist.mean()), "lgbm")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="B1·B3 생략")
    a = ap.parse_args()

    print("=" * 78)
    print("파이프라인 재현성 검증")
    print("=" * 78)
    print(f"  플랫폼   {platform.platform()}")
    print(f"  머신     {platform.machine()} · Python {platform.python_version()}")
    print(f"  numpy {np.__version__} · pandas {pd.__version__}")
    try:
        import lightgbm
        print(f"  lightgbm {lightgbm.__version__}", end="")
    except ImportError:
        print("  lightgbm 없음", end="")
    try:
        import torch
        print(f" · torch {torch.__version__} (CUDA {torch.cuda.is_available()})")
    except ImportError:
        print(" · torch 없음")
    print(f"  기준값   2026-08-24 Apple M4 Pro / Python 3.12.13\n")

    t0 = time.time()
    stages = [("분할", v_splits), ("정본·B0", v_base_and_b0), ("수평선", v_horizon),
              ("지형·녹지", v_terrain_green), ("경계", v_boundary), ("격자", v_grid),
              ("분포 모델", v_dist)]
    if not a.quick:
        stages += [("B1", v_b1), ("B3", v_b3)]
    for label, fn in stages:
        print(f"[{label}]", file=sys.stderr)
        fn()

    print(f"\n  {'항목':<22}{'기준':>14}{'실측':>14}{'':>4}  차이")
    print("  " + "-" * 74)
    def fmt(x):
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return str(x)
        if isinstance(x, int):
            return f"{x:,}"
        return f"{x:.4g}" if abs(x) >= 1e-4 or x == 0 else f"{x:.2e}"
    for name, exp, act, mark, d in _rows:
        print(f"  {name:<22}{fmt(exp):>14}{fmt(act):>14}{mark:>4}  {d}")

    bad = [r[0] for r in _rows if r[3] == "실패"]
    print(f"\n  총 {len(_rows)}개 · 통과 {sum(1 for r in _rows if r[3]=='통과')} · "
          f"실패 {len(bad)} · 정보 {sum(1 for r in _rows if r[3]=='정보')}")
    print(f"  소요 {(time.time()-t0)/60:.1f}분")
    if bad:
        print(f"\n  실패: {', '.join(bad)}")
        print("  → **어느 쪽 수치를 신청서에 쓸지 먼저 정한다.** 조용히 새 값을 쓰면")
        print("     record.md 이력과 신청서가 어긋난다. 정정 시 record.md에 남긴다.")
        return 1
    print("\n  전 항목 재현 — 이 플랫폼에서 계속 진행 가능")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
