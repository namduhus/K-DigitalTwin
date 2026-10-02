from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from core.eval.dataset import (
    N_FOLDS,
    iter_folds,
    load_base,
    load_regional,
    load_sensor_meta,
    save_preds,
)

CELL_M = 5000.0   # 기상청 동네예보 격자 해상도


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=float, default=CELL_M, help="격자 한 변 (m, EPSG:5186)")
    a = ap.parse_args()

    base = load_base()
    regional = load_regional()
    meta = load_sensor_meta()

    # 5km 격자 칸 부여. EPSG:5186 미터 좌표를 격자 크기로 내림한다.
    xy = meta.loc[meta.index.intersection(base["sn"].unique()), ["x", "y"]]
    cell = (np.floor(xy["x"] / a.cell).astype(int).astype(str) + "_"
            + np.floor(xy["y"] / a.cell).astype(int).astype(str))
    cell.name = "cell"

    occupied = cell.nunique()
    print(f"\n격자 {a.cell:.0f} m — 센서가 있는 칸 **{occupied}개** · 센서 {len(xy):,}")
    per = cell.value_counts()
    print(f"  칸당 센서: 중위 {per.median():.0f} · 최소 {per.min()} · 최대 {per.max()}")
    print(f"  서울 범위 x {xy['x'].min():.0f}~{xy['x'].max():.0f} "
          f"({(xy['x'].max()-xy['x'].min())/1000:.1f} km) · "
          f"y {xy['y'].min():.0f}~{xy['y'].max():.0f} "
          f"({(xy['y'].max()-xy['y'].min())/1000:.1f} km)")

    sn_cell = base["sn"].map(cell).to_numpy()
    fold_of = base["fold"].to_numpy()
    ts = base["ts"].to_numpy()

    y_b0a = np.full(len(base), np.nan, np.float32)
    y_b0b = np.full(len(base), np.nan, np.float32)

    print(f"\n{'fold':<6}{'test행':>12}{'학습센서':>10}{'대체된 (칸,시각)':>18}{'비율':>8}")
    print("-" * 56)
    for fd in iter_folds(base, regional):
        m_test = fd.is_test
        y_b0a[m_test] = fd.t_reg[m_test]

        # 칸별 · 시각별 학습센서 편차 중위 → test 행에 조회
        tr = fd.is_train
        cell_med = (pd.DataFrame({"cell": sn_cell[tr], "ts": ts[tr], "d": fd.delta[tr]})
                    .groupby(["cell", "ts"], observed=True)["d"].median())
        idx = pd.MultiIndex.from_arrays([sn_cell[m_test], ts[m_test]])
        adj = cell_med.reindex(idx).to_numpy(np.float32)

        n_miss = int(np.isnan(adj).sum())
        # 학습센서가 없는 (칸, 시각)은 서울 전체 기준값으로 대체 = B0a와 동일
        adj = np.nan_to_num(adj, nan=0.0)
        y_b0b[m_test] = fd.t_reg[m_test] + adj

        print(f"{fd.f:<6}{int(m_test.sum()):>12,}{int(base.loc[fd.is_train,'sn'].nunique()):>10,}"
              f"{n_miss:>18,}{n_miss/max(int(m_test.sum()),1)*100:>7.2f}%")

    for name, y in (("b0a", y_b0a), ("b0b", y_b0b)):
        if np.isnan(y).any():
            # always-train 센서(fold=-1)는 어느 fold에서도 test가 아니므로 예측이 없다
            n = int(np.isnan(y).sum())
            print(f"  [{name}] 예측 없는 행 {n:,} — always-train 센서 "
                  f"{int((fold_of == -1).sum()):,}행과 일치해야 한다")
        p = save_preds(name, base["sn"], base["ts"], y)
        print(f"→ {p}")

    # 즉석 확인 — B0a MAE는 mean(|Δ|)와 같아야 한다 (하네스 검산은 evaluate.py가 한다)
    m = fold_of >= 0
    mae_a = float(np.abs(y_b0a[m] - base["tp"].to_numpy()[m]).mean())
    mae_b = float(np.abs(y_b0b[m] - base["tp"].to_numpy()[m]).mean())
    print(f"\n전체 MAE(참고, fold 접기 전)  B0a {mae_a:.3f}℃ · B0b {mae_b:.3f}℃ "
          f"· 개선 {(1-mae_b/mae_a)*100:+.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
