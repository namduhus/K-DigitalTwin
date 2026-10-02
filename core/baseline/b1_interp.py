from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from core.eval.dataset import (
    MIN_OBS_TRUST,
    N_FOLDS,
    iter_folds,
    load_base,
    load_regional,
    load_sensor_meta,
    save_preds,
    trusted_sensors,
)

POWERS = (1.0, 2.0, 3.0)
KS = (5, 10, 20, 50, 0)      # 0 = 학습 센서 전체
EPS_M = 1.0                  # 거리 0 방지 (동일 좌표 센서가 있으면 발산한다)


def idw_weights(d: np.ndarray, p: float, k: int) -> np.ndarray:
    w = 1.0 / np.maximum(d, EPS_M) ** p
    if k and k < d.shape[0]:
        # 각 test 열에서 가까운 k개만 남긴다
        cut = np.partition(d, k - 1, axis=0)[k - 1]     # k번째 최근접 거리
        w = np.where(d <= cut, w, 0.0)
    return w.astype(np.float32)


def predict(M: np.ndarray, W: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(M)
    num = np.nan_to_num(M) @ W
    den = ok.astype(np.float32) @ W
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(den > 0, out, 0.0)      # 이웃이 전무하면 Δ̂=0 (= B0a로 후퇴)


def pivot(fd, sn_col: np.ndarray, ts_row: np.ndarray, n_ts: int, n_sn: int) -> np.ndarray:
    M = np.full((n_ts, n_sn), np.nan, np.float32)
    M[ts_row, sn_col] = fd.delta
    return M


def sensor_grid(base):
    sns = np.array(sorted(base["sn"].unique()))
    tss = np.array(sorted(base["ts"].unique()))
    sn_col = base["sn"].map({s: i for i, s in enumerate(sns)}).to_numpy()
    ts_row = base["ts"].map({t: i for i, t in enumerate(tss)}).to_numpy()
    return sns, tss, sn_col, ts_row


def idw_oof(fd, base, D, sensor_fold, sn_col, ts_row, n_ts, untrusted,
            p: float, k: int) -> np.ndarray:
    M = pivot(fd, sn_col, ts_row, n_ts, len(D))
    tr = np.setdiff1d(np.where(sensor_fold != fd.f)[0], untrusted)
    W = idw_weights(D[np.ix_(tr, np.arange(len(D)))], p, k)     # (학습 × 전체센서)
    for i, s in enumerate(tr):
        W[i, s] = 0.0                                          # LOO — 자기 자신 제거
    pr = predict(M[:, tr], W)                                  # (시각 × 전체센서)
    return pr[ts_row, sn_col].astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-krige", action="store_true", help="배리오그램 생략")
    a = ap.parse_args()

    base = load_base()
    regional = load_regional()
    meta = load_sensor_meta()

    sns = np.array(sorted(base["sn"].unique()))
    tss = np.array(sorted(base["ts"].unique()))
    sn_ix = {s: i for i, s in enumerate(sns)}
    ts_ix = {t: i for i, t in enumerate(tss)}
    sn_col = base["sn"].map(sn_ix).to_numpy()
    ts_row = base["ts"].map(ts_ix).to_numpy()
    n_sn, n_ts = len(sns), len(tss)

    xy = meta.loc[sns, ["x", "y"]].to_numpy(np.float64)
    D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))    # (센서 × 센서) m
    nn = np.sort(D + np.eye(n_sn) * 1e9, axis=1)[:, 0]
    print(f"\n센서 {n_sn:,} · 시각 {n_ts:,} · 편차 행렬 {n_ts * n_sn * 4 / 1e6:.0f} MB")
    print(f"최근접 센서 거리: 중위 {np.median(nn):.0f} m · P05 {np.percentile(nn,5):.0f} "
          f"· P95 {np.percentile(nn,95):.0f} · 최대 {nn.max():.0f} m")

    sensor_fold = base.groupby("sn")["fold"].first().reindex(sns).to_numpy()

    # 미판정 센서를 이웃 풀에서 뺀다. `build_dataset`의 센서 QC가 관측 200개 미만을
    # "표본 부족으로 판정하지 않는다"로 통과시키는데, 편차 평균 극단값이 전부 거기서
    # 나온다 (`OC3CL200195` 관측 8개에 -10.59℃). 판정 안 한 센서를 이웃으로 믿을 수 없다.
    # 전부 always-train이라 평가 행 집합은 바뀌지 않는다.
    trust = set(trusted_sensors(base))
    untrusted = np.array([i for i, s in enumerate(sns) if s not in trust])
    if len(untrusted):
        print(f"  이웃 풀에서 제외: 미판정 센서 {len(untrusted)}개 "
              f"(관측 {MIN_OBS_TRUST} 미만, 전부 always-train)")

    y_pred = np.full(len(base), np.nan, np.float32)
    chosen = []

    print(f"\n{'fold':<6}{'학습':>7}{'test':>6}{'내부검증':>9}{'  최적 (p, k)':<16}{'내부 MAE':>10}{'초':>7}")
    print("-" * 62)
    for fd in iter_folds(base, regional):
        t0 = time.time()
        M = pivot(fd, sn_col, ts_row, n_ts, n_sn)
        tr = np.setdiff1d(np.where(sensor_fold != fd.f)[0], untrusted)
        te = np.where(sensor_fold == fd.f)[0]

        # 내부 검증 = 학습 센서 중 fold (f+1)%5 에 속한 것들 (fold f 기준 여전히 학습 대상)
        inner_val = np.setdiff1d(np.where(sensor_fold == (fd.f + 1) % N_FOLDS)[0], untrusted)
        inner_tr = np.setdiff1d(tr, inner_val)

        best, best_mae = None, np.inf
        for p in POWERS:
            for k in KS:
                W = idw_weights(D[np.ix_(inner_tr, inner_val)], p, k)
                pr = predict(M[:, inner_tr], W)
                truth = M[:, inner_val]
                m = ~np.isnan(truth)
                mae = float(np.abs(pr[m] - truth[m]).mean())
                if mae < best_mae:
                    best_mae, best = mae, (p, k)
        p, k = best
        chosen.append(best)

        # 최종 예측 — fold f의 전체 학습 센서로
        W = idw_weights(D[np.ix_(tr, te)], p, k)
        pr = predict(M[:, tr], W)                        # (시각 × test센서)
        m_test = fd.is_test
        y_pred[m_test] = fd.t_reg[m_test] + pr[ts_row[m_test], np.searchsorted(te, sn_col[m_test])]

        print(f"{fd.f:<6}{len(tr):>7}{len(te):>6}{len(inner_val):>9}"
              f"{f'  p={p:.0f}, k={k or 'all'}':<16}{best_mae:>10.4f}{time.time()-t0:>7.1f}")

    p = save_preds("b1_idw", base["sn"], base["ts"], y_pred)
    print(f"→ {p}")

    m = base["fold"].to_numpy() >= 0
    mae = float(np.abs(y_pred[m] - base["tp"].to_numpy()[m]).mean())
    print(f"\n전체 MAE(참고, fold 접기 전) B1 {mae:.3f}℃")
    print(f"선택된 (p, k): {chosen}")

    if not a.no_krige:
        variogram(base, regional, meta, sns)
    return 0


def variogram(base, regional, meta, sns) -> None:
    fd = next(iter_folds(base, regional))                  # fold 0 기준 편차
    z_all = pd.Series(fd.delta).groupby(base["sn"].to_numpy()).mean()
    trust = trusted_sensors(base)
    keep = [s for s in sns if s in set(trust)]
    z = z_all.reindex(keep).to_numpy()
    xy = meta.loc[keep, ["x", "y"]].to_numpy(np.float64)
    print(f"\n배리오그램 — 신뢰 센서 {len(z):,}개 (미판정 {len(sns)-len(keep)}개 제외)")
    print(f"  평균 편차 범위 {z.min():+.2f} ~ {z.max():+.2f}℃ · 분산 {z.var():.3f}")

    iu = np.triu_indices(len(z), k=1)
    d = np.sqrt(((xy[iu[0]] - xy[iu[1]]) ** 2).sum(1))
    g = 0.5 * (z[iu[0]] - z[iu[1]]) ** 2
    print(f"  쌍 {len(d):,}개 · 거리 {d.min():.0f} ~ {d.max():.0f} m")

    edges = np.r_[0, 100, 200, 300, 400, 500, 700, 1000, 1500, 2000, 3000, 5000, 8000, 12000]
    print(f"\n  {'거리 구간 (m)':<16}{'쌍':>9}{'세미분산 γ':>12}{'γ/전체분산':>12}")
    print("  " + "-" * 49)
    tot = float(z.var())
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & (d < hi)
        if m.sum() < 30:
            continue
        gg = float(g[m].mean())
        rows.append((lo, hi, int(m.sum()), gg))
        print(f"  {f'{lo:,}–{hi:,}':<16}{int(m.sum()):>9,}{gg:>12.3f}{gg/tot:>12.2f}")

    # nugget을 최단 구간 하나로 추정하지 않는다. 0–100m는 쌍이 49개뿐이라
    # 소수점을 믿을 수 없다 — 작은 n에 정밀도를 붙이는 오류를 이미 두 번 저질렀다
    # (`공원 -1.36℃` n=9, QC 임계값 근거). 짧은 거리 구간이 사실상 평탄하므로
    # **0–700m 전체를 쌍 수로 가중 평균**해 견고하게 잡는다.
    SHORT_M = 700
    short = [r for r in rows if r[1] <= SHORT_M]
    wn = sum(r[2] for r in short)
    nugget = sum(r[3] * r[2] for r in short) / wn
    plateau = max(r[3] for r in rows)
    at300 = next(r[3] for r in rows if r[0] <= 300 < r[1])

    print(f"\n  단거리 nugget (0–{SHORT_M}m, 쌍 {wn:,}개 가중평균) γ = **{nugget:.3f}**")
    print(f"     최대 γ(8–12km) = {plateau:.3f} → **nugget 비율 {nugget/plateau*100:.0f}%**")
    print(f"     최단 구간 하나(0–100m, 쌍 {rows[0][2]}개)는 γ={rows[0][3]:.3f}이지만 표본이 적어 인용하지 않는다")
    print(f"\n  실측 최근접 센서 거리 중위 **300m** 구간에서 γ/전체분산 = **{at300/tot:.2f}**")
    print(f"     = 가장 가까운 이웃으로부터도 편차 분산의 {at300/tot*100:.0f}%를 설명하지 못한다")
    print(f"     (`plan.md`의 '평균 740m 간격'은 면적÷센서수 명목값이다. 보간이 실제로")
    print(f"      건너야 하는 거리는 최근접 거리이므로 300m로 비교한다)")
    print(f"\n  명확한 plateau가 없다 — γ가 12km까지 단조 증가한다. 대규모 트렌드")
    print(f"     (표고·도심↔외곽 구배)가 있다는 뜻이고, IDW가 p=1·k=50을 고른 것과 정합한다:")
    print(f"     국지 구조가 약하니 멀리까지 넓게 평균하는 쪽이 이긴다")


if __name__ == "__main__":
    raise SystemExit(main())
