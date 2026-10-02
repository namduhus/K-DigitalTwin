from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from core.eval.dataset import N_FOLDS, PREDS, hot_timestamps, load_base

DIST = Path("data/metric/traj_dist.npz")
OUT = Path("data/metric/calib.npz")

LEVELS = [0.50, 0.80, 0.90, 0.95]
N_SAMP = 200          # CRPS·적중률용 표본 수
N_SIGMA_BIN = 5
SEED = 20260826


def crps_from_samples(X, y):
    m = X.shape[1]
    Xs = np.sort(X, axis=1)
    term1 = np.abs(Xs - y[:, None]).mean(1)
    # E|X−X'| = (2/m²)·Σ_k (2k − m + 1)·x_(k)   (k는 0부터)
    k = np.arange(m, dtype=np.float64)
    term2 = (Xs * (2 * k - m + 1)).sum(1) * (2.0 / (m * m))
    return term1 - 0.5 * term2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-rows", type=int, default=1_200_000,
                    help="CRPS 계산 표본 행 수 (0이면 전체)")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)

    z = np.load(DIST)
    R, nu, z_sd = z["R"], float(z["nu"]), float(z["z_sd"])

    base = load_base(verbose=False)
    base = base[base.fold >= 0]
    b3 = pd.read_parquet(PREDS / "b3_hybrid.parquet"); b3["sn"] = b3["sn"].astype(str)
    sc = pd.read_parquet(PREDS / "b3_scale.parquet"); sc["sn"] = sc["sn"].astype(str)
    d = (base.merge(b3, on=["sn", "ts"]).merge(sc[["sn", "ts", "sigma"]], on=["sn", "ts"])
         .dropna(subset=["y_pred", "sigma"]))
    y = d.tp.to_numpy(np.float64)
    mu = d.y_pred.to_numpy(np.float64)
    sg = d.sigma.to_numpy(np.float64)
    fold = d.fold.to_numpy()
    E = np.abs(y - mu) / sg                       # 척도로 나눈 점수
    n = len(d)
    print(f"[행] {n:,} · 센서 {d.sn.nunique():,} · σ 중위 {np.median(sg):.3f}℃")
    print(f"     |잔차|/σ  중위 {np.median(E):.3f} · P90 {np.quantile(E,.9):.3f} · "
          f"최대 {E.max():.1f}")

    # ── ① 보정 전 — 모수적 구간의 실제 적중률 ──
    from scipy import stats
    print(f"\n{'='*76}\n① 보정 전 — 모수 가정만으로 만든 구간이 이름값을 지키는가\n{'='*76}")
    print(f"  {'이름값':>8}{'가우시안(σ 그대로)':>20}{'가우시안(×z_sd)':>18}"
          f"{'t(ν=4)(×z_sd)':>18}")
    print("  " + "-" * 62)
    pre = {}
    for lv in LEVELS:
        p = 1 - (1 - lv) / 2
        q_n, q_t = stats.norm.ppf(p), stats.t.ppf(p, nu) * np.sqrt((nu - 2) / nu)
        c_raw = float((E <= q_n).mean())
        c_gau = float((E <= q_n * z_sd).mean())
        c_t = float((E <= q_t * z_sd).mean())
        pre[lv] = (c_raw, c_gau, c_t)
        print(f"  {lv*100:>7.0f}%{c_raw*100:>19.1f}%{c_gau*100:>17.1f}%{c_t*100:>17.1f}%")
    print(f"\n  σ를 그대로 쓰면 크게 모자란다. `z_sd`({z_sd:.3f}배)를 곱하면 가까워지지만")
    print(f"    **이름값을 정확히 맞추지는 못한다** — 그것을 conformal이 한다")

    # ── ② split conformal (leave-one-fold-out) ──
    print(f"\n{'='*76}\n② split conformal — 보정과 평가를 다른 센서에서 한다\n{'='*76}")
    print(f"  {'이름값':>8}{'보정계수 Q (fold 평균)':>24}{'보정 후 적중률':>18}{'평균 구간폭':>14}")
    print("  " + "-" * 64)
    Qs, post = {}, {}
    for lv in LEVELS:
        qf, cov, wid = [], [], []
        for f in range(N_FOLDS):
            cal, tst = fold != f, fold == f
            m = int(cal.sum())
            # 유한표본 보정 (1−α)(1+1/n) — 이것이 conformal의 보증을 만든다
            q = float(np.quantile(E[cal], min(lv * (1 + 1 / m), 1.0)))
            qf.append(q)
            cov.append(float((E[tst] <= q).mean()))
            wid.append(float((2 * q * sg[tst]).mean()))
        Qs[lv], post[lv] = float(np.mean(qf)), float(np.mean(cov))
        print(f"  {lv*100:>7.0f}%{np.mean(qf):>19.3f} ± {np.std(qf):.3f}"
              f"{np.mean(cov)*100:>17.1f}%{np.mean(wid):>13.3f}℃")
    print(f"\n  90% 보정계수 **{Qs[0.90]:.3f}** vs 전역 배수 z_sd×1.645 = "
          f"{z_sd*1.645:.3f} — conformal이 더 {'크다' if Qs[0.90] > z_sd*1.645 else '작다'}")

    # ── ③ σ 구간별 조건부 적중률 — 이중 계상 판정 ──
    print(f"\n{'='*76}\n③ σ 구간별 조건부 적중률 — **격자 이중 계상 의심을 판정한다**\n{'='*76}")
    print(f"  척도가 큰 구간(외삽 셀에 해당)에서 과대면, 전역 배수를 격자에 쓰면 안 된다\n")
    qb = pd.qcut(sg, N_SIGMA_BIN, labels=False, duplicates="drop")
    q90 = Qs[0.90]
    print(f"  {'σ 구간':>8}{'σ 범위 (℃)':>18}{'행':>12}{'보정 전 90%':>14}{'보정 후 90%':>14}")
    print("  " + "-" * 66)
    cond = []
    for b in range(int(qb.max()) + 1):
        m = qb == b
        lo, hi = sg[m].min(), sg[m].max()
        c0 = float((E[m] <= stats.t.ppf(0.95, nu) * np.sqrt((nu - 2) / nu) * z_sd).mean())
        c1 = float((E[m] <= q90).mean())
        cond.append(c1)
        print(f"  {'Q'+str(b+1):>8}{f'{lo:.2f}~{hi:.2f}':>18}{int(m.sum()):>12,}"
              f"{c0*100:>13.1f}%{c1*100:>13.1f}%")
    spread = max(cond) - min(cond)
    print(f"\n  보정 후 구간별 적중률 편차 **{spread*100:.1f}%p**")
    if spread < 0.05:
        print(f"    → 척도에 관계없이 고르다. **전역 배수를 격자에 써도 된다** "
              f"(이중 계상 아님)")
    else:
        print(f"    → 기울어져 있다. **σ 구간별 배수**가 필요하다 (이중 계상 실재)")

    # ── ③-b Mondrian conformal — σ 구간마다 따로 보정한다 ──
    #
    # 전역 Q 하나로는 σ가 작은 곳에서 좁고(80.1%) 큰 곳에서 넓다(94.5%).
    # 즉 **분위 GBM의 σ가 차이를 과장한다** — 실제 오차 척도는 σ만큼 벌어지지 않는다.
    # 격자 셀은 σ가 큰 쪽(중위 0.872℃)에 몰려 있으므로 전역 배수를 쓰면 **과대**가 된다.
    # 구간마다 Q를 따로 잡으면 각 구간이 이름값을 지키고, 격자에는 자기 구간의 배수가 붙는다.
    NB = 10
    edges = np.quantile(sg, np.linspace(0, 1, NB + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bidx = np.clip(np.searchsorted(edges, sg, side="right") - 1, 0, NB - 1)

    # 폭염 축을 하나 더 넣는다 (8/26 추가).
    #
    # σ 구간만으로 보정했더니 **폭염 시각 적중률이 87.4%**로 90%에 못 미쳤다
    # (`core/metrics/accuracy_table.py` 표 2). 폭염은 이 제안의 **핵심 용도**이므로
    # 거기서 구간이 좁으면 안 된다. conformal은 조건을 나눠 보정할 수 있으므로
    # (σ 구간 × 폭염 여부)로 격자를 2차원으로 만든다.
    #
    # 격자 추론에도 그대로 적용된다 — 폭염 판정은 `t_reg >= 33℃`이고 그 시각별
    # 광역값은 격자 쪽도 갖고 있다.
    hot_ts = set(hot_timestamps(d))
    hot = d.ts.isin(hot_ts).to_numpy()
    print(f"\n  폭염 시각 행 {int(hot.sum()):,} ({hot.mean()*100:.1f}%)")
    print(f"\n{'='*76}\n③-b Mondrian conformal — σ 구간 {NB}개에 각각 보정\n{'='*76}")
    print(f"  {'σ 구간':>7}{'σ 범위 (℃)':>16}{'행':>11}{'Q(90%)':>10}"
          f"{'적중률':>10}{'평균 폭':>11}")
    print("  " + "-" * 63)
    # Qb[hot, bin] — 폭염 여부 × σ 구간
    Qb = np.zeros((2, NB))
    cov_b, wid_b, nb = [], [], []
    for h in (0, 1):
        for b in range(NB):
            qf, cv, wd = [], [], []
            for f in range(N_FOLDS):
                cal = (bidx == b) & (hot == bool(h)) & (fold != f)
                tst = (bidx == b) & (hot == bool(h)) & (fold == f)
                if cal.sum() < 100 or tst.sum() == 0:
                    continue
                m = int(cal.sum())
                q = float(np.quantile(E[cal], min(0.90 * (1 + 1 / m), 1.0)))
                qf.append(q); cv.append(float((E[tst] <= q).mean()))
                wd.append(float((2 * q * sg[tst]).mean()))
            if not qf:                       # 표본이 없으면 폭염 아닌 쪽 값을 쓴다
                Qb[h, b] = Qb[0, b] if h else np.nan
                continue
            Qb[h, b] = np.mean(qf)
            cov_b.append(np.mean(cv)); wid_b.append(np.mean(wd))
            m = (bidx == b) & (hot == bool(h))
            nb.append(int(m.sum()))
            print(f"  {('폭염 ' if h else '평시 ')+'B'+str(b+1):>7}"
                  f"{f'{sg[m].min():.2f}~{sg[m].max():.2f}':>16}"
                  f"{int(m.sum()):>11,}{Qb[h, b]:>10.3f}{np.mean(cv)*100:>9.1f}%"
                  f"{np.mean(wd):>10.3f}℃")
    sp2 = max(cov_b) - min(cov_b)
    w_glob = float((2 * q90 * sg).mean())
    w_mond = float(np.average(wid_b, weights=nb))
    print(f"\n  구간별 적중률 편차 {spread*100:.1f}%p → **{sp2*100:.1f}%p**")
    print(f"  평균 구간폭 전역 {w_glob:.3f}℃ → Mondrian **{w_mond:.3f}℃** "
          f"({(w_mond-w_glob)/w_glob*100:+.1f}%)")
    print(f"  Q가 σ와 함께 {Qb[0,0]:.2f} → {Qb[0,-1]:.2f}로 변한다 "
          f"— σ가 클수록 배수가 작다. 분위 GBM이 차이를 과장한다는 뜻이다")
    r_hot = np.nanmean(Qb[1] / Qb[0])
    print(f"  폭염 시각의 배수가 평시의 **{r_hot:.2f}배** — "
          f"{'폭염에 더 넓은 구간이 필요하다' if r_hot > 1 else '폭염이 오히려 좁다'}")

    # ── ④ CRPS ──
    print(f"\n{'='*76}\n④ CRPS — 분포 전체의 정확도. 결정론적 예측의 CRPS는 MAE와 같다\n{'='*76}")
    ix = (rng.choice(n, a.sample_rows, replace=False) if 0 < a.sample_rows < n
          else np.arange(n))
    ys, mus, sgs = y[ix], mu[ix], sg[ix]
    rows = []
    tq = np.sqrt((nu - 2) / nu)
    for lab, draw in (
        ("B3 (점추정)", None),
        ("가우시안 (σ 그대로)", lambda: rng.standard_normal((len(ix), N_SAMP)) * sgs[:, None]),
        (f"가우시안 (×z_sd)", lambda: rng.standard_normal((len(ix), N_SAMP)) * (sgs * z_sd)[:, None]),
        (f"t(ν={nu:.0f}) (×z_sd)",
         lambda: rng.standard_t(nu, (len(ix), N_SAMP)) * tq * (sgs * z_sd)[:, None]),
        (f"t(ν={nu:.0f}) + conformal",
         lambda: rng.standard_t(nu, (len(ix), N_SAMP)) * tq
                 * (sgs * (q90 / (stats.t.ppf(0.95, nu) * tq)))[:, None]),
    ):
        if draw is None:
            c = float(np.abs(mus - ys).mean())
        else:
            c = float(crps_from_samples(mus[:, None] + draw(), ys).mean())
        rows.append((lab, c))
        print(f"  {lab:<24}{c:>10.4f}℃")
    best = min(rows[1:], key=lambda r: r[1])
    b3c = rows[0][1]
    print(f"\n  최저 CRPS **{best[0]}** {best[1]:.4f}℃ — B3 점추정 대비 "
          f"**{(b3c-best[1])/b3c*100:+.1f}%**")
    print(f"    (점추정은 분포를 못 내므로 CRPS에서 불리하다. 이것이 분포를 내는 이유다)")

    np.savez(OUT, Q=np.array([Qs[l] for l in LEVELS]), levels=np.array(LEVELS),
             nu=nu, z_sd=z_sd, cond_cov=np.array(cond),
             sigma_edges=edges, Q_bin=Qb, cov_bin=np.array(cov_b), hot_c=33.0,
             crps=np.array([r[1] for r in rows], dtype=float))
    print(f"\n→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
