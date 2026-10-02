from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from core.eval.dataset import PREDS, load_base

OUT = Path("data/metric/traj_probe.parquet")
SEED = 20260826
N_PC = 5              # GMM을 태울 주성분 수
K_MAX = 6
HOT_C = 33.0


# ─────────────────────────────────────────────── GMM (EM, 전체 공분산)
# sklearn이 없다. 의존성을 늘리기보다 직접 쓴다 — 60줄이고 이 시점에는 더 투명하다.

def _logpdf(X, mu, cov):
    from scipy.linalg import solve_triangular
    d = X.shape[1]
    L = np.linalg.cholesky(cov)
    z = solve_triangular(L, (X - mu).T, lower=True)
    logdet = 2.0 * np.log(np.diag(L)).sum()
    return -0.5 * (d * np.log(2 * np.pi) + logdet + (z ** 2).sum(0))


def gmm_fit(X, k, seed=SEED, iters=300, tol=1e-7, reg=1e-6):
    n, d = X.shape
    rng = np.random.default_rng(seed)
    # k-means++ 유사 초기화 — 무작위 초기화는 성분이 겹친 채 수렴하는 일이 잦다
    mu = X[rng.choice(n, 1)]
    for _ in range(k - 1):
        d2 = ((X[:, None, :] - mu[None]) ** 2).sum(-1).min(1)
        mu = np.vstack([mu, X[rng.choice(n, p=d2 / d2.sum())]])
    cov = np.stack([np.cov(X.T) + reg * np.eye(d) for _ in range(k)])
    w = np.full(k, 1.0 / k)

    prev = -np.inf
    for _ in range(iters):
        lp = np.stack([np.log(w[j]) + _logpdf(X, mu[j], cov[j]) for j in range(k)], 1)
        m = lp.max(1, keepdims=True)
        ll_i = m[:, 0] + np.log(np.exp(lp - m).sum(1))
        ll = ll_i.mean()
        R = np.exp(lp - ll_i[:, None])
        Nk = R.sum(0) + 1e-12
        w = Nk / n
        mu = (R.T @ X) / Nk[:, None]
        for j in range(k):
            Xc = X - mu[j]
            cov[j] = (Xc * R[:, j:j + 1]).T @ Xc / Nk[j] + reg * np.eye(d)
        if abs(ll - prev) < tol:
            break
        prev = ll
    n_par = k * (d + d * (d + 1) / 2) + (k - 1)
    bic = -2 * ll * n + n_par * np.log(n)
    ent = -(R * np.log(np.clip(R, 1e-300, None))).sum()
    return w, mu, cov, ll * n, bic, bic + 2 * ent


# ─────────────────────────────────────────────── 궤적 조립

def build(verbose=True):
    base = load_base(["hour"], verbose=verbose)
    base = base[base.fold >= 0]                       # always-train은 예측이 없다
    p = pd.read_parquet(PREDS / "b3_hybrid.parquet")
    p["sn"] = p["sn"].astype(str)
    base = base.merge(p, on=["sn", "ts"], how="inner")
    base["day"] = base.ts.dt.floor("D")

    piv = lambda c: base.pivot_table(index=["sn", "day"], columns="hour",
                                     values=c, aggfunc="mean")
    T, Y = piv("tp"), piv("y_pred")
    T = T.reindex(columns=range(24))
    Y = Y.reindex(columns=range(24))
    full = T.notna().all(axis=1) & Y.notna().all(axis=1)   # 24시간 완전한 것만
    if verbose:
        print(f"[궤적] (센서,날짜) {len(T):,}개 → **24시간 완전 {int(full.sum()):,}개** "
              f"({full.mean()*100:.1f}%)")
    T, Y = T[full], Y[full]
    return T.to_numpy(np.float64), Y.to_numpy(np.float64), T.index


def derived(A):
    return {"33℃ 초과(h)": (A >= HOT_C).sum(1).astype(float),
            "일최고 시각": A.argmax(1).astype(float),
            "일최저(℃)": A.min(1),
            "일교차(℃)": A.max(1) - A.min(1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsim", type=int, default=5, help="가우시안 샘플 배수")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)

    T, Y, idx = build()
    R = T - Y                                          # 잔차 = p(Δ|c)의 모양
    n = len(R)
    print(f"       잔차 전체 표준편차 {R.std():.3f}℃ · "
          f"시각별 {R.std(0).min():.3f}~{R.std(0).max():.3f}℃")

    # 날짜 표준화 — 이분산이 다봉으로 위장하는 것을 걷어낸다.
    #
    # 처음에 **시각별·날짜별** 표준편차로 나누고 하한을 `1e-6`으로 뒀다가 망했다:
    # 그 시각 센서가 적은 날은 표준편차가 0.01 규모로 작아져 **나눈 값이 100배로
    # 폭발**했고, PC1 첨도가 **41,620**, 성분 평균이 **309℃**로 나왔다. 값이 터진 것을
    # "다봉"으로 읽었으면 정반대 결론을 냈을 것이다.
    #
    # 고친 방식 — ① 날짜당 **스칼라** 척도(24시간 전체 MAD 기반)로 나눈다. 시각별로
    # 나누면 궤적의 시간 구조까지 지워버린다. ② 하한을 **0.2℃**로 둔다. 잔차 전체
    # 표준편차가 0.797℃이므로 그보다 4배 작은 날은 척도 추정 자체가 못 믿을 것이다.
    dayk = idx.get_level_values(1)
    mad = pd.Series(np.abs(R - np.median(R, axis=1, keepdims=True)).mean(1),
                    index=dayk).groupby(level=0).median()
    scale = np.maximum(mad.reindex(dayk).to_numpy() * 1.4826, 0.2)
    ok = np.isfinite(scale)
    R_std = R[ok] / scale[ok, None]
    print(f"       날짜 표준화 — 날짜당 스칼라 척도(MAD, 하한 0.2℃) · "
          f"척도 {np.nanmin(scale):.2f}~{np.nanmax(scale):.2f}℃ · "
          f"하한 적용 {int((scale <= 0.2001).sum()):,}행")

    rows = []
    for tag, X in (("원본 잔차", R), ("날짜 표준화", R_std)):
        print(f"\n{'='*72}\n[{tag}]  n = {len(X):,} × 24차원\n{'='*72}")
        Xc = X - X.mean(0)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        evr = S ** 2 / (S ** 2).sum()
        print(f"  PCA 설명력  " + " · ".join(f"PC{i+1} {evr[i]*100:.1f}%" for i in range(5))
              + f"  (상위 5개 합 {evr[:5].sum()*100:.1f}%)")
        Z = U[:, :N_PC] * S[:N_PC]                     # 주성분 점수

        from scipy import stats
        print(f"  {'':4}{'왜도':>8}{'첨도(초과)':>12}   ← 0에 가까우면 정규에 가깝다")
        for i in range(3):
            print(f"  PC{i+1} {stats.skew(Z[:, i]):>8.3f}{stats.kurtosis(Z[:, i]):>12.3f}")

        print(f"\n  {'k':>3}{'BIC':>14}{'ICL':>14}{'ΔBIC(vs k-1)':>15}{'최소 가중치':>12}")
        print("  " + "-" * 58)
        best, prev_bic, fits, icls = None, None, {}, {}
        for k in range(1, K_MAX + 1):
            w, mu, cov, ll, bic, icl = gmm_fit(Z, k)
            fits[k], icls[k] = (w, mu, cov), icl
            d = "" if prev_bic is None else f"{bic - prev_bic:>+15.0f}"
            print(f"  {k:>3}{bic:>14.0f}{icl:>14.0f}{d:>15}{w.min()*100:>11.1f}%")
            if best is None or bic < best[1]:
                best = (k, bic)
            prev_bic = bic
        kbest, kicl = best[0], min(icls, key=icls.get)
        print(f"\n  → BIC 최소 k = **{kbest}** · ICL 최소 k = **{kicl}**")
        if kbest != kicl:
            print(f"     둘이 갈린다. **BIC는 표본이 크면 미세한 비정규성에도 성분을**")
            print(f"     **더 붙인다**(n={len(X):,}). ICL은 성분이 겹치면 벌점을 주므로,")
            print(f"     이 불일치 자체가 *봉우리가 아니라 비정규 꼬리*라는 신호다.")

        # 모드인가 분산혼합인가 — 평균이 붙어 있는데 퍼짐만 다르면 봉우리가 아니다
        for k in sorted({kbest, kicl}):
            if k == 1:
                continue
            w, mu, cov = fits[k]
            back = mu @ Vt[:N_PC]                      # 24차원 ℃ 공간으로 되돌린다
            sep = max(np.sqrt(((back[i] - back[j]) ** 2).mean())
                      for i in range(k) for j in range(i + 1, k))
            scale = np.array([np.sqrt(np.trace(cov[j]) / N_PC) for j in range(k)])
            shift = np.sqrt((back ** 2).mean(1))       # 전체 평균에서 떨어진 거리
            print(f"\n  [k={k}] 성분별")
            print(f"    {'가중치':>8}{'평균 이동(℃)':>14}{'퍼짐(주성분 σ)':>16}")
            for j in np.argsort(-w):
                print(f"    {w[j]*100:>7.1f}%{shift[j]:>14.3f}{scale[j]:>16.3f}")
            print(f"    성분 간 평균 최대 차이 **{sep:.3f}℃** (RMS, 24시간) — "
                  f"잔차 σ {X.std():.3f}의 {sep/X.std():.2f}배")
            print(f"    퍼짐 비율 최대/최소 **{scale.max()/scale.min():.2f}배** "
                  f"— 이 값이 크고 평균 이동이 작으면 **분산혼합**이다")
        rows.append({"set": tag, "n": len(X), "k_bic": kbest, "k_icl": kicl,
                     "evr5": float(evr[:5].sum()),
                     "skew_pc1": float(stats.skew(Z[:, 0])),
                     "kurt_pc1": float(stats.kurtosis(Z[:, 0]))})

    # ─────────────── ③ 실질 검사 — 가우시안이 파생량을 재현하는가
    print(f"\n{'='*72}\n③ 실질 검사 — 다변량 가우시안 잔차가 궤적 파생량을 재현하는가\n{'='*72}")
    mu_r, cov_r = R.mean(0), np.cov(R.T)
    sim = rng.multivariate_normal(mu_r, cov_r, size=n * a.nsim)
    T_sim = np.repeat(Y, a.nsim, axis=0) + sim
    d_true, d_gau, d_b3 = derived(T), derived(T_sim), derived(Y)

    print(f"  {'파생량':<14}{'실측':>22}{'가우시안 잔차':>22}{'B3 점추정':>20}")
    print(f"  {'':<14}{'평균 / 표준편차':>22}{'평균 / 표준편차':>22}{'평균 / 표준편차':>20}")
    print("  " + "-" * 76)
    for k in d_true:
        t, g, b = d_true[k], d_gau[k], d_b3[k]
        print(f"  {k:<14}{t.mean():>11.2f} /{t.std():>8.2f}"
              f"{g.mean():>13.2f} /{g.std():>8.2f}{b.mean():>11.2f} /{b.std():>8.2f}")
        ks = stats.ks_2samp(t, g).statistic
        print(f"  {'':<14}{'':>22}{'KS = ' + format(ks, '.4f'):>22}")

    pd.DataFrame(rows).to_parquet(OUT)
    print(f"\n→ {OUT}")
    print("\n  판정 기준 (`plan.md` 「왜 FM인가」에 미리 적어둠)")
    print("   · BIC k=1 이거나, k≥2여도 성분 차이가 잔차 표준편차 대비 작다 → **단봉**")
    print("   · 날짜 표준화 후 다봉성이 사라진다 → 봉우리가 아니라 **이분산**이었다")
    print("   · ③에서 가우시안이 파생량을 재현한다 → **FM의 실익이 없다**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
