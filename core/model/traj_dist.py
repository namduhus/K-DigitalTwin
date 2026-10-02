from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core.baseline.b1_interp import idw_oof, sensor_grid
from core.baseline.b2_gbm import PARAMS
from core.baseline.b3_hybrid import (
    ALL_F, BASE_F, EARLY, FEATS, IDW_IX, IDW_K, IDW_P, N_ROUND, TREG_IX,
)
from core.eval.dataset import (
    N_FOLDS, PREDS, iter_folds, load_base, load_regional, load_sensor_meta,
    trusted_sensors,
)

OUT_SCALE = PREDS / "b3_scale.parquet"      # sn, ts, q10, q90, sigma
OUT_DIST = Path("data/metric/traj_dist.npz")  # R (24×24), nu, 진단

QUANTILES = {"q10": 0.10, "q90": 0.90}
NORMAL_IQ = 2.5631031310892016              # 정규분포의 (q90 − q10) / σ
NU_GRID = np.array([2.5, 3, 3.5, 4, 5, 6, 8, 10, 15, 20, 30, 50, 100])


# ─────────────────────────────────────────────── 조건부 척도

def fit_scale(a) -> pd.DataFrame:
    import lightgbm as lgb

    base = load_base(BASE_F)
    regional = load_regional()
    meta = load_sensor_meta()
    sns, tss, sn_col, ts_row = sensor_grid(base)
    xy = meta.loc[sns, ["x", "y"]].to_numpy(np.float64)
    D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    sensor_fold = base.groupby("sn")["fold"].first().reindex(sns).to_numpy()
    trust = set(trusted_sensors(base))
    untrusted = np.array([i for i, s in enumerate(sns) if s not in trust])

    X = np.zeros((len(base), len(ALL_F) + 1), np.float32)
    X[:, [ALL_F.index(c) for c in BASE_F]] = base[BASE_F].to_numpy(np.float32)
    cols = [ALL_F.index(c) for c in FEATS["b3_hybrid"]] + [IDW_IX]

    out = {k: np.full(len(base), np.nan, np.float32) for k in QUANTILES}
    folds = [0] if a.probe else list(range(N_FOLDS))
    print(f"\n{'fold':<5}{'분위':<6}{'학습':>10}{'트리':>6}{'핀볼손실':>11}{'초':>6}")
    print("-" * 44)
    for fd in iter_folds(base, regional):
        if fd.f not in folds:
            continue
        X[:, IDW_IX] = idw_oof(fd, base, D, sensor_fold, sn_col, ts_row,
                               len(tss), untrusted, IDW_P, IDW_K)
        X[:, TREG_IX] = fd.t_reg
        y = fd.delta
        inner = np.where(sensor_fold == (fd.f + 1) % N_FOLDS)[0]
        m_te = fd.is_test
        m_va = np.isin(sn_col, inner) & ~m_te
        m_tr = ~m_te & ~m_va

        for name, q in QUANTILES.items():
            t0 = time.time()
            p = dict(PARAMS, objective="quantile", alpha=q)
            dtr = lgb.Dataset(X[m_tr][:, cols], label=y[m_tr])
            dva = lgb.Dataset(X[m_va][:, cols], label=y[m_va], reference=dtr)
            bst = lgb.train(p, dtr, num_boost_round=N_ROUND, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(EARLY, verbose=False)])
            pr = bst.predict(X[m_te][:, cols], num_iteration=bst.best_iteration)
            out[name][m_te] = pr
            e = y[m_te] - pr
            pin = float(np.maximum(q * e, (q - 1) * e).mean())
            print(f"{fd.f:<5}{name:<6}{int(m_tr.sum()):>10,}{bst.best_iteration:>6}"
                  f"{pin:>11.4f}{time.time()-t0:>6.1f}")

    d = pd.DataFrame({"sn": base["sn"].astype("string"), "ts": base["ts"],
                      "q10": out["q10"], "q90": out["q90"]})
    d["sigma"] = (d.q90 - d.q10) / NORMAL_IQ
    ok = d.sigma.notna()
    # 분위 교차(q90 < q10)는 분위 회귀에서 실제로 생긴다 — 두 모델이 독립이라
    # 단조성이 보장되지 않는다. 음수 척도를 그대로 두면 샘플러가 조용히 망가진다.
    bad = int((d.sigma <= 0).sum())
    if bad:
        print(f"\n분위 교차 {bad:,}행 ({bad/int(ok.sum())*100:.3f}%) — "
              f"하한 {0.05:.2f}℃로 자른다")
    d["sigma"] = d.sigma.clip(lower=0.05)
    OUT_SCALE.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT_SCALE, index=False)
    print(f"\n→ {OUT_SCALE}  σ 중위 {d.sigma.median():.3f}℃ · "
          f"P05 {d.sigma.quantile(.05):.3f} · P95 {d.sigma.quantile(.95):.3f}")
    return d


# ─────────────────────────────────────────────── 다변량 t

def mvt_loglik(Z, R, nu):
    from scipy.linalg import solve_triangular
    from scipy.special import gammaln
    n, d = Z.shape
    S = R * (nu - 2) / nu
    L = np.linalg.cholesky(S)
    q = (solve_triangular(L, Z.T, lower=True) ** 2).sum(0)
    logdet = 2.0 * np.log(np.diag(L)).sum()
    return float(n * (gammaln((nu + d) / 2) - gammaln(nu / 2)
                      - d / 2 * np.log(nu * np.pi) - 0.5 * logdet)
                 - (nu + d) / 2 * np.log1p(q / nu).sum())


def mvn_loglik(Z, R):
    from scipy.linalg import solve_triangular
    n, d = Z.shape
    L = np.linalg.cholesky(R)
    q = (solve_triangular(L, Z.T, lower=True) ** 2).sum(0)
    logdet = 2.0 * np.log(np.diag(L)).sum()
    return float(-0.5 * (n * (d * np.log(2 * np.pi) + logdet) + q.sum()))


def sample(mu, sigma, R, nu, n_draw, rng):
    n, d = mu.shape
    L = np.linalg.cholesky(R * (nu - 2) / nu)
    g = rng.standard_normal((n_draw, n, d)) @ L.T
    w = rng.chisquare(nu, size=(n_draw, n, 1)) / nu
    return mu + sigma * (g / np.sqrt(w))


# ─────────────────────────────────────────────── 궤적 조립 + 추정

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--reuse-scale", action="store_true",
                    help="이미 만든 b3_scale.parquet을 다시 쓴다")
    a = ap.parse_args()
    rng = np.random.default_rng(20260826)

    if a.reuse_scale and OUT_SCALE.exists():
        sc = pd.read_parquet(OUT_SCALE)
        print(f"[척도] 재사용 {OUT_SCALE}")
    else:
        sc = fit_scale(a)

    # ── 궤적 조립: 실측 · B3 평균 · 척도 ──
    base = load_base(["hour"], verbose=False)
    base = base[base.fold >= 0]
    b3 = pd.read_parquet(PREDS / "b3_hybrid.parquet")
    b3["sn"] = b3["sn"].astype(str)
    sc["sn"] = sc["sn"].astype(str)
    d = (base.merge(b3, on=["sn", "ts"], how="inner")
             .merge(sc[["sn", "ts", "sigma"]], on=["sn", "ts"], how="inner"))
    d["day"] = d.ts.dt.floor("D")

    piv = lambda c: (d.pivot_table(index=["sn", "day"], columns="hour", values=c,
                                   aggfunc="mean").reindex(columns=range(24)))
    T, M, S = piv("tp"), piv("y_pred"), piv("sigma")
    full = T.notna().all(axis=1) & M.notna().all(axis=1) & S.notna().all(axis=1)
    T, M, S = T[full].to_numpy(), M[full].to_numpy(), S[full].to_numpy()
    print(f"\n[궤적] 24시간 완전 **{len(T):,}개**")

    Rres = T - M
    Z = Rres / S
    z_sd = Z.std()
    Z = Z / z_sd                       # 전체 분산을 1로 — σ는 비례만 하면 된다
    print(f"       잔차 σ {Rres.std():.3f}℃ · 조건부 척도로 나눈 뒤 σ {z_sd:.3f} → 정규화")
    print(f"       표준화 잔차 시각별 σ {Z.std(0).min():.3f}~{Z.std(0).max():.3f} "
          f"(1에 가까울수록 척도 모델이 이분산을 잘 잡은 것)")

    # ── 상관행렬 ──
    R = np.corrcoef(Z.T)
    off = R[~np.eye(24, dtype=bool)]
    lag1 = np.mean([R[i, i + 1] for i in range(23)])
    lag6 = np.mean([R[i, i + 6] for i in range(18)])
    lag12 = np.mean([R[i, i + 12] for i in range(12)])
    print(f"\n[상관] 인접 시각 {lag1:.3f} · 6시간 {lag6:.3f} · 12시간 {lag12:.3f} · "
          f"최소 {off.min():.3f}")
    print(f"       시각 간 상관이 이만큼 강하다는 것이 **결합분포가 필요한 이유**다")

    # ── 자유도 ──
    print(f"\n[꼬리] 다변량 t 프로파일 로그가능도 (공분산은 R로 고정 — 꼬리만 비교)")
    ll_n = mvn_loglik(Z, R)
    lls = np.array([mvt_loglik(Z, R, nu) for nu in NU_GRID])
    best = int(lls.argmax())
    nu_hat = float(NU_GRID[best])
    print(f"  {'ν':>7}{'로그가능도':>16}{'가우시안 대비':>16}")
    print("  " + "-" * 37)
    for nu, ll in zip(NU_GRID, lls):
        mark = "  ←" if nu == nu_hat else ""
        print(f"  {nu:>7.1f}{ll:>16,.0f}{ll - ll_n:>+16,.0f}{mark}")
    print(f"  {'가우시안':>7}{ll_n:>16,.0f}{0:>+16.0f}")
    print(f"\n  → ν = **{nu_hat:.1f}** · 가우시안 대비 로그가능도 **{lls[best]-ll_n:+,.0f}**")
    print(f"     (ν가 작을수록 꼬리가 두껍다. ν→∞가 가우시안)")

    # ── 검산: 90% 구간 적중률 ──
    print(f"\n[검산] 90% 구간 적중률 — 이름값대로 0.90이 나와야 한다")
    for lab, nu in (("가우시안", np.inf), (f"t(ν={nu_hat:.1f})", nu_hat)):
        if np.isinf(nu):
            draws = M + S * z_sd * (rng.standard_normal((100, *M.shape))
                                    @ np.linalg.cholesky(R).T)
        else:
            draws = sample(M, S * z_sd, R, nu, 100, rng)
        lo, hi = np.percentile(draws, [5, 95], axis=0)
        cov = float(((T >= lo) & (T <= hi)).mean())
        width = float((hi - lo).mean())
        print(f"  {lab:<12} 적중률 {cov*100:>5.1f}%  ·  평균 구간폭 {width:.3f}℃")

    OUT_DIST.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIST, R=R, nu=nu_hat, z_sd=z_sd,
             ll_gauss=ll_n, ll_t=lls[best], n_traj=len(T))
    print(f"\n→ {OUT_DIST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
