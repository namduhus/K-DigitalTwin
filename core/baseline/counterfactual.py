from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from core.baseline.b1_interp import idw_oof, sensor_grid
from core.baseline.b2_gbm import PARAMS
from core.baseline.b3_hybrid import ALL_F, BASE_F, FEATS, IDW_IX, IDW_K, IDW_P, TREG_IX
from core.eval.dataset import (
    N_FOLDS,
    iter_folds,
    load_base,
    load_regional,
    load_sensor_meta,
    trusted_sensors,
)

N_ROUND, EARLY = 800, 100

# 시나리오 — (피처, 증분) 목록. 면적 피처는 서로 상쇄시켜 총합을 보존한다.
# `pave_frac`은 개입 레버로 쓸 수 없다 — **92% 센서가 이미 0.05 미만**이라
# -0.05를 적용하면 0으로 클립되어 변화가 없다 (중위 0.002, P99 0.135).
# 「광장 녹화」 시나리오를 폐기한 이유다.
#
# 치환 크기를 훑는다. GBM은 **구간 분할** 모델이라 작은 치환이 분할 경계를 못 넘으면
# 예측이 안 변하고, 엉뚱하게 넘으면 부호가 뒤집힌다. 실제로 `grn_frac +0.05`만 준
# 시나리오에서 +0.014℃(더워짐)가 나왔다. 어느 크기부터 신뢰할 수 있는지 확인한다.
SCENARIOS = {
    "녹지 +0.05":   [("grn_frac", +0.05)],
    "녹지 +0.10":   [("grn_frac", +0.10)],
    "녹지 +0.20":   [("grn_frac", +0.20)],
    "녹지 +0.40":   [("grn_frac", +0.40)],
    "소공원 (복합)": [("grn_frac", +0.10), ("bld_ar_frac", -0.05),
                  ("grn_d_min", "→0"), ("grn_n", +1)],
    "차양 svf−0.05": [("svf", -0.05), ("hz_mean", +3.0)],
}
# 물리적 범위 — 치환 후 클립한다
BOUNDS = {"grn_frac": (0, 1), "pave_frac": (0, 1), "bld_ar_frac": (0, 1),
          "svf": (0, 1), "grn_d_min": (0, None), "grn_n": (0, None), "hz_mean": (0, 90)}

# 매칭 대조에서 통제할 교란 변수 (녹지 관련은 제외한다)
CONFOUND = ["bld_ar_frac", "bld_d_min", "bld_h_mean", "elev", "svf", "terr_dominant"]
MATCH_MAX_DIST = 0.5      # 표준화 공간에서의 최대 거리 (교란 조건이 "비슷하다"의 기준)
MATCH_MIN_DGRN = 0.10     # 녹지비 차이가 이 이상인 쌍만 본다


def apply_scenario(X: np.ndarray, spec: list) -> np.ndarray:
    Xc = X.copy()
    for feat, delta in spec:
        j = ALL_F.index(feat)
        Xc[:, j] = 0.0 if delta == "→0" else Xc[:, j] + delta
        lo, hi = BOUNDS.get(feat, (None, None))
        if lo is not None:
            Xc[:, j] = np.maximum(Xc[:, j], lo)
        if hi is not None:
            Xc[:, j] = np.minimum(Xc[:, j], hi)
    return Xc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
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
    night = base["is_night"].to_numpy().astype(bool)

    X = np.zeros((len(base), len(ALL_F) + 1), np.float32)
    X[:, [ALL_F.index(c) for c in BASE_F]] = base[BASE_F].to_numpy(np.float32)
    cols = [ALL_F.index(c) for c in FEATS["b3_hybrid"]] + [IDW_IX]

    # 학습 분포 — 치환값이 내삽인지 판정할 기준
    q = {f: np.percentile(base[f].dropna(), [1, 50, 99]) for f, _ in
         {k: v for s in SCENARIOS.values() for k, v in [(x[0], x[1]) for x in s]}.items()
         if f in base.columns}
    print("\n학습 분포 (P01 / 중위 / P99) — 치환값이 이 안에 있어야 내삽이다")
    for f, v in q.items():
        print(f"  {f:<14}{v[0]:>9.3f}{v[1]:>9.3f}{v[2]:>9.3f}")

    folds = [0] if a.probe else list(range(N_FOLDS))
    rows, oob = [], []
    print(f"\n{'fold':<5}{'시나리오':<14}{'test행':>10}{'효과 ℃':>9}{'주간':>9}{'야간':>9}"
          f"{'분포밖':>8}{'초':>6}")
    print("-" * 72)
    for fd in iter_folds(base, regional):
        if fd.f not in folds:
            continue
        t0 = time.time()
        X[:, IDW_IX] = idw_oof(fd, base, D, sensor_fold, sn_col, ts_row,
                               len(tss), untrusted, IDW_P, IDW_K)
        X[:, TREG_IX] = fd.t_reg
        inner = np.isin(sn_col, np.where(sensor_fold == (fd.f + 1) % N_FOLDS)[0]) & ~fd.is_test
        m_tr = ~fd.is_test & ~inner
        bst = lgb.train(PARAMS, lgb.Dataset(X[m_tr][:, cols], label=fd.delta[m_tr]),
                        num_boost_round=N_ROUND, valid_sets=[
                            lgb.Dataset(X[inner][:, cols], label=fd.delta[inner])],
                        callbacks=[lgb.early_stopping(EARLY, verbose=False)])
        te = fd.is_test
        base_pred = bst.predict(X[te][:, cols], num_iteration=bst.best_iteration)
        fit_s = time.time() - t0

        for name, spec in SCENARIOS.items():
            t1 = time.time()
            Xc = apply_scenario(X[te], spec)
            cf = bst.predict(Xc[:, cols], num_iteration=bst.best_iteration)
            eff = cf - base_pred                       # 음수면 시원해진다
            # 치환값이 학습 분포 밖으로 나간 비율
            out = np.zeros(len(eff), bool)
            for f, _ in spec:
                if f in q:
                    j = ALL_F.index(f)
                    out |= (Xc[:, j] > q[f][2]) | (Xc[:, j] < q[f][0])
            nt = night[te]
            rows.append({"fold": fd.f, "scenario": name, "n": len(eff),
                         "eff": eff.mean(), "eff_day": eff[~nt].mean(),
                         "eff_night": eff[nt].mean(), "oob": out.mean()})
            print(f"{fd.f:<5}{name:<14}{len(eff):>10,}{eff.mean():>+9.3f}"
                  f"{eff[~nt].mean():>+9.3f}{eff[nt].mean():>+9.3f}"
                  f"{out.mean()*100:>7.1f}%{time.time()-t1:>6.1f}")

    r = pd.DataFrame(rows)
    print(f"\n{'='*66}\n개입 효과 (5-fold 평균 ± 표준편차, 음수 = 시원해진다)\n{'='*66}")
    print(f"  {'시나리오':<14}{'전체':>16}{'주간':>16}{'야간':>16}{'분포밖':>8}")
    print("  " + "-" * 62)
    for name, g in r.groupby("scenario", sort=False):
        print(f"  {name:<14}{g.eff.mean():>+10.3f} ±{g.eff.std():.3f}"
              f"{g.eff_day.mean():>+10.3f} ±{g.eff_day.std():.3f}"
              f"{g.eff_night.mean():>+10.3f} ±{g.eff_night.std():.3f}"
              f"{g.oob.mean()*100:>7.1f}%")

    if len(folds) == N_FOLDS:
        slopes(base, regional, r)
    return 0


def slopes(base, regional, r) -> None:
    fd = next(iter_folds(base, regional))
    sen = base.groupby("sn")[CONFOUND + ["grn_frac"]].first()
    sen["delta"] = pd.Series(fd.delta).groupby(base["sn"].to_numpy()).mean()
    sen = sen.dropna()
    g, y = sen["grn_frac"].to_numpy(), sen["delta"].to_numpy()

    print(f"\n{'='*70}\n녹지 효과 기울기 — 세 방법 (℃ / 녹지비 1.0)\n{'='*70}")

    # ① 분위 단순 비교
    qi = pd.qcut(sen["grn_frac"], 5, labels=False, duplicates="drop")
    lo, hi = sen[qi == 0], sen[qi == qi.max()]
    sl1 = (hi["delta"].mean() - lo["delta"].mean()) / (hi["grn_frac"].median() - lo["grn_frac"].median())
    print(f"  ① 분위 단순 비교 (통제 없음)      {sl1:>+8.3f}   "
          f"최저분위 {lo['delta'].mean():+.3f} → 최고분위 {hi['delta'].mean():+.3f}℃")

    # ② 매칭 대조
    Z = ((sen[CONFOUND] - sen[CONFOUND].mean()) / sen[CONFOUND].std()).to_numpy()
    d = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)) / np.sqrt(len(CONFOUND))
    iu = np.triu_indices(len(sen), k=1)
    dist, dg, dy = d[iu], g[iu[1]] - g[iu[0]], y[iu[1]] - y[iu[0]]
    print(f"  ② 매칭 대조 (교란 {len(CONFOUND)}변수)")
    rng = np.random.default_rng(0)
    for md in (0.3, 0.5, 0.8):
        m = (dist <= md) & (np.abs(dg) >= MATCH_MIN_DGRN)
        if m.sum() < 30:
            continue
        s = np.polyfit(dg[m], dy[m], 1)[0]
        bs = [np.polyfit(dg[m][i], dy[m][i], 1)[0] for i in
              (rng.integers(0, int(m.sum()), int(m.sum())) for _ in range(1000))]
        q = np.percentile(bs, [2.5, 97.5])
        print(f"       거리≤{md}  {int(m.sum()):>7,}쌍   {s:>+8.3f}   CI [{q[0]:+.3f}, {q[1]:+.3f}]")

    # ③ 다변량 회귀
    Xr = np.c_[np.ones(len(sen)), sen[CONFOUND + ["grn_frac"]].to_numpy()]
    b, *_ = np.linalg.lstsq(Xr, y, rcond=None)
    res = y - Xr @ b
    se = np.sqrt(np.diag(np.linalg.inv(Xr.T @ Xr)) * res.var(ddof=Xr.shape[1]))
    tv = b[-1] / se[-1]
    print(f"  ③ 다변량 회귀 (교란 {len(CONFOUND)}변수 선형 통제)  {b[-1]:>+8.3f}   t={tv:+.2f}"
          f" {'***' if abs(tv) > 2.58 else '**' if abs(tv) > 1.96 else ''}")

    # ④ GBM 반사실 — 치환 크기별로 기울기 환산
    print(f"  ④ GBM 반사실 (치환 크기별 기울기 환산)")
    for name, gg in r.groupby("scenario", sort=False):
        inc = dict(SCENARIOS[name]).get("grn_frac")
        if not isinstance(inc, float):
            continue
        print(f"       {name:<12}{gg.eff.mean()/inc:>+8.3f}   "
              f"(효과 {gg.eff.mean():+.4f}℃ ÷ {inc:.2f})")

    print(f"\n  통제를 강하게 할수록 효과가 줄어든다 → **겉보기 효과의 상당 부분이 교란**"
          f"(표고·산근접·저밀도)이다")
    print(f"  신청서에는 단일 숫자가 아니라 **범위**로 쓴다: 녹지비 +0.10당 "
          f"약 −0.025 ~ −0.057℃")
    print(f"  어느 방법도 인과 식별이 아니다. 개입 실험이 아니라 공간 연관 기반 추정이다")
    print(f"  쌍이 독립이 아니므로(한 센서가 여러 쌍에 등장) 매칭 CI는 참고용이다")

if __name__ == "__main__":
    raise SystemExit(main())
