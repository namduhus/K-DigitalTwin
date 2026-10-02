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
    N_FOLDS, iter_folds, load_base, load_regional, load_sensor_meta, trusted_sensors,
)

OUT = Path("data/metric/treg_sensitivity.parquet")
SEED = 20260828

BIAS = [0.0, +0.5, +1.0, +2.0, -1.0]        # 계통 편차 (℃)
NOISE = [0.0, 0.5, 1.0, 2.0]                # 시각별 무작위 오차 표준편차 (℃)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    import lightgbm as lgb
    rng = np.random.default_rng(SEED)

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
    y_true = base["tp"].to_numpy(np.float64)

    # 오차는 **시각 단위**로 만든다. 행마다 독립이면 평균에서 상쇄돼 감도를 과소평가한다.
    ts_codes, ts_uniq = pd.factorize(base["ts"])
    noise_by_ts = {s: rng.normal(0.0, s, size=len(ts_uniq)) for s in NOISE if s > 0}

    cols_idw = [ALL_F.index(c) for c in FEATS["b3_hybrid"]] + [IDW_IX]
    cols_noidw = [ALL_F.index(c) for c in FEATS["b3_hybrid"]]      # 예보 상황 = IDW 없음

    folds = [0] if a.probe else list(range(N_FOLDS))
    rows = []
    print(f"\n{'fold':<5}{'모델':<10}{'학습':>10}{'트리':>6}{'기준 MAE':>10}{'초':>6}")
    print("-" * 47)
    for fd in iter_folds(base, regional):
        if fd.f not in folds:
            continue
        X[:, IDW_IX] = idw_oof(fd, base, D, sensor_fold, sn_col, ts_row,
                               len(tss), untrusted, IDW_P, IDW_K)
        X[:, TREG_IX] = fd.t_reg
        yv = fd.delta

        inner = np.where(sensor_fold == (fd.f + 1) % N_FOLDS)[0]
        m_te = fd.is_test
        m_va = np.isin(sn_col, inner) & ~m_te
        m_tr = ~m_te & ~m_va

        # 기준선 — **예보를 그대로 쓴 경우** (모든 골목에 같은 값).
        # 이게 없으면 *"예보로 돌리면 88% 나빠진다"*만 남아 오해를 부른다.
        # 비교 대상은 실황이 아니라 **같은 예보를 다운스케일링 없이 쓴 것**이다.
        treg_te0 = fd.t_reg[m_te].astype(np.float64)
        yt0 = y_true[m_te]
        for kind, vals in (("bias", BIAS), ("noise", NOISE)):
            for v in vals:
                if kind == "noise" and v == 0:
                    continue
                eps = (np.full(m_te.sum(), v, np.float64) if kind == "bias"
                       else noise_by_ts[v][ts_codes[m_te]])
                mae = float(np.abs(treg_te0 + eps - yt0).mean())
                rows.append({"fold": fd.f, "model": "예보 그대로", "kind": kind,
                             "eps": v, "mae": mae, "mae_ref": mae})

        for name, cols in (("IDW 있음", cols_idw), ("IDW 없음", cols_noidw)):
            t0 = time.time()
            dtr = lgb.Dataset(X[m_tr][:, cols], label=yv[m_tr])
            dva = lgb.Dataset(X[m_va][:, cols], label=yv[m_va], reference=dtr)
            bst = lgb.train(PARAMS, dtr, num_boost_round=N_ROUND, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(EARLY, verbose=False)])
            ni = bst.best_iteration

            Xt = X[m_te][:, cols].copy()
            tcol = cols.index(TREG_IX)          # 오차를 넣을 열 위치
            treg_te = fd.t_reg[m_te].astype(np.float64)
            yt = y_true[m_te]
            base_mae = None

            # 기준 예측 — 광역값이 정확할 때
            pred0 = treg_te + bst.predict(Xt, num_iteration=ni)

            for kind, vals in (("bias", BIAS), ("noise", NOISE)):
                for v in vals:
                    if kind == "noise" and v == 0:
                        continue                # bias 0과 같다
                    if kind == "bias":
                        eps = np.full(m_te.sum(), v, np.float64)
                    else:
                        eps = noise_by_ts[v][ts_codes[m_te]]
                    Xt[:, tcol] = (treg_te + eps).astype(np.float32)
                    # 기준점도 오차값을 쓴다 — 운용에서는 참값을 모른다
                    pred = (treg_te + eps) + bst.predict(Xt, num_iteration=ni)
                    mae = float(np.abs(pred - yt).mean())
                    # 참조 — **모델이 `t_reg`에 전혀 반응하지 않는 경우.**
                    # 오차가 기준점으로만 더해진 결과다. 실제와 이걸 비교해야
                    # *"피처로서의 `t_reg`가 오차를 증폭하는가"*를 알 수 있다.
                    # MAE는 오차가 독립일 때 단순 합이 아니라 직교합으로 커지므로
                    # `MAE 증가 ÷ σ`를 「증폭비」로 읽으면 **항상 1보다 작게 나온다.**
                    mae_ref = float(np.abs(pred0 + eps - yt).mean())
                    if kind == "bias" and v == 0:
                        base_mae = mae
                    rows.append({"fold": fd.f, "model": name, "kind": kind,
                                 "eps": v, "mae": mae, "mae_ref": mae_ref})
            print(f"{fd.f:<5}{name:<10}{int(m_tr.sum()):>10,}{ni:>6}"
                  f"{base_mae:>10.4f}{time.time()-t0:>6.1f}")

    r = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    r.to_parquet(OUT, index=False)

    piv = r.groupby(["model", "kind", "eps"]).mae.agg(["mean", "std"])

    def g(model, kind, eps):
        try:
            return piv.loc[(model, kind, eps), "mean"]
        except KeyError:
            return np.nan

    pivr = r.groupby(["model", "kind", "eps"]).mae_ref.mean()

    def gr(model, kind, eps):
        try:
            return pivr.loc[(model, kind, eps)]
        except KeyError:
            return np.nan

    b0 = g("IDW 있음", "bias", 0.0)
    n0 = g("IDW 없음", "bias", 0.0)

    print(f"\n{'='*72}\n① IDW를 잃는 비용 — 예보에는 실시간 이웃 관측이 없다\n{'='*72}")
    print(f"  실황 (IDW 있음, 광역값 정확)   MAE **{b0:.4f}℃**")
    print(f"  예보 (IDW 없음, 광역값 정확)   MAE **{n0:.4f}℃**   "
          f"({(n0-b0)/b0*100:+.1f}%)")
    print(f"  IDW 하나를 잃는 비용이 **{n0-b0:+.4f}℃**다")

    print(f"\n{'='*72}\n② 광역값 계통 편차 — 예보가 일관되게 ε만큼 높다/낮다\n{'='*72}")
    print(f"  {'편차 ε':>8}{'IDW 있음':>12}{'참조':>10}{'비':>7}"
          f"{'IDW 없음':>12}{'참조':>10}{'비':>7}")
    print("  " + "-" * 64)
    for v in BIAS:
        a1, r1 = g("IDW 있음", "bias", v), gr("IDW 있음", "bias", v)
        a2, r2 = g("IDW 없음", "bias", v), gr("IDW 없음", "bias", v)
        f1 = a1 / r1 if r1 else np.nan
        f2 = a2 / r2 if r2 else np.nan
        print(f"  {v:>+7.1f}℃{a1:>12.4f}{r1:>10.4f}{f1:>7.3f}"
              f"{a2:>12.4f}{r2:>10.4f}{f2:>7.3f}")

    print(f"\n{'='*72}\n③ 광역값 무작위 오차 — 예보가 시각마다 N(0,σ)로 흔들린다\n{'='*72}")
    print(f"  {'σ':>7}{'IDW 있음':>12}{'참조':>10}{'비':>7}"
          f"{'IDW 없음':>12}{'참조':>10}{'비':>7}")
    print("  " + "-" * 63)
    for v in NOISE:
        if v == 0:
            continue
        a1, r1 = g("IDW 있음", "noise", v), gr("IDW 있음", "noise", v)
        a2, r2 = g("IDW 없음", "noise", v), gr("IDW 없음", "noise", v)
        print(f"  {v:>6.1f}℃{a1:>12.4f}{r1:>10.4f}{a1/r1:>7.3f}"
              f"{a2:>12.4f}{r2:>10.4f}{a2/r2:>7.3f}")
    print(f"\n  **참조 = 모델이 `t_reg`에 전혀 반응하지 않고 기준점으로만 오차가 더해진 경우.**")
    print(f"    비 = 실제 ÷ 참조.  **1.00이면 피처 역할이 오차를 키우지도 줄이지도 않는다.**")
    print(f"    1보다 크면 증폭, 작으면 모델이 일부 보정한다는 뜻이다.")
    print(f"\n  `MAE 증가 ÷ σ`를 「증폭비」로 읽으면 안 된다 — 오차가 독립일 때")
    print(f"    MAE는 단순 합이 아니라 **직교합**으로 커지므로 항상 1보다 작게 나온다.")

    print(f"\n{'='*72}\n④ 비교 기준을 바로잡는다 — 실황이 아니라 **같은 예보를 그대로 쓴 것**\n{'='*72}")
    print(f"  {'예보 오차 σ':>12}{'예보 그대로':>13}{'우리 다운스케일링':>18}{'개선 +':>9}")
    print("  " + "-" * 52)
    for v in [0.0] + [x for x in NOISE if x > 0]:
        raw = g("예보 그대로", "bias", 0.0) if v == 0 else g("예보 그대로", "noise", v)
        ours = n0 if v == 0 else g("IDW 없음", "noise", v)
        # 부호 규약: **개선율 = (기준 − 대상)/기준**. 양수면 오차가 줄었다는 뜻이다.
        # 이전에는 `(ours-raw)/raw`라 라벨이 「개선」인데 좋을수록 음수로 찍혔다 (9/5 정정).
        print(f"  {v:>11.1f}℃{raw:>13.4f}{ours:>18.4f}{(raw-ours)/raw*100:>+8.1f}%")
    print(f"\n  **예보가 부정확해도 우리가 더 낫다.** 다만 상대 이득은 줄어든다 —")
    print(f"    광역 오차가 **모든 셀에 공통**으로 실려 절대 MAE를 지배하기 때문이다.")

    print(f"\n{'='*72}\n⑤ 그런데 상대 판단에는 광역 오차가 **전혀** 영향이 없다\n{'='*72}")
    print(f"  광역 오차 ε는 **모든 셀에 같은 값**으로 더해진다. 그래서:")
    print(f"    · *\"어느 골목이 더 더운가\"*  → 순위 불변. **완전 무영향**")
    print(f"    · *\"이 골목이 옆보다 몇 도 낮은가\"* → 차이 불변. **완전 무영향**")
    print(f"    · *\"33℃를 넘는가\"*          → 임계값 판단이라 **ε가 그대로 들어온다**")
    print(f"\n  → 운용 서사를 나눠 쓴다. **그늘 경로 선택·개입 우선순위는 예보 오차와 무관**하고,")
    print(f"    **절대 임계값 경보만 예보 정확도에 걸린다.**")
    print(f"\n→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
