from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core.baseline.b2_gbm import PARAMS
from core.baseline.b3_hybrid import ALL_F, BASE_F, EARLY, FEATS, N_ROUND, TREG_IX
from core.baseline.grid_infer import AZ_STEP, N_AZ, QUANTILES, sun_features
from core.eval.dataset import (
    N_FOLDS, iter_folds, load_base, load_regional, sensor_fold_map,
)

KMA = Path("data/processing/kma_treg.parquet")
HORIZON = Path("data/processing/sensor_horizon.parquet")
TRAIN = Path("data/processing/train.parquet")
# 검증 관측은 **학습 스냅샷과 분리해서** 읽는다 (9/5 확정).
#
# 이 실험이 성립하는 근거는 **예보 구간이 학습에 없다**는 것 하나다. 그런데 관측을
# `train.parquet`에서 읽으면, 데이터를 갱신할 때마다 학습 스냅샷도 같이 밀려
# **검증 구간이 학습에 들어가 버린다** — 그것도 조용히. 그래서 관측만 갱신되는
# `sdot_summer.parquet`에서 읽고 `train.parquet`은 8/18 스냅샷으로 동결한다.
# → `decisions.md` ADR-021
OBS = Path("data/processing/sdot_summer.parquet")
OUT = Path("data/metric/forecast_sensors.parquet")

TIME_F_NAMES = ["hour", "doy", "year", "sun_el", "sun_az", "is_night", "t_reg"]
SEED = 20260829


def train_one(X, y, cols, m_tr, m_va, quant=None):
    import lightgbm as lgb
    p = dict(PARAMS) if quant is None else dict(PARAMS, objective="quantile", alpha=quant)
    dtr = lgb.Dataset(X[m_tr][:, cols], label=y[m_tr])
    dva = lgb.Dataset(X[m_va][:, cols], label=y[m_va], reference=dtr)
    return lgb.train(p, dtr, num_boost_round=N_ROUND, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(EARLY, verbose=False)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="관측과 대조 (9/5 이후)")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)

    if a.verify:
        return verify()

    kma = pd.read_parquet(KMA).sort_values("ts")
    want = pd.DatetimeIndex(kma.ts)
    treg_map = pd.Series(kma.t_reg.to_numpy(), index=want)
    print(f"[예보] 발표 {kma.base.iloc[0]} · 시각 {len(want)}개 "
          f"({want.min():%m-%d %H시} ~ {want.max():%m-%d %H시})")

    # ── 센서 정적 피처 ──
    base = load_base(BASE_F)
    regional = load_regional()
    static = [c for c in ALL_F if c not in TIME_F_NAMES + ["hz_sun", "shadow_margin",
                                                           "is_shadow"]]
    sen = base.groupby("sn")[static].first().dropna()
    hz = pd.read_parquet(HORIZON).set_index("sn")
    hzc = [f"hz_{i:03d}" for i in range(N_AZ)]
    sen = sen.join(hz[hzc], how="inner")
    fmap = sensor_fold_map()
    sen["fold"] = fmap.reindex(sen.index).fillna(-1).astype(int)
    print(f"[센서] 정적 피처 완전 **{len(sen):,}개** · "
          f"평가 대상(fold≥0) {int((sen.fold >= 0).sum()):,} · "
          f"always-train {int((sen.fold < 0).sum())}")

    # ── 모델 학습 ──
    X = np.zeros((len(base), len(ALL_F)), np.float32)
    X[:, [ALL_F.index(c) for c in BASE_F]] = base[BASE_F].to_numpy(np.float32)
    cols = [ALL_F.index(c) for c in FEATS["b3_hybrid"]]      # IDW 없음
    sn_arr = base["sn"].to_numpy()
    fd0 = next(iter_folds(base, regional))
    X[:, TREG_IX] = fd0.t_reg
    y = fd0.delta

    print(f"\n모델 학습 — IDW 없음 · 피처 {len(cols)}개")
    models = {}
    # ① 전체 센서 (시간 외삽만 측정)
    val_sn = rng.choice(np.unique(sn_arr), size=max(len(np.unique(sn_arr)) // 5, 1),
                        replace=False)
    m_va = np.isin(sn_arr, val_sn)
    t0 = time.time()
    models["all"] = {k: train_one(X, y, cols, ~m_va, m_va,
                                  None if k == "point" else QUANTILES[k])
                     for k in ("point", "q10", "q90")}
    print(f"  전체 센서 학습 · {time.time()-t0:.0f}초")

    # ② fold별 (이중 hold-out)
    sensor_fold = pd.Series(sen.fold, index=sen.index)
    for f in range(N_FOLDS):
        te_sn = sensor_fold.index[sensor_fold == f]
        m_te = np.isin(sn_arr, te_sn)
        inner = sensor_fold.index[sensor_fold == (f + 1) % N_FOLDS]
        m_va = np.isin(sn_arr, inner) & ~m_te
        t0 = time.time()
        models[f] = {k: train_one(X, y, cols, ~m_te & ~m_va, m_va,
                                  None if k == "point" else QUANTILES[k])
                     for k in ("point", "q10", "q90")}
        print(f"  fold {f} 학습 (test 센서 {len(te_sn):,}개 제외) · {time.time()-t0:.0f}초")

    # ── 예보 시각마다 예측 ──
    S = sen[static].to_numpy(np.float32)
    HZ = sen[hzc].to_numpy(np.float32)
    folds = sen.fold.to_numpy()
    sun = sun_features(want).set_index("ts")
    n = len(sen)
    six = {c: ALL_F.index(c) for c in static}

    recs, t0 = [], time.time()
    for k, ts in enumerate(want):
        s = sun.loc[ts]
        ai = int(s.sun_az / AZ_STEP) % N_AZ
        hz_sun = HZ[:, ai]
        night = s.sun_el <= 0
        margin = np.full(n, np.nan, np.float32)
        shadow = np.full(n, np.nan, np.float32)
        if not night:
            margin = (s.sun_el - hz_sun).astype(np.float32)
            shadow = (margin < 0).astype(np.float32)
        t_reg = float(treg_map.loc[ts])

        Xf = np.empty((n, len(ALL_F)), np.float32)
        for c, j in six.items():
            Xf[:, j] = S[:, static.index(c)]
        Xf[:, ALL_F.index("hour")] = ts.hour
        Xf[:, ALL_F.index("doy")] = ts.dayofyear
        Xf[:, ALL_F.index("year")] = ts.year
        Xf[:, ALL_F.index("sun_el")] = s.sun_el
        Xf[:, ALL_F.index("sun_az")] = s.sun_az
        Xf[:, ALL_F.index("is_night")] = float(night)
        Xf[:, ALL_F.index("hz_sun")] = hz_sun
        Xf[:, ALL_F.index("shadow_margin")] = margin
        Xf[:, ALL_F.index("is_shadow")] = shadow
        Xf[:, TREG_IX] = t_reg
        Xc = Xf[:, cols]

        r = {"sn": sen.index.to_numpy(), "ts": np.repeat(ts, n),
             "fold": folds, "t_reg": np.full(n, t_reg, np.float32)}
        for tag, key in (("all", "all"),):
            for q in ("point", "q10", "q90"):
                m = models[tag][q]
                r[f"{key}_{q}"] = (t_reg + m.predict(Xc, num_iteration=m.num_trees())
                                   ).astype(np.float32)
        # fold별 — 각 센서를 그 센서를 안 본 모델로
        for q in ("point", "q10", "q90"):
            out = np.full(n, np.nan, np.float32)
            for f in range(N_FOLDS):
                sel = folds == f
                if not sel.any():
                    continue
                m = models[f][q]
                out[sel] = t_reg + m.predict(Xc[sel], num_iteration=m.num_trees())
            r[f"cv_{q}"] = out
        recs.append(pd.DataFrame(r))
        print(f"  {k+1}/{len(want)} · {time.time()-t0:.0f}초", end="\r")
    print()

    df = pd.concat(recs, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\n→ {OUT} ({OUT.stat().st_size/1e6:.1f} MB · {len(df):,}행)")
    print(f"   센서 {df.sn.nunique():,} × 시각 {df.ts.nunique()}")
    print(f"   `all_*` = 전체 센서 학습 (시간 외삽만) · "
          f"`cv_*` = fold별 (이중 hold-out, always-train은 NaN)")
    # 관측이 어디까지 있는지 미리 알려준다 — 9/5에 무엇을 받아야 하는지가 분명해진다
    last_obs = pd.read_parquet(OBS, columns=["ts"]).ts.max()
    gap = (want.min() - last_obs).total_seconds() / 3600
    print(f"\n  현재 관측 마지막 시각 {last_obs:%Y-%m-%d %H시} · "
          f"예보 시작까지 공백 **{gap:.0f}시간**")
    print(f"  → 9/5 갱신에서 이 구간이 채워져야 대조가 성립한다 "
          f"(API 32일 윈도우가 덮는다)")
    print(f"\n  9/5에 `--verify`로 관측과 대조한다.")
    print(f"    그 전에 `scripts/fetch_sdot.py fetch`로 API 구간을 받아야 한다 —")
    print(f"    **주간 파일은 약 1주 지연이라 예보 구간이 아직 없을 수 있다.**")
    return 0


def verify() -> int:
    if not OUT.exists():
        raise SystemExit(f"[error] {OUT} 없음 — 먼저 예측을 만든다")
    p = pd.read_parquet(OUT)
    obs = pd.read_parquet(OBS, columns=["sn", "ts", "tp", "sensor_ok"])
    obs = obs[obs.sensor_ok & obs.tp.notna()][["sn", "ts", "tp"]]
    obs["sn"] = obs["sn"].astype(str)
    p["sn"] = p["sn"].astype(str)

    # 시각을 정확히 맞춰 조인하면 **매칭이 0이 된다.**
    # S-DoT 관측은 전부 `:07분`이고 기상청 예보는 `:00분`이다(실측 확인).
    # 8/24 격자 추론에서도 같은 함정을 겪었다.
    # **시(hour) 단위로 내려 붙인다.**
    p["h"] = p.ts.dt.floor("h")
    obs["h"] = obs.ts.dt.floor("h")
    d = p.merge(obs[["sn", "h", "tp"]], on=["sn", "h"], how="inner")
    print(f"[대조] 예측 {len(p):,} × 관측 → **매칭 {len(d):,}행** "
          f"({len(d)/len(p)*100:.1f}%) · 센서 {d.sn.nunique():,} · 시각 {d.h.nunique()}")
    if not len(d):
        print("\n  매칭이 0이다 — 예보 구간 관측이 아직 안 들어왔다.")
        print("     `scripts/download_sdot_files.py get --weeks` + "
              "`scripts/fetch_sdot.py fetch` 후 `build_dataset.py`·`build_train.py` 재실행")
        return 1

    # ① 예보 자체의 오차 σ
    reg = d.groupby("h").agg(t_reg=("t_reg", "first"), obs_med=("tp", "median"))
    err = reg.t_reg - reg.obs_med
    print(f"\n{'='*70}\n① 기상청 예보 오차 — 이게 감도표에 대입할 σ다\n{'='*70}")
    print(f"  편향(평균) **{err.mean():+.3f}℃** · **σ = {err.std():.3f}℃** · "
          f"MAE {err.abs().mean():.3f}℃")
    print(f"  범위 {err.min():+.1f} ~ {err.max():+.1f}℃ · 시각 {len(err)}개")

    # ② 우리 산출물의 MAE
    print(f"\n{'='*70}\n② 예보 다운스케일링 성능\n{'='*70}")
    print(f"  {'모델':<26}{'행':>10}{'MAE':>10}{'80% 적중':>11}")
    print("  " + "-" * 57)
    rows = []
    for tag, lab in (("all", "전체 센서 학습 (시간 외삽만)"),
                     ("cv", "fold별 (시간+공간 이중)")):
        m = d[f"{tag}_point"].notna()
        e = (d.loc[m, f"{tag}_point"] - d.loc[m, "tp"]).abs()
        cov = ((d.loc[m, "tp"] >= d.loc[m, f"{tag}_q10"]) &
               (d.loc[m, "tp"] <= d.loc[m, f"{tag}_q90"])).mean()
        print(f"  {lab:<26}{int(m.sum()):>10,}{e.mean():>10.4f}{cov*100:>10.1f}%")
        rows.append((lab, e.mean(), cov))
    # 기준선 — 예보를 그대로 (모든 센서에 같은 값)
    e0 = (d.t_reg - d.tp).abs().mean()
    print(f"  {'예보 그대로 (다운스케일링 없음)':<26}{len(d):>10,}{e0:>10.4f}{'—':>11}")
    # 부호 규약: **개선율 = (기준 − 대상)/기준**. 양수면 오차가 줄었다 (`CLAUDE.md`).
    print(f"\n  다운스케일링 개선 — 전체학습 {(e0-rows[0][1])/e0*100:+.1f}% · "
          f"이중 hold-out {(e0-rows[1][1])/e0*100:+.1f}%  (양수면 좋아진 것)")
    print(f"\n  공간 in-sample과 이중 hold-out의 차이가 **공간 성능의 낙관 정도**다.")

    # ③ 오차 분해 — ②를 그대로 읽으면 결론을 잘못 내린다 (9/5 실측).
    #
    # ②에서 다운스케일링 이득이 0에 가깝게 나오는데, 그것은 우리 모델이 무력해서가
    # 아니라 **앵커가 통째로 밀려 있어서**다. 산출물은 T = t_reg + Δ̂ 구조라
    # `t_reg`의 계통 편향이 모든 센서에 같은 부호로 실린다. 그러면 오차의 대부분이
    # 공통항이 되고, 우리 모델과 「예보 그대로」의 차이(= Δ̂의 몫)가 상대적으로
    # 묻힌다 — **둘 다 같은 편향을 지고 있으니 비교 자체가 편향에 둔감해진다.**
    #
    # 그래서 세 층으로 나눠 본다.
    #   ⓐ 앵커=예보 그대로     ← ②와 같음. 운용 그대로의 값
    #   ⓑ 앵커=예보−편향       ← 편향만 상수로 걷어냈을 때
    #   ⓒ 앵커=관측 중위(진값)  ← 예보가 완벽하다면. **공간 몫만 남는다**
    #
    # ⓒ가 핵심이다. 지금까지 평가는 전부 기간이 겹친 지점 hold-out이었으므로
    # *"학습에 없는 날짜"*에서 공간 이득이 남는지는 여기서 처음 확인된다.
    med = d.groupby("h").tp.median().rename("obs_med")
    d = d.join(med, on="h")
    bias_h = d.groupby("h").t_reg.first() - med
    b = bias_h.mean()
    by_hour = bias_h.groupby(bias_h.index.hour).mean()

    print(f"\n{'='*70}\n③ 오차 분해 — 앵커 편향과 공간 몫을 나눈다\n{'='*70}")
    print(f"  앵커 편향 b = **{b:+.3f}℃** (시각별 σ {bias_h.std():.3f}) · "
          f"야간 최대 {by_hour.min():+.2f}℃({by_hour.idxmin()}시) · "
          f"주간 최소 {by_hour.max():+.2f}℃({by_hour.idxmax()}시)")
    print(f"  → 편향이 시간대에 의존한다. 상수 하나로는 절반만 걷힌다")

    dev_all, dev_cv = d.all_point - d.t_reg, d.cv_point - d.t_reg
    layers = [
        ("ⓐ 앵커=예보", [("예보 그대로", d.t_reg), ("다운스케일", d.all_point),
                       ("이중 hold-out", d.cv_point)]),
        ("ⓑ 앵커=예보−편향", [("예보 그대로", d.t_reg - b), ("다운스케일", d.all_point - b),
                          ("이중 hold-out", d.cv_point - b)]),
        ("ⓒ 앵커=관측중위(진값)", [("광역 균일", d.obs_med), ("다운스케일", d.obs_med + dev_all),
                             ("이중 hold-out", d.obs_med + dev_cv)]),
    ]
    print(f"\n  {'층':<20}{'광역 균일':>11}{'다운스케일':>12}{'이중 hold-out':>15}{'개선 +':>10}")
    print("  " + "-" * 68)
    for lab, items in layers:
        v = []
        for _, pr in items:
            m = pr.notna()
            v.append(float((pr[m] - d.tp[m]).abs().mean()))
        # 개선율 = (광역 균일 − 이중 hold-out)/광역 균일. **양수면 좋아진 것**이다.
        print(f"  {lab:<20}{v[0]:>11.4f}{v[1]:>12.4f}{v[2]:>15.4f}"
              f"{(v[0]-v[2])/v[0]*100:>+9.1f}%")

    print(f"\n  ⓒ의 개선폭이 **시간 외삽에서도 공간 상세화가 남는가**에 대한 답이다.")
    print(f"  ⓐ와 ⓒ의 차이는 우리 모델이 아니라 **광역 앵커 품질**이 만든 것이다 —")
    print(f"    운용에서는 앵커를 실황(관측 중위)이나 편향 보정 예보로 받아야 한다.")

    # 구간은 편차의 불확실성만 담고 있어 앵커 편향을 못 덮는다. 보정 후로 다시 잰다.
    print(f"\n  80% 구간 적중 — 편향 보정 후")
    for tag in ("all", "cv"):
        lo, hi = d[f"{tag}_q10"] - b, d[f"{tag}_q90"] - b
        m = lo.notna()
        c = ((d.tp[m] >= lo[m]) & (d.tp[m] <= hi[m])).mean() * 100
        print(f"    {tag:<4}{c:>7.1f}%  (평균 폭 {(hi[m]-lo[m]).mean():.2f}℃)")
    print(f"  적중이 80%에 못 미친다 — conformal 보정은 학습 기간 안에서 잡은 것이라")
    print(f"    미래 구간·예보 앵커에는 그대로 옮겨지지 않는다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
