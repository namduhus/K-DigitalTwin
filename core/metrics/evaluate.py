from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from core.eval.dataset import (
    N_FOLDS,
    PREDS,
    feature_bins,
    hot_timestamps,
    iter_folds,
    load_base,
    load_regional,
    load_sensor_meta,
    sparse_gu,
)

OUT = Path("data/metric/metrics.parquet")

BIN_FEATURES = ["grn_frac", "svf", "bld_d_min"]

# 음영 관련 행 단위 슬라이스. 전체 MAE에서 음영·SVF 기여가 1.1%로 작게 나오는데,
# 그것이 "기여가 없다"는 뜻인지 "쉬운 행이 분모를 채운다"는 뜻인지 가려야 한다.
# 그늘 여부가 실제로 갈리는 행만 골라 재면 후자임을 확인할 수 있다.
#
#   shade      그늘/양지 (주간 행만 — 야간은 음영이 정의되지 않는다)
#   sun_el     태양고도 구간. 그늘 효과가 -0.40 → -1.10℃로 커지는 구간이다
#   margin     shadow_margin = 태양고도 − 장애물앙각. **경계(-10~+10°)가 정보가 결정적인 곳**
SHADE_SLICES = ["shade", "sun_el_band", "margin_band"]
SUN_EL_EDGES = [0, 10, 30, 50, 90]
MARGIN_EDGES = [-90, -30, -10, 10, 30, 90]


# ─────────────────────────────────────────────────────────── 슬라이스

def build_slices(base: pd.DataFrame) -> dict[str, pd.Series]:
    meta = load_sensor_meta()
    hot = set(hot_timestamps(base))
    sparse = set(sparse_gu(meta))
    gu = base["sn"].map(meta["gu"])

    sl: dict[str, pd.Series] = {
        "all": pd.Series("all", index=base.index),
        "hot": pd.Series(np.where(base["ts"].isin(hot), "hot", None), index=base.index),
        "daynight": pd.Series(np.where(base["is_night"], "night", "day"), index=base.index),
        "sparse_gu": pd.Series(np.where(gu.isin(sparse), "sparse", None), index=base.index),
    }
    for c in BIN_FEATURES:
        if c in base.columns:
            sl[c] = base["sn"].map(feature_bins(base, c)).astype("object")

    # 음영 슬라이스 — 야간 행은 음영이 정의되지 않으므로 None으로 빠진다
    if "is_shadow" in base.columns:
        sh = base["is_shadow"]
        sl["shade"] = pd.Series(np.where(sh.isna(), None,
                                np.where(sh.fillna(False).astype(bool), "그늘", "양지")),
                                index=base.index)
    if "sun_el" in base.columns:
        lab = pd.cut(base["sun_el"], SUN_EL_EDGES,
                     labels=[f"{a}~{b}°" for a, b in zip(SUN_EL_EDGES[:-1], SUN_EL_EDGES[1:])])
        sl["sun_el_band"] = lab.astype("object").where(lab.notna(), None)
    if "shadow_margin" in base.columns:
        lab = pd.cut(base["shadow_margin"], MARGIN_EDGES,
                     labels=["-30↓", "-30~-10", "경계 -10~+10", "+10~+30", "+30↑"])
        sl["margin_band"] = lab.astype("object").where(lab.notna(), None)
    return sl


# ─────────────────────────────────────────────────────────── 지표

def _agg(err: np.ndarray) -> tuple[int, float, float, float]:
    n = err.size
    if n == 0:
        return 0, np.nan, np.nan, np.nan
    return n, float(np.abs(err).mean()), float(np.sqrt((err ** 2).mean())), float(err.mean())


def score(base: pd.DataFrame, preds: dict[str, np.ndarray],
          slices: dict[str, pd.Series]) -> pd.DataFrame:
    y_true = base["tp"].to_numpy(np.float32)
    fold_of = base["fold"].to_numpy()
    rows = []
    for model, y_pred in preds.items():
        err_all = y_pred - y_true
        for f in range(N_FOLDS):
            m_test = fold_of == f
            for kind, lab in slices.items():
                labv = lab.to_numpy()
                for val in pd.unique(labv[m_test]):
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        continue
                    m = m_test & (labv == val)
                    n, mae, rmse, bias = _agg(err_all[m])
                    rows.append((model, f, kind, str(val), n, mae, rmse, bias))
    return pd.DataFrame(rows, columns=["model", "fold", "slice_kind", "slice_val",
                                       "n", "mae", "rmse", "bias"])


def fold_summary(m: pd.DataFrame) -> pd.DataFrame:
    g = m.groupby(["model", "slice_kind", "slice_val"], observed=True)
    return g.agg(n=("n", "sum"), mae=("mae", "mean"), mae_sd=("mae", "std"),
                 rmse=("rmse", "mean"), bias=("bias", "mean")).reset_index()


# ─────────────────────────────────────────────────────────── 출력

def report(s: pd.DataFrame, kinds: list[str] | None = None) -> None:
    models = list(dict.fromkeys(s["model"]))
    for kind in (kinds or ["all", "hot", "daynight", "sparse_gu"] + BIN_FEATURES + SHADE_SLICES):
        sub = s[s["slice_kind"] == kind]
        if sub.empty:
            continue
        piv = sub.pivot(index="slice_val", columns="model", values="mae")
        sd = sub.pivot(index="slice_val", columns="model", values="mae_sd")
        nn = sub.pivot(index="slice_val", columns="model", values="n")
        print(f"\n  [{kind}]  MAE ℃ (5-fold 평균 ± 표준편차)")
        head = f"{'구간':<10}{'행':>12}" + "".join(f"{m:>18}" for m in models)
        print("  " + head); print("  " + "-" * len(head))
        for v in piv.index:
            line = f"{v:<10}{int(nn.loc[v, models[0]]):>12,}"
            for m in models:
                line += f"{piv.loc[v, m]:>12.3f} ±{sd.loc[v, m]:.3f}"
            print("  " + line)


# ─────────────────────────────────────────────────────────── 검산

def verify(base: pd.DataFrame, preds: dict[str, np.ndarray], regional) -> bool:
    if "b0a" not in preds:
        print("[verify] b0a 예측이 없어 건너뜀")
        return True
    print("\n[verify] B0a는 Δ̂=0 이므로 MAE가 mean(|Δ|)와 같아야 한다")
    ok = True
    fold_of = base["fold"].to_numpy()
    y_true = base["tp"].to_numpy(np.float32)
    for fd in iter_folds(base, regional):
        m = fold_of == fd.f
        independent = float(np.abs(fd.delta[m]).mean())      # |tp - t_reg|
        harness = float(np.abs(preds["b0a"][m] - y_true[m]).mean())
        d = abs(independent - harness)
        flag = "통과" if d < 1e-4 else "실패"
        if d >= 1e-4:
            ok = False
        print(f"  fold {fd.f}: 독립 {independent:.6f} · 하네스 {harness:.6f} · 차 {d:.2e} {flag}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None, help="기본: data/metric/preds/* 전부")
    ap.add_argument("--verify", action="store_true", help="하네스 검산 수행")
    a = ap.parse_args()

    files = ([PREDS / f"{m}.parquet" for m in a.models] if a.models
             else sorted(PREDS.glob("*.parquet")))
    files = [f for f in files if f.exists()]
    if not files:
        sys.exit(f"[error] {PREDS} 에 예측이 없다 — 먼저 baseline을 돌린다")
    print(f"예측 {len(files)}개: {', '.join(f.stem for f in files)}")

    feats = BIN_FEATURES + ["is_night", "is_shadow", "sun_el", "shadow_margin"]
    base = load_base(feats)
    regional = load_regional()

    key = pd.MultiIndex.from_arrays([base["sn"], base["ts"]])
    preds: dict[str, np.ndarray] = {}
    for f in files:
        p = pd.read_parquet(f)
        s = pd.Series(p["y_pred"].to_numpy(np.float32),
                      index=pd.MultiIndex.from_arrays([p["sn"].astype(str), p["ts"]]))
        v = s.reindex(key).to_numpy(np.float32)
        # always-train 센서(fold=-1)는 어느 fold에서도 test가 아니므로 예측이 없는 게 정상이다.
        # 그 수를 넘어서는 결측만 문제다 — 조인 키가 어긋났다는 뜻이다.
        n_at = int((base["fold"].to_numpy() == -1).sum())
        miss = int(np.isnan(v).sum())
        if miss == n_at:
            print(f"  {f.stem}: 예측 {len(v)-miss:,}행 (always-train {n_at:,}행 제외 — 정상)")
        elif miss > n_at:
            print(f"  {f.stem}: 평가 대상인데 예측 없는 행 {miss - n_at:,} — 조인 키 확인")
        else:
            print(f"  {f.stem}: 결측 {miss:,} < always-train {n_at:,} — always-train에 예측이 붙었다")
        preds[f.stem] = v

    if a.verify and not verify(base, preds, regional):
        sys.exit("[error] 하네스 검산 실패 — 조인·마스크·fold 배정을 확인한다")

    slices = build_slices(base)
    m = score(base, preds, slices)
    s = fold_summary(m)
    report(s)

    contribution(s)

    m.to_parquet(OUT, index=False)
    print(f"\n→ {OUT} ({len(m):,}행)")
    return 0


def contribution(s: pd.DataFrame, a: str = "b3_noshade", b: str = "b3_hybrid") -> None:
    have = set(s["model"])
    if not {a, b} <= have:
        return
    A = s[s.model == a].set_index(["slice_kind", "slice_val"])["mae"]
    B = s[s.model == b].set_index(["slice_kind", "slice_val"])["mae"]
    N = s[s.model == b].set_index(["slice_kind", "slice_val"])["n"]
    d = (A - B).dropna()

    print(f"\n{'='*64}\n음영·SVF 기여 = {a} − {b}  (양수면 음영을 넣어 좋아졌다)\n{'='*64}")
    print(f"  {'슬라이스':<14}{'구간':<16}{'행':>11}{'기여 ℃':>9}{'%':>8}")
    print("  " + "-" * 58)
    for (kind, val) in d.sort_values(ascending=False).index:
        if kind not in (["all", "hot", "daynight"] + SHADE_SLICES):
            continue
        print(f"  {kind:<14}{val:<16}{int(N[(kind,val)]):>11,}"
              f"{d[(kind,val)]:>+9.4f}{d[(kind,val)]/A[(kind,val)]*100:>+8.2f}%")
    base_pct = d[("all", "all")] / A[("all", "all")] * 100
    print(f"\n  전체 기여 {base_pct:+.2f}% 를 기준으로, 조건별 배수:")
    for kind in SHADE_SLICES:
        sub = [(v, d[(kind, v)] / A[(kind, v)] * 100) for (k, v) in d.index if k == kind]
        for v, pct in sorted(sub, key=lambda x: -x[1])[:3]:
            print(f"    {kind:<14}{v:<16}{pct:+.2f}%  = 전체의 {pct/base_pct:.1f}배")


if __name__ == "__main__":
    raise SystemExit(main())
