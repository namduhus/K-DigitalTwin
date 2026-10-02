from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from core.eval.dataset import (  # 상수는 로더와 한 곳에서 공유한다 — 두 곳이 어긋나면 조용히 틀린다
    EXCLUDE_SN,
    LOCATIONS,
    MIN_HOURS,
    MIN_TRAJ_EVAL,
    N_FOLDS,
    REQUIRED_FEATURES,
    SEED,
    TRAIN,
)

OUT_SPLITS = Path("data/metric/splits.json")
OUT_REGIONAL = Path("data/metric/regional.parquet")

# 자치구는 `train.parquet`의 `gu_addr`이 아니라 위치파일에서 조인한다.
# `gu_addr`은 109개 센서가 결측이다 — 원본 xlsx 주소에 `서울측별시` 오타가 있어
# 상류 정규식이 못 뽑았다. `prep_locations.py`는 고쳤지만 `train.parquet`은
# 다음 리빌드(Phase 2.5) 때 반영되므로, 그때까지 위치파일이 권위 있는 소스다.
#
# `train.parquet`의 `gu`(API 응답 필드)도 쓰면 안 된다. 8개 센서에서 행마다 값이
# 흔들리고 존재하지 않는 자치구 `Seoul_Grand_Park`가 섞여 고유값이 26개가 된다.

# `EXCLUDE_SN`(과천 서울대공원 4센서)과 `REQUIRED_FEATURES`(조건 피처 전무 센서
# 자동 제외)는 `core/eval/dataset.py`에 있다 — 로더와 두 곳에 두면 어긋난다.


def assign_folds(sen: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    out = pd.Series(-1, index=sen.index, dtype=int)
    for rgn, grp in sen.groupby("rgn"):
        # 자치구별로 섞고, 자치구를 번갈아 꺼내 한 줄로 만든다
        by_gu = {g: list(rng.permutation(list(idx)))
                 for g, idx in grp.groupby("gu_addr").groups.items()}
        order = []
        while by_gu:
            for g in sorted(by_gu):
                order.append(by_gu[g].pop())
                if not by_gu[g]:
                    del by_gu[g]
        # 시작 fold를 유형마다 돌려 잔여분이 한 fold에 쌓이지 않게 한다
        offset = rng.integers(N_FOLDS)
        for i, sn in enumerate(order):
            out.loc[sn] = int((i + offset) % N_FOLDS)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args()

    if not TRAIN.exists():
        sys.exit(f"[error] {TRAIN} 없음")

    d = pd.read_parquet(TRAIN, columns=["sn", "rgn", "ts", "tp", "sensor_ok"] + REQUIRED_FEATURES)
    n_all = len(d)
    d = d[d["sensor_ok"] & d["tp"].notna()]
    print(f"관측 {n_all:,}행 → QC 통과 {len(d):,}행 · 센서 {d['sn'].nunique():,}")

    n_ex = int(d["sn"].isin(EXCLUDE_SN).sum())
    d = d[~d["sn"].isin(EXCLUDE_SN)]
    print(f"서울대공원(과천 소재) {len(EXCLUDE_SN)}센서 제외 → {n_ex:,}행 제거, 센서 {d['sn'].nunique():,}")

    nofeat = d.groupby("sn")[REQUIRED_FEATURES].apply(lambda g: g.isna().all().any())
    drop = sorted(nofeat.index[nofeat])
    if drop:
        n_d = int(d["sn"].isin(drop).sum())
        d = d[~d["sn"].isin(drop)]
        print(f"조건 피처 전무 센서 {len(drop)}개 제외 → {n_d:,}행 제거, 센서 {d['sn'].nunique():,}")
        print(f"  {drop}")

    gu = pd.read_csv(LOCATIONS).set_index("sn")["gu"]
    d["gu_addr"] = d["sn"].map(gu)

    # 일관성 검사에 결측을 반드시 포함한다. `nunique()`는 NaN을 무시하므로
    # 전부 결측인 센서는 nunique()==0이 되어 ">1" 조건을 통과한다. 실제로 그 때문에
    # 109개 센서의 자치구 결측이 검사를 통과했고, groupby가 말없이 버렸다.
    for c in ("gu_addr", "rgn"):
        cnt = d.groupby("sn")[c].agg(["nunique", "count", "size"])
        multi = cnt[cnt["nunique"] > 1]
        miss = cnt[cnt["count"] < cnt["size"]]
        if len(multi):
            sys.exit(f"[error] {c}가 센서당 일관하지 않다 ({len(multi)}개): {list(multi.index[:5])}")
        if len(miss):
            sys.exit(f"[error] {c} 결측 센서 {len(miss)}개: {list(miss.index[:5])}. 층화 불가")
    print(f"층화 키 검사 통과 — gu_addr({d['gu_addr'].nunique()}구) · rgn({d['rgn'].nunique()}종) "
          f"모두 센서당 고유값 1, 결측 0")

    d["date"] = d["ts"].dt.normalize()
    hours = d.groupby(["sn", "date"]).size()
    traj = hours[hours >= MIN_HOURS].groupby("sn").size()

    sen = (d.groupby("sn")[["gu_addr", "rgn"]].first()
             .assign(n_traj=traj).fillna({"n_traj": 0}))
    sen["n_traj"] = sen["n_traj"].astype(int)

    thin = sorted(sen.index[sen["n_traj"] < MIN_TRAJ_EVAL])
    elig = sen[sen["n_traj"] >= MIN_TRAJ_EVAL]
    print(f"\n궤적({MIN_HOURS}h+) 총 {int(sen['n_traj'].sum()):,}개")
    print(f"  평가 대상 센서 {len(elig):,} (궤적 {MIN_TRAJ_EVAL}개 이상)")
    print(f"  always-train 센서 {len(thin):,} (궤적 중위 {sen.loc[thin,'n_traj'].median():.0f})")

    fold = assign_folds(elig, np.random.default_rng(SEED))
    assert (fold >= 0).all(), "배정 누락"

    print(f"\n{'fold':<6}{'test 센서':>10}{'test 궤적':>11}{'자치구':>8}{'지역유형':>9}")
    print("-" * 45)
    for f in range(N_FOLDS):
        ss = elig[fold == f]
        print(f"{f:<6}{len(ss):>10,}{ss['n_traj'].sum():>11,}{ss['gu_addr'].nunique():>8}{ss['rgn'].nunique():>9}")

    print(f"\n지역유형 × fold (test 센서 수) — 희소 유형이 모든 fold에 있어야 한다")
    tab = pd.crosstab(elig["rgn"], fold)
    tab["합"] = tab.sum(axis=1)
    tab["always-train"] = sen.loc[thin].groupby("rgn").size()
    print(tab.fillna(0).astype(int).to_string())

    zero = [(r, f) for r in tab.index for f in range(N_FOLDS) if tab.loc[r, f] == 0]
    print(f"\n  test 0개인 (유형, fold) 조합: {len(zero)}개" + (f" → {zero}" if zero else ""))

    gt = pd.crosstab(elig["gu_addr"], fold)
    print(f"\n자치구 × fold — 25구 전부가 모든 fold에 test를 갖는가")
    print(f"  test 0개인 (자치구, fold) 조합: {int((gt == 0).sum().sum())}개")
    print(f"  자치구별 총 test 센서: 최소 {gt.sum(axis=1).min()} · 중위 {gt.sum(axis=1).median():.0f} · 최대 {gt.sum(axis=1).max()}")

    # fold별 광역 기준값 — 해당 fold의 학습 센서(= 다른 fold + always-train)만의 중위
    print(f"\n광역 기준값 — fold별 학습 센서 중위")
    regs = []
    for f in range(N_FOLDS):
        tr = set(elig.index[fold != f]) | set(thin)
        r = (d[d["sn"].isin(tr)].groupby("ts")
             .agg(t_reg=("tp", "median"), n_sensor=("tp", "size")).reset_index())
        r["fold"] = f
        regs.append(r)
        print(f"  fold {f}: 학습센서 {len(tr):,} · {len(r):,}시각 · 시각당 기여센서 중위 {r['n_sensor'].median():.0f} 최소 {r['n_sensor'].min()}")
    reg = pd.concat(regs, ignore_index=True)

    lean = reg.groupby("ts")["n_sensor"].min().pipe(lambda s: int((s < 30).sum()))
    print(f"  기여 센서 30개 미만 시각 {lean:,}개 — 중위가 불안정하므로 학습에서 제외 검토"
          if lean else "  기여 센서 30개 미만 시각 없음")

    meta = {
        "scheme": "sensor-level 5-fold CV",
        "seed": SEED,
        "n_folds": N_FOLDS,
        "min_hours": MIN_HOURS,
        "min_traj_eval": MIN_TRAJ_EVAL,
        "stratify": "rgn (1차) + gu_addr 라운드로빈 (2차)",
        "n_traj_total": int(sen["n_traj"].sum()),
        "n_sensor_eval": len(elig),
        "n_sensor_always_train": len(thin),
        "folds": {str(f): sorted(elig.index[fold == f]) for f in range(N_FOLDS)},
        "always_train": thin,
    }
    OUT_SPLITS.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    reg.to_parquet(OUT_REGIONAL, index=False)
    print(f"\n→ {OUT_SPLITS}")
    print(f"→ {OUT_REGIONAL} ({OUT_REGIONAL.stat().st_size/1e3:.0f} KB, {len(reg):,}행 = 시각 × fold)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
