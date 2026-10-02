from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

TRAIN = Path("data/processing/train.parquet")
LOCATIONS = Path("data/processing/sdot_locations.csv")
SPLITS = Path("data/metric/splits.json")
REGIONAL = Path("data/metric/regional.parquet")
PREDS = Path("data/metric/preds")

MIN_HOURS = 20          # 하루 24시간 중 이만큼 관측되면 궤적 1개로 인정
MIN_TRAJ_EVAL = 20      # 평가 대상 최소 궤적 수. 이하는 always-train
N_FOLDS = 5
SEED = 20260821

# 서울대공원 4센서 — 주소가 문자 그대로 '서울대공원'이고 과천 소재다. 서울 센서 중
# 최남단이며 자치구가 없어 층화가 불가능하다. "서울 한정" 제안에서 과천 데이터로
# 학습했다는 지적을 받을 이유가 없다. 제외해도 `parks` 관측가중 편차는
# -1.363 → -1.333℃로 거의 같다.
EXCLUDE_SN = ["OC3CL200061", "OC3CL200010", "OC3CL200065", "OC3CL200011"]

# 조건 피처가 없는 센서는 학습도 평가도 불가능하므로 자동으로 뺀다. 하드코딩보다
# 낫다 — 같은 일이 다시 생겨도 걸린다. 실제로 `OC3DL220000`·`OC3DL220001`이 여기서
# 걸린다: `sn_alias`가 1:1이 아니라 8~10개 센서가 같은 옛 시리얼을 공유해
# 물리적 위치를 특정할 수 없고, 위치 조인이 실패해 공간 피처가 전부 NaN이다.
REQUIRED_FEATURES = ["svf", "elev", "bld_d_min"]

# 폭염 시각 판정 — 그 시각 서울 센서 기온 중위값 기준. 근거 수치가 이 정의로 산출됐다.
HOT_MEDIAN_C = 33.0

# 이웃·참조로 신뢰할 수 있는 최소 관측 수. `build_dataset.py`의 `SENSOR_MIN_OBS`와 같은 값이다.
#
# 그 QC는 관측 200개 미만 센서를 "표본 부족으로 판정하지 않는다"로 통과시킨다.
# 판정 자체는 타당하다 — 관측 8개로 고장을 단정할 수 없다. 그런데 **"판정 안 함"이
# "신뢰 가능"은 아니다.** 그 센서들이 IDW 이웃 풀과 배리오그램에 그대로 들어가 있었다.
#
# 실측: 10개 센서(총 950행 = 전체의 0.03%)가 여기 걸리고, 전부 always-train이다.
# 편차 평균 극단값이 **전부 이들에서 나온다** — 판정 대상(200+)은 -2.86 ~ +1.69℃인데
# 미판정은 -10.59 ~ +0.36℃다. `OC3CL200195`는 관측 8개에 -10.59℃다.
#
# 평가 행 집합은 바뀌지 않는다(전부 always-train). 바뀌는 것은 **참조로 쓰는지**다.
MIN_OBS_TRUST = 200


def trusted_sensors(base: pd.DataFrame) -> np.ndarray:
    n = base.groupby("sn")["tp"].size()
    return np.array(sorted(n.index[n >= MIN_OBS_TRUST]))

# 희소 자치구 — 센서 수 하위 N개. "센서 희소 지역 오차를 별도 보고한다"는 리스크 대응.
SPARSE_GU_N = 5


# ─────────────────────────────────────────────────────────── 기본 로더

def load_splits() -> dict:
    if not SPLITS.exists():
        raise FileNotFoundError(f"{SPLITS} 없음 — 먼저 `uv run python -m core.eval.build_splits`")
    return json.loads(SPLITS.read_text())


def sensor_fold_map(splits: dict | None = None) -> pd.Series:
    sp = splits or load_splits()
    out = {}
    for f in range(N_FOLDS):
        for sn in sp["folds"][str(f)]:
            out[sn] = f
    for sn in sp["always_train"]:
        out[sn] = -1
    return pd.Series(out, dtype="int8", name="fold")


def load_sensor_meta() -> pd.DataFrame:
    loc = pd.read_csv(LOCATIONS).set_index("sn")
    return loc[["x", "y", "gu", "lat", "lon"]]


def load_regional() -> pd.DataFrame:
    reg = pd.read_parquet(REGIONAL)
    return reg.pivot(index="ts", columns="fold", values="t_reg").sort_index()


def load_base(features: Sequence[str] = (), *, verbose: bool = True) -> pd.DataFrame:
    fold = sensor_fold_map()
    cols = ["sn", "ts", "tp", "sensor_ok"] + [c for c in features if c not in ("sn", "ts", "tp")]
    d = pd.read_parquet(TRAIN, columns=list(dict.fromkeys(cols)))
    n0 = len(d)

    d = d[d["sensor_ok"] & d["tp"].notna()].drop(columns="sensor_ok")
    n1 = len(d)

    # splits.json에 없는 센서는 이미 걸러진 것이다 (과천 4 + 조건피처 전무 2)
    d = d[d["sn"].isin(fold.index)]
    d["fold"] = d["sn"].map(fold).astype("int8")

    if verbose:
        print(f"[base] 관측 {n0:,} → QC {n1:,} → 유효센서 {len(d):,}행 · "
              f"센서 {d['sn'].nunique():,} · 시각 {d['ts'].nunique():,}")
        drop = n1 - len(d)
        if drop:
            print(f"       splits.json 외 센서 {drop:,}행 제외 (과천 서울대공원 · 조건피처 전무)")
    return d.reset_index(drop=True)


# ─────────────────────────────────────────────────────────── fold 순회

class Fold:
    __slots__ = ("f", "base", "is_test", "t_reg", "delta")

    def __init__(self, f: int, base: pd.DataFrame, t_reg: np.ndarray):
        self.f = f
        self.base = base
        self.t_reg = t_reg
        self.delta = base["tp"].to_numpy(np.float32) - t_reg
        # always-train(-1)은 어느 fold에서도 test가 아니다
        self.is_test = (base["fold"].to_numpy() == f)

    @property
    def is_train(self) -> np.ndarray:
        return ~self.is_test

    def train(self, cols: Sequence[str] | None = None) -> pd.DataFrame:
        return self._slice(self.is_train, cols)

    def test(self, cols: Sequence[str] | None = None) -> pd.DataFrame:
        return self._slice(self.is_test, cols)

    def _slice(self, mask: np.ndarray, cols: Sequence[str] | None) -> pd.DataFrame:
        out = self.base.loc[mask, list(cols) if cols else self.base.columns].copy()
        out["t_reg"] = self.t_reg[mask]
        out["delta"] = self.delta[mask]
        return out


def iter_folds(base: pd.DataFrame, regional: pd.DataFrame | None = None) -> Iterator[Fold]:
    reg = regional if regional is not None else load_regional()
    ts = pd.DatetimeIndex(base["ts"])
    for f in range(N_FOLDS):
        t_reg = reg[f].reindex(ts).to_numpy(np.float32)
        if np.isnan(t_reg).any():
            n = int(np.isnan(t_reg).sum())
            raise ValueError(f"fold {f}: 광역 기준값이 없는 시각 {n:,}행 — regional.parquet 확인")
        yield Fold(f, base, t_reg)


# ─────────────────────────────────────────────────────────── 슬라이스 정의

def hot_timestamps(base: pd.DataFrame) -> pd.DatetimeIndex:
    med = base.groupby("ts")["tp"].median()
    return pd.DatetimeIndex(med.index[med >= HOT_MEDIAN_C])


def sparse_gu(meta: pd.DataFrame, splits: dict | None = None) -> list[str]:
    fold = sensor_fold_map(splits)
    cnt = meta.loc[meta.index.intersection(fold.index), "gu"].value_counts()
    return sorted(cnt.tail(SPARSE_GU_N).index)


def feature_bins(base: pd.DataFrame, col: str, q: int = 5) -> pd.Series:
    per = base.groupby("sn")[col].first().dropna()
    labels = [f"Q{i+1}" for i in range(q)]
    return pd.qcut(per, q, labels=labels, duplicates="drop")


# ─────────────────────────────────────────────────────────── 예측 저장

def save_preds(name: str, sn: pd.Series, ts: pd.Series, y_pred: np.ndarray) -> Path:
    PREDS.mkdir(parents=True, exist_ok=True)
    out = PREDS / f"{name}.parquet"
    pd.DataFrame({
        "sn": pd.Series(np.asarray(sn), dtype="string"),
        "ts": pd.Series(np.asarray(ts)),
        "y_pred": np.asarray(y_pred, dtype=np.float32),
    }).to_parquet(out, index=False)
    return out
