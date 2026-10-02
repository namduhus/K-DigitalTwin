from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import sdot_io  # noqa: E402

RAW = Path("data/raw")
LOC = Path("data/processing/sdot_locations.csv")
OUT = Path("data/processing/sdot_summer.parquet")

DAILY_RE = re.compile(r"PUBDATA_(\d{4})(\d{2})(\d{2})\.csv$")
WEEKLY_RE = re.compile(r"S-DoT_NATURE_(\d{4})\.(\d{2})\.(\d{2})-(\d{2})\.(\d{2})\.csv$")


def discover(months: set[int]) -> tuple[list[Path], list[Path]]:
    csvs: list[Path] = []
    for p in sorted(RAW.rglob("*.csv")):
        if m := DAILY_RE.search(p.name):
            if int(m.group(2)) in months:
                csvs.append(p)
        elif m := WEEKLY_RE.search(p.name):
            if int(m.group(2)) in months or int(m.group(4)) in months:
                csvs.append(p)
    apis = sorted(RAW.glob("sdot_*.parquet"))
    return csvs, apis


# 센서 결함 센티널. 정확히 이 값으로 13,921건이 들어 있다 (미측정 표시).
SENTINELS = (-40.0,)

# 여름철 서울에서 물리적으로 가능한 범위. 밖의 값은 센서 오류로 보고 NaN 처리한다.
# 상한 45℃는 여유를 둔 값이다 — 센서가 지표 근처 복사열을 받아 기상관측값보다 높게 나올 수 있다.
QC_RANGE = {"tp": (5.0, 45.0), "hum": (5.0, 100.0), "illu": (0.0, 2e5), "uv": (0.0, 20.0)}

# 센서 단위 이상 판정 기준.
#
# 값 단위 QC(센티널·물리범위)를 통과해도 **특정 센서가 지속적으로 틀린** 경우가 남는다.
# 판단 근거는 비대칭이다:
#   - 더운 쪽은 평균 편차 최대 +1.7℃, 자기 표준편차 0.7~1.3 → 안정적이고 물리적으로 설명된다
#   - 시원한 쪽은 -10.3℃까지 가고 자기 표준편차가 6.97까지 간다 → 설명 불가
# 지역 유형 중 가장 시원한 '공원'의 평균 편차가 -1.36℃이므로,
# 시각별 중위 대비 평균 -3℃ 이하는 어떤 냉각 현상으로도 설명할 수 없다.

SENSOR_MIN_ANOM = -3.0  # 시각별 중위 기온 대비 평균 편차 하한
SENSOR_MAX_STD = 4.0  # 센서 자기 편차의 표준편차 상한
SENSOR_MIN_OBS = 200  # 이 미만이면 표본 부족으로 판정하지 않는다


def apply_qc(df: pd.DataFrame) -> pd.DataFrame:
    for col, (lo, hi) in QC_RANGE.items():
        if col not in df:
            continue
        bad = df[col].isin(SENTINELS) | ~df[col].between(lo, hi)
        df.loc[bad & df[col].notna(), col] = np.nan
    return df


# NWS 열지수 표의 최대값(137℉ ≈ 58.3℃). 표는 T 80~110℉ / RH 40~100%에서 정의되지만 T=110℉ & RH=100% 같은 조합은 자연에 없어 표에 없다. 
# Rothfusz 식을 그 영역으로 외삽하면 발산한다 — 실제로 기온 42.6℃ + 습도 100%에서 열지수 130℃가 나왔다.

HI_MAX = 58.3


def flag_bad_sensors(df: pd.DataFrame) -> set[str]:
    d = df.dropna(subset=["tp"])
    anom = d["tp"] - d.groupby("ts")["tp"].transform("median")
    per = anom.groupby(d["sn"]).agg(["count", "mean", "std"])
    per = per[per["count"] >= SENSOR_MIN_OBS]
    bad = per[(per["mean"] < SENSOR_MIN_ANOM) | (per["std"] > SENSOR_MAX_STD)]
    return set(bad.index)


def heat_index(tp: pd.Series, hum: pd.Series) -> tuple[pd.Series, int]:
    t = tp * 9 / 5 + 32  # ℉
    r = hum
    rothfusz = (
        -42.379 + 2.04901523 * t + 10.14333127 * r
        - 0.22475541 * t * r - 6.83783e-3 * t**2 - 5.481717e-2 * r**2
        + 1.22874e-3 * t**2 * r + 8.5282e-4 * t * r**2 - 1.99e-6 * t**2 * r**2
    )
    valid = (t >= 80) & (r >= 40)
    hi = pd.Series((np.where(valid, rothfusz, t) - 32) * 5 / 9, index=tp.index)
    clipped = int((hi > HI_MAX).sum())
    return hi.clip(upper=HI_MAX), clipped


def coord_map(loc: pd.DataFrame) -> dict[str, int]:
    m = dict(zip(loc["sn"], loc.index))
    m.update(
        {a: i for a, i in zip(loc["sn_alias"], loc.index) if isinstance(a, str) and a.strip()}
    )
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, nargs="*", default=[6, 7, 8, 9])
    ap.add_argument("-n", "--dry-run", action="store_true")
    a = ap.parse_args()
    months = set(a.months)

    if not LOC.exists():
        sys.exit(f"[error] {LOC} 없음 — 먼저 prep_locations.py 실행")

    csvs, apis = discover(months)
    if not csvs and not apis:
        sys.exit(f"[error] 월 {sorted(months)}에 해당하는 데이터가 {RAW}에 없다")

    files = csvs + apis
    size = sum(p.stat().st_size for p in files) / 1e9
    print(f"대상 CSV {len(csvs)}개 + API parquet {len(apis)}개 · {size:.2f} GB · 월 {sorted(months)}")
    for p in apis:
        print(f"  (API) {p.name}")
    if a.dry_run:
        for p in csvs[:3]:
            print(f"  {p.relative_to(RAW)}")
        if len(csvs) > 3:
            print(f"  ... 외 {len(csvs) - 3}개")
        return 0

    loc = pd.read_csv(LOC)
    cmap = coord_map(loc)

    frames, unmatched = [], set()
    for i, p in enumerate(files, 1):
        df = sdot_io.read_api(p) if p.suffix == ".parquet" else sdot_io.read(p)
        idx = df["sn"].map(cmap)
        unmatched |= set(df.loc[idx.isna(), "sn"].unique())
        df = df[idx.notna()].copy()
        j = loc.loc[idx.dropna().astype(int).values]
        df["lat"] = j["lat"].to_numpy()
        df["lon"] = j["lon"].to_numpy()
        df["gu_addr"] = j["gu"].to_numpy()  # 원본 자치구 필드는 비정상 값이 섞여 있어 주소 기준을 쓴다
        df["src"] = "api" if p.suffix == ".parquet" else "file"
        frames.append(df)
        print(f"  [{i}/{len(files)}] {p.name[:52]:<52} {len(df):>8,}행", end="\r")

    print()
    out = pd.concat(frames, ignore_index=True)
    out = out[out["ts"].dt.month.isin(months)]  # 주간 파일이 경계를 넘는 부분 정리

    # 파일과 API가 겹치는 구간을 정리한다. 교차 검증에서 두 경로의 값이 100% 일치함을 확인했으므로 어느 쪽을 남겨도 같다. 파일을 원본으로 우선한다.
    dup = out.duplicated(subset=["sn", "ts"], keep=False).sum()
    # "file"이 먼저 오도록 명시한다. 문자열 정렬이면 "api" < "file"이라 반대가 된다.
    out["src"] = pd.Categorical(out["src"], categories=["file", "api"], ordered=True)
    out = out.sort_values("src").drop_duplicates(subset=["sn", "ts"], keep="first")
    print(f"중복 {dup:,}건 발견 → {len(out):,}행으로 정리 (파일 우선)")

    before = {c: out[c].notna().sum() for c in ["tp", "hum", "illu", "uv"]}
    out = apply_qc(out)

    # 습도 포화는 제거하지 않고 표시만 한다. 열대야·안개의 실제 100%와 센서 포화가 섞여 있어 일괄 제거는 부당하다 (전 기온대에 균일하게 2.8%).
    out["hum_sat"] = out["hum"].eq(100)

    out["hi"], hi_clipped = heat_index(out["tp"], out["hum"])

    # 센서 단위 이상 판정. 행을 버리지 않고 플래그만 남겨 downstream이 선택하게 한다.
    bad_sn = flag_bad_sensors(out)
    out["sensor_ok"] = ~out["sn"].isin(bad_sn)

    if unmatched:
        print(f"[warn] 좌표 미매칭 센서 {len(unmatched)}개 제외: {sorted(unmatched)[:5]}")

    print(f"\n{len(out):,}행 · 센서 {out['sn'].nunique()}개 · 자치구 {out['gu_addr'].nunique()}개")
    print(f"기간 {out['ts'].min()} ~ {out['ts'].max()}")

    print(f"\n{'항목':<6} {'결측률':>8} {'QC제거':>9}  범위")
    for c in ["tp", "hum", "hi", "illu", "uv", "gt"]:
        s = out[c]
        qc = f"{before[c] - s.notna().sum():>9,}" if c in before else " " * 9
        rng = f"{s.min():.1f} ~ {s.max():.1f}" if s.notna().any() else "-"
        print(f"{c:<6} {1 - s.notna().mean():>7.1%} {qc}  {rng}")

    print(
        f"\n열지수 클립({HI_MAX}℃ 상한, NWS 표 최대값): {hi_clipped:,}건 "
        f"— 주 타깃은 기온이고 열지수는 보조 지표다"
    )
    print(f"습도 포화(100%) 표시: {out['hum_sat'].sum():,}건 (제거하지 않음)")

    # 센서별 QC 불량률 — 결함 센서를 학습에서 가중 조정하거나 제외할 때 쓴다
    bad = out.groupby("sn")["tp"].apply(lambda s: s.isna().mean()).sort_values(ascending=False)
    print(f"기온 결측률 50% 초과 센서: {(bad > 0.5).sum()}개 / {len(bad)}개")

    print(
        f"센서 단위 이상 판정: {len(bad_sn)}개 제외 대상 "
        f"(평균편차 < {SENSOR_MIN_ANOM}℃ 또는 자기 표준편차 > {SENSOR_MAX_STD}℃) "
        f"— 관측 {out['sensor_ok'].mean():.2%} 유지"
    )

    # ── 다운스케일링의 근거가 되는 핵심 수치 ──────────────────────────────
    # 최대-최소는 이상값 하나에 좌우되므로 분위 기반을 주 지표로 쓴다.
    # 신청서 배경 및 필요성에 인용할 값이라 견고해야 한다.
    ok = out[out["sensor_ok"]].dropna(subset=["tp"])

    def spread(d: pd.DataFrame) -> pd.DataFrame:
        p = d.groupby("ts")["tp"].agg(
            count="count", std="std", lo="min", hi="max",
            p01=lambda s: s.quantile(0.01), p05=lambda s: s.quantile(0.05),
            p95=lambda s: s.quantile(0.95), p99=lambda s: s.quantile(0.99),
        )
        return p[p["count"] > 500]

    piv = spread(ok)
    r95, r99 = piv["p95"] - piv["p05"], piv["p99"] - piv["p01"]
    print(f"\n[핵심] 동일 시각 서울 내 기온 편차 (시각 {len(piv):,}개, 각 500개 이상 관측)")
    print(
        f"  P95-P05  평균 {r95.mean():.2f}℃ · 중위 {r95.median():.2f}℃ · "
        f"상위 5% 시각 {r95.quantile(0.95):.2f}℃"
    )
    print(f"  P99-P01  평균 {r99.mean():.2f}℃ · 상위 5% 시각 {r99.quantile(0.95):.2f}℃")
    print(f"  최대-최소 평균 {(piv['hi'] - piv['lo']).mean():.2f}℃ (이상값에 취약 — 인용하지 않는다)")
    print(f"  시각별 표준편차 평균 {piv['std'].mean():.2f}℃")

    # 폭염 시각에서 편차가 커지는지 — 제안의 핵심 논리
    hot_ts = ok.groupby("ts")["tp"].median()
    hot = spread(ok[ok["ts"].isin(hot_ts[hot_ts >= 33].index)])
    if len(hot):
        h = hot["p95"] - hot["p05"]
        print(
            f"  폭염 시각만 (서울 중위 33℃ 이상, {len(hot):,}시각): "
            f"P95-P05 평균 {h.mean():.2f}℃ · 최대 {h.max():.2f}℃"
        )

    # 위치가 편차를 얼마나 설명하는가 = 공간 정보로 예측 가능한 몫, 모델 성능 상한 시사
    anom = ok["tp"] - ok.groupby("ts")["tp"].transform("median")
    fixed = anom.groupby(ok["sn"]).transform("mean")
    print(f"  센서 고정효과 설명력 {fixed.var() / anom.var():.1%} (공간 정보로 예측 가능한 몫)")

    # 지역 유형별 체계적 차이 — 신호가 실재한다는 증거
    rg = anom.groupby(ok["rgn"]).agg(["size", "mean"])
    rg = rg[rg["size"] >= 5000].sort_values("mean", ascending=False)
    print("\n  지역 유형별 평균 편차:")
    for name, row in rg.iterrows():
        print(f"    {str(name):<22} {row['mean']:+.2f}℃")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"\n→ {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
