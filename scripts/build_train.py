from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

OBS = Path("data/processing/sdot_summer.parquet")
# 건물+지형 합산본을 우선 쓴다. `sensor_horizon`은 평지 가정이라
# 표고·경사·지형차폐가 빠지는데, 검증에서 표고가 가장 강한 예측자 중 하나였다
# (센서 단위 다변량 표준화 계수 주간 -0.187 / 야간 -0.277).
FEATURES = Path("data/processing/sensor_features.parquet")
HZ_FALLBACK = Path("data/processing/sensor_horizon.parquet")
GREEN = Path("data/processing/sensor_green.parquet")   # 있으면 녹지 피처를 함께 붙인다
OUT = Path("data/processing/train.parquet")

# 서울 중심 (시청). 태양 위치 계산 기준점.
LAT, LON, TZ = 37.5665, 126.9780, "Asia/Seoul"
AZ_STEP = 2.0
N_AZ = 180


def main() -> int:
    if not OBS.exists():
        sys.exit(f"[error] {OBS} 없음")
    src = FEATURES if FEATURES.exists() else HZ_FALLBACK
    if not src.exists():
        sys.exit(f"[error] {FEATURES} / {HZ_FALLBACK} 둘 다 없음")
    if src is HZ_FALLBACK:
        print(f"[warn] {FEATURES} 없음 → {HZ_FALLBACK} 사용 (평지 가정, 표고·지형 피처 없음)")

    obs = pd.read_parquet(OBS)
    hz = pd.read_parquet(src)
    kind = "건물+지형 합산" if src is FEATURES else "건물만"
    print(f"관측 {len(obs):,}행 · 수평선 {len(hz):,}센서 ({kind})")

    # ── 태양 위치 ─────────────────────────────────────────────────────
    # 관측 시각은 KST 지역시각(naive)이다. pvlib에 넘기려면 tz를 명시해야 한다.
    uniq = pd.DatetimeIndex(sorted(obs["ts"].unique())).tz_localize(TZ, nonexistent="shift_forward")
    sp = pvlib.solarposition.spa_python(uniq, LAT, LON)
    sun = pd.DataFrame({
        "ts": uniq.tz_localize(None),
        "sun_el": sp["apparent_elevation"].to_numpy(),  # 대기굴절 보정된 겉보기 고도
        "sun_az": sp["azimuth"].to_numpy(),
    })
    print(f"태양 위치 계산 {len(sun):,}시각 (NREL SPA, 서울 중심 기준)")
    print(f"  고도 {sun.sun_el.min():.1f} ~ {sun.sun_el.max():.1f}° · "
          f"주간(고도>0) {(sun.sun_el > 0).mean():.1%}")

    df = obs.merge(sun, on="ts", how="left")

    # ── 음영 판정 ─────────────────────────────────────────────────────
    hz_cols = [f"hz_{i:03d}" for i in range(N_AZ)]
    hz_mat = hz[hz_cols].to_numpy(np.float32)          # (센서, 방위)
    sn_pos = {s: i for i, s in enumerate(hz["sn"])}

    row = df["sn"].map(sn_pos)
    ok = row.notna()
    if (~ok).any():
        print(f"[warn] 수평선 없는 센서의 관측 {(~ok).sum():,}행 — 음영 결측 처리")

    ri = row.fillna(0).astype(np.int32).to_numpy()
    ai = (df["sun_az"].to_numpy() / AZ_STEP).astype(np.int32) % N_AZ
    obstacle = hz_mat[ri, ai]
    obstacle[~ok.to_numpy()] = np.nan

    df["hz_sun"] = obstacle                                   # 태양 방향 장애물 앙각
    df["shadow_margin"] = df["sun_el"] - df["hz_sun"]

    # 해가 지평선 아래면 음영 개념이 없다. 야간을 별도 플래그로 두고
    # 음영 피처는 결측으로 만든다 — 0으로 채우면 "양지"로 오해된다.
    df["is_night"] = df["sun_el"] <= 0
    df.loc[df["is_night"], "shadow_margin"] = np.nan
    # nullable boolean이어야 야간 결측을 담을 수 있다 (순수 bool은 NA 불가)
    df["is_shadow"] = (df["shadow_margin"] < 0).astype("boolean")
    df.loc[df["is_night"] | df["shadow_margin"].isna(), "is_shadow"] = pd.NA

    # ── 정적 공간 피처 조인 ───────────────────────────────────────────
    static = [
        # 천공률 — 주야 부호가 반대다 (주간 +0.067 / 야간 -0.137, 둘 다 p<0.001)
        "svf", "svf_bld", "svf_terr",
        "hz_mean", "hz_max",
        # 건물 — 최근접 거리가 대로변의 "SVF 낮은데 시원함"을 설명한다
        "bld_n", "bld_d_min", "bld_h_max", "bld_h_mean", "bld_ar_frac",
        # 지형 — 표고가 가장 강한 예측자 중 하나. terr_dominant는 산지 근접 대리변수
        "elev", "slope", "aspect", "th_mean", "th_max", "terr_dominant",
    ]
    have = [c for c in static if c in hz.columns]
    if missing := [c for c in static if c not in hz.columns]:
        print(f"[warn] 정적 피처 누락: {', '.join(missing)}")
    df = df.merge(hz[["sn"] + have], on="sn", how="left")

    # 녹지 — 야간 기온에 유의하고(t=-3.79), `pave_frac`은 주야 모두 유의하다.
    # R² 기여는 작지만(+0.007 주간 / +0.013 야간) 「활용분야」의 그린인프라 논리와
    # 개입 반사실(처방) 확장에 필수다.
    if GREEN.exists():
        gr = pd.read_parquet(GREEN)
        gcols = [c for c in gr.columns if c != "sn"]
        df = df.merge(gr, on="sn", how="left")
        have += gcols
        print(f"녹지 피처 {len(gcols)}개 조인: {', '.join(gcols)}")
    else:
        print(f"[warn] {GREEN} 없음 — 녹지 피처 제외")

    # ── 시간 피처 ─────────────────────────────────────────────────────
    df["hour"] = df["ts"].dt.hour
    df["month"] = df["ts"].dt.month
    df["doy"] = df["ts"].dt.dayofyear
    df["year"] = df["ts"].dt.year

    # ── 리포트 ────────────────────────────────────────────────────────
    day = df[~df["is_night"]]
    print(f"\n주간 관측 {len(day):,}행 ({len(day)/len(df):.1%})")
    sh = day["is_shadow"].astype("boolean")
    print(f"  그늘 판정 {sh.sum():,}행 ({sh.mean():.1%})")
    print(f"  shadow_margin 중위 {day['shadow_margin'].median():.1f}° · "
          f"1%분위 {day['shadow_margin'].quantile(.01):.1f}° · "
          f"99%분위 {day['shadow_margin'].quantile(.99):.1f}°")

    print(f"\n시각별 그늘 비율:")
    byh = day.groupby("hour")["is_shadow"].mean()
    for h in sorted(byh.index):
        bar = "█" * int(byh[h] * 40)
        print(f"  {h:>2}시 {byh[h]:>6.1%} {bar}")

    print(f"\n정적 공간 피처 {len(have)}개 조인:")
    for c in have:
        s = df[c]
        print(f"  {c:<14} 중위 {s.median():>9.3f} · 범위 {s.min():>9.3f} ~ {s.max():>9.3f} "
              f"· 결측 {s.isna().mean():.2%}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\n→ {OUT} ({OUT.stat().st_size / 1e6:.0f} MB, {len(df):,}행 × {len(df.columns)}컬럼)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
