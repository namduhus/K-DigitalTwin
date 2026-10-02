from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import from_wkt

SRC = Path("data/raw/buildings_use_seoul.parquet")
OUT = Path("data/processing/buildings.parquet")

# 서울 최고층 건물(롯데월드타워 555m)에 여유를 둔 상한. 이 이상은 입력 오류로 본다.
HEIGHT_MAX = 600.0

# 층고 학습 표본에서 받아들일 범위(m/층). 밖은 높이 또는 층수 입력 오류다.
FLOOR_H_RANGE = (2.0, 6.0)

# 폴리곤 면적 하한(㎡). 이보다 작으면 형상 오류로 보고 음영 계산에서 제외한다.
AREA_MIN = 0.5

FALLBACK_FLOOR_H = 3.35  # 용도분류를 모를 때 쓸 전체 중위값


def main() -> int:
    if not SRC.exists():
        sys.exit(f"[error] {SRC} 없음 — 먼저 fetch_buildings.py 실행")

    df = pd.read_parquet(SRC)
    print(f"원본 {len(df):,}동")

    # ── 지오메트리 ────────────────────────────────────────────────────
    geom = from_wkt(df["wkt"].to_numpy())
    df["poly_ar"] = [p.area if p is not None else np.nan for p in geom]
    df["cx"] = [p.centroid.x if p is not None else np.nan for p in geom]
    df["cy"] = [p.centroid.y if p is not None else np.nan for p in geom]
    valid = np.array([p is not None and p.is_valid for p in geom])
    df["geom_ok"] = valid & (df["poly_ar"] >= AREA_MIN).to_numpy()
    print(f"지오메트리 유효 {df['geom_ok'].sum():,}동 "
          f"(무효·{AREA_MIN}㎡ 미만 {(~df['geom_ok']).sum():,}동 제외 대상)")

    # ── 높이 이상치 제거 ──────────────────────────────────────────────
    h = pd.to_numeric(df["height"], errors="coerce")
    gf = pd.to_numeric(df["gf"], errors="coerce")
    bad_h = h > HEIGHT_MAX
    if bad_h.any():
        print(f"높이 {HEIGHT_MAX:.0f}m 초과 {bad_h.sum()}동 무효화 "
              f"(최대 {h.max():,.0f}m) — 층수 기반 추정으로 넘긴다")
    h = h.where(~bad_h & (h > 0))

    # ── 용도분류별 층고 학습 ──────────────────────────────────────────
    # 층고가 비현실적인 표본은 높이나 층수가 잘못 들어간 것이므로 학습에서 뺀다.
    train = pd.DataFrame({"use_class": df["use_class"], "per": h / gf}).dropna()
    lo, hi = FLOOR_H_RANGE
    train = train[train["per"].between(lo, hi)]
    floor_h = train.groupby("use_class")["per"].median()
    n_per_use = train.groupby("use_class")["per"].size()

    print(f"\n층고 학습 표본 {len(train):,}동 (층당 {lo}~{hi}m 범위만)")
    for uc in n_per_use.sort_values(ascending=False).index[:8]:
        print(f"  {str(uc):<12} {n_per_use[uc]:>7,}동 · {floor_h[uc]:.2f} m/층")

    # ── 높이 완성 ─────────────────────────────────────────────────────
    per_use = df["use_class"].map(floor_h).fillna(FALLBACK_FLOOR_H)
    # 층수가 0이거나 결측이면 추정할 수 없다. `gf * per_use`를 그대로 쓰면
    # gf=0에서 높이 0m이 나오고 그게 "추정됨"으로 잘못 분류된다.
    est = (gf.where(gf > 0) * per_use)

    df["height_m"] = h.where(h.notna(), est)
    df["height_src"] = np.select(
        [h.notna(), est.notna()], ["measured", "estimated"], default="unknown"
    )

    # ── 건축면적 완성 ─────────────────────────────────────────────────
    # bldg_ar이 0/결측인 경우 폴리곤 면적으로 대체한다.
    # 둘 다 있는 표본에서 폴리곤/건축면적 비가 중위 1.04로 거의 1:1임을 확인했다.
    ba = pd.to_numeric(df["bldg_ar"], errors="coerce")
    ba = ba.where(ba > 0)
    df["bldg_ar_m2"] = ba.where(ba.notna(), df["poly_ar"])
    df["bldg_ar_src"] = np.where(ba.notna(), "measured", "polygon")

    # ── 리포트 ────────────────────────────────────────────────────────
    print(f"\n{'높이 출처':<12} {'건물수':>9} {'비율':>7} {'중위 높이':>9}")
    print("-" * 42)
    for src in ["measured", "estimated", "unknown"]:
        m = df["height_src"] == src
        med = f"{df.loc[m, 'height_m'].median():.1f} m" if m.any() and src != "unknown" else "-"
        print(f"{src:<12} {m.sum():>9,} {m.mean():>6.1%} {med:>9}")

    # 추정 높이의 타당성 — 층수 분포와 맞아야 한다
    e = df[df["height_src"] == "estimated"]
    if len(e):
        print(f"\n추정 높이 검산: 층수 중위 {e['gf'].median():.0f}층 × 층고 → 높이 중위 "
              f"{e['height_m'].median():.1f} m (실측군 층수 중위 "
              f"{df.loc[df.height_src=='measured','gf'].median():.0f}층)")

    df["shadow_ok"] = df["height_m"].notna() & (df["height_m"] > 0) & df["geom_ok"]
    ok = df["shadow_ok"]
    print(f"\n음영 계산 가능 {ok.sum():,}동 ({ok.mean():.2%})")
    print(f"  높이 범위 {df.loc[ok,'height_m'].min():.1f} ~ {df.loc[ok,'height_m'].max():.1f} m "
          f"· 중위 {df.loc[ok,'height_m'].median():.1f} m")
    print(f"  제외 {(~ok).sum():,}동 — 지오메트리 불량 {(~df['geom_ok']).sum():,} / "
          f"높이 미확정 {(df['height_src']=='unknown').sum():,}")
    print(f"건축면적 출처: {df['bldg_ar_src'].value_counts().to_dict()}")

    keep = ["gid", "pnu", "use_class", "use_main", "struct", "approved",
            "gf", "uf", "height_m", "height_src",
            "bldg_ar_m2", "bldg_ar_src", "poly_ar", "totar", "far", "bcr",
            "cx", "cy", "geom_ok", "shadow_ok", "wkt"]
    out = df[[c for c in keep if c in df.columns]]
    out.to_parquet(OUT, index=False)
    print(f"\n→ {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
