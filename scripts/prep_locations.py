import glob
import sys

import pandas as pd
from pyproj import Transformer

SRC_GLOB = "data/*설치 위치정보*.xlsx"
OUT = "data/processing/sdot_locations.csv"

# 서울 행정구역 경계 대략값. 이 범위를 벗어나면 좌표 오류로 본다.
SEOUL_BBOX = {"lat": (37.41, 37.72), "lon": (126.73, 127.20)}

# 원본 위치정보의 명백한 입력 오류를 고친다.
# OC3KL240004: 주소는 은평구 녹번동인데 위도가 37.06195로 적혀 있다(서울 최남단이 약 37.42).
# 37.60195의 자릿수 오타로 보이며, 보정하면 은평구 녹번동 실제 위치와 맞는다.
# 버리면 관측 1개 센서를 잃으므로 고쳐서 쓴다.
LAT_LON_FIX = {"OC3KL240004": (37.60195, 126.921107)}


def main() -> int:
    paths = glob.glob(SRC_GLOB)
    if not paths:
        print(f"[error] {SRC_GLOB} 없음", file=sys.stderr)
        return 1

    df = pd.read_excel(paths[0], header=0)
    df.columns = [str(c).strip() for c in df.columns]

    out = pd.DataFrame(
        {
            "sn": df["모델 시리얼(*)"].astype(str).str.strip(),
            "sn_alias": df["변경 전 시리얼(데이터 상 표기)"].astype("string").str.strip(),
            "address": df["주소"].astype(str).str.strip(),
            "crs_code": df["좌표 구분코드"].astype(str).str.strip(),
            "lat": pd.to_numeric(df["위도"], errors="coerce"),
            "lon": pd.to_numeric(df["경도"], errors="coerce"),
        }
    )
    # `서울특별시`로 고정 매칭하면 안 된다. 원본 xlsx에 오타가 섞여 있다 —
    # 지번 주소 계열 110건이 `서울측별시`(특→측)이고 `서울측별시울`(울 중복)도 있다.
    # 고정 매칭 시 110개 센서의 자치구가 조용히 결측되고, 그 상태로 층화 분할을
    # 하면 groupby가 해당 센서를 말없이 버린다. `서울` 뒤는 무엇이든 허용한다.
    out["gu"] = out["address"].str.extract(r"^서울\S*\s+(\S+구)\s")[0]

    # 결측을 절대 조용히 넘기지 않는다. 위 오타가 발견된 경위가 정확히 "조용히 넘어감"이었다.
    unmatched = out[out["gu"].isna()]
    if len(unmatched):
        print(f"\n[warn] 주소에서 자치구를 못 뽑은 센서 {len(unmatched)}건 — 확인 필요", file=sys.stderr)
        for sn, ad in unmatched[["sn", "address"]].itertuples(index=False):
            print(f"  {sn}  {ad!r}", file=sys.stderr)

    # 좌표계 확인 — W84(WGS84)가 아니면 변환이 필요하다
    crs = out["crs_code"].value_counts().to_dict()
    if set(crs) != {"W84"}:
        print(f"[warn] 좌표 구분코드가 W84 단일이 아님: {crs}", file=sys.stderr)

    # 알려진 입력 오류 보정
    for sn, (lat, lon) in LAT_LON_FIX.items():
        hit = out["sn"] == sn
        if hit.any():
            before = out.loc[hit, ["lat", "lon"]].iloc[0].tolist()
            out.loc[hit, ["lat", "lon"]] = [lat, lon]
            print(f"[fix] {sn} 좌표 보정 {before} → [{lat}, {lon}]")

    # 서울 경계 밖 좌표 = 오류로 판정하고 분리
    in_bbox = (
        out["lat"].between(*SEOUL_BBOX["lat"])
        & out["lon"].between(*SEOUL_BBOX["lon"])
    )
    bad = out[~in_bbox]
    clean = out[in_bbox].copy()

    print(f"전체 {len(out)}개 · 고유 시리얼 {out['sn'].nunique()}개")
    print(f"자치구 {clean['gu'].nunique()}개 · 시리얼 변경 이력 {out['sn_alias'].notna().sum()}건")

    if len(bad):
        print(f"\n[warn] 서울 경계 밖 좌표 {len(bad)}건 — 제외함")
        print(bad[["sn", "address", "lat", "lon"]].to_string(index=False))

    # WGS84 → EPSG:5186 (GRS80 TM중부, 미터). 건물 데이터와 같은 좌표계로 맞춘다.
    tr = Transformer.from_crs("EPSG:4326", "EPSG:5186", always_xy=True)
    clean["x"], clean["y"] = tr.transform(clean["lon"].to_numpy(), clean["lat"].to_numpy())
    print(f"\n좌표 변환 EPSG:4326 → EPSG:5186 (미터)")
    print(f"  x {clean['x'].min():.0f} ~ {clean['x'].max():.0f}  "
          f"(폭 {(clean['x'].max()-clean['x'].min())/1000:.1f} km)")
    print(f"  y {clean['y'].min():.0f} ~ {clean['y'].max():.0f}  "
          f"(높이 {(clean['y'].max()-clean['y'].min())/1000:.1f} km)")

    # 역변환으로 왕복 오차를 확인한다. 좌표계를 잘못 지정하면 여기서 드러난다.
    back = Transformer.from_crs("EPSG:5186", "EPSG:4326", always_xy=True)
    lon2, lat2 = back.transform(clean["x"].to_numpy(), clean["y"].to_numpy())
    err_m = (((lat2 - clean["lat"]) ** 2 + (lon2 - clean["lon"]) ** 2) ** 0.5) * 111_000
    print(f"  왕복 오차 최대 {err_m.max():.4f} m (0에 가까워야 정상)")

    # 자치구별 센서 밀도 — 학습/검증 분할 설계에 필요
    counts = clean["gu"].value_counts()
    print(
        f"\n자치구별 센서 수: 최다 {counts.idxmax()} {counts.max()}개 / "
        f"최소 {counts.idxmin()} {counts.min()}개 (약 {counts.max() / counts.min():.1f}배 편차)"
    )

    clean.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n→ {OUT} ({len(clean)}행)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
