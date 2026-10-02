from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import dotenv_values

OUT = Path("data/raw")
BASE = "https://api.vworld.kr/ned/wfs"
MAX_FEATURES = 1000  # API 하드 상한
MIN_TILE = 100.0  # 이보다 작게는 쪼개지 않는다 (m). 밀집지에서 무한 분할 방지

# 서울 행정구역 외곽 bbox — EPSG:5186 (미터)
SEOUL_BBOX = (176_000.0, 534_500.0, 217_700.0, 569_000.0)

# 데이터셋. 높이 채움률은 둘 다 약 55%로 비슷하고, 용도별건물이 용도 분류까지 주므로 기본값.
DATASETS = {
    "use": ("getBuildingUseWFS", "dt_d198"),   # 용도별건물정보
    "age": ("getBuildingAgeWFS", "dt_d196"),   # 건축물연령정보
}

# GML 속성 → 짧은 이름
FIELDS = {
    "gis_idntfc_no": "gid",
    "pnu": "pnu",
    "buld_idntfc_no": "bld_id",
    "buld_totar": "totar",
    "buld_bildng_ar": "bldg_ar",
    "buld_plot_ar": "plot_ar",
    "measrmt_rt": "far",           # 용적률
    "btl_rt": "bcr",               # 건폐율
    "strct_code_nm": "struct",
    "main_prpos_code_nm": "use_main",
    "detail_prpos_code_nm": "use_detail",
    "buld_prpos_cl_code_nm": "use_class",
    "buld_hg": "height",           # 건물높이(m) — 약 55%만 채워짐
    "ground_floor_co": "gf",       # 지상층수 — 약 99.9% 채워짐
    "undgrnd_floor_co": "uf",      # 지하층수
    "use_confm_de": "approved",
    "buld_age": "age",             # age 데이터셋에만 있음
}

COORD_RE = re.compile(r"<gml:coordinates[^>]*>([^<]+)</gml:coordinates>")


def key() -> str:
    for k, v in dotenv_values(".env").items():
        if "WORLD" in k.upper() and v and v.strip():
            return v.strip()
    sys.exit("[error] .env에서 브이월드 인증키를 찾을 수 없다")


def request_tile(sess: requests.Session, k: str, op: str, typename: str,
                 box: tuple[float, float, float, float],
                 tries: int = 5) -> str | None:
    params = {
        "key": k, "domain": "localhost", "typename": typename,
        "bbox": ",".join(f"{v:.1f}" for v in box),
        "maxFeatures": str(MAX_FEATURES),
        "resultType": "results", "srsName": "EPSG:5186",
    }
    for attempt in range(tries):
        try:
            r = sess.get(f"{BASE}/{op}", params=params, timeout=120)
            r.encoding = "utf-8"  # 서버가 charset을 안 주면 한글이 깨진다
            if r.status_code == 200 and "FeatureCollection" in r.text:
                return r.text
        except requests.RequestException:
            pass
        if attempt < tries - 1:
            time.sleep(min(2 ** attempt * 2, 60))  # 2·4·8·16초 — 긴 끊김도 견딘다
    return None


def parse(xml: str, typename: str) -> list[dict]:
    rows = []
    for chunk in xml.split("<gml:featureMember>")[1:]:
        rec: dict = {}
        for src, dst in FIELDS.items():
            m = re.search(rf"<sop:{src}>([^<]*)</sop:{src}>", chunk)
            if m:
                rec[dst] = m.group(1)
        rings = COORD_RE.findall(chunk)
        if not rec.get("gid") or not rings:
            continue
        # 외곽링 하나만 쓴다. 내부 홀은 음영 계산에 영향이 미미하고 파싱을 단순하게 유지한다.
        rec["wkt"] = "POLYGON((" + ", ".join(
            p.replace(",", " ") for p in rings[0].split()
        ) + "))"
        rec["n_rings"] = len(rings)
        rows.append(rec)
    return rows


def grid(box: tuple[float, float, float, float], step: float) -> list[tuple[float, float, float, float]]:
    x1, y1, x2, y2 = box
    tiles = []
    y = y1
    while y < y2:
        x = x1
        while x < x2:
            tiles.append((x, y, min(x + step, x2), min(y + step, y2)))
            x += step
        y += step
    return tiles


def tkey(b: tuple[float, float, float, float]) -> str:
    return ",".join(f"{v:.0f}" for v in b)


def collect(k: str, op: str, typename: str,
            box: tuple[float, float, float, float], step: float,
            ckpt: Path) -> tuple[list[dict], list[str]]:
    sess = requests.Session()
    out: list[dict] = []
    done: set[str] = set()
    failed: list[str] = []

    # 이어받기
    pending: list[str] = []
    if ckpt.exists():
        state = json.loads(ckpt.read_text())
        done = set(state["done"])
        failed = list(state.get("failed", []))
        pending = state.get("pending", [])
        part = ckpt.with_suffix(".part.parquet")
        if part.exists():
            out = pd.read_parquet(part).to_dict("records")
        print(f"[resume] 완료 타일 {len(done):,}개 · 수집 {len(out):,}건 이어받음", file=sys.stderr)

    # 격자 중 미완료 + 이전 실행에서 분할되어 대기 중이던 타일 + 실패 타일 재시도
    seen = done | set(pending)
    stack = [b for b in grid(box, step) if tkey(b) not in seen]
    stack += [tuple(float(v) for v in s.split(",")) for s in pending + failed]
    failed = []  # 재시도하므로 초기화
    print(f"초기 타일 {len(stack):,}개 ({step:.0f}m 격자)", file=sys.stderr)

    tiles = truncated = empty = 0

    def save() -> None:
        pd.DataFrame(out).to_parquet(ckpt.with_suffix(".part.parquet"), index=False)
        ckpt.write_text(json.dumps({
            "done": sorted(done), "failed": failed,
            "pending": [tkey(b) for b in stack],
        }))

    while stack:
        b = stack.pop()
        xml = request_tile(sess, k, op, typename, b)
        tiles += 1
        w, h = b[2] - b[0], b[3] - b[1]

        if xml is None:
            # 요청 실패. 빈 타일로 오해하지 않도록 별도 기록하고 계속한다.
            failed.append(tkey(b))
            print(f"\n[fail] 타일 요청 실패 — 기록 후 계속: {tkey(b)}", file=sys.stderr)
            save()
            continue

        rows = parse(xml, typename)

        # 1000개는 상한에 걸렸다는 뜻이므로 더 쪼갠다
        if len(rows) >= MAX_FEATURES and min(w, h) > MIN_TILE:
            truncated += 1
            mx, my = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            stack += [
                (b[0], b[1], mx, my), (mx, b[1], b[2], my),
                (b[0], my, mx, b[3]), (mx, my, b[2], b[3]),
            ]
        else:
            out += rows
            done.add(tkey(b))
            if not rows:
                empty += 1
            if len(rows) >= MAX_FEATURES:
                print(f"\n[warn] 최소 타일({MIN_TILE}m)에서 {len(rows)}개 — 일부 누락 가능: {tkey(b)}",
                      file=sys.stderr)

        if tiles % 100 == 0:
            save()
        print(f"  타일 {tiles:>5} (분할 {truncated:>4} · 빈 {empty:>4} · 실패 {len(failed):>3}) · "
              f"대기 {len(stack):>5} · 수집 {len(out):>7,}", end="\r", file=sys.stderr)
        time.sleep(0.15)

    save()
    print(f"\n타일 {tiles:,}개 처리 (분할 {truncated:,} · 빈 {empty:,} · 실패 {len(failed):,})",
          file=sys.stderr)
    return out, failed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default="use")
    ap.add_argument("--probe", action="store_true", help="서울시청 주변 1km만")
    ap.add_argument("--bbox", help="x1,y1,x2,y2 (EPSG:5186)")
    ap.add_argument("--tile", type=float, default=1000.0, help="초기 격자 크기(m)")
    a = ap.parse_args()

    op, typename = DATASETS[a.dataset]
    if a.probe:
        box = (196_800.0, 550_800.0, 197_800.0, 551_800.0)
        tag = "probe"
    elif a.bbox:
        box = tuple(float(v) for v in a.bbox.split(","))  # type: ignore[assignment]
        tag = "bbox"
    else:
        box = SEOUL_BBOX
        tag = "seoul"

    print(f"데이터셋 {a.dataset}({typename}) · bbox {box}")
    ckpt = OUT / f"buildings_{a.dataset}_{tag}.ckpt.json"
    rows, failed = collect(key(), op, typename, box, a.tile, ckpt)
    if not rows:
        sys.exit("[error] 수집 결과 0건")

    if failed:
        print(f"\n[fail] 요청 실패 타일 {len(failed)}개 — 데이터에 구멍이 있다.")
        print(f"       같은 명령을 다시 실행하면 체크포인트에서 이어받는다: {ckpt}")
        for t in failed[:10]:
            print(f"         {t}")

    df = pd.DataFrame(rows)
    before = len(df)
    df = df.drop_duplicates(subset=["gid"])  # 타일 경계 건물이 중복 수집된다
    for c in ["height", "totar", "bldg_ar", "plot_ar", "far", "bcr"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["gf", "uf", "age", "n_rings"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    print(f"\n{before:,}건 → 중복 제거 후 {len(df):,}건")
    if "height" in df:
        hpos = df["height"] > 0
        print(f"높이>0 {hpos.sum():,}건 ({hpos.mean():.1%}) · 중위 {df.loc[hpos,'height'].median():.1f}m")
    if "gf" in df:
        fpos = df["gf"] > 0
        print(f"지상층수>0 {fpos.sum():,}건 ({fpos.mean():.1%}) · 중위 {df.loc[fpos,'gf'].median():.0f}층")
    if "use_class" in df:
        print("\n용도분류:", df["use_class"].value_counts().head(6).to_dict())

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"buildings_{a.dataset}_{tag}.parquet"
    df.to_parquet(dest, index=False)
    print(f"\n→ {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
