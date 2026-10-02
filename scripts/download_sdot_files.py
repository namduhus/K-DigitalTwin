from __future__ import annotations

import argparse
import html
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import requests

INF_ID = "OA-15969"
PAGE = f"https://data.seoul.go.kr/dataList/{INF_ID}/S/1/datasetView.do"
POST = "https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false"
RAW = Path("data/raw")

# 폭염 분석은 6~9월이 핵심이다. 주간 파일은 파일명의 기간으로 판정한다.
SUMMER_MONTHS = {6, 7, 8, 9}


@dataclass
class FileRow:
    seq: str
    name: str
    size_mb: str
    updated: str

    @property
    def is_data(self) -> bool:
        return self.name.startswith("S-DoT_NATURE_")

    @property
    def is_yearly(self) -> bool:
        return self.name.endswith(".zip")

    @property
    def is_summer(self) -> bool:
        if self.is_yearly:
            return True
        m = re.search(r"(\d{4})\.(\d{2})\.\d{2}-(\d{2})\.\d{2}", self.name)
        if not m:
            return False
        return int(m.group(2)) in SUMMER_MONTHS or int(m.group(3)) in SUMMER_MONTHS


def fetch_list() -> list[FileRow]:
    r = requests.get(PAGE, timeout=60)
    r.raise_for_status()
    out: list[FileRow] = []
    for tr in re.findall(r'<tr id="fileTr_\d+">(.*?)</tr>', r.text, re.S):
        seq = re.search(r"downloadFile\('(\d+)'\)", tr)
        tds = [re.sub(r"<[^>]*>", "", t).strip() for t in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        if seq and len(tds) >= 5:
            out.append(FileRow(seq.group(1), html.unescape(tds[2]), tds[3], tds[4]))
    if not out:
        sys.exit("[error] 파일 목록 파싱 실패 — 페이지 구조가 바뀐 듯하다")
    return out


def report_coverage(targets: list[FileRow]) -> None:
    weeks = sorted((r for r in targets if not r.is_yearly), key=lambda r: r.name)
    years = sorted((r for r in targets if r.is_yearly), key=lambda r: r.name)
    total = sum(float(r.size_mb) for r in targets)

    print(f"대상 {len(targets)}개 · 약 {total:,.0f} MB")

    if weeks:
        span = re.findall(r"(\d{4})\.(\d{2}\.\d{2})-(\d{2}\.\d{2})", weeks[0].name + weeks[-1].name)
        if len(span) == 2:
            print(
                f"  주간 CSV {len(weeks)}개  →  {span[0][0]}.{span[0][1]} ~ {span[1][0]}.{span[1][2]}"
                f"  ({sum(float(r.size_mb) for r in weeks):,.0f} MB)"
            )
    if years:
        print(
            f"  연도별 zip {len(years)}개  →  {years[0].name[13:17]} ~ {years[-1].name[13:17]}"
            f"  ({sum(float(r.size_mb) for r in years):,.0f} MB)"
            "  ※ 각 연도 전체가 들어 있다. 압축 해제 후 필요한 달만 골라 쓴다"
        )

    have = {p.name for p in RAW.glob("*")} if RAW.exists() else set()
    if skip := [r for r in targets if r.name in have]:
        print(f"  이미 있음 {len(skip)}개 — 건너뜀")

    print(
        "  ※ 주간 파일은 약 1주 지연 발행이다. 최근 구간은 "
        "`fetch_sdot.py`(API, 32일 롤링)로 메운다"
    )


def download(row: FileRow) -> None:
    RAW.mkdir(exist_ok=True)
    dest = RAW / row.name
    if dest.exists():
        print(f"  건너뜀 (있음)  {row.name}")
        return

    data = {"infId": INF_ID, "seqNo": "", "seq": row.seq, "infSeq": "3"}
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.post(POST, data=data, stream=True, timeout=180) as r:
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        if "text/html" in ctype:
            print(f"  [실패] {row.name} — HTML 응답 (다운로드 거부)", file=sys.stderr)
            return
        got = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                got += len(chunk)
                print(f"  {row.name[:46]:<46} {got / 1e6:>7.1f} MB", end="\r")
    tmp.rename(dest)
    print(f"  {row.name[:46]:<46} {got / 1e6:>7.1f} MB  완료")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "get"])
    ap.add_argument("--seq", nargs="*", help="내려받을 seq 목록")
    ap.add_argument(
        "--summer",
        action="store_true",
        help="6~9월 주간 CSV + 연도별 zip 전부 (zip은 연도 전체가 들어 있다)",
    )
    ap.add_argument("--weeks", action="store_true", help="2026 주간 CSV 중 6~9월만 (zip 제외)")
    ap.add_argument("--years", action="store_true", help="연도별 zip만")
    ap.add_argument("--all", action="store_true", help="데이터 파일 전부")
    ap.add_argument("-n", "--dry-run", action="store_true", help="받지 않고 대상만 보여준다")
    a = ap.parse_args()

    rows = fetch_list()

    if a.cmd == "list":
        print(f"{'seq':<12} {'파일명':<50} {'MB':>8}  갱신")
        print("-" * 82)
        for r in rows:
            mark = "*" if r.is_summer and r.is_data else " "
            print(f"{r.seq:<12} {mark}{r.name[:49]:<49} {r.size_mb:>8}  {r.updated}")
        data = [r for r in rows if r.is_data]
        summer = [r for r in data if r.is_summer]
        print(f"\n데이터 파일 {len(data)}개 · * 여름철 관련 {len(summer)}개")
        print("* 만 받으려면:  uv run python scripts/download_sdot_files.py get --summer")
        return 0

    data = [r for r in rows if r.is_data]
    if a.all:
        targets = data
    elif a.summer:
        targets = [r for r in data if r.is_summer]
    elif a.weeks:
        targets = [r for r in data if r.is_summer and not r.is_yearly]
    elif a.years:
        targets = [r for r in data if r.is_yearly]
    elif a.seq:
        want = set(a.seq)
        targets = [r for r in rows if r.seq in want]
        if missing := want - {r.seq for r in targets}:
            print(f"[warn] 없는 seq: {', '.join(sorted(missing))}", file=sys.stderr)
    else:
        sys.exit("[error] --summer / --weeks / --years / --all / --seq 중 하나를 지정")

    if not targets:
        sys.exit("[error] 대상 없음")

    report_coverage(targets)
    if a.dry_run:
        return 0
    print()
    for r in targets:
        download(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
