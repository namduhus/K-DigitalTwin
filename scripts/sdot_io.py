from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

# UTF-8을 먼저 본다. 순서를 바꾸면 CP949가 UTF-8 파일을 깨진 채로 받아들인다.
ENCODINGS = ("utf-8-sig", "utf-8", "cp949")
PROBE_BYTES = 1 << 18  # 판별용으로 읽는 앞부분 크기
N_COLS = 58  # 헤더가 정의하는 유효 컬럼 수


def detect_encoding(path: str | Path) -> str:
    head = Path(path).open("rb").read(PROBE_BYTES)
    for enc in ENCODINGS:
        try:
            head.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"{path} 인코딩 판별 실패")

# 분석에 쓰는 컬럼의 짧은 이름
RENAME = {
    "모델번호": "mdl",
    "시리얼": "sn",
    "측정시간": "ts",
    "지역": "rgn",
    "자치구": "gu",
    "행정동": "dong",
    "온도 평균(℃)": "tp",
    "습도 평균(%)": "hum",
    "풍속 평균(m/s)": "wspd",
    "조도 평균(lux)": "illu",
    "자외선 평균(UV)": "uv",
    "소음 평균(dB)": "nis",
    "흑구온도 평균(℃)": "gt",
}
NUMERIC = ["tp", "hum", "wspd", "illu", "uv", "nis", "gt"]


def header(path: str | Path, enc: str | None = None) -> list[str]:
    enc = enc or detect_encoding(path)
    with open(path, encoding=enc, errors="replace", newline="") as fh:
        return next(csv.reader(fh))[:N_COLS]


def read(path: str | Path, slim: bool = True) -> pd.DataFrame:
    enc = detect_encoding(path)
    names = header(path, enc)
    df = pd.read_csv(
        path,
        encoding=enc,
        header=0,
        names=names,
        usecols=range(N_COLS),  # 이름 없는 후행 6개 컬럼을 버린다
        low_memory=False,
    )
    if not slim:
        return df

    df = df[[c for c in RENAME if c in df.columns]].rename(columns=RENAME)
    for c in NUMERIC:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "ts" in df:
        df["ts"] = pd.to_datetime(df["ts"], format="%Y-%m-%d_%H:%M:%S", errors="coerce")
    return df


def read_many(paths: list[str | Path], slim: bool = True) -> pd.DataFrame:
    return pd.concat([read(p, slim) for p in paths], ignore_index=True)


# ── API parquet ──────────────────────────────────────────────────────────────
# fetch_sdot.py가 저장한 parquet은 컬럼명이 영문이다(CSV는 한글).
# 두 경로를 같은 스키마로 맞춰야 하나의 데이터셋으로 합칠 수 있다.
# API는 32일 롤링 윈도우라 주간 파일이 아직 발행되지 않은 최근 구간의 유일한 사본이다.
API_RENAME = {
    "MDL_NO": "mdl",
    "SN": "sn",
    "MSRMT_HR": "ts",
    "RGN": "rgn",
    "CGG": "gu",
    "DONG": "dong",
    "AVG_TP": "tp",
    "AVG_HUM": "hum",
    "AVG_WSPD": "wspd",
    "AVG_INILLU": "illu",
    "AVG_UV": "uv",
    "AVG_NIS": "nis",
    "AVG_GT": "gt",
}


def read_api(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[[c for c in API_RENAME if c in df.columns]].rename(columns=API_RENAME)
    for c in NUMERIC:
        if c in df:
            # API는 미측정을 빈 문자열로 준다
            df[c] = pd.to_numeric(df[c].replace("", None), errors="coerce")
    df["ts"] = pd.to_datetime(df["ts"], format="%Y-%m-%d_%H:%M:%S", errors="coerce")
    return df
