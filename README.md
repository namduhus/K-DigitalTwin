# 서울 생활권 체감더위 골목 단위 시간별 추정

서울시 도시데이터센서(S-DoT) 관측값에 3D 건물 음영, 천공률(SVF), 지형, 녹지 정보를 더해 서울 전역 25m 격자(969,212칸)의 시간별 기온과 불확실성을 추정한다.

기상청 5km 격자로는 서울이 43칸이다. 2026-08-06 16시 추정값을 보면 5km 한 칸 안에서도 25m 셀 기온이 P95−P05 기준 2.40℃ 차이 난다.

## 방법

```
T(셀, 시각) = t_reg(시각) + Δ(셀, 시각)

t_reg : 서울 광역값
Δ     : 동시각 편차 (추정 대상)
```

`t_reg`는 그 시각 서울 전체를 대표하는 값 하나다. 학습과 평가에는 센서 중위를, 운용에는 기상청 단기예보 중위를 쓴다.

`Δ` 점추정은 LightGBM이다. 같은 시각 이웃 센서의 IDW 보간값과 정적 공간 피처를 입력으로 쓴다. 학습 센서의 IDW는 자기 관측을 빼고(leave-one-out) 계산한다. 그러지 않으면 정답이 피처에 들어간다.

정적 공간 피처는 다음과 같다.

- 음영: 센서와 격자 셀마다 180방위 수평선 프로파일을 미리 구해 두고, 시각별 태양 위치와 비교해 `shadow_margin`(태양 고도 − 장애물 앙각)을 만든다. 416만 행마다 광선을 쏘는 대신 배열 조회로 처리한다.
- 천공률(SVF), 건물 밀도
- 표고, 경사, 지형 차폐 (DEM 90m)
- 녹지·포장면 비율

센서 피처와 격자 피처는 같은 코드(`scripts/horizon_core.py`)로 만든다. 학습은 센서 값으로, 추론은 격자 값으로 하기 때문이다.

불확실성은 24시간 궤적의 조건부 분포로 낸다. 이분산 척도, 시각 간 상관, t(ν=4) 꼬리를 쓰고 Mondrian split conformal로 보정한다. 처음에는 Flow Matching을 쓰려 했는데, 87,444개 궤적을 확인해 보니 다봉이 아니라 분산 혼합이어서 모수 분포로 바꿨다(`core/model/traj_probe.py`).

검증은 지점 단위 5-fold로 한다. 평가 센서는 학습에서 통째로 빠진다. 모델 비교는 fold별 짝지은 차이의 `t(4)`로 한다.

## 결과

따로 적지 않으면 MAE(℃)다. 개선율은 `(기준 − 대상) ÷ 기준`이고, 양수면 오차가 줄었다는 뜻이다.

| 모델 | 방식 | MAE |
|---|---|---|
| B0a 서울 단일값 | 모든 지점에 그 시각 서울 광역값 `t_reg`를 그대로 쓴다(`Δ = 0`) | 0.762 ± 0.028 |
| B0b 5km 격자 | 서울을 5km 칸으로 나누고 칸마다 학습 센서 편차의 중위를 더한다. 기상청 격자 해상도에 해당한다 | 0.662 ± 0.032 |
| B1 IDW 공간 보간 | 같은 시각 이웃 센서의 편차를 거리 역가중(IDW)으로 보간한다 | 0.622 ± 0.031 |
| B3 IDW + 정적 공간 피처 | B1의 보간값을 피처로 넣고 음영, SVF, 지형, 녹지 피처와 함께 LightGBM으로 학습한다. 본 모델 | 0.5582 ± 0.0256 |
| 참고: 오라클 정적 지도 | 지점별 실제 평균 편차를 안다고 가정한 지도 (in-sample) | 0.5995 |
| 참고: 오차 하한 | 105m 이내 센서 쌍 118개의 동시각 편차 차이(중위 0.6435℃)를 √2로 나눈 값. 계기 잡음과 100m 이하 미세 변동이 함께 들어 있다 | 약 0.455 |

B2(IDW 없이 정적 피처만 쓴 LightGBM)는 0.662 ± 0.027로 B1보다 나빴다. 그래서 B3는 IDW 보간값을 피처로 함께 넣었다. B3는 B1 대비 개선 +10.2%(`t(4)=12.74`), B0b 대비 +15.7%, B0a 대비 +26.8%다. 오차 하한의 1.23배 수준이다.

| 분포 지표 | 값 |
|---|---|
| conformal 보정 후 적중률 | 명목 50 / 80 / 90 / 95% → 실측 50.0 / 80.0 / 90.0 / 95.0% |
| CRPS | 점추정 0.5580 → 분포 0.4133 (개선 +25.9%) |
| 33℃ 초과 시간 CRPS | 점추정 0.608 → 분포 0.428 (개선 +29.6%) |

음영·SVF 피처의 기여는 1.4%로 작지만 5개 fold 모두 같은 방향이었다(`t(4)=4.11`).

전향 검증도 했다. 2026-08-28 예보로 만든 예측을 고정해 두고, 9/5까지 쌓인 관측과 비교했다. 학습에 없던 날짜(56,547행, 센서 780개)에서 광역 균일값 0.6083 대비 0.5533℃로 개선 +9.0%였다.

## 활용 데이터

| 데이터 | 제공 | 라이선스 |
|---|---|---|
| S-DoT 도시데이터센서 환경정보 (파일 OA-15969, API IotVdata017) | 서울 열린데이터광장 | 공공누리 1유형 |
| S-DoT 설치 위치정보, API 명세서 | 서울 열린데이터광장 | 공공누리 1유형 |
| 도시계획시설(공간시설) `UPIS_C_UQ153` | 서울 열린데이터광장 OA-21129 | 공공누리 1유형 |
| 생활권계획 시설(공원) `UPIS_SHP_ZON216` | 서울 열린데이터광장 OA-15529 | 공공누리 1유형 |
| 국가중점데이터 건물정보 (WFS) | 브이월드, 공공데이터포털 15123970 | 이용허락범위 제한 없음 |
| 행정구역도 시군구 (WFS) | 국가공간정보센터, 공공데이터포털 15059008 | 공공누리 1유형 |
| 수치표고모델(DEM) 90m | 국토지리정보원, 브이월드 | 이용허락범위 제한 없음 |
| 인구 격자통계(100m), 집계구 성연령별, 집계구 경계 | 통계청 SGIS | 이용허락범위 제한 없음 |
| 단기예보 조회서비스 | 기상청, 공공데이터포털 | 공공누리 1유형 |

## 폴더 구조

```
scripts/       데이터 파이프라인 (수집, 정제, 피처 → train.parquet, 25m 격자 피처, 데모 데이터)
               파일별 설명은 scripts/README.md
core/          모델링·평가 (train.parquet 이후)
  eval/          지점 단위 5-fold 분할, 버퍼 CV
  baseline/      B0~B3, 격자 추론, 예보 다운스케일링, 개입 반사실
  model/         조건부 궤적 분포, conformal 보정
  metrics/       지표, 정확도 표, 그림
               파일별 설명은 core/README.md
demo/          발표 데모 (deck.gl 정적 페이지, 오프라인 동작)
figures/       그림 PNG (1~9, 11은 core/metrics/make_figures.py, 10은 scripts/capture_demo.py)
```

## 실행

`uv`를 쓰고, Python은 3.12로 고정돼 있다(`pyproject.toml`의 `>=3.12,<3.13`). 명령은 모두 리포 루트에서 실행한다.

```bash
uv sync                    # 기본 (torch 제외)
uv sync --extra fm         # torch 포함, CUDA 12.1 서버용
cp .env.example .env       # API 키 입력 (scripts/README.md의 인증키 항목)
```

```bash
# 1. 데이터 → train.parquet, 격자 피처 (앞 단계를 포함한 전체 순서는 scripts/README.md)
uv run python scripts/build_train.py

# 2. 분할 → 기준선·모델 → 지표 (core/는 -m으로 실행, 전체 순서는 core/README.md)
uv run python -m core.eval.build_splits
uv run python -m core.baseline.b0_regional
uv run python -m core.baseline.b3_hybrid
uv run python -m core.metrics.evaluate --verify

# 3. 재현성 검증 (플랫폼을 옮기거나 계산 코드를 고친 뒤)
uv run python scripts/verify_pipeline.py            # 기준값과 전 항목 비교
uv run python scripts/verify_pipeline.py --quick    # B1·B3 생략
```

### 데모

```bash
uv run python scripts/build_demo.py              # 구역 3D (관악구 600m) → demo/data/district/
uv run python scripts/build_overview.py          # 서울 전역 25m 래스터 → demo/data/overview/
python3 -m http.server 8899 --directory demo     # http://localhost:8899
```

`index.html`을 파일로 바로 열면 `fetch`가 CORS에 막히니 HTTP 서버로 띄운다. deck.gl은 `demo/vendor/`에 들어 있어서 인터넷 없이도 동작한다.

멀리서 보면 서울 전역 2D 래스터가, 확대하면 건물 3D와 시각별 그림자가 나온다. 날짜는 폭염일 2026-08-06과 평온일 2026-07-15 두 개다.

## 표기

- 온도 수치는 대부분 동시각 편차(같은 시각 서울 센서 중위와의 차이)다. `−1.36℃`는 영하가 아니라 중위보다 1.36℃ 낮다는 뜻이다. 절대 기온은 여름 데이터라 항상 영상이다.
- 퍼센트는 개선율로만 쓰고, `개선 +15.7%`, `악화 −17.1%`처럼 방향을 단어로 붙인다.
- 공간 연산은 EPSG:5186(미터)에서 한다. 서울시 UPIS 자료(EPSG:5174, Bessel 타원체)는 `pyproj`로 변환한다. 좌표를 그대로 넣으면 100~200m 어긋난다.

## 라이선스

| 대상 | 라이선스 | 원문 |
|---|---|---|
| 코드 (`scripts/`, `core/`, `demo/index.html`) | MIT | [`LICENSE`](LICENSE) |
| 파생 데이터·그림 (`demo/data/`, `figures/`) | CC BY 4.0 | [`LICENSE-DATA`](LICENSE-DATA) |
| deck.gl 9.0.33 번들 (`demo/vendor/deck.gl.min.js`) | MIT (vis.gl contributors) | [`demo/vendor/LICENSE-deck.gl`](demo/vendor/LICENSE-deck.gl) |

`demo/data/`와 `figures/`는 [활용 데이터](#활용-데이터) 표의 기관이 제공한 공공데이터를 가공한 것이다. 공공누리 제1유형 자료는 출처표시 조건에 따라 이용했다. 이 저장소의 데이터나 그림을 다시 쓸 때는 원 출처 기관도 함께 표시해야 한다.

원본 데이터는 저장소에 넣지 않았다. 받는 곳과 둘 경로는 [`scripts/README.md`](scripts/README.md)에 있다.
