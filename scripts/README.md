# scripts/ — 데이터 파이프라인

원본 수집부터 **`data/processing/train.parquet`**(학습 데이터)과 **`grid_features.parquet`**(25m 격자 추론 입력)까지 만든다. 그 이후의 분할·모델·지표는 `core/`가 맡는다.

예외는 **데모 스크립트 3개**다. `core/`가 만든 격자 예측과 궤적 분포를 읽어 화면용 데이터로 굽는다 (아래 [5. 데모](#5-데모--core-산출-이후)).

```bash
uv run python scripts/<파일>.py      # 리포 루트에서. core/와 달리 -m 없이 경로로 실행한다
```

이 문서는 파일마다 **무엇을 읽어 무엇을 만드는지**와 그렇게 만든 이유를 정리한다.

---

## 흐름

```
[수집]                         [정제]                    [센서 피처]                        [학습 데이터]
S-DoT 위치 xlsx ─ prep_locations ─> sdot_locations.csv ─┬─────────────────────────┐
S-DoT CSV ─ download_sdot_files ─┐                      │                         │
                                 ├─ build_dataset ───> sdot_summer.parquet ───────┼──────────┐
S-DoT API ─ fetch_sdot ──────────┘  (API parquet은 읽지 않음)                     │          │
                                                        │                         │          │
브이월드 WFS ─ fetch_buildings ─ prep_buildings ─> buildings.parquet              │          │
                                                        ├─ build_horizon ─> sensor_horizon ──┤
                                                  DEM ──┴─ build_terrain ─> sensor_features ─┼─ build_train ─> train.parquet
                                         UPIS 공원·녹지 ─── build_green ───> sensor_green ───┘                  ──> core/

[25m 격자]
브이월드 WFS ─ fetch_boundary ─> seoul_boundary.parquet
buildings + 경계 ─ build_grid (horizon_core) ─> grid_features.parquet ─ build_grid_terrain (지형·녹지 열 추가)
SGIS 인구 + 격자 ─ build_population ─> grid_population.parquet                                    ──> core/

[운용]  기상청 단기예보 ─ fetch_kma ─> kma_treg.parquet ──> core/baseline/forecast_infer.py
[데모]  core/ 산출 ─ build_demo · build_overview ─> demo/data/ ─ capture_demo ─> figures/fig10
```

---

## 실행 순서

```bash
# 1. 수집
uv run python scripts/prep_locations.py
uv run python scripts/download_sdot_files.py get --summer
uv run python scripts/fetch_sdot.py fetch                 # 최근 약 32일 — 지나면 다시 못 받는다
uv run python scripts/fetch_buildings.py                  # 서울 전역
uv run python scripts/fetch_boundary.py

# 2. 정제
uv run python scripts/build_dataset.py                    # 신청서 근거 수치를 매번 출력한다
uv run python scripts/prep_buildings.py

# 3. 센서 피처 → train.parquet
uv run python scripts/build_horizon.py
uv run python scripts/build_terrain.py
uv run python scripts/build_green.py
uv run python scripts/build_train.py

# 4. 25m 격자 피처
uv run python scripts/build_grid.py                       # 병렬. Mac 6워커 약 35분 / 서버 30워커 약 13분
uv run python scripts/build_grid_terrain.py --check       # 센서 재현을 먼저 확인하고
uv run python scripts/build_grid_terrain.py
uv run python scripts/build_population.py

# 5. 검증 — 플랫폼을 옮기거나 계산 코드를 고친 뒤
uv run python scripts/verify_pipeline.py
```

3단계의 `build_horizon` · `build_terrain` · `build_green`은 서로 독립이다. 단 `build_terrain`은 `sensor_horizon.parquet`을 읽으므로 `build_horizon` 뒤에 돌린다.

---

## 파일별 역할

### 공용 모듈 — 직접 실행하지 않는다

| 파일 | 역할 |
|---|---|
| **`sdot_io.py`** | **S-DoT CSV·API 로더.** `read()`(CSV) · `read_api()`(API parquet). **맨손 `pd.read_csv`를 쓰면 안 된다** — 인코딩이 파일마다 CP949/UTF-8-SIG로 섞여 있고, 주간 파일은 헤더 58 / 데이터 64 필드라 그냥 읽으면 **전 컬럼이 한 칸씩 밀린 채 에러 없이** 읽힌다 |
| **`horizon_core.py`** | **수평선 프로파일 · SVF · 건물 밀도 계산 엔진.** 센서와 격자가 **같은 코드**를 쓰게 하려고 분리했다 — 다르면 추론 때 피처 분포가 어긋난다. 전 건물 외곽선을 한 번만 densify(11.7백만 점)하고 `cKDTree`로 질의해 셀당 비용을 벡터 연산으로 줄였다. `build_grid.py`와 `verify_pipeline.py`가 임포트한다 |

### 1. 수집

| 파일 | 역할 | 입력 | 산출 |
|---|---|---|---|
| `prep_locations.py` | S-DoT 설치 위치정보 정제. 위경도를 EPSG:5186 `x, y`로 변환하고, 시리얼이 바뀐 센서의 옛 시리얼을 `sn_alias`에 담는다 | `data/*설치 위치정보*.xlsx` | `data/processing/sdot_locations.csv` |
| `download_sdot_files.py` | S-DoT 과거 파일(연도별 zip · 주간 CSV) 다운로드. 파일 목록을 데이터셋 페이지에서 매번 파싱하므로 새 주간 파일이 자동으로 잡힌다. `list`(여름철 파일에 `*` 표시) / `get --summer · --seq · --all` | 서울 열린데이터광장 | `data/raw/` |
| `fetch_sdot.py` | S-DoT OpenAPI(`IotVdata017`) 수집과 결측률 점검. `probe` / `missing` / `fetch` | OpenAPI | `data/raw/sdot_api_*.parquet` |
| `fetch_buildings.py` | 브이월드 WFS로 서울 건물 수집. 요청 상한이 1000개이고 페이징이 없어서 **1000개가 오면 잘린 것으로 보고 bbox를 4분할해 재귀**한다. EPSG:5186으로 직접 요청한다 | 브이월드 WFS | `data/raw/buildings_use_seoul.parquet` |
| `fetch_boundary.py` | 서울 자치구 경계 수집 — 25m 격자 마스크. bbox로 격자를 만들면 1.65배가 되고 서울 밖까지 칠해진다 | 브이월드 WFS | `data/raw/vworld_adsigg.gml` · `data/processing/seoul_boundary.parquet` |
| `fetch_kma.py` | 기상청 단기예보 → 서울 광역 기준값 `t_reg`. 운용(예보 다운스케일링)과 전향 검증의 입력이고 학습에는 쓰지 않는다. `--probe`로 발표 시각 가용성만 확인 | 기상청 API · `seoul_boundary.parquet` | `data/raw/kma/` (원본 응답) · `data/processing/kma_treg.parquet` |

- **`data/raw/sdot_api_*.parquet`은 복구할 수 없다. 삭제하지 않는다.** OpenAPI는 약 32일 롤링 윈도우이고 주간 파일은 약 1주 늦게 올라온다
- **기상청 과거 예보는 약 2일치만 남는다.** 그래서 `fetch_kma.py`는 원본 응답을 `data/raw/kma/`에 따로 보관한다

### 2. 정제

| 파일 | 역할 | 입력 | 산출 |
|---|---|---|---|
| **`build_dataset.py`** | 흩어진 S-DoT CSV를 하나로. 대상 월(기본 6~9월)만 골라 `sdot_io.read()`로 읽고, 좌표를 조인(본 시리얼 + `sn_alias`)하고, QC로 센티널(−40)·물리범위 밖 값을 NaN 처리한다(**행은 지우지 않는다**). 열지수 `hi`를 보조로 계산한다. **신청서 근거 수치를 매 실행마다 출력한다** — 다시 계산하지 말고 이 출력을 쓴다. `--months` · `--dry-run` | `data/raw/` CSV · `sdot_locations.csv` | `data/processing/sdot_summer.parquet` |
| `prep_buildings.py` | 건물 원본 정제. 높이가 60.1%만 채워져 있고 최대 11,700m 같은 오류가 있다 → 이상치를 거르고, 용도별 층고를 37만 동에서 학습해 **`층수 × 층고`로 결측을 채우고 출처를 플래그로 남긴다** | `data/raw/buildings_use_seoul.parquet` | `data/processing/buildings.parquet` |

- `build_dataset.py`는 **API parquet을 읽지 않는다** — 컬럼명이 영문(`AVG_TP` 등)이라 별도 정규화가 필요하다
- **주 타깃은 기온 `tp`다.** 흑구온도는 결측이 너무 많아(2026년 96.8%) WBGT를 만들 수 없다

### 3. 센서 피처 → `train.parquet`

| 파일 | 역할 | 입력 | 산출 |
|---|---|---|---|
| `build_horizon.py` | 센서별 **건물** 수평선 프로파일(2° × 180방위 장애물 앙각)과 SVF · 건물 밀도. 센서를 점이 아니라 **반경 5m 9점**으로 보고 평균한다 — 점 판정은 건물 경계에서 1m 차이로 그늘/양지가 뒤집힌다. `--probe`(센서 30개) | `buildings.parquet` · `sdot_locations.csv` | `data/processing/sensor_horizon.parquet` |
| `build_terrain.py` | DEM으로 표고·경사·사면향과 **지형 차폐** 앙각을 구해 건물 수평선과 합친다(`max`). 북한산 836m는 2km 거리에서 앙각 21.4°라 건물 평균 앙각(18.7°)과 맞먹는다. `--probe` | DEM 90m · `sensor_horizon.parquet` · `sdot_locations.csv` | `data/processing/sensor_features.parquet` |
| `build_green.py` | 공원·녹지 면적비와 거리 · 포장면 비율. 광장·공공공지는 포장면이라 녹지와 따로 센다 | UPIS 공원·녹지 SHP · `sdot_locations.csv` | `data/processing/sensor_green.parquet` |
| **`build_train.py`** | 관측(y)에 정적 피처(X)를 붙이고, 시각별 태양 위치(`pvlib`)를 수평선과 대조해 **`is_shadow` · `shadow_margin`**(태양 고도 − 장애물 앙각)을 만든다. 태양 위치는 서울 중심에서 한 번만 계산한다 — 동서 끝의 방위 차가 약 0.47°로 2° 구간 안이다 | `sdot_summer` · `sensor_horizon` · `sensor_features` · `sensor_green` | **`data/processing/train.parquet`** |

- **UPIS 공원·녹지는 EPSG:5174(Bessel 1841)다.** False Northing 차이(10만 m)는 바로 드러나지만, **타원체 차이로 생기는 100~200m datum shift는 티가 안 난다.** `geopandas.to_crs`(pyproj 정식 변환)를 거친다
- 서울시 가로수 위치정보는 공공누리 4유형이라 쓰지 않는다

### 4. 25m 격자 피처

| 파일 | 역할 | 입력 | 산출 |
|---|---|---|---|
| `build_grid.py` | 서울 경계 안 **969,212셀**의 수평선 · SVF · 건물 밀도. `horizon_core`를 센서와 공유하고 9점 샘플도 똑같이 쓴다. 건물 내부 셀은 지우지 않고 `in_building` 플래그만 단다. 전역 사전계산(274 MB)을 워커마다 다시 만들지 않도록 **fork로 공유**한다(macOS 기본값이 spawn이라 명시). 끝나면 격자 피처가 센서 분포 밖에 있는 비율(외삽 구간)을 출력한다. `--probe`(1,000셀) · `--cell` · `--jobs` | `buildings.parquet` · `seoul_boundary.parquet` · `sensor_horizon.parquet` | `data/processing/grid_features.parquet` |
| `build_grid_terrain.py` | 격자에 지형·녹지 열을 붙인다(제자리 갱신). **`--check`로 센서 좌표에서 같은 코드를 돌려 `sensor_features` · `sensor_green`을 재현하는지 먼저 확인**한다. 항목마다 `통과`/`불일치`를 찍고, 전부 맞으면 `전 항목 재현`을 출력한다 — `verify_pipeline.py`가 이 문구로 판정하므로 바꾸지 않는다. 지형 차폐는 셀을 청크로 나눠 벡터화한다(청크 1,000 = 120 MB) | `grid_features.parquet` · DEM · UPIS SHP · 센서 피처 | `grid_features.parquet`에 열 추가 |
| `build_population.py` | 25m 격자 인구·고령인구 — 처방 지도의 노출 가중. SGIS 100m 격자는 연령이 없고 집계구는 연령은 있지만 해상도가 낮다 → **비율은 집계구, 인구수는 격자**에서 가져온다. 비율로 쓰면 SGIS 비식별 노이즈가 분자·분모에서 상쇄된다. 원본은 EPSG:5179인데 5186과 같은 GRS80 타원체라 datum shift가 없다. 집계구 조인율을 매번 출력한다 | SGIS 원본 4종 · `grid_features` · `seoul_boundary` | `data/processing/grid_population.parquet` |

- **격자 피처는 반드시 센서 피처와 같은 코드로 만든다.** 학습은 센서 값으로, 예측은 격자 값으로 하기 때문이다. `horizon_core` 리팩터 첫 시도에서 앙각이 최대 2.37° 어긋난 적이 있다

### 5. 데모 — `core/` 산출 이후

`core/baseline/grid_infer.py`(격자 예측) · `core/model/traj_dist.py` · `calibrate.py`(궤적 분포)를 먼저 돌려야 한다.

| 파일 | 역할 | 입력 | 산출 |
|---|---|---|---|
| `build_demo.py` | **구역 3D** 데이터 — 관악구 600m 창의 건물 폴리곤·높이, 시각별 셀 기온, 셀별 궤적 분위(팬차트). 무거운 계산을 여기서 끝내서 브라우저는 그리기만 한다. 좌표는 5186 그대로 두고 중심만 WGS84로 바꿔 deck.gl `METER_OFFSETS`에 넘긴다. `--days` | `grid_pred_<날짜>.parquet` · `traj_dist.npz` · `calib.npz` · `buildings` · `grid_features` · `grid_population` | `demo/data/district/<날짜>.json` |
| `build_overview.py` | **서울 전역 2D** — 25m 예측을 시각별 래스터 24장으로 굽는다. 기온을 8bit 회색조로 양자화해 **색이 아니라 값**을 담고, 색은 뷰어가 입힌다. 네 모서리만 주면 위치가 수십 m 밀리므로 `rasterio.warp`로 EPSG:4326에 맞춰 재투영한다. 동시각 중위는 옥외 셀로만 잡는다. 약 8분. `--days` | `grid_pred_<날짜>.parquet` · `grid_features` · `seoul_boundary` | `demo/data/overview/<날짜>/` |
| `capture_demo.py` | 데모 화면 캡처 → 신청서 그림 10. 로컬 서버와 헤드리스 크롬을 직접 띄우고 09시·16시를 찍어 나란히 붙인다. 색 범위·셀 수는 페이지에서 읽어 캡션에 넣는다. CDP용 WebSocket을 표준 라이브러리로 구현해 의존성을 늘리지 않았다. **Mac 전용**(Chrome 필요). `--hours` · `--keep-shots` | `demo/` | `figures/fig10_demo_<날짜>.png` (· `data/processing/demo_shots/`) |

- **날짜를 추가하면 두 날을 한 번에 굽는다.** 편차 색 범위가 두 날 공통이어야 「폭염일에 격차가 벌어진다」가 보인다. 절대 기온 범위는 날마다 따로 잡는다
- **전역과 구역의 「편차」는 기준이 다르다** — 전역은 서울 전역 중위, 구역은 창 안 중위 대비다
- `build_overview.py`는 33℃ 초과 시간을 굽지 않는다. 구역 3D의 값은 궤적 200샘플에서 나오는데, 전역에서 점추정을 이어 붙여 세면 **같은 이름의 다른 양**이 화면에 뜬다

### 6. 검증

| 파일 | 역할 |
|---|---|
| **`verify_pipeline.py`** | **재현성 검증.** 센서·격자·궤적 수, 기하(면적·앙각·SVF), B0·B1·B3 MAE, 분포 모델 지표(ν · 상관 · conformal 배수 · CRPS · 적중률)를 기준값(2026-08-24, Apple M4 Pro / Python 3.12.13)과 대조하고 항목마다 `통과`/`실패`를 찍는다. 허용 오차는 출처별로 다르다 — 정수 카운트는 **완전 일치**, 기하는 상대 1e-4, numpy 유래는 절대 1e-3, LightGBM 유래는 절대 5e-3. `--quick`은 B1·B3를 건너뛴다(Mac M4 Pro에서 23항목 · 0.4분) |

플랫폼을 옮기거나 계산 코드를 고쳤으면 **반드시 돌린다.** 값이 어긋나면 어느 쪽 수치를 쓸지 먼저 정한다 — 조용히 새 값으로 바꾸면 이전 결과와 어긋난다.

---

## 직접 받아야 하는 원본

다음은 스크립트가 받지 않는다. 받아서 아래 경로에 둔다.

| 데이터 | 경로 | 쓰는 곳 |
|---|---|---|
| S-DoT 설치 위치정보 xlsx | `data/*설치 위치정보*.xlsx` | `prep_locations` |
| 수치표고모델 DEM 90m (브이월드 데이터 다운로드) | `data/raw/dem90/한반도90m_GRS80.img` | `build_terrain` · `build_grid_terrain` |
| 서울시 UPIS `UPIS_C_UQ153` · `UPIS_SHP_ZON216` | `data/raw/park/*.shp` | `build_green` · `build_grid_terrain` |
| SGIS 100m 격자 인구 CSV · 격자 경계 SHP · 집계구 성연령별·총괄 CSV · 집계구 경계 SHP | `data/raw/` 아래 (정확한 폴더명은 `build_population.py` 상단 상수) | `build_population` |

## 인증키 — `.env`

`.env.example`을 복사해서 채운다. `.env`는 버전 관리에서 빠져 있다.

| 키 | 쓰는 스크립트 | 비고 |
|---|---|---|
| `SMART_SEOUL_ENV` (없으면 `SMART_SEOUL_ENV_LIVE`) | `fetch_sdot` | 서울 열린데이터광장 |
| 이름에 `WORLD`가 들어간 키 | `fetch_buildings` · `fetch_boundary` | 브이월드. 도메인 `localhost`로 발급. 셸 호환을 위해 `VWORLD_API_KEY`처럼 하이픈 없는 이름을 권장한다 |
| 이름에 `KMA`가 들어간 키 | `fetch_kma` | 공공데이터포털. Encoding 키(`%3D`로 끝남)를 넣어도 코드가 디코딩해서 쓴다 — 그대로 `params`에 넣으면 이중 인코딩돼 `SERVICE_KEY_IS_NOT_REGISTERED_ERROR`가 난다 |

---

## 주의

- **모든 공간 연산은 EPSG:5186(미터)에서 한다.** S-DoT(WGS84) 쪽을 변환해 건물에 맞춘다. 반대로 하면 건물 수백만 좌표를 다시 투영해야 한다. EPSG:4326에서 거리·반경을 계산하지 않는다
- **이상값은 지우지 않고 플래그를 단다.** 42℃대 기온과 고온에 습도 100%는 신호다 — S-DoT은 2~4m 높이에서 아스팔트 복사열을 직접 받는다
- **매칭에서 결측이 생기면 매칭되지 않은 항목을 전부 출력한다.** 원본 주소의 `서울측별시` 오타 110건을 고정 정규식이 놓친 적이 있다
- **백그라운드 실행 출력을 `grep`으로 파이프하지 않는다.** 파이프 마지막 명령의 종료코드가 보고돼 Python이 죽어도 `exit 0`으로 보인다. 로그 파일로 리다이렉트하고 종료코드를 따로 확인한다
