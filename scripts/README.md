# scripts/

원본 수집부터 `data/processing/train.parquet`(학습 데이터)과 `grid_features.parquet`(25m 격자 추론 입력)을 만드는 데까지 맡는다. 분할, 모델, 지표는 `core/`에서 한다.

데모 스크립트 3개만 예외다. `core/`가 만든 격자 예측과 궤적 분포를 읽어 화면용 데이터를 만든다(아래 5. 데모).

```bash
uv run python scripts/<파일>.py      # 리포 루트에서 실행. core/와 달리 -m 없이 경로로 실행한다
```

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

## 실행 순서

```bash
# 1. 수집
uv run python scripts/prep_locations.py
uv run python scripts/download_sdot_files.py get --summer
uv run python scripts/fetch_sdot.py fetch                 # OpenAPI는 최근 약 32일치만 받을 수 있다
uv run python scripts/fetch_buildings.py                  # 서울 전역
uv run python scripts/fetch_boundary.py

# 2. 정제
uv run python scripts/build_dataset.py                    # 실행할 때마다 데이터 요약 수치를 출력
uv run python scripts/prep_buildings.py

# 3. 센서 피처 → train.parquet
uv run python scripts/build_horizon.py
uv run python scripts/build_terrain.py
uv run python scripts/build_green.py
uv run python scripts/build_train.py

# 4. 25m 격자 피처
uv run python scripts/build_grid.py                       # 병렬. Mac 6워커 약 35분, 서버 30워커 약 13분
uv run python scripts/build_grid_terrain.py --check       # 센서 피처 재현 확인
uv run python scripts/build_grid_terrain.py
uv run python scripts/build_population.py

# 5. 검증 (플랫폼을 옮기거나 계산 코드를 고친 뒤)
uv run python scripts/verify_pipeline.py
```

3단계의 `build_horizon`, `build_terrain`, `build_green`은 서로 독립이지만, `build_terrain`이 `sensor_horizon.parquet`을 읽으므로 `build_horizon`을 먼저 돌린다.

## 파일별 역할

### 공용 모듈 (직접 실행하지 않음)

| 파일 | 역할 |
|---|---|
| `sdot_io.py` | S-DoT 로더. `read()`는 CSV, `read_api()`는 API parquet용이다. 파일마다 인코딩이 CP949와 UTF-8-SIG로 섞여 있고, 주간 파일은 헤더가 58필드, 데이터가 64필드다. `pd.read_csv`로 바로 읽으면 오류 없이 컬럼이 한 칸씩 밀리므로 이 함수로 읽는다 |
| `horizon_core.py` | 수평선 프로파일, SVF, 건물 밀도 계산. 센서와 격자가 같은 코드를 쓰도록 따로 뺐다. 건물 외곽선 전체를 한 번 densify(1,170만 점)하고 `cKDTree`로 조회한다. `build_grid.py`와 `verify_pipeline.py`가 임포트한다 |

### 1. 수집

| 파일 | 역할 | 입력 | 산출 |
|---|---|---|---|
| `prep_locations.py` | S-DoT 설치 위치 정제. 위경도를 EPSG:5186 `x, y`로 바꾸고, 시리얼이 바뀐 센서는 옛 시리얼을 `sn_alias`에 넣는다 | `data/*설치 위치정보*.xlsx` | `data/processing/sdot_locations.csv` |
| `download_sdot_files.py` | S-DoT 과거 파일(연도별 zip, 주간 CSV) 다운로드. 파일 목록을 데이터셋 페이지에서 매번 읽어 오므로 새로 올라온 주간 파일도 받는다. `list`(여름철 파일에 `*` 표시), `get --summer`/`--seq`/`--all` | 서울 열린데이터광장 | `data/raw/` |
| `fetch_sdot.py` | S-DoT OpenAPI(`IotVdata017`) 수집과 결측률 확인. `probe`, `missing`, `fetch` | OpenAPI | `data/raw/sdot_api_*.parquet` |
| `fetch_buildings.py` | 브이월드 WFS로 서울 건물 수집. 한 번에 1,000건까지만 오고 페이징이 없어서, 1,000건이 오면 bbox를 4분할해 다시 요청한다. 좌표계는 EPSG:5186으로 요청한다 | 브이월드 WFS | `data/raw/buildings_use_seoul.parquet` |
| `fetch_boundary.py` | 서울 자치구 경계. 25m 격자 마스크로 쓴다. bbox로 격자를 만들면 셀이 1.65배로 늘고 서울 밖까지 들어간다 | 브이월드 WFS | `data/raw/vworld_adsigg.gml`, `data/processing/seoul_boundary.parquet` |
| `fetch_kma.py` | 기상청 단기예보로 서울 광역 기준값 `t_reg`를 만든다. 예보 다운스케일링과 전향 검증에만 쓰고 학습에는 쓰지 않는다. `--probe`는 발표 시각 가용 여부만 확인한다 | 기상청 API, `seoul_boundary.parquet` | `data/raw/kma/`(원본 응답), `data/processing/kma_treg.parquet` |

- `data/raw/sdot_api_*.parquet`은 다시 받을 수 없으니 지우지 않는다. OpenAPI는 최근 약 32일치만 제공하고, 주간 파일은 약 1주 늦게 올라온다.
- 기상청 과거 예보는 약 2일치만 남아 있다. 그래서 `fetch_kma.py`가 원본 응답을 `data/raw/kma/`에 따로 저장한다.

### 2. 정제

| 파일 | 역할 | 입력 | 산출 |
|---|---|---|---|
| `build_dataset.py` | S-DoT CSV를 하나로 합친다. 대상 월(기본 6~9월) 파일을 `sdot_io.read()`로 읽고, 좌표를 조인하고(본 시리얼과 `sn_alias`), 센티널(−40)과 물리 범위 밖 값을 NaN으로 바꾼다. 행은 지우지 않는다. 열지수 `hi`를 보조 변수로 계산하고, 실행할 때마다 요약 수치를 출력한다. `--months`, `--dry-run` | `data/raw/` CSV, `sdot_locations.csv` | `data/processing/sdot_summer.parquet` |
| `prep_buildings.py` | 건물 원본 정제. 높이 값이 60.1%만 채워져 있고 11,700m 같은 오류도 있다. 이상치를 거른 뒤 용도별 층고를 37만 동에서 추정해 결측을 `층수 × 층고`로 채우고, 값의 출처를 플래그로 남긴다 | `data/raw/buildings_use_seoul.parquet` | `data/processing/buildings.parquet` |

- `build_dataset.py`는 API parquet을 읽지 않는다. 컬럼명이 영문(`AVG_TP` 등)이라 따로 맞춰야 한다.
- 타깃은 기온 `tp`다. 흑구온도는 2026년 결측이 96.8%라 WBGT를 계산할 수 없다.

### 3. 센서 피처 → `train.parquet`

| 파일 | 역할 | 입력 | 산출 |
|---|---|---|---|
| `build_horizon.py` | 센서별 건물 수평선 프로파일(2° 간격 180방위의 장애물 앙각), SVF, 건물 밀도. 센서 위치를 점 하나가 아니라 반경 5m 안 9점의 평균으로 계산한다. 점 하나로 보면 건물 경계 근처에서 1m 차이로 그늘과 양지가 바뀐다. `--probe`(센서 30개) | `buildings.parquet`, `sdot_locations.csv` | `data/processing/sensor_horizon.parquet` |
| `build_terrain.py` | DEM으로 표고, 경사, 사면향, 지형 차폐 앙각을 구하고 건물 수평선과 방위별 `max`로 합친다. 북한산(836m)은 2km 거리에서 앙각이 21.4°로 건물 평균 앙각(18.7°)보다 높다. `--probe` | DEM 90m, `sensor_horizon.parquet`, `sdot_locations.csv` | `data/processing/sensor_features.parquet` |
| `build_green.py` | 공원·녹지 면적비, 최근접 녹지 거리, 포장면 비율. 광장과 공공공지는 포장면이라 녹지와 따로 센다 | UPIS 공원·녹지 SHP, `sdot_locations.csv` | `data/processing/sensor_green.parquet` |
| `build_train.py` | 관측(y)에 정적 피처를 붙이고, 시각별 태양 위치(`pvlib`)를 수평선과 비교해 `is_shadow`, `shadow_margin`(태양 고도 − 장애물 앙각)을 만든다. 태양 위치는 서울 중심 한 곳에서만 계산한다. 동서 끝의 방위 차이가 약 0.47°로 방위 간격 2°보다 작기 때문이다 | `sdot_summer`, `sensor_horizon`, `sensor_features`, `sensor_green` | `data/processing/train.parquet` |

- UPIS 공원·녹지 SHP는 EPSG:5174(Bessel 1841)다. False Northing이 10만 m 다르기 때문에 좌표를 그대로 넣으면 바로 알 수 있지만, 타원체 차이에서 오는 100~200m 어긋남은 눈에 잘 띄지 않는다. `geopandas.to_crs`(pyproj)로 변환한다.
- 서울시 가로수 위치정보는 공공누리 4유형(상업적 이용 금지, 변경 금지)이라 쓰지 않았다.

### 4. 25m 격자 피처

| 파일 | 역할 | 입력 | 산출 |
|---|---|---|---|
| `build_grid.py` | 서울 경계 안 969,212셀의 수평선, SVF, 건물 밀도. `horizon_core`와 9점 샘플링을 센서와 똑같이 쓴다. 건물 안 셀도 지우지 않고 `in_building` 플래그만 단다. 전역 사전계산 결과(274 MB)는 워커마다 다시 만들지 않고 fork로 공유한다(macOS 기본값이 spawn이라 명시적으로 지정). 끝나면 센서 피처 분포 밖에 있는 셀 비율을 출력한다. `--probe`(1,000셀), `--cell`, `--jobs` | `buildings.parquet`, `seoul_boundary.parquet`, `sensor_horizon.parquet` | `data/processing/grid_features.parquet` |
| `build_grid_terrain.py` | 격자에 지형·녹지 열을 추가한다(파일을 그대로 갱신). `--check`는 같은 코드를 센서 좌표에서 돌려 `sensor_features`, `sensor_green`이 재현되는지 본다. 항목마다 `통과`/`불일치`를 찍고 전부 맞으면 `전 항목 재현`을 출력하는데, `verify_pipeline.py`가 이 문구로 판정하므로 바꾸면 안 된다. 지형 차폐는 1,000셀(약 120 MB)씩 나눠 벡터 연산한다 | `grid_features.parquet`, DEM, UPIS SHP, 센서 피처 | `grid_features.parquet`에 열 추가 |
| `build_population.py` | 25m 격자 인구와 고령인구. 노출 가중치로 쓴다. SGIS 100m 격자에는 연령이 없고, 집계구에는 연령이 있지만 해상도가 낮다. 그래서 연령 비율은 집계구에서, 인구수는 격자에서 가져온다. 비율로 쓰면 SGIS 비식별 노이즈가 분자와 분모에서 상쇄된다. 원본은 EPSG:5179인데 5186과 같은 GRS80 타원체라 datum 차이는 없다. 실행할 때마다 집계구 조인율을 출력한다 | SGIS 원본 4종, `grid_features`, `seoul_boundary` | `data/processing/grid_population.parquet` |

- 격자 피처는 센서 피처와 같은 코드로 만든다. 학습은 센서 값, 예측은 격자 값으로 하기 때문이다. `horizon_core`로 코드를 합치던 첫 시도에서 둘의 앙각이 최대 2.37° 달랐던 적이 있다.

### 5. 데모 (`core/` 산출 이후)

`core/baseline/grid_infer.py`(격자 예측)와 `core/model/traj_dist.py`, `calibrate.py`(궤적 분포)를 먼저 돌려야 한다.

| 파일 | 역할 | 입력 | 산출 |
|---|---|---|---|
| `build_demo.py` | 관악구 600m 구역의 3D 데이터. 건물 폴리곤과 높이, 시각별 셀 기온, 셀별 궤적 분위(팬차트)를 담는다. 계산은 여기서 끝내고 브라우저는 그리기만 한다. 좌표는 5186 그대로 두고 중심점만 WGS84로 바꿔 deck.gl `METER_OFFSETS`에 넘긴다. `--days` | `grid_pred_<날짜>.parquet`, `traj_dist.npz`, `calib.npz`, `buildings`, `grid_features`, `grid_population` | `demo/data/district/<날짜>.json` |
| `build_overview.py` | 서울 전역 2D. 25m 예측을 시각별 래스터 24장으로 만든다. 기온을 8bit 회색조로 양자화해 값을 그대로 저장하고, 색은 뷰어에서 입힌다. 네 모서리 좌표만 주면 위치가 수십 m 밀려서 `rasterio.warp`로 EPSG:4326에 재투영한다. 동시각 중위는 옥외 셀로만 계산한다. 약 8분 걸린다. `--days` | `grid_pred_<날짜>.parquet`, `grid_features`, `seoul_boundary` | `demo/data/overview/<날짜>/` |
| `capture_demo.py` | 데모 화면을 캡처해 그림 10을 만든다. 로컬 서버와 헤드리스 크롬을 띄우고 09시와 16시 화면을 찍어 나란히 붙인다. 색 범위와 셀 수는 페이지에서 읽어 캡션에 넣는다. CDP용 WebSocket은 표준 라이브러리로 구현해서 의존성을 늘리지 않았다. Chrome이 필요해 Mac에서만 돌린다. `--hours`, `--keep-shots` | `demo/` | `figures/fig10_demo_<날짜>.png`(와 `data/processing/demo_shots/`) |

- 날짜를 추가하면 두 날을 한 번에 다시 만든다. 편차 색 범위를 두 날 공통으로 잡아야 폭염일에 격차가 벌어지는 게 보인다. 절대 기온 범위는 날마다 따로 잡는다.
- 전역 화면의 편차는 서울 전역 중위 기준이고, 구역 화면의 편차는 600m 창 안 중위 기준이다.
- `build_overview.py`는 33℃ 초과 시간을 만들지 않는다. 구역 3D의 이 값은 궤적 200개 샘플에서 계산하는데, 전역에서 점추정으로 세면 이름만 같고 다른 값이 된다.

### 6. 검증

| 파일 | 역할 |
|---|---|
| `verify_pipeline.py` | 재현성 검증. 센서·격자·궤적 수, 기하(면적, 앙각, SVF), B0·B1·B3 MAE, 분포 모델 지표(ν, 상관, conformal 배수, CRPS, 적중률)를 기준값(2026-08-24, Apple M4 Pro, Python 3.12.13)과 비교해 항목마다 `통과`/`실패`를 찍는다. 허용 오차는 정수 카운트 완전 일치, 기하 상대 1e-4, numpy 계산 절대 1e-3, LightGBM 결과 절대 5e-3이다. `--quick`은 B1·B3를 건너뛴다(Mac M4 Pro에서 23항목, 0.4분) |

플랫폼을 옮기거나 계산 코드를 고쳤으면 돌린다. 값이 다르게 나오면 어느 쪽을 쓸지 먼저 정하고 기록해 둔다. 그냥 새 값으로 바꾸면 이전 결과와 맞지 않게 된다.

## 직접 받아야 하는 원본

아래 파일은 스크립트가 받지 않는다. 직접 받아서 해당 경로에 둔다.

| 데이터 | 경로 | 쓰는 곳 |
|---|---|---|
| S-DoT 설치 위치정보 xlsx | `data/*설치 위치정보*.xlsx` | `prep_locations` |
| 수치표고모델 DEM 90m (브이월드 데이터 다운로드) | `data/raw/dem90/한반도90m_GRS80.img` | `build_terrain`, `build_grid_terrain` |
| 서울시 UPIS `UPIS_C_UQ153`, `UPIS_SHP_ZON216` | `data/raw/park/*.shp` | `build_green`, `build_grid_terrain` |
| SGIS 100m 격자 인구 CSV, 격자 경계 SHP, 집계구 성연령별·총괄 CSV, 집계구 경계 SHP | `data/raw/` 아래 (폴더명은 `build_population.py` 상단 상수 참고) | `build_population` |

## 인증키 (`.env`)

`.env.example`을 복사해서 채운다. `.env`는 버전 관리에서 빠져 있다.

| 키 | 쓰는 스크립트 | 비고 |
|---|---|---|
| `SMART_SEOUL_ENV` (없으면 `SMART_SEOUL_ENV_LIVE`) | `fetch_sdot` | 서울 열린데이터광장 |
| 이름에 `WORLD`가 들어간 키 | `fetch_buildings`, `fetch_boundary` | 브이월드. 도메인을 `localhost`로 발급받는다. 셸에서 쓰기 쉽게 `VWORLD_API_KEY`처럼 하이픈 없는 이름이 낫다 |
| 이름에 `KMA`가 들어간 키 | `fetch_kma` | 공공데이터포털. Encoding 키(`%3D`로 끝나는 것)를 넣어도 코드에서 디코딩한다. 인코딩된 키를 `params`에 그대로 넣으면 이중 인코딩되어 `SERVICE_KEY_IS_NOT_REGISTERED_ERROR`가 난다 |

## 주의

- 공간 연산은 모두 EPSG:5186(미터)에서 한다. S-DoT 좌표(WGS84)를 5186으로 바꿔 건물에 맞춘다. 반대로 하면 건물 좌표 수백만 개를 다시 투영해야 한다. EPSG:4326에서 거리나 반경을 계산하지 않는다.
- 이상값은 지우지 않고 플래그만 단다. 42℃대 기온이나 고온에서 습도 100%는 실제 관측이다. S-DoT 센서는 지상 2~4m에 있어서 아스팔트 복사열을 그대로 받는다.
- 매칭에서 결측이 생기면 매칭되지 않은 항목을 모두 출력한다. 원본 주소의 `서울측별시` 오타 110건을 정규식이 놓친 적이 있다.
- 백그라운드로 돌릴 때 출력을 `grep`에 파이프하면 종료코드가 `grep` 것으로 바뀌어서 Python이 죽어도 `exit 0`이 나온다. 로그 파일로 리다이렉트하고 종료코드는 따로 확인한다.
