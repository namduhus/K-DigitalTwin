# core/

`scripts/`가 만든 `data/processing/train.parquet`부터 이어받는다. 분할, 기준선과 모델, 분포 보정, 지표와 그림 순서로 진행하고 산출물은 모두 `data/metric/`에 저장한다.

```bash
uv run python -m core.<폴더>.<파일>     # 리포 루트에서 -m으로 실행한다 (패키지 내부를 임포트하기 때문)
```

| 폴더 | 내용 |
|---|---|
| `eval/` | 평가 설계. 지점 단위 5-fold 분할, fold별 광역 기준값, 평가 행 로더, 버퍼 CV |
| `baseline/` | 비교 기준 B0~B3, 25m 격자 추론, 예보 다운스케일링, 개입 반사실 |
| `model/` | 24시간 궤적의 조건부 분포(척도, 시각 간 상관, t 꼬리), conformal 보정, 격자 궤적 샘플 |
| `metrics/` | 지표 계산, 정확도 표, 궤적 파생량 평가, 진단, 그림 |

## 흐름

```
train.parquet
   │
   ├─ eval/build_splits ─────────> splits.json · regional.parquet        (분할 · fold별 t_reg)
   │
   ├─ baseline/b0 · b1 · b2 · b3 ─> preds/<모델>.parquet                  (sn, ts, y_pred = 절대 기온)
   │        └─ metrics/evaluate ─> metrics.parquet
   │
   ├─ model/traj_dist ───────────> traj_dist.npz · preds/b3_scale.parquet (척도 · 상관 R · 자유도 ν)
   │  model/calibrate ───────────> calib.npz                              (conformal 배수)
   │        └─ metrics/accuracy_table · traj_score
   │
   └─ baseline/grid_infer ───────> grid_pred_<날짜>.parquet               (969,212셀 × 시각)
            ├─ model/grid_traj ──> grid_traj_<날짜>.parquet               (셀마다 궤적 30개)
            ├─ metrics/fan_extract · make_figures
            └─ scripts/build_demo · build_overview                        (데모)
```

## 공통 규칙

- 분할은 지점 단위다. 시각을 무작위로 나누면 같은 센서가 학습과 평가에 같이 들어간다. 센서 고정효과가 편차 분산의 37.5%를 설명하기 때문에 그만큼 점수가 부풀려진다.
- 광역 기준값 `t_reg`는 fold마다 학습 센서만으로 중위를 낸다. 전체 센서로 내면 평가 센서가 자기 기준선 계산에 들어간다.
- 학습은 편차로 하고 저장은 절대 기온으로 한다. `preds/<모델>.parquet`은 모두 `(sn, ts, y_pred)` 형식이고 `y_pred`는 절대 기온(℃)이다. 그래서 지표 코드는 모델 내부 표현을 몰라도 된다.
- 모든 모델을 같은 행으로 평가한다. 로더는 `eval/dataset.py` 하나이고 제외 규칙은 `splits.json`에만 있다.
- IDW 이웃은 그 fold의 학습 센서로만 잡는다. 학습 센서 자신의 IDW는 자기 관측을 빼고(leave-one-out) 계산한다.
- 튜닝은 fold 안에서 한다. fold f의 학습 센서 중 fold (f+1)%5에 속한 센서를 내부 검증에 쓴다.
- 모델 비교는 fold별 짝지은 차이의 `t(4)`로 한다. LightGBM 학습 자체는 결정론적이지만 `feature_fraction=0.8`이라 피처 구성이 바뀌면 결과가 0.003 정도 움직인다. 그래서 0.001~0.002 차이로는 설계를 바꾸지 않는다.

## 실행 순서

```bash
# 분할
uv run python -m core.eval.build_splits

# 기준선 → 지표
uv run python -m core.baseline.b0_regional
uv run python -m core.baseline.b1_interp
uv run python -m core.baseline.b2_gbm
uv run python -m core.baseline.b3_hybrid
uv run python -m core.metrics.evaluate --verify

# 분포 모델 (b3_hybrid 예측 필요)
uv run python -m core.model.traj_dist
uv run python -m core.model.calibrate
uv run python -m core.metrics.accuracy_table
uv run python -m core.metrics.traj_score

# 25m 격자 (traj_dist, calib 필요)
uv run python -m core.baseline.grid_infer
uv run python -m core.model.grid_traj
uv run python -m core.metrics.fan_extract

# 그림 (한글 폰트 필요)
uv run python -m core.metrics.make_figures            # 전부
uv run python -m core.metrics.make_figures 2 3        # 일부
```

대부분 `--probe` 옵션(fold 0이나 일부만 실행)이 있다. 처음 돌릴 때는 `--probe`로 걸리는 시간을 먼저 보는 게 좋다.

## 파일별 역할

### `eval/`

| 파일 | 역할 | 산출 |
|---|---|---|
| `dataset.py` | 평가 행과 fold별 기준값 로더(`load_base`, `load_regional`, `iter_folds`, `save_preds`). `iter_folds`는 fold마다 `t_reg`와 `delta`를 다시 채운다. fold 0을 돌릴 때는 train과 test 행 모두 fold 0의 기준값을 쓴다. 직접 실행하지 않는다 | — |
| `build_splits.py` | 지점 단위 5-fold. 지역유형 `rgn`으로 먼저 층화하고, 유형 안에서는 주소에서 뽑은 자치구 `gu_addr` 순으로 돌아가며 배정한다. API의 `gu` 필드는 같은 센서인데도 값이 바뀌어서 쓰지 않는다. 궤적이 너무 적은 센서는 평가에 넣지 않고 학습에만 쓴다. 과천 서울대공원 4개와 위치를 특정할 수 없는 2개 센서는 제외한다 | `splits.json`, `regional.parquet` |
| `buffer_cv.py` | 관측에서 멀어질수록 얼마나 나빠지는지 본다. 평가 센서 반경 r 안의 센서를 IDW 이웃과 GBM 학습 행 양쪽에서 뺀다. 학습 표본이 줄어드는 효과와 구분하려고 같은 개수를 무작위로 빼는 대조군을 같이 돌린다. 거리는 명목 r 대신 실제 최근접 거리로 기록한다. `--dry`, `--probe`, `--radii` | `buffer_cv.parquet` |

### `baseline/`

| 파일 | 역할 | 산출 |
|---|---|---|
| `b0_regional.py` | B0a는 서울 단일값(`Δ = 0`), B0b는 5km 정방 격자 칸별 학습 센서 편차 중위다. B0b를 기상청 해상도로 얻을 수 있는 상한으로 본다. 실제 기상청 격자(LCC)와 완전히 같지는 않다. 센서가 없는 칸은 `t_reg`로 채우고 그 칸 수를 로그에 남긴다. `--cell` | `preds/b0a`, `b0b` |
| `b1_interp.py` | B1, 편차 IDW. 편차를 (시각 × 센서) 행렬로 만들어 행렬곱 한 번으로 예측하고, 시각마다 관측된 센서만으로 가중치를 다시 정규화한다. 고정 가중치로 나누면 결측 센서가 0℃로 들어간다. `p`, `k`는 fold 안에서 중첩 튜닝한다. 크리깅은 배리오그램 확인용으로 한 번만 돌린다. `idw_oof` 등은 다른 파일에서 가져다 쓴다. `--no-krige` | `preds/b1_idw` |
| `b2_gbm.py` | B2, LightGBM과 정적 피처. 피처 묶음을 하나씩 더하는 절제 실험 4단계를 같은 코드로 돌린다(A1 시간, A2 건물 밀도, A3 DEM·녹지, A4 음영·SVF). 피처 묶음과 `PARAMS`는 다른 파일에서도 쓴다. `--probe` | `preds/b2_a1_time`, `b2_a2_bld`, `b2_a3_terr`, `b2_gbm`(= A4) |
| `b3_hybrid.py` | B3, 본 모델. IDW 예측값을 피처로 넣고 정적 피처와 함께 학습한다. B2가 B1보다 못했던 건 두 모델이 쓰는 정보가 달랐기 때문이라, 보간(B1)에 공간 피처를 더하는 구조로 바꿨다. 주야는 따로 학습하지 않고 `is_night`을 피처로 넣는다(분리 학습 결과는 `b3_split`으로 남긴다). IDW는 `p=1, k=50`으로 고정했다(B1 중첩 튜닝에서 5개 fold 중 4개가 고른 값). `--probe` | `preds/b3_hybrid`, `b3_noshade`(음영·SVF 제외), `b3_split`(주야 분리) |
| `grid_infer.py` | 25m 격자 추론. 평가는 fold별 모델로 하고, 지도는 전체 센서로 다시 학습한 모델로 만든다. 시간 피처(태양 위치, `shadow_margin`, `is_shadow`)는 `scripts/build_train.py`와 같은 식으로 계산한다. 분위 GBM q10/q50/q90과 녹지 개입 시나리오 `cf_park`도 같이 낸다. 기본 날짜는 폭염일 2026-08-06과 평온일 2026-07-15이고, 열대야 구간까지 보려고 00시부터 다음날 09시까지 34시각을 계산한다. `--probe`, `--days` | `grid_pred_<날짜>.parquet` |
| `counterfactual.py` | 개입 반사실. 학습된 모델에 조건만 바꿔 다시 예측한다. 피처 하나만 바꾸면 면적 비율 합이 100%를 넘으므로, 녹지가 늘면 포장면과 건물이 줄고 최근접 녹지 거리도 바뀌도록 묶어서 바꾼다. 바꾼 값이 학습 분포 밖에 있는 비율을 같이 출력하고, 교란 조건이 비슷한 센서 쌍을 매칭해 모델 효과와 비교한다. 인과 효과를 식별하는 것은 아니다. 쿨루프는 대응하는 피처가 없어 계산하지 않는다. 시나리오 정의는 `grid_infer`에서도 쓴다. `--probe` | 콘솔 출력 |
| `forecast_infer.py` | 예보 다운스케일링. 기상청 단기예보 중위를 `t_reg`로 넣고 25m 격자 궤적을 만든다. 미래 시각에는 이웃 관측이 없으므로 IDW 피처를 뺀 모델을 쓴다. `--probe`, `--hours` | `forecast_grid.parquet` |
| `forecast_sensors.py` | 전향 검증. 예보로 센서 위치의 값을 미리 예측해 두고, 관측이 쌓인 뒤 `--verify`로 비교한다. 전체 센서로 학습한 모델(시간 외삽만)과 fold별 모델(시간과 공간 모두 hold-out)을 나란히 내서, 공간 성능이 얼마나 낙관적으로 잡혔는지 본다 | `forecast_sensors.parquet` |
| `probe_stidw.py` | 시간 이력 피처를 넣으면 B3가 더 좋아지는지 본 실험. 이웃 센서의 과거 IDW(lag 1, 2, 24h)와 광역값 변화율만 넣었다. 평가 센서 자신의 과거값은 격자 셀에는 없는 정보이고, 지점 hold-out으로는 이 시간 방향 누수를 막을 수 없어서 넣지 않았다. 결과는 차이 없음(`t(4)=−0.35`). `preds/`와 `metrics.parquet`은 건드리지 않는다. `--fold`, `--paired` | `probe_stidw.parquet` |

### `model/`

```
Δ(셀, 24시간) = μ(c)  +  D(c) · z,     z ~ t_ν(0, R)

μ : B3 예측
D : 조건부 척도
R : 24시각 상관,  ν : 꼬리 두께
```

| 파일 | 역할 | 산출 |
|---|---|---|
| `traj_probe.py` | 궤적 잔차가 다봉인지 확인한다. 표본이 크면 BIC는 거의 항상 k≥2를 고르므로 BIC만으로 판단하지 않았다. PCA + GMM(BIC/ICL), 성분 간 평균 차이(℃), 파생량 재현을 함께 보고, 이분산 때문에 다봉처럼 보이는 것은 아닌지 날짜별 표준화 잔차로 다시 확인했다. 결과는 분산 혼합이었고, 그래서 Flow Matching 대신 다변량 t 분포를 쓴다. `--nsim` | `traj_probe.parquet` |
| `traj_dist.py` | 분포 추정. 척도 `D`는 B3와 같은 5-fold, 피처, IDW로 분위 GBM을 out-of-fold 학습한 뒤 `(q90 − q10) / 2.563`으로 구한다. `grid_infer`와 정의가 같아서 격자에 그대로 쓸 수 있다. 표준화 잔차로 24×24 상관 `R`을 구하고, 자유도 `ν`는 프로파일 최대우도로 추정한다. `--probe`, `--reuse-scale` | `traj_dist.npz`, `preds/b3_scale.parquet` |
| `calibrate.py` | conformal 보정. 점수 `E = |y − μ| / σ`로 곱셈 배수 `Q`를 정한다. 예측이 모두 out-of-fold라서 fold f를 평가할 때 나머지 4개 fold를 보정 집합으로 쓴다(leave-one-fold-out). σ 구간별로 따로 보정하는 Mondrian 방식과 폭염 축을 포함하고, CRPS는 정렬 표본 공식으로 계산한다. `--sample-rows` | `calib.npz` |
| `grid_traj.py` | 격자 궤적 샘플. 셀마다 24시간 궤적 30개를 뽑아 시각별 분위와 파생량 분포(33℃ 초과 시간 등)를 낸다. 파생량은 궤적의 비선형 함수라 시각별 분위를 이어 붙여서는 구할 수 없다. 분위 GBM 구간은 명목 80%의 실제 적중률이 60.3%라서 Mondrian 보정을 거친다. 열대야(18시~다음날 09시)는 상관행렬이 달력일 기준이라 아직 넣지 않았다. `--probe`, `--days`, `--no-hourly` | `grid_traj_<날짜>.parquet`, `grid_traj_q_<날짜>.parquet` |

### `metrics/`

| 파일 | 역할 | 산출 |
|---|---|---|
| `evaluate.py` | 지표 계산. `preds/*`만 읽고 모델 내부는 보지 않는다. 슬라이스는 전체, 폭염 시각, 주/야, 센서가 적은 자치구, 녹지비·SVF·최근접 건물거리 5분위다. 칸마다 5-fold 평균 ± 표준편차를 낸다. `--verify`는 B0a MAE가 따로 계산한 `mean(|Δ|)`와 같은지 확인한다. `--models` | `metrics.parquet` |
| `accuracy_table.py` | 정확도 표 3개(MAE 슬라이스, 분포 지표인 CRPS·적중률·구간 폭, 피처 분위별 MAE). 결정론적 예측의 CRPS는 MAE와 같아서 점추정 모델이 그대로 기준선이 된다. 지역유형 범주는 안에서 편차가 커서 연속 피처의 분위로 나눈다. `--samples` | `accuracy_table.parquet` |
| `trajectory.py` | 궤적 파생량 오차. 열대야 지속시간, 일최고 발생시각, 33℃ 초과 시간을 점추정 궤적에서 센다. 관측이 비어 있으면 파생량이 왜곡되므로 완결성 기준(하루 20시간 이상 등)을 둔다. `--models` | `trajectory.parquet` |
| `traj_score.py` | 파생량을 분포로 평가한다(CRPS, 80% 구간 적중률). 점추정 궤적에서 센 값은 `f(E[X]) ≠ E[f(X)]`라 한쪽으로 치우친다. `--draws` | `traj_score.parquet` |
| `treg_sensitivity.py` | 광역값 `t_reg`가 틀렸을 때 얼마나 나빠지는지 본다. `t_reg`는 기준점이면서 입력 피처라 영향이 단순하지 않다. 계통 편차와 시각별 무작위 오차를 시각 단위로 넣고(행 단위로 넣으면 평균에서 상쇄된다), IDW를 잃는 비용은 따로 떼어 본다. 학습은 참값으로, 추론에만 오차를 넣는다. `--probe` | `treg_sensitivity.parquet` |
| `shadow_probe.py` | 400m 창 안에서 기온이 그림자를 따라가는지 본다. 창이 좁으면 반경 600m 집계 피처가 거의 상수라서 동네 간 차이가 통제된다. 모델이 음영을 피처로 쓰기 때문에 독립 검증은 아니고 메커니즘 확인이다. `--day`, `--win`, `--hour` | `shadow_probe.parquet` |
| `globe_check.py` | 흑구온도로 복사 부하(`gt − tp`)를 직접 확인하려 했다. 야간에도 −2.75℃이고 주야 차이가 0.00℃라 복사에 반응하지 않는 데이터로 보고 검증에 쓰지 않았다 | `globe_check.parquet` |
| `siting.py` | 외삽 정도와 다음 센서 후보 위치. 정적 피처 22개를 센서 분포로 표준화한 공간에서 격자 셀과 최근접 센서 사이 거리를 재고, 외삽 정도 × 인구로 우선순위를 매긴다. 최적 배치를 푸는 게 아니라 어디가 비어 있는지 보여 주는 용도다. `--topn` | `siting.parquet` |
| `fan_extract.py` | 팬차트용 셀 몇 개의 평균 `t_hat`과 보정 척도 `sigma_cal`만 뽑는다. 격자 궤적 원본이 수백 MB라서 그림 그리는 머신으로 옮기지 않으려고 둔 단계다 | `fan_cells.parquet` |
| `make_figures.py` | 그림 1~9, 11(그림 10은 `scripts/capture_demo.py`). 한글 폰트가 필요하다. 그림 번호를 인자로 주면 그 그림만 그린다 | `figures/*.png` |

## `data/metric/` 산출물

| 파일 | 만드는 곳 | 내용 |
|---|---|---|
| `splits.json` | `eval/build_splits` | fold별 센서 목록, 항상 학습에만 쓰는 센서 |
| `regional.parquet` | `eval/build_splits` | 시각 × fold 광역 기준값 `t_reg` |
| `preds/<모델>.parquet` | `baseline/b0~b3` | `sn, ts, y_pred` (절대 기온) |
| `preds/b3_scale.parquet` | `model/traj_dist` | `sn, ts, q10, q90, sigma` (out-of-fold 척도) |
| `metrics.parquet` | `metrics/evaluate` | long 형식 `model, fold, slice_kind, slice_val, n, mae, rmse, bias` |
| `traj_dist.npz` | `model/traj_dist` | 상관 `R`, 자유도 `ν` 등 |
| `calib.npz` | `model/calibrate` | conformal 배수 `Q` (수준별, Mondrian 구간별) |
| `grid_pred_<날짜>.parquet` | `baseline/grid_infer` | `gx, gy, ts, t_hat, q10, q50, q90, cf_park` |
| `grid_traj_<날짜>.parquet`, `grid_traj_q_<날짜>.parquet` | `model/grid_traj` | 셀별 파생량 분포, 시각별 분위 |
| `forecast_grid.parquet`, `forecast_sensors.parquet` | `baseline/forecast_*` | 예보 다운스케일링, 전향 검증 |
| 그 밖의 `<이름>.parquet` | 같은 이름의 `metrics/`, `eval/`, `model/` 파일 | 각 진단 결과 |

## 주의

- `grid_pred`의 `q10`/`q90`은 보정 전 값이다. 명목 80%의 실제 적중률이 60.3%이므로 `calib.npz`로 보정한 값을 쓴다.
- conformal 적중률(90% → 90.0%)은 실황 기준값을 쓸 때만 성립한다. 학습 기간 안에서 보정했기 때문이다. 예보로 미래를 돌린 전향 검증에서는 80% 구간 적중률이 52.9%(편향 보정 후)였다.
- 예보를 기준값으로 쓰면 오차는 주로 모델이 아니라 `t_reg`에서 온다. 전향 검증에서 예보 기준값의 편향이 −1.670℃(야간 −2.78℃)였다.
- `make_figures`는 한글 폰트가 없으면 경고만 내고 라벨이 깨진 채로 그린다.
- `counterfactual` 효과는 500m 블록 단위로 본다. 25m 셀 단위에서는 부호가 반대로 나오는 셀이 30.4%라 불안정하다.
