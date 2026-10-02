# core/ — 모델링·평가

`scripts/`가 만든 `data/processing/train.parquet` 이후를 맡는다. 분할 → 기준선·모델 → 분포 보정 → 지표·그림 순서이고, 산출물은 전부 `data/metric/`에 쓴다.

```bash
uv run python -m core.<폴더>.<파일>     # 리포 루트에서, 반드시 -m으로 (패키지 내부를 임포트한다)
```

| 폴더 | 하는 일 |
|---|---|
| `eval/` | 평가 설계 — 지점 단위 5-fold 분할, fold별 광역 기준값, 평가 행 집합 로더, 버퍼 CV |
| `baseline/` | 비교 기준 B0~B3, 25m 격자 추론, 예보 다운스케일링, 개입 반사실 |
| `model/` | 24시간 궤적의 조건부 분포 (척도 · 시각 간 상관 · t 꼬리), conformal 보정, 격자 궤적 샘플 |
| `metrics/` | 지표 하네스, 정확도 표, 궤적 파생량 평가, 진단, 그림 |

---

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

---

## 공통 설계 — 모든 파일이 지키는 것

- **지점 단위로 나눈다.** 시각을 무작위로 나누면 같은 센서가 학습과 평가에 동시에 들어간다. 센서 고정효과가 편차 분산의 37.5%를 설명하므로 그 누수가 곧 점수가 된다
- **광역 기준값 `t_reg`를 fold마다 따로 만든다.** 학습 센서만의 중위를 쓴다. 전체 센서 중위를 쓰면 test 센서가 자기 정답의 기준선을 만든다
- **편차로 학습하고 절대 기온으로 저장한다.** `preds/<모델>.parquet`은 모두 `(sn, ts, y_pred)` 형식이고 `y_pred`는 ℃ 절대 기온이다. 지표 계층은 모델 내부 표현을 몰라도 된다
- **모든 모델이 같은 행에서 평가된다.** `eval/dataset.py`가 유일한 로더이고, 제외 규칙은 `splits.json` 한 곳에만 있다
- **IDW 피처는 누수를 막는다.** 이웃은 그 fold의 학습 센서뿐이고, 학습 센서 자신의 IDW는 자기 관측을 빼고(leave-one-out) 계산한다
- **튜닝은 fold 안에서 한다.** fold f의 학습 센서 중 fold (f+1)%5 소속을 내부 검증으로 쓴다
- **비교는 fold별 짝지은 차이와 `t(4)`로 한다.** 평균 두 개를 나란히 놓고 고르지 않는다. LightGBM 학습은 결정론적이지만 피처 구성이 바뀌면(`feature_fraction=0.8`) 0.003 규모로 움직이므로, 0.001~0.002 차이로 설계를 정하지 않는다

---

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

# 분포 모델 (b3_hybrid 예측이 있어야 한다)
uv run python -m core.model.traj_dist
uv run python -m core.model.calibrate
uv run python -m core.metrics.accuracy_table
uv run python -m core.metrics.traj_score

# 25m 격자 (traj_dist · calib가 있어야 한다)
uv run python -m core.baseline.grid_infer
uv run python -m core.model.grid_traj
uv run python -m core.metrics.fan_extract

# 그림 — 한글 폰트가 있는 머신에서
uv run python -m core.metrics.make_figures            # 전부
uv run python -m core.metrics.make_figures 2 3        # 일부
```

대부분 `--probe`(fold 0 또는 일부만)가 있다. 처음 돌릴 때는 `--probe`로 시간을 재고 시작한다.

---

## 파일별 역할

### `eval/` — 평가 설계

| 파일 | 역할 | 산출 |
|---|---|---|
| `dataset.py` | **평가 행 집합과 fold별 기준값 로더.** `load_base` · `load_regional` · `iter_folds` · `save_preds`. `iter_folds`가 fold마다 `t_reg`와 `delta`를 다시 채운다 — fold 0을 돌릴 때는 train·test 모든 행이 fold 0의 기준값을 쓴다. 직접 실행하지 않는다 | — |
| `build_splits.py` | 지점 단위 5-fold. 지역유형 `rgn`으로 1차 층화하고 유형 안에서는 주소 유래 자치구 `gu_addr` 순으로 라운드로빈 배정한다(API 필드 `gu`는 센서마다 흔들려서 쓰지 않는다). 궤적이 너무 적은 센서는 test에 넣지 않고 항상 학습에만 쓴다. 과천 서울대공원 4센서와 위치를 특정할 수 없는 2센서는 제외한다 | `splits.json` · `regional.parquet` |
| `buffer_cv.py` | **관측에서 멀어지면 얼마나 나빠지는가.** test 센서 반경 r 안의 센서를 IDW 이웃 풀과 GBM 학습 행 **양쪽에서** 뺀다. 표본이 줄어드는 효과와 구분하려고 같은 수를 무작위로 빼는 대조군을 함께 돌리고, 명목 r이 아니라 **실현 최근접 거리**를 기록한다. `--dry` · `--probe` · `--radii` | `buffer_cv.parquet` |

### `baseline/` — 비교 기준과 추론

| 파일 | 역할 | 산출 |
|---|---|---|
| `b0_regional.py` | **B0a** 서울 단일값(`Δ = 0`)과 **B0b** 5km 정방 격자 칸별 학습센서 편차 중위. B0b가 기상청 해상도로 도달 가능한 상한이다. 실제 기상청 격자(LCC)와 정확히 같지는 않다. 센서 없는 칸은 `t_reg`로 채우고 몇 칸인지 로그에 남긴다. `--cell` | `preds/b0a` · `b0b` |
| `b1_interp.py` | **B1** 편차에 IDW. 편차를 (시각 × 센서) 행렬로 펼쳐 행렬곱 하나로 예측하고, 시각마다 관측된 센서만으로 분모를 다시 정규화한다(고정 가중치로 나누면 결측 센서가 0℃로 취급된다). `p` · `k`는 fold 안에서 중첩 튜닝한다. 크리깅은 배리오그램을 얻으려고 한 번만 돌린다. `idw_oof` 등을 다른 파일에 제공한다. `--no-krige` | `preds/b1_idw` |
| `b2_gbm.py` | **B2** LightGBM + 정적 피처. 피처 묶음을 쌓아 **절제 4단**을 같은 코드로 낸다 — A1 시간 → A2 건물 밀도 → A3 DEM·녹지 → A4 음영·SVF. 피처 묶음과 `PARAMS`를 다른 파일에 제공한다. `--probe` | `preds/b2_a1_time` · `b2_a2_bld` · `b2_a3_terr` · `b2_gbm`(= A4) |
| `b3_hybrid.py` | **B3 (본선)** IDW 예측을 피처로 넣고 정적 피처와 함께 학습한다. B2가 B1에 진 이유는 두 모델이 쓰는 정보가 달라서였다 — 주장은 「보간 + 공간 피처가 보간보다 낫다」여야 한다. 주야는 나눠 학습하지 않고 `is_night`을 피처로 준다(분리 학습은 판단 근거로만 남긴다). IDW는 `p=1, k=50` 고정(B1 중첩 튜닝에서 5 fold 중 4개가 고른 값). `--probe` | `preds/b3_hybrid` · `b3_noshade`(음영·SVF 제외) · `b3_split`(주야 분리) |
| `grid_infer.py` | **25m 격자 추론.** 평가는 fold별 모델로, 지도는 **전체 센서로 다시 학습한** 모델로 만든다. 시간 피처(태양 위치·`shadow_margin`·`is_shadow`)는 `scripts/build_train.py`와 같은 식으로 만든다. 분위 GBM q10/q50/q90과 녹지 개입 시나리오 `cf_park`를 함께 낸다. 기본 날짜는 폭염일 2026-08-06과 평온일 2026-07-15이고, 열대야 구간까지 덮도록 00시~다음날 09시 34시각을 계산한다. `--probe` · `--days` | `grid_pred_<날짜>.parquet` |
| `counterfactual.py` | **개입 반사실.** 학습된 모델에 조건만 바꿔 다시 예측한다. 한 피처만 바꾸면 면적 합이 100%를 넘으므로 녹지가 늘면 포장면·건물이 줄고 최근접 녹지 거리도 바뀌는 **일관된 시나리오**로 묶는다. 치환값이 학습 분포 밖인 비율을 함께 보고하고, 교란 조건이 비슷한 센서 쌍의 **매칭 대조**로 모델 효과를 점검한다. 인과 식별이 아니다. 쿨루프는 대응 피처가 없어 계산하지 않는다. `grid_infer`가 시나리오 정의를 가져다 쓴다. `--probe` | 콘솔 출력 |
| `forecast_infer.py` | **예보 다운스케일링.** 기상청 단기예보 중위를 `t_reg` 자리에 넣고 25m 격자 궤적을 만든다. 미래에는 이웃 관측이 없으므로 **IDW를 뺀 모델**을 쓴다. `--probe` · `--hours` | `forecast_grid.parquet` |
| `forecast_sensors.py` | **전향 검증.** 예보로 센서 위치를 미리 예측해 두고, 관측이 쌓이면 `--verify`로 대조한다. 전체 센서 학습(시간 외삽만)과 fold별 학습(시간·공간 이중 hold-out)을 나란히 내서 공간 성능이 얼마나 낙관적이었는지 드러낸다 | `forecast_sensors.parquet` |
| `probe_stidw.py` | **프로브** 시간 이력 피처가 B3를 더 내리는가. 이웃 센서의 과거 IDW(lag 1·2·24h)와 광역값 변화율만 넣는다 — test 센서 자신의 과거는 격자 셀에 없고, 지점 hold-out이 그 시간 축 누수를 못 잡으므로 넣지 않는다. 결과는 구별 불가(`t(4)=−0.35`). `preds/`와 `metrics.parquet`은 건드리지 않는다. `--fold` · `--paired` | `probe_stidw.parquet` |

### `model/` — 조건부 궤적 분포

```
Δ(셀, 24시간) = μ(c)  +  D(c) · z,     z ~ t_ν(0, R)
               B3 예측   조건부 척도    24시각 상관 R + 두꺼운 꼬리 ν
```

| 파일 | 역할 | 산출 |
|---|---|---|
| `traj_probe.py` | **검증 2 — 궤적 잔차가 다봉인가.** 표본이 크면 BIC가 거의 항상 k≥2를 고르므로 BIC 하나로 판정하지 않는다. PCA + GMM(BIC/ICL), 성분 간 평균 차이를 ℃로 잰 값, 파생량 재현을 함께 보고, 이분산이 다봉처럼 보이는지 날짜별 표준화 잔차로 다시 확인한다. 결론은 다봉이 아니라 **분산 혼합** → Flow Matching 대신 다변량 t를 쓴다. `--nsim` | `traj_probe.parquet` |
| `traj_dist.py` | 분포 추정. 척도 `D`는 B3와 같은 5-fold·피처·IDW로 분위 GBM을 out-of-fold 학습해 `(q90 − q10) / 2.563`으로 얻는다(격자 쪽 `grid_infer`와 같은 정의라 그대로 재사용된다). 표준화 잔차로 24×24 상관 `R`을 구하고 자유도 `ν`를 프로파일 최대우도로 추정한다. `--probe` · `--reuse-scale` | `traj_dist.npz` · `preds/b3_scale.parquet` |
| `calibrate.py` | **conformal 보정.** 점수 `E = |y − μ| / σ`로 곱셈형 배수 `Q`를 잡는다. 예측이 전부 out-of-fold이므로 fold f를 평가할 때 나머지 4개 fold를 보정 집합으로 쓴다(leave-one-fold-out). σ 구간별로 따로 보정하는 **Mondrian** 변형과 폭염 축을 포함하고, CRPS는 정렬 표본 공식으로 계산한다. `--sample-rows` | `calib.npz` |
| `grid_traj.py` | **격자 궤적 샘플.** 셀마다 24시간 궤적 30개를 뽑아 시각별 분위와 **파생량 분포**(예: 33℃ 초과 시간)를 낸다. 파생량은 궤적의 비선형 함수라 시각별 분위를 이어 붙여서는 나오지 않는다. 분위 GBM 구간을 그대로 쓰면 명목 80%가 실제 60.3%라서 Mondrian 보정을 거친다. 열대야(18시~다음날 09시)는 상관행렬이 달력일 기준이라 아직 넣지 않았다. `--probe` · `--days` · `--no-hourly` | `grid_traj_<날짜>.parquet` · `grid_traj_q_<날짜>.parquet` |

### `metrics/` — 지표 · 진단 · 그림

| 파일 | 역할 | 산출 |
|---|---|---|
| `evaluate.py` | **지표 하네스.** `preds/*`만 읽고 모델 내부를 모른다. 슬라이스는 전체 · 폭염 시각 · 주/야 · 센서 희소 자치구 · 녹지비·SVF·최근접 건물거리 5분위. 각 칸은 5-fold 평균 ± 표준편차. `--verify`는 B0a MAE가 독립 계산한 `mean(|Δ|)`와 일치하는지 대조한다. `--models` | `metrics.parquet` |
| `accuracy_table.py` | 정확도 표 3개 — MAE 슬라이스, 분포 지표(CRPS · 적중률 · 구간 폭), 피처 분위별 MAE. 결정론적 예측의 CRPS는 MAE와 같으므로 점추정 모델이 자동으로 기준선이 된다. 지역유형 범주가 아니라 연속 피처 분위로 본다(범주 내부가 이질적이다). `--samples` | `accuracy_table.parquet` |
| `trajectory.py` | 궤적 파생량 오차 — 열대야 지속시간 · 일최고 발생시각 · 33℃ 초과 시간을 점추정 궤적에서 센다. 구간이 비면 파생량이 왜곡되므로 관측 완결성 기준(하루 20시간 이상 등)을 건다. `--models` | `trajectory.parquet` |
| `traj_score.py` | 파생량을 **분포로** 평가 — CRPS와 80% 구간 적중률. 점추정 궤적에서 센 값은 `f(E[X]) ≠ E[f(X)]` 때문에 체계적으로 치우친다. `--draws` | `traj_score.parquet` |
| `treg_sensitivity.py` | **광역값이 틀리면 얼마나 망가지는가.** `t_reg`는 기준점이자 입력 피처라 감도가 자명하지 않다. 계통 편차와 시각별 무작위 오차를 **시각 단위로** 넣고(행마다 넣으면 평균에서 상쇄된다), IDW를 잃는 비용을 따로 분리한다. 학습은 참값, 추론만 오차값. `--probe` | `treg_sensitivity.parquet` |
| `shadow_probe.py` | **검증 1 — 400m 창 안에서 기온이 그림자를 따라가는가.** 창을 좁히면 반경 600m 집계 피처가 거의 상수가 되어 동네 차이를 통제한다. 모델이 음영을 피처로 쓰므로 이것은 독립 검증이 아니라 **메커니즘 확인**이다. `--day` · `--win` · `--hour` | `shadow_probe.parquet` |
| `globe_check.py` | 흑구온도로 복사 부하(`gt − tp`)를 직접 검증하려 한 시도. 야간에도 −2.75℃이고 주야 차이가 0.00℃라 **복사에 반응하지 않는 데이터**로 판정했다 — 검증 불가 | `globe_check.parquet` |
| `siting.py` | **외삽 정량 + 다음 센서 위치.** 정적 피처 22개를 센서 분포로 표준화한 공간에서 격자 셀 → 최근접 센서 거리를 재고, 우선순위를 외삽 정도 × 인구로 매긴다. 최적 설계가 아니라 어디가 비어 있는지만 보인다. `--topn` | `siting.parquet` |
| `fan_extract.py` | 팬차트용 셀 몇 개의 평균 `t_hat`과 보정 척도 `sigma_cal`만 뽑는다. 원본 격자 궤적 파일이 수백 MB라 그림 머신으로 옮기지 않으려고 둔 단계다 | `fan_cells.parquet` |
| `make_figures.py` | 그림 1~9, 11 (그림 10은 `scripts/capture_demo.py`). 한글 폰트가 필요하다. 인자로 그림 번호를 주면 그것만 그린다 | `figures/*.png` |

---

## `data/metric/` 산출물

| 파일 | 만드는 곳 | 내용 |
|---|---|---|
| `splits.json` | `eval/build_splits` | fold별 센서 목록 + 항상 학습 센서 |
| `regional.parquet` | `eval/build_splits` | 시각 × fold 광역 기준값 `t_reg` |
| `preds/<모델>.parquet` | `baseline/b0~b3` | `sn, ts, y_pred` — **절대 기온** |
| `preds/b3_scale.parquet` | `model/traj_dist` | `sn, ts, q10, q90, sigma` — out-of-fold 척도 |
| `metrics.parquet` | `metrics/evaluate` | long 형식: `model, fold, slice_kind, slice_val, n, mae, rmse, bias` |
| `traj_dist.npz` | `model/traj_dist` | 상관 `R` · 자유도 `ν` 등 |
| `calib.npz` | `model/calibrate` | conformal 배수 `Q` (수준별, Mondrian 구간별) |
| `grid_pred_<날짜>.parquet` | `baseline/grid_infer` | `gx, gy, ts, t_hat, q10, q50, q90, cf_park` |
| `grid_traj_<날짜>.parquet` · `grid_traj_q_<날짜>.parquet` | `model/grid_traj` | 셀별 파생량 분포 · 시각별 분위 |
| `forecast_grid.parquet` · `forecast_sensors.parquet` | `baseline/forecast_*` | 예보 다운스케일링 · 전향 검증 |
| 그 밖의 `<이름>.parquet` | 같은 이름의 `metrics/` · `eval/` · `model/` 파일 | 각 진단의 결과 표 |

---

## 주의

- **`grid_pred`의 `q10`/`q90`을 불확실성으로 그대로 쓰지 않는다.** 명목 80%의 실제 적중률이 60.3%다. `calib.npz`를 거친 값을 쓴다
- **conformal 적중률(90% → 90.0%)은 실황 앵커 조건에서만 성립한다.** 학습 기간 안에서 보정했기 때문에, 예보로 미래를 돌린 전향 검증에서는 80% 명목 구간이 52.9%(편향 보정 후)였다
- **예보를 앵커로 쓰면 병목은 모델이 아니라 `t_reg`다.** 전향 검증에서 예보 앵커 편향이 −1.670℃(야간 −2.78℃)였다
- **`make_figures`는 한글 폰트가 있는 머신에서 돌린다.** 폰트가 없으면 경고만 내고 라벨이 깨진 채 그려진다
- **`counterfactual`의 효과는 500m 블록 단위로 쓴다.** 25m 셀 단위에서는 부호 역전이 30.4%라 불안정하다
