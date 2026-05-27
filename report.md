# InF FR1 Indoor Positioning — Bias-Corrected Weighted Least Squares (BC-WLS)

## 1. 모티베이션 & 인트로

(TODO: 중간발표까지의 실험 결과와 거기서 본 알고리즘 아이디어가 도출된 흐름)

본 프로젝트는 18개 기지국이 측정한 RTT(d_hat)로부터 Indoor Factory FR1 환경에서 사용자의 2D 위치를 추정한다. 제공된 학습 데이터 700명의 측정값과 실제 위치를 분석한 결과, 다음과 같은 핵심 관찰을 얻었다.

| 지표 | 값 |
|------|-----|
| 측정거리 최댓값 | 약 307 m |
| 방의 기하학적 최대 거리 (대각선) | 약 126 m |
| 측정거리 - 실제거리 평균 | +15.9 m |
| 측정거리 - 실제거리 표준편차 | 20.2 m |
| 5% 분위 / 95% 분위 | -0.9 m / +53.6 m |

즉, RTT 측정값은 평균적으로 약 16m 길게 측정되며 분포가 강하게 비대칭이다. 이는 indoor factory 환경의 multipath 및 NLOS 전파 특성에서 비롯한 잘 알려진 현상으로, 신호가 직진 경로를 잡지 못한 채 벽·구조물에 반사·회절하면서 도달 시간이 늘어나기 때문이다.

이 관찰로부터 다음 아이디어를 도출하였다: **18개 BS 측정을 모두 동일하게 신뢰하는 일반적 LS 측위는 NLOS BS 쪽으로 추정값을 끌어당겨 큰 오차를 낳는다. 따라서 각 BS의 측정값에 포함된 NLOS bias 자체를 별도로 추정하여 보정한 후 LS를 수행하면 측위 정확도를 크게 개선할 수 있다.**

본 알고리즘 BC-WLS (Bias-Corrected Weighted Least Squares) 는 다음 네 단계로 구성된다:
1. 18개 BS 측정에 대한 closed-form LS로 초기 위치 `p₀` 추정
2. `p₀`에 기반하여 각 BS에 대한 8차원 feature 추출
3. 작은 MLP가 각 BS의 NLOS bias를 예측
4. 보정된 거리값과 residual 기반 가중치로 weighted nonlinear LS 수행

## 2. 알고리즘 설명

### 2.1 초기 위치 추정 — 선형화 Closed-Form LS

i번째 기지국의 측정 거리 dᵢ와 위치 pbsᵢ 사이에는 ||p - pbsᵢ||² = dᵢ² 의 관계가 이상적으로 성립한다. 이를 전개하면 ||p||² - 2 p · pbsᵢ + ||pbsᵢ||² = dᵢ² 이며, 임의의 기준 기지국(ref) 식을 다른 모든 식에서 빼면 비선형 항 ||p||²이 소거되어 다음 선형 식을 얻는다.

-2 p · (pbsᵢ - pbsref) = dᵢ² - dref² - ||pbsᵢ||² + ||pbsref||²

i = 1, ..., 17에 대해 이를 모아 행렬형 Ap = b 로 표현하고 표준 LS로 풀어 초기 추정값 p₀을 얻는다. 이는 폐형식 해이므로 매우 빠르며, NLOS bias가 존재해도 평균적으로 방의 중앙 근처로 추정값을 떨어뜨려 이후 단계의 좋은 시작점이 된다.

### 2.2 Per-BS Feature 추출

(TODO: 본인이 추가/수정한 feature가 있으면 반영)

각 BS i에 대해 다음 8차원 feature 벡터를 구성한다.

| # | Feature | 의미 |
|---|---------|------|
| 1 | dᵢ_hat | 원시 RTT 측정값 |
| 2 | dᵢ_pred = ‖p₀ - pbsᵢ‖ | 초기 추정에서 본 기하 거리 |
| 3 | residᵢ = dᵢ_hat - dᵢ_pred | 부호 있는 잔차 |
| 4 | \|residᵢ\| | 잔차 크기 |
| 5 | median(resid) | 18개 잔차의 중앙값 (글로벌 컨텍스트) |
| 6 | MAD(resid) | 중앙값 기반 강건 산포 |
| 7 | pbsᵢ.x | BS x 좌표 |
| 8 | pbsᵢ.y | BS y 좌표 |

5·6번 feature를 모든 BS에 동일하게 부여함으로써 MLP가 "다른 BS들과 비교했을 때 이 BS만 유난히 큰 잔차인지" 같은 상대적 판단을 할 수 있게 한다.

### 2.3 MLP를 통한 Bias 예측

8차원 입력 → 은닉층 64 → 은닉층 64 → 스칼라 출력의 단순 MLP 사용 (활성함수 ReLU). 출력 b̂ᵢ는 i번째 BS 측정값에 포함된 NLOS bias의 추정값이며, 학습 시 사용한 라벨은 b_true,ᵢ = dᵢ_hat - ‖pgt - pbsᵢ‖ 이다. 손실 함수는 SmoothL1 (Huber loss)을 사용하여 극단적 NLOS 측정에 학습이 휘둘리지 않도록 한다.

학습 데이터는 700명 사용자 × 18 BS = 12,600 (user, BS) 쌍이며, 사용자 단위로 80:20 train:val 분할하여 같은 사용자의 다른 BS 측정이 양쪽 split에 섞이지 않도록 한다.

### 2.4 최종 측위 — Bias-Corrected Weighted LS

각 BS의 보정 거리를 dᵢ_corr = max(dᵢ_hat - b̂ᵢ, 0.1) 로 정의한다 (음수 거리 방지). 이후 p₀에서 본 dᵢ_corr와의 새로운 잔차 nresᵢ = |dᵢ_corr - ‖p₀ - pbsᵢ‖| 로부터 가중치 wᵢ = 1/(1 + nresᵢ)를 부여하고, Levenberg-Marquardt를 이용해 다음 목적함수를 최소화한다.

Σᵢ wᵢ · (‖p - pbsᵢ‖ - dᵢ_corr)²

해 p̂이 본 알고리즘의 최종 출력이다.

## 3. Agent AI 활용 방안

(TODO: 본인이 실제로 활용한 내용에 맞추어 작성)

예시 — 본인이 한 일과 AI가 한 일을 구분해서 작성:
- **데이터 탐색 및 NLOS bias 발견**: 본인이 산점도, 히스토그램, BS별 잔차 분석 수행
- **알고리즘 설계**: 본인이 "bias를 직접 예측해서 빼는" 핵심 아이디어 결정. AI(예: Claude)는 그 아이디어의 구체화 (feature 후보 brainstorm, MLP 구조 의견) 보조
- **코드 구현**: AI가 초기 boilerplate 생성 → 본인이 검증·수정 → 디버깅 및 성능 튜닝은 본인 주도
- **레퍼런스 조사**: 본인이 IEEE/3GPP 문헌 검색

## 4. 결과 도출 & 디스커션

### 4.1 평가 방식

학생 제공 700명을 사용자 단위로 5-fold cross-validation 하여 측정한 결과는 다음과 같다.

| 방식 | 평균 오차 (m) | 중간값 오차 (m) | 95% 오차 (m) |
|------|--------------|----------------|--------------|
| Closed-form LS (uniform weight) | TODO | TODO | TODO |
| Nonlinear LS + Huber loss | TODO | TODO | TODO |
| **BC-WLS (본 알고리즘)** | **TODO** | **TODO** | **TODO** |

### 4.2 Fairness 디스커션

(TODO: 본인의 평가 방식이 공정한가에 대한 논의)
- baseline과 본 알고리즘이 동일한 입력·동일 데이터 분할에서 비교되는가
- 평가 metric(평균/중간값/분위) 선택 근거
- hidden test set에 대한 일반화 보장 — 본 알고리즘이 feature를 "위치 자체"가 아닌 "초기 추정에 대한 상대값"으로 구성했으므로 hidden 사용자에도 transfer 기대됨

### 4.3 장점 / 단점 / Future Work

장점 (TODO 본인 시각으로 보강):
- 물리 모델(LS)과 학습 기반 보정을 결합하여 학습 데이터가 적어도 안정적
- 모델이 가벼워 실시간 측위에 적합
- NLOS 영향이 큰 환경일수록 개선폭이 큼

단점:
- 초기 추정 p₀이 크게 왜곡되면 feature 품질이 떨어져 bias 예측이 부정확해질 수 있음 → 향후 iterative refinement 적용 가능
- (TODO)

Future work:
- p̂으로 다시 feature를 만들어 bias 예측을 1~2회 반복하는 iterative scheme
- BS subset selection (top-k LOS) 과 결합
- (TODO)

## 5. Reference

(TODO: 참고한 논문 추가. 참고가 없다면 본 섹션 삭제 가능)

예시 형식:
- [1] Author, "Title", Journal, Year.
- 본 논문 대비 본 알고리즘의 차이점: 위 §2에 기재.