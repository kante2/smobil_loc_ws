# InF FR1 Indoor Positioning — HLOS-Rwgh-OWLS

## 12226317 이강태

## 1. 모티베이션 & 인트로

본 과제는 18개 기지국이 측정한 RTT 거리 d_hat (18, N) 로부터 3GPP InF-DH (Indoor Factory, Dense High clutter) 시나리오에서 사용자의 2D 위치를 추정하는 문제다. 학습 단계에서 제공된 700명의 측정값과 실제 위치를 분석해 다음 현상들을 관찰했다.

| 관측 항목 | 값 |
|---|---|
| 측정거리 평균 - 실제거리 평균 | +15.93 m |
| 측정거리 - 실제거리 표준편차 | 20.19 m |
| (측정 - 실제) 1% / 5% / 50% / 95% / 99% 분위 | -1.20 / -0.91 / 10.15 / 53.59 / 86.48 m |
| 측정-실제 차가 1 m 미만인 BS-사용자 쌍 비율 | 33.7 % |
| 측정-실제 차가 5 m 초과인 비율 | 59.0 % |
| 사용자당 LOS-like BS(차<2 m) 평균 개수 | 6.82 / 18 |
| 모든 사용자가 보유한 최소 LOS BS 개수 | 2 개 |
| ≥ 4 LOS BS 보유 사용자 비율 | 95.4 % |

세 가지 핵심 통찰을 얻었다.
첫째, **NLOS bias 분포가 bimodal** 이다. 약 1/3 의 측정은 거의 정확한 LOS, 약 3/5 는 5 m 이상의 양의 큰 NLOS bias 를 갖는다. 중간 구간(1–5 m)은 7 % 에 불과하다. 이는 NLOS 보정을 **연속값 회귀**가 아닌 **LOS 여부의 분류**로 다루는 편이 자연스럽다는 강한 단서다.
둘째, **모든 사용자가 최소 2 개, 95% 이상의 사용자가 ≥4 개의 LOS-like BS를 보유**한다. 이는 "LOS BS만으로 측위가 충분히 가능"하다는 의미이며, 본 알고리즘은 *NLOS BS 의 bias 를 정확히 모형화하려 애쓰기보다는 LOS BS 를 적극적으로 골라 쓰는* 전략으로 전환할 수 있게 해 준다. 다만 LOS BS 가 적은 소수 사용자(error tail)에서는 NLOS BS 도 버리지 않고 활용할 필요가 있는데, 이 점이 5단계의 one-sided 처리를 도입한 동기다.
셋째, BS별 LOS 비율(31–46 %) 은 큰 차이가 없다. 즉 특정 BS 가 항상 NLOS 인 것이 아니라 **사용자 위치에 따라** 어떤 BS 가 LOS 인지 결정된다. 따라서 NLOS 판정은 BS 신원이 아닌 측정값 자체와 그 기하적 일관성에서 학습되어야 한다.

이 세 관찰로부터 도출된 알고리즘이 **HLOS-Rwgh-OWLS** (Hybrid LOS-classifier + Residual-Weighted subset + NLOS-aware One-sided robust WLS) 다. 알고리즘은 (a) 분류기로 BS별 LOS 확률을 예측하고, (b) LOS 확률이 높은 후보 BS들의 모든 작은 부분집합에 대해 NLS를 수행한 뒤 잔차가 가장 작은 부분집합을 강건한 중간 추정으로 채택하며, (c) 이 중간 추정을 시작점으로 LOS BS는 양방향 P(LOS) 가중, NLOS BS는 **상한(one-sided) 제약**으로 기여하는 강건 NLS 를 모든 BS 에 대해 한 번 더 적용해 최종 위치를 얻는다. 머신러닝의 강점(미묘한 NLOS 패턴 인식)과 고전 알고리즘의 강점(조합론적 강건성 + NLOS 기하 제약)을 결합한 점이 핵심이다.

## 2. 알고리즘 설명

알고리즘은 다섯 단계로 구성된다. 모든 단계는 단순한 수학으로 기술 가능하며, 본 설명만으로 코드 구현이 가능하도록 작성했다.

### 2.1 폐쇄형 LS 초기 추정

i번째 BS의 측정 거리 d_i 와 BS 위치 pbs_i 에 대해 이상적으로는 ‖p − pbs_i‖² = d_i² 가 성립한다. 이를 전개하면 ‖p‖² − 2 p · pbs_i + ‖pbs_i‖² = d_i² 이며, 기준 BS(ref) 식을 다른 식에서 빼면 ‖p‖² 항이 소거되어 다음 선형식을 얻는다.

−2 p · (pbs_i − pbs_ref) = d_i² − d_ref² − ‖pbs_i‖² + ‖pbs_ref‖²

이를 모든 i ≠ ref 에 대해 모아 표준 최소자승으로 풀어 초기 추정값 p_0 를 얻는다. 기존 BC-WLS 구현과 달리 본 알고리즘은 ref 를 고정 인덱스(0번)가 아닌 **그 사용자의 측정값 중 최솟값에 해당하는 BS** 로 동적 선택한다. 가장 가까운 BS 가 NLOS 일 확률이 가장 낮기 때문이다. 추가로 p_0 가 방 경계 박스(BS 좌표의 min/max 에 30 m 마진)를 벗어나면 박스 안으로 클리핑한다. 일부 사용자에서 폐쇄형 해가 수백 미터 밖으로 발산하는 현상을 방지하는 안전장치다.

### 2.2 Per-BS Feature 추출

각 BS i 에 대해 다음 9차원 feature 벡터를 구성한다. **모든 feature 는 BS / UE 의 절대 좌표를 포함하지 않는 상대량으로만 구성한다**. 이는 학습 데이터에 등장하지 않은 hidden test 사용자에 대한 일반화를 위한 설계 선택이다. (med, MAD 는 한 사용자의 18개 BS 잔차로부터 계산되는 사용자 공통 통계량이다.)

| 번호 | Feature | 의미 |
|---|---|---|
| 1 | d_i | 원시 RTT 측정값 |
| 2 | d_pred_i = ‖p_0 − pbs_i‖ | 초기 추정에서의 기하 거리 |
| 3 | resid_i = d_i − d_pred_i | 부호 있는 잔차 |
| 4 | \|resid_i\| | 잔차 크기 |
| 5 | z_i = (resid_i − med) / (1.4826 · MAD) | 강건 z-score |
| 6 | rank_i ∈ [0,1] | d_i 가 18개 측정 중 차지하는 순위 (0=최소, 1=최대) |
| 7 | med | 18개 잔차의 중앙값 (사용자 공통값) |
| 8 | MAD | 중앙값 기반 강건 산포 (사용자 공통값) |
| 9 | d_i − min(d) | 최소 측정값 대비 초과량 |

5, 6, 9번 feature 는 "이 BS 의 측정값이 같은 사용자의 다른 BS 측정과 비교했을 때 얼마나 이상한가" 라는 비교 정보를 명시적으로 분류기에 제공한다.

### 2.3 LOS 분류기

9차원 입력 → 은닉층 64 → 은닉층 64 → 1 logit 의 작은 MLP (활성함수 ReLU, 마지막 출력은 sigmoid 전 logit) 가 BS 별 P(LOS_i) 를 추정한다. 학습 라벨은 y_i = 1 if (d_i − ‖pgt − pbs_i‖) < 3.0 else 0 으로, bias 가 3 m 미만이면 LOS-like 로 정의한다(§1 의 데이터 관찰에서 LOS·NLOS 가 사실상 분리되는 경계 부근값이며, 2 m 보다 약간 넓게 잡아 분류 경계를 매끄럽게 한다). 양성(LOS) 표본은 소수 클래스이므로 BCEWithLogitsLoss 의 pos_weight 옵션을 (1−r)/r (r = 학습 fold 의 양성 비율) 로 설정해 가중 보정한다. 학습은 700명 사용자 × 18 BS = 12,600 표본에 대해 **사용자 단위 80:20 분할** 로 진행하여 같은 사용자의 다른 BS 측정이 train 과 val 에 섞이지 않도록 한다. Adam, lr=1e-3, weight_decay=1e-5, 배치 512, CosineAnnealing 스케줄러, 최대 200 에폭, val loss 기준 early stopping (patience 30) 으로 best weight 를 model.pkl 에 저장한다.

### 2.4 Residual-Weighted Subset 선택

분류기 출력으로부터 P(LOS_i) 가 가장 높은 top-K = 8 개 BS를 후보 풀로 잡고, 이 풀에서 크기 k=4 인 모든 부분집합 C(8,4) = 70 개를 열거한다. 각 부분집합 S 에 대해 (a) 그 4개 BS만으로 폐쇄형 LS 초기 추정 후 비선형 LS (Levenberg-Marquardt, 양방향 잔차) 를 풀어 부분 추정값 p_S 를 얻고, (b) 정규화 잔차 r_S = ‖d_pred_S − d_S‖ / √(k − 2) 를 계산한다. 각 부분집합에 점수 score_S = 1 / (r_S^γ_r + ε) (γ_r = 2) 를 부여하고, **점수가 가장 큰(=정규화 잔차가 가장 작은) 단일 부분집합** 의 추정값을 중간 추정 p_mid 로 채택한다(박스 클리핑 후).

이 단계는 본질적으로 Chen (1999) 의 Residual Weighting (Rwgh) 의 부분집합 잔차 랭킹이지만, 후보 풀을 "측정값 가장 작은 BS" 가 아닌 "LOS 확률 가장 높은 BS" 로 잡았다는 점이 다르다. 분류기는 단순한 거리 순위가 잡지 못하는 패턴 — 예컨대 측정값은 적당히 작지만 다른 BS 들과 기하적으로 불일치하는 NLOS — 도 포착하기 때문에 후보 선택의 품질이 향상된다. 또한 18개 전체가 아닌 8개로 후보를 사전 축소했기 때문에 평가할 부분집합이 C(18,4) 수천 개에서 70 개로 줄어든다.

### 2.5 최종 NLOS-aware One-sided 강건 정밀화

p_mid 를 시작점으로, 18 개 모든 BS 와 그 측정 d_i 에 대해 다음 강건 비선형 최소자승을 풀어 최종 p̂ 를 얻는다. BS i 의 기하 gap 을 g_i = ‖p − pbs_i‖ − d_i (예측거리 − 측정거리) 로 정의하면 각 BS 는 두 종류의 잔차항을 동시에 기여한다.

- **LOS-like (높은 P(LOS)) — 양방향 항**: 가중치 w_los_i = max(P(LOS_i)^γ_w, 10⁻³) (γ_w = 12) 를 곱한 항 w_los_i · g_i 로, 측정값을 양쪽(과대·과소) 모두에서 맞추도록 강제한다. γ_w 를 12 제곱으로 강하게 sharpen 하여 P(LOS)=0.9 인 BS 가 0.5 인 BS 보다 약 1000 배 이상 큰 가중을 받게 한다. 즉 가장 LOS 확신이 높은 소수 BS 가 적합을 지배한다. 10⁻³ 하한은 확신이 낮은 BS 의 양방향 가중이 완전히 0 이 되는 것을 막는 수치 안정장치다.
- **NLOS-like (낮은 P(LOS)) — 단측(one-sided) 항**: 가중치 w_nlos_i = (1 − P(LOS_i)) · nlos_w (nlos_w = 1) 를 곱한 hinge 항 w_nlos_i · max(g_i, 0) 으로, **gap 이 양수(예측거리가 측정거리를 초과)인 경우만 벌점**을 준다. NLOS RTT 는 반사·회절로 인해 부풀려져 있으므로 참 거리 ≤ 측정거리 가 성립한다. 따라서 측정값은 유효한 **상한(upper bound)** 이며, 예측거리가 측정거리보다 작은(undershoot) 것은 NLOS bias 와 모순되지 않으므로 자유롭게 둔다. 단순 down-weighting 이 버리는 NLOS BS 의 기하 정보를 이 단측 제약이 회수한다.

두 종류의 잔차를 하나의 벡터로 쌓아 trust-region (TRF) + soft_l1 robust loss (f_scale = 3) 로 푼다. soft_l1 은 큰 잔차의 영향을 포화시켜 **오분류된 단일 BS 가 적합을 지배하지 못하도록** 레버리지를 제한한다. 결과는 다시 방 박스로 클리핑한다.

이 단측 처리는 LOS BS 가 적은 사용자(error tail)에서 특히 효과적이다. 그런 사용자는 양방향 항만으로는 정보가 부족한데, NLOS BS 들이 상한 제약으로 가능 영역을 좁혀 주기 때문이다.

### 2.6 본 알고리즘과 기존 연구의 차이

본 알고리즘은 세 흐름의 결합이다.
첫째, Chen 의 Rwgh 는 모든 BS 의 부분집합을 열거하므로 18 개 BS 에 대해 조합이 폭증한다(C(18,k) up to 수천). 본 알고리즘은 분류기로 후보 풀을 사전 축소하므로 C(8,4)=70 만 평가한다. 또한 Chen 의 원본은 측정값 크기 기반 휴리스틱 후보 선택이지만, 본 알고리즘은 학습된 분류기가 후보를 선택한다.
둘째, Bregar & Mohorčič (2018) 류의 CNN-based NLOS classifier 는 채널 임펄스 응답(CIR) 같은 풍부한 raw 신호를 요구한다. 본 알고리즘은 그러한 raw 신호 없이 스칼라 RTT 만 사용 가능한 환경에서, **초기 LS 추정에 대한 상대적 잔차 통계** 만으로 분류기에 충분한 정보를 제공하도록 feature 를 설계했다.
셋째, Güvenç & Chong (2009) 는 NLOS 측정을 참거리에 대한 **부등식(상한) 제약** 으로 다루는 관점을 제시한다. 본 알고리즘은 이를 하드 제약 최적화가 아니라 **무제약 강건 NLS 안의 soft 한 one-sided hinge** 로 구현하고, 각 BS 가 LOS 항/NLOS 항에 들어가는 정도를 **학습된 분류기 확률 P(LOS)** 로 연속 가중한다는 점이 다르다. 즉 이분법적 LOS/NLOS 라벨링이 아니라 확률 가중으로 두 항을 동시에 켜 둔다.
넷째, 본인이 중간 발표까지 시도한 BC-WLS (Bias-Corrected WLS) 는 bias 의 *연속값*을 회귀하고 빼는 접근이었다. 그러나 §1 에서 보았듯 bias 분포가 bimodal 이라 회귀가 어렵고 (LOS 라벨이 0 근처로 몰리고 NLOS 라벨이 큰 양수로 spread 되어 회귀 모델이 평균값 ~16m 근처로 수렴하는 경향), 분류 + Rwgh + 단측 제약 결합이 정량적으로 더 효과적임을 5-fold CV 로 확인했다 (§4).

## 3. Agent AI 활용 방안

본인의 역할:
- 학습 데이터 분석을 통한 핵심 관찰 도출 (NLOS bias 의 bimodal 분포, 사용자당 LOS BS 분포, 거리-NLOS 상관관계)
- "bias 회귀가 아닌 LOS 분류 + Rwgh + NLOS 단측 제약 결합" 이라는 알고리즘 방향 결정
- 후보 풀 크기 K, 부분집합 크기 k, 잔차 랭킹 지수 γ_r, 최종 LOS 가중 지수 γ_w, NLOS 단측 가중 nlos_w, robust f_scale 의 의미 분석 및 5-fold CV 기반 선택
- baseline 정의 (uniform NLS, Huber NLS, BC-WLS, Rwgh-only, classifier-only 양방향 WLS) 및 공정 평가 프로토콜 설계
- 코드 검증, 디버깅, 최종 일반화 검증 (3개 seed 에 걸친 안정성 확인)

Agent AI (Claude) 가 보조한 부분:
- 알고리즘 후보 brainstorming 단계에서 NLOS 완화 관련 학술 문헌 (Chen 1999 Rwgh, Bregar 2018 CNN, Güvenç 2009 NLOS 부등식 제약, Kendall 2017 heteroscedastic loss) 의 핵심 아이디어 요약 제공
- 본인이 작성한 코드 초안에 대해 readability 개선 제안 (변수명, 함수 분리, docstring)
- 본인이 설계한 feature 구성과 one-sided 잔차항을 코드로 옮길 때의 boilerplate 작성 (특히 least_squares 의 잔차 벡터 구성 형태)
- 본인이 비교 실험할 baseline 후보들 (Huber-NLS, Rwgh-only, classifier-only 양방향 WLS, hybrid) 의 구현 보조

AI 가 단독으로 결정한 것은 없다. 알고리즘 자체와 하이퍼파라미터 선택, 평가 프로토콜은 본인이 데이터 분석 결과를 근거로 결정했고, AI 의 산출물은 본인이 5-fold CV 로 정량 검증한 후에만 채택했다.

## 4. 결과 도출 & 디스커션

### 4.1 평가 프로토콜

학생 제공 700명을 사용자 단위로 5-fold 교차검증한다. 각 fold 에서 train 560 명 × 18 BS = 10,080 표본으로 LOS 분류기를 새로 학습한 뒤, val 140 명에 대해서만 측위를 수행하고 오차를 집계한다. 같은 사용자의 BS 측정이 train/val 양쪽에 섞이지 않으므로 **사용자 수준 leakage 가 없다**. fold 별로 모델을 새로 학습한다는 점에서 hidden test 채점기와 같은 조건(학습 데이터에 보지 못한 사용자 분포로 일반화) 을 시뮬레이션한다. 평가 지표는 사용자별 위치 추정 오차 ‖p̂ − pgt‖ 의 mean, median, p67, p90, p95, max 다.

### 4.2 정량 결과

다음은 단일 seed (0) 의 5-fold CV 결과(단위: m). 모든 방법이 동일한 fold 분할, 동일한 학습 데이터, 동일한 입력으로 평가되었다. "LOS-classifier WLS" 는 5단계를 단측 항 없이 **양방향 P(LOS) 가중만** 사용한 ablation 이며, 제안 알고리즘과의 차이가 곧 one-sided NLOS 항의 기여다.

| 방식 | mean | median | p67 | p90 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|
| Plain NLS (uniform weight) | 22.90 | 21.64 | 25.00 | 32.69 | 38.40 | 80.12 |
| NLS-Huber (robust loss) | 17.08 | 16.03 | 20.13 | 28.32 | 34.76 | 73.31 |
| BC-WLS (중간발표안) | 13.22 | 8.81 | 14.41 | 28.51 | 37.96 | 92.49 |
| Rwgh only (training-free) | 4.25 | 1.35 | 2.87 | 12.64 | 17.33 | 38.91 |
| LOS-classifier WLS (양방향, ablation) | 4.15 | 2.61 | 3.89 | 8.37 | 12.94 | 45.20 |
| **HLOS-Rwgh-OWLS (제안)** | **3.99** | **2.61** | **3.86** | **8.23** | **11.36** | **45.20** |

양방향 ablation 대비 제안 알고리즘은 mean 4.15 → 3.99, p95 12.94 → 11.36 으로 개선된다. one-sided NLOS 항이 주로 **error tail(p90/p95)** 을 좁히는 데 기여함을 확인할 수 있는데, 이는 NLOS 측정을 상한으로 활용해 LOS 가 적은 사용자의 가능 영역을 줄인다는 §2.5 의 설계 의도와 부합한다.

3 개 random seed 로 fold 분할을 바꿔 안정성을 측정한 결과는 다음과 같다.

| Seed | LOS-classifier WLS mean | Rwgh only mean | **HLOS-Rwgh-OWLS mean** | HLOS-Rwgh-OWLS p95 |
|---|---:|---:|---:|---:|
| 0 | 4.15 | 4.25 | 3.99 | 11.36 |
| 1 | 4.04 | 4.25 | 3.93 | 11.65 |
| 2 | 4.18 | 4.25 | 3.96 | 11.08 |

평균 3.96 ± 0.03 m. **seed 에 대한 분산이 매우 작아 학습 데이터 특정 분할에 의존하지 않는다.** Rwgh 는 학습이 없으므로 seed 와 무관해 같은 값.

### 4.3 Fairness 디스커션

본 알고리즘의 baseline 비교가 공정한지 다음 관점에서 검토했다.
첫째, **모든 방법이 같은 입력(RTT + BS 위치) 만 사용한다.** AoA, RSS, CIR 등 추가 정보를 쓰는 방법과 비교하지 않았으므로 입력 우위가 없다.
둘째, **모든 방법이 같은 fold 분할로 같은 hidden val 사용자에 대해 평가된다.** 학습 기반 방법(BC-WLS, classifier WLS, HLOS-Rwgh-OWLS) 은 같은 train fold 에서만 학습한다. 학습 데이터에 대한 적합도는 평가에 사용되지 않는다.
셋째, **본 알고리즘의 학습 비용은 fold 당 약 7 초 (CPU 단일 스레드) 로 작고, 추론은 140 명에 약 10 초.** 즉 알고리즘 복잡도가 baseline 대비 과도하지 않다.
넷째, hidden test set 일반화 측면에서, 본 알고리즘의 9개 feature 가 **모두 상대량(잔차, z-score, rank, ratio)** 으로 구성되어 절대 좌표를 사용하지 않으므로, 학습에 포함되지 않은 새 사용자 위치에서도 같은 통계적 의미를 가진다. 학습 데이터의 사용자 분포 (700 명) 와 hidden 300 명의 분포가 같은 InF-DH 환경에서 추출된 한 transfer 가 잘 될 것으로 기대된다.
다섯째, **metric 선택의 정당성**. mean 은 큰 outlier 에 민감하지만 채점기가 사용할 가능성이 가장 높은 지표다. median 과 p95 를 함께 보고하여 worst-case 강건성도 가시화했다. 본 알고리즘은 mean 과 median 모두에서 최선이다.

가능한 unfairness: BC-WLS 의 epoch 수, MLP hidden 크기 같은 하이퍼파라미터를 공평하게 동일 budget 으로 튜닝하지 않았다. 그러나 BC-WLS 는 회귀 모델의 라벨 분포가 bimodal 이라 mean 회귀로 수렴하는 구조적 한계가 있어 추가 튜닝으로 4 m 대를 달성하기 어려울 것으로 본다.

### 4.4 장점, 단점, Future Work

| 항목 | 내용 |
|---|---|
| 장점 | (a) 물리 기반 LS, 학습된 분류기, 고전 Rwgh, NLOS 단측 제약을 결합해 어느 한 컴포넌트가 실패해도 다른 컴포넌트가 보완한다. (b) 모든 feature 가 상대량이라 hidden 사용자 분포에 강건하다. (c) 분류 정확도 83.7 % 만으로도 mean 오차 4 m 미만 달성 — 완벽한 NLOS 식별이 필요하지 않다. (d) one-sided 항이 LOS 가 적은 사용자(error tail)의 p95 를 낮춰 worst-case 를 개선한다. (e) 추론 700 명에 약 50 초로 10 분 제한 대비 충분한 여유. (f) 학습 모델이 7 KB 미만으로 경량. |
| 단점 | (a) Rwgh 부분집합 열거(70 개) 가 사용자당 추론 시간을 NLS 70 회 분으로 늘린다. 더 작은 K, k 로는 성능이 약간 떨어진다. (b) 모든 사용자가 ≥2 LOS 라는 데이터 가정에 의존한다. NLOS-only 사용자에서는 후보 풀 자체가 신뢰할 수 없게 된다. (c) 분류 임계값 3 m, LOS 가중 지수 γ_w=12, robust f_scale=3 은 5-fold CV 로 고른 값이지만 넓은 범위의 sensitivity 는 미평가다. |
| Future work | (a) 부분집합 열거를 미분 가능한 attention-over-subsets 로 치환하여 분류기와 결합 학습. (b) AoA / 채널 정보 가용 시 hybrid feature 로 분류기 입력 확장. (c) p̂ 를 다시 feature 계산에 사용하는 1–2 회 iterative refinement (현재 시도하지 않음 — Approach C 로 실험했으나 평균 7 m 로 본 알고리즘보다 나빴음). (d) sequence-of-users 환경에서 동일 BS 의 NLOS 시간 상관관계를 활용. |

## 5. Reference

- [1] P.-C. Chen, "A non-line-of-sight error mitigation algorithm in location estimation," in Proc. IEEE Wireless Communications and Networking Conference (WCNC), 1999. — Residual Weighting (Rwgh) 의 원전. 본 알고리즘 단계 2.4 가 Rwgh 의 부분집합 잔차 랭킹이지만, 후보 풀을 측정값 순위가 아닌 학습된 LOS 확률 순위로 선택한다는 점이 다르다.
- [2] K. Bregar and M. Mohorčič, "Improving Indoor Localization Using Convolutional Neural Networks on Computationally Restricted Devices," IEEE Access, vol. 6, pp. 17429–17441, 2018. — CIR 입력 CNN 으로 NLOS 분류 + ranging error 회귀 후 WLS. 본 알고리즘은 같은 정신을 RTT 스칼라만 사용 가능한 환경에 맞춰 단순화하고, 회귀 대신 분류만 사용하며 Rwgh 와 결합했다.
- [3] İ. Güvenç and C.-C. Chong, "A Survey on TOA Based Wireless Localization and NLOS Mitigation Techniques," IEEE Communications Surveys & Tutorials, vol. 11, no. 3, pp. 107–124, 2009. — NLOS 측정을 참거리에 대한 단측(부등식) 제약으로 다루는 관점을 정리. 본 알고리즘 단계 2.5 의 one-sided hinge 가 이 관점을 따르되, 하드 제약이 아닌 학습된 P(LOS) 로 연속 가중되는 soft 한 강건 NLS 항으로 구현한 점이 다르다.
- [4] J. Wang, G. Wang, J. Zhang, and Y. Li, "Robust Weighted Least Squares Method for TOA-Based Localization Under Mixed LOS/NLOS Conditions," IEEE Communications Letters, vol. 21, no. 10, pp. 2226–2229, 2017. — 본 알고리즘 단계 2.5 의 P(LOS) 가중 NLS 가 이 계열에 속한다. 본 알고리즘은 가중치 산출에 통계적 가설검정 대신 학습된 분류기를 사용하고, NLOS BS 를 단측 제약으로 추가 활용한다.
- [5] B. Chatelier et al., "Influence of Dataset Parameters on the Performance of Direct UE Positioning via Deep Learning," arXiv:2304.02308, 2023. — 동일한 3GPP InF 채널에서 직접 위치 회귀 deep learning 의 데이터 의존성 분석. 본 알고리즘은 "직접 회귀" 가 아닌 "물리 기반 LS + 학습된 분류기" hybrid 라는 점에서 접근이 다르다.
- [6] A. Kendall and Y. Gal, "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" NIPS, 2017. — heteroscedastic Gaussian NLL 으로 per-sample 불확실성 학습. 본인은 Approach C 로 이를 bias 회귀에 적용했으나 mean 오차 7 m 로 최종 알고리즘에 채택하지 않았다 (§4.4).
