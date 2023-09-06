# 2023 제3회 CJ 대한통운 미래기술 챌린지 
### 물류 센터의 출고 검수 시간 단축을 위해 비전을 활용한 스마트 검수 알고리즘 개발

#### 일정
1. 참가신청 : 2023.06.05(월) ~ 06.30(금)
2. 데이터 제공 : 2023.07.03(월)
3. 과제 수행 : 2023.07.03(월) ~ 2023.08.15(화)
4. 1차 평가 : 2023.08.21(월)
5. 1차 결과 발표 : 2023.08.25(금)
6. 온라인 PT : 2023.09.05(화)
7. 최종 결과 발표 : 2023.09.05(화)

#### 결과
본선진출 / 최종 수상 실패

#### 배경 및 필요성
* 출고 물품 검수 시간 단축

#### 목적
Object Detection 모델을 이용하여 정확한 검출을 통해 검수 시간 단축

#### 개발환경
* 인공지능 모델
  * Python 3.10.12
  * Cuda : 11.8
  * Pytorch : 2.0.1

#### 데이터
* 제공 데이터
  * Train : 단일 객체에 대한 이미지 (대략 3000 x 3000 resolution) 
  * Validation : 실제 물품들이 적재된 카트 이미지 (대략 1024 x 1024 resolution)
* 외부 데이터
  * 데이터 불균형 해소(상대적으로 구하기 쉬운 라면 Box 위주로 크롤링 진행)

#### 시스템 구현
* 필요한 전처리 정의
  * 논문을 통한 Train 이미지의 배경 Noise 제거 및 확대 (Rembg 라이브러리 활용)
* 기존의 Object Detection 방식으로는 학습 X
  → Two-Stage Detector 개발
    1. Pre-Train Detector(Detic) 모델을 통해 개별 Instance Detection
    2. Detection Box Crop
    3. DINOv2 모델을 통해 개별 Instance에 대한 Train Label 학습
       * Train 이미지 간 바라보는 View의 차이가 매우 심함
         → CrossEntropy Loss 대신
         → "ArcFace Loss" 기반 유사도 학습  
    4. Train Data를 통해 추출된 Embedding Vector DB 생성
    5. Embedding Vector DB 기반으로 Inference 시 KNN으로 Class 예측
    6. 음료수의 경우 묶음 단위로 예측을 해야 함
       → Instance를 Grouping 하는 Post-processing 진행

#### 결론
```
실제 Cart 이미지를 입력으로 받았을 때, 학습된 물품에 대해서는 Bounding Box 및 Class를 제공
```

#### 필요한 개선 사항
1. 1차 평가 시 39 / 53 (73%)의 매우 준수한 성능을 보여줬으나
  Iou > 0.5 기준이 적용되고 나서 26 / 53 (49%)의 낮은 성능이 되었음
  → Post-processing의 빈약함을 보여줌
  → Post-processing 고도화 필요 (Wegithed Boxes Fusion)
2. False Positive (모델은 객체가 있다고 판단한 지점이 Training data가 아닐 때)가 많음
   → 이는 추후 Unsupervised Anomaly Detection 기법인 Isolation Forest 등 모델 활용
3. 모델이 상당히 무거움(Inference Time : 15s(average))
   → DINOv2 distillation을 통해 성능은 유지되면서, 파라미터 수를 줄이는 기법 제안
