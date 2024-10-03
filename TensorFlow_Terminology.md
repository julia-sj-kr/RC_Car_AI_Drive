----------------
### 손실함수

딥러닝 및 머신러닝에서 모델이 예측한 값과 실제 값 사이의 차이를 측정하는 함수입니다. `모델의 성능을 평가`하기 위해 사용되며, 학습 과정에서 이 값을 최소화하는 것이 목표입니다. 즉, 손실 함수는 모델이 얼마나 잘못 예측했는지를 수치로 표현해줍니다.
- __모델 평가__: 모델이 예측한 결과와 실제 결과 간의 차이를 측정합니다. 이 차이가 크다면 모델이 제대로 학습되지 않았음을 의미합니다.
- __학습의 목표__: 딥러닝 모델을 학습할 때 손실 함수는 매우 중요한 역할을 하며, 최적화 알고리즘(예: 경사 하강법, Adam Optimizer)은 이 손실 함수의 값을 최소화하여 모델의 성능을 개선합니다. 손실 함수가 최소화되면 모델의 예측 정확도는 증가합니다.
- __손실 함수의 종류__:
   - 회귀(Regression) 문제에서 사용하는 손실 함수(MeanSquaredError(), MeanAbsoluteError())
   - 분류(Classification) 문제에서 사용하는 손실 함수(BinaryCrossentropy(), CategoricalCrossentropy())
- __TensorFlow에서 손실 함수 사용 예시__
  ```
  import tensorflow as tf
  
  # 이진 분류 문제에서 사용할 예시 손실 함수
  loss_fn = tf.keras.losses.BinaryCrossentropy()
  
  # 실제 값 (예: 레이블)
  y_true = [0, 1, 0, 1]
  
  # 예측 값 (예: 모델의 출력)
  y_pred = [0.1, 0.9, 0.2, 0.8]
  
  # 손실 계산
  loss = loss_fn(y_true, y_pred)
  print('Loss:', loss.numpy())
  ```
  - 출력  
  ```
  Loss: 0.16425234
  ```
  이 예제에서 모델이 예측한 값과 실제 값의 차이를 이진 교차 엔트로피로 측정한 결과, 손실 값이 0.164로 계산되었습니다. 이 값이 작아질수록 모델이 더 정확하게 예측하고 있음을 나타냅니다.

----------------
### 주요메서드  
TensorFlow의 메서드는 여러 모듈로 나뉘어 있으며 여기서는 주로 자주 사용되는 메서드들과 주요 모듈을 소개하겠습니다.
#### Tensor 메서드
- tf.constant() : 상수 텐서를 생성합니다.
- tf.Variable() : 변수 텐서를 생성합니다. 학습 과정에서 값이 변하는 파라미터(예: 가중치, 편향)를 정의할 때 사용됩니다.
- tensor.numpy() : 텐서를 NumPy 배열로 변환합니다.
- tf.convert_to_tensor() : NumPy 배열 또는 리스트를 Tensor로 변환합니다.
#### 연산 관련 메서드
- tf.add() : 두 텐서를 더합니다.
- tf.subtract() : 두 텐서를 뺍니다.
- tf.multiply() : 두 텐서를 곱합니다.
- tf.matmul() : 두 텐서의 행렬 곱셈을 수행합니다.
- tf.reduce_mean() : 텐서의 평균을 구합니다.
- tf.reduce_sum() : 텐서의 합을 구합니다.
#### 신경망 모델 구축을 위한 메서드 (Keras API)
- tf.keras.Sequential() : 신경망 모델을 정의할 때 사용합니다.
- tf.keras.layers.Dense() : 완전 연결층을 정의합니다.
- model.compile() : 모델을 컴파일합니다. 손실 함수와 옵티마이저를 설정할 수 있습니다.
- model.fit() : 모델을 학습시킵니다.
- model.evaluate() : 모델의 성능을 평가합니다.
- model.predict() : 새로운 입력 데이터에 대한 예측을 수행합니다.
#### 자동 미분 및 최적화 관련 메서드
- tf.GradientTape() : 그래디언트(미분)를 계산하는 데 사용됩니다.
- optimizer.apply_gradients() : 계산된 그래디언트를 적용하여 모델의 가중치를 업데이트합니다.
#### 데이터 처리 관련 메서드
- tf.data.Dataset.from_tensor_slices() : 텐서에서 데이터셋을 생성합니다.
- dataset.batch() : 데이터셋을 배치로 묶습니다.
- dataset.shuffle() : 데이터셋을 섞습니다.
#### 모델 저장 및 불러오기 관련 메서드
- model.save() : 학습된 모델을 저장합니다.
- tf.keras.models.load_model() : 저장된 모델을 다시 불러옵니다.
#### GPU/TPU 관련 메서드
- tf.config.list_physical_devices('GPU') : 사용 가능한 GPU 장치를 확인합니다.
- tf.device() : 특정 장치에서 연산을 수행합니다.
#### 랜덤 관련 메서드
- tf.random.normal() : 정규 분포를 따르는 난수를 생성합니다.
- tf.random.set_seed() : 난수 생성기의 시드를 설정합니다.

이 외에도 TensorFlow에는 매우 많은 메서드들이 있으며, 이를 통해 딥러닝 모델을 구축, 학습, 최적화할 수 있습니다. TensorFlow 공식 문서에서는 각 메서드에 대한 상세한 설명과 예제를 제공합니다.

----------------
