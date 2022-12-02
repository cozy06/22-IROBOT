# 텐서플로 임포트
import tensorflow as tf
# 매트플랏립 임포트
import matplotlib.pyplot as plt
# 사이킷런의 데이터 임포트
from sklearn.model_selection import train_test_split
# 텐서플로 케라스의 Sequential 임포트
from tensorflow.keras import Sequential
# 텐서플로 케라스의 Dense 임포트
from tensorflow.keras.layers import Dense

# MNIST 데이터 불러오기
(x_train_all, y_train_all), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# 이미지 확인
plt.imshow(x_train_all[0], cmap='gray')
plt.show()

# 클래스 네임 붙이기
class_names = ['티셔츠/윗도리', '바지', '스웨터', '드레스', '코트', '샌들', '셔츠', '스니커즈', '가방', '앵클부츠']

# 클래스 네임 확인
print(class_names[y_train_all[0]])

# 훈련 세트와 검증 세트 고르게 나누기
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, \
                                                  stratify=y_train_all, test_size=0.2, random_state=42)

x_train = [1,2,3,4,5,6]

# 타깃을 원-핫 인코딩으로 변환
y_train_encoded = tf.keras.utils.to_categorical(y_train)
y_val_encoded = tf.keras.utils.to_categorical(y_val)

# Sequential 클래스의 객체 모델 만들기
model = Sequential()

# 은닉층을 모델에 추가
# 은닉층의 유닛 개수 100개
# 은닉층의 활성화 함수 시그모이드 함수
# 입력 데이터 크기 28x28 = 784
model.add(Dense(100, activation='sigmoid', input_shape=(1,)))

# 출력층을 모델에 추가
# 출력층의 유닛 개수 10개
# 출력층의 활성화 함수 소프트 맥스 함수
model.add(Dense(10, activation='softmax'))

# 최적화 알고리즘 : 경사 하강법
# 손실 함수 : 크로스 엔트로피 손실 함수
# metrics 매개변수는 훈련 과정 기록으로 정확도를 남기기 위해 추가
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델을 훈련하고 예측
# 40번의 에포크 동안 훈련
# 측정한 값들을 History 클래스 객체에 담아 반환
history = model.fit(x_train, y_train_encoded, epochs=40, \
                    validation_data=(x_val, y_val_encoded))

# 손실 및 정확도 그래프 그리기
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()